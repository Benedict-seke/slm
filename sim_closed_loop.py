"""Mô phỏng VÒNG KÍN + FAULT INJECTION (phần mềm thuần, không cần board/camera/model).   python sim_closed_loop.py --all

Mô hình 3 lớp theo sơ đồ team:
  Plant  : băng tải (vận tốc, dòng, nhiệt) + lỗi chèn được (kẹt, quá tải, quá nhiệt, mất encoder, NaN, camera, E-stop)
  MCU    : STM32 — an toàn THỜI GIAN THỰC, ĐỘC LẬP với Linux/AI: quá dòng (debounce), quá nhiệt, mất encoder/NaN,
           kẹt (stall), E-stop, watchdog heartbeat từ Linux -> SAFE STOP; tự kiểm lại điều kiện trước khi nhận START
  Linux  : orchestrator — telemetry qua bridge, camera, arbiter.py (watchdog_action / decide), lệnh giọng nói

Đo: fault-to-stop (lệnh dừng + vận tốc về 0), lớp nào dừng, chẩn đoán sau dừng, restart qua Arbiter,
    START cưỡng bức (giả lập Arbiter bị lỗi), và số lần dừng nhầm/giờ khi KHÔNG có lỗi (soak).

!! Mọi tham số vật lý và NGƯỠNG yêu cầu bên dưới là GIẢ ĐỊNH của mô hình — kết quả chứng minh LOGIC an toàn,
   KHÔNG thay cho số đo thật trên UNO Q (chỉnh CFG, ghi nguồn từng tham số vào báo cáo).
"""
import argparse
import csv
import math
import random
from pathlib import Path

import arbiter as AR

CFG = dict(
    dt=0.01, v_nom=0.8, v_stopped=0.05,
    oc_ms=50, ot_ms=500, enc_ms=100, stall_ms=300, stall_grace_s=1.0,   # ngưỡng/debounce của MCU
    wd_ms=500, hb_ms=100,                                                # watchdog Linux heartbeat
    bridge_ms=5, tele_ms=50, tick_ms=50,                                 # bridge, telemetry, vòng lặp Linux
    cam_delay_s=1.1, cam_clear_s=0.3,                                    # độ trễ camera (lấy cỡ cv_blockage ~1.0-1.2 s)
    cam_debounce_ms=300,                                                 # Linux chịu mất camera ngắn trước khi dừng
    voice_fast_ms=500, voice_slm_ms=1500, tts_ttfa_ms=70,                # thay bằng số đo của bạn
    hb_coupled=True,    # True: heartbeat phát TRONG vòng kiểm tra an toàn (AI treo => mất heartbeat => watchdog dừng)
)
# Yêu cầu đặt trước (GIẢ ĐỊNH — đây là thứ phải thống nhất với team/giảng viên)
LIMIT_MS = lambda c: {"JAM": 300, "OVERLOAD": 200, "OVERHEAT": c["ot_ms"] + 150, "ENC_DROP": c["enc_ms"] + 150,   # noqa: E731
                      "NAN_CURRENT": c["oc_ms"] + 100, "ESTOP": 50, "CAM_DROP": c["cam_debounce_ms"] + 200}
TRUTH = {"JAM": "BLOCKAGE", "OVERLOAD": "OVERLOAD", "OVERHEAT": "OVERHEAT", "ENC_DROP": "SENSOR_FAULT",
         "NAN_CURRENT": "SENSOR_FAULT", "ESTOP": "ESTOP", "CAM_DROP": "SENSOR_FAULT"}
fin = lambda x: x is not None and math.isfinite(x)  # noqa: E731


class Plant:
    def __init__(s, rng, c):
        s.r, s.c = rng, c
        s.v, s.i, s.T, s.bump = 0.0, 0.05, 45.0, 0.0
        s.obstacle = s.overload = s.overheat = s.enc_fail = s.cam_fail = s.estop = s.nan_current = False
        s.obs_t = s.clear_t = None

    def step(s, run):
        c, dt = s.c, s.c["dt"]
        run = run and not s.estop
        tgt = c["v_nom"] if run else 0.0
        if run and s.obstacle:
            tgt = 0.03
        s.v += (tgt - s.v) * dt / (0.3 if tgt > s.v else 0.12)
        itgt = 0.05
        if run:
            itgt = 0.3 + s.v / c["v_nom"] + s.bump
            if s.obstacle:
                itgt = 3.5
            if s.overload:
                itgt = 3.0
        s.i += (itgt - s.i) * dt / 0.05
        if s.overheat and run:
            s.T += 4.0 * dt
        else:
            s.T += ((52.0 if run else 40.0) - s.T) * dt / 60.0

    def meas(s):
        i = float("nan") if s.nan_current else s.i + s.r.gauss(0, 0.03)
        return i, s.T + s.r.gauss(0, 0.2), (None if s.enc_fail else s.v + s.r.gauss(0, 0.01))

    def camera(s, t, glitch=False):
        if s.cam_fail or glitch:
            return "unavailable"
        if s.obstacle and t - s.obs_t >= s.c["cam_delay_s"]:
            return "product stationary on belt"
        if (not s.obstacle) and s.clear_t is not None and t - s.clear_t < s.c["cam_clear_s"]:
            return "product stationary on belt"
        return "products moving" if s.v > 0.1 else "belt empty"


class MCU:
    def __init__(s, c):
        s.c, s.run, s.reason, s.last_hb = c, False, None, 0.0
        s.cnt = dict(oc=0.0, ot=0.0, enc=0.0, stall=0.0)
        s.run_since, s.events, s.rejected = 0.0, [], []

    def _stop(s, t, reason):
        if s.run:
            s.run, s.reason = False, reason
            s.events.append((t, "STOP", reason))

    def _can_start(s, t, i, T, enc, estop):
        return (not estop and fin(i) and i < AR.CURRENT_LIMIT_A and fin(T) and T < AR.TEMP_LIMIT_C and fin(enc)
                and t - s.last_hb <= s.c["wd_ms"] / 1000)

    def step(s, t, i, T, enc, estop, cmds):
        c, dt = s.c, s.c["dt"]
        for kind, src, code in cmds:
            if kind == "HB":
                s.last_hb = t
            elif kind == "STOP":
                s._stop(t, code)
            elif kind == "START":
                if s._can_start(t, i, T, enc, estop):
                    s.run, s.reason, s.run_since = True, None, t
                    s.cnt = dict.fromkeys(s.cnt, 0.0)
                    s.events.append((t, "START", src))
                else:
                    s.rejected.append((t, src))
        if estop:
            s._stop(t, "ESTOP")
            return
        if not s.run:
            return
        k = s.cnt
        k["oc"] = k["oc"] + dt if (not fin(i) or i >= AR.CURRENT_LIMIT_A) else 0.0
        k["ot"] = k["ot"] + dt if (not fin(T) or T >= AR.TEMP_LIMIT_C) else 0.0
        k["enc"] = k["enc"] + dt if not fin(enc) else 0.0
        stalled = fin(enc) and enc < 0.3 * c["v_nom"] and t - s.run_since > c["stall_grace_s"]
        k["stall"] = k["stall"] + dt if stalled else 0.0
        if k["enc"] * 1000 >= c["enc_ms"]:
            s._stop(t, "ENC")
        elif k["oc"] * 1000 >= c["oc_ms"]:
            s._stop(t, "SENSOR" if not fin(i) else "OC")
        elif k["ot"] * 1000 >= c["ot_ms"]:
            s._stop(t, "OT")
        elif k["stall"] * 1000 >= c["stall_ms"]:
            s._stop(t, "STALL")
        elif t - s.last_hb > c["wd_ms"] / 1000:
            s._stop(t, "WATCHDOG")


def reason_code(reasons):
    r = set(reasons)
    if "ESTOP" in r:
        return "ESTOP"
    if r & {"SENSOR_FAULT", "CAMERA_UNAVAILABLE", "CAMERA_UNKNOWN", "NO_CONTEXT"}:
        return "SENSOR"
    if "BLOCKAGE" in r:
        return "BLOCKAGE"
    if "OVERCURRENT" in r:
        return "OC"
    if "OVERTEMP" in r:
        return "OT"
    if "STALE_CONTEXT" in r:
        return "STALE"
    return "OPERATOR" if "OPERATOR_REQUEST" in r else "OTHER"


def diagnose(ctx):
    """Chẩn đoán dựa trên LÝ DO DỪNG ĐƯỢC MCU CHỐT (không dựa số đo tức thời — sau khi dừng dòng/nhiệt đã đổi)."""
    r, cam = ctx.get("last_stop_reason"), ctx.get("camera")
    if ctx.get("estop") or r == "ESTOP":
        return "ESTOP"
    if r in ("ENC", "SENSOR"):
        return "SENSOR_FAULT"
    if r == "OT":
        return "OVERHEAT"
    if r in ("OC", "STALL", "BLOCKAGE"):
        return "BLOCKAGE" if cam == "product stationary on belt" else "OVERLOAD"
    if r == "OPERATOR":
        return "OPERATOR_STOP"
    return "UNKNOWN"


class Sim:
    def __init__(s, cfg=None, seed=0, duration=10.0, faults=(), voice=(), start_running=True, nominal=None):
        s.c = dict(CFG, **(cfg or {}))
        s.rng = random.Random(seed)
        s.plant, s.mcu = Plant(s.rng, s.c), MCU(s.c)
        s.dur, s.faults, s.voice_q = duration, sorted(faults), sorted((t, tool) for t, tool in voice)
        s.nominal = nominal or {}
        s.cmdq, s.teleq, s.pending_voice, s.log = [], [], [], []
        s.alive = s.bridge = s.ai_alive = True
        s.t_onset = None
        s.wait_overheat = False
        s.stop_event = s.t_vstop = None
        s.unsafe_linux_starts = 0
        s.forced = None
        s.pkt = None
        s.ctx = None
        s.cam_bad_since, s.cam_good = None, "belt empty"
        s.spurious = 0
        s.stopped_time = 0.0
        s.answers = []
        s.alert_t = None
        s.diag_samples = {}
        s.v_at = {}
        if start_running:
            s.mcu.run, s.mcu.last_hb, s.mcu.run_since = True, 0.0, -10.0
            s.plant.v, s.plant.i = s.c["v_nom"], 1.3
            s.plant.T = 52.0

    # ---- lỗi chèn
    def _fault(s, t, name, arg):
        p = s.plant
        s.log.append((t, f"FAULT {name}"))
        onset = True
        if name == "JAM":
            p.obstacle, p.obs_t = True, t
        elif name == "REMOVE_OBSTACLE":
            p.obstacle, p.clear_t, onset = False, t, False
        elif name == "OVERLOAD":
            p.overload = True
        elif name == "OVERHEAT":
            p.overheat, p.T, s.wait_overheat, onset = True, max(p.T, 66.0), True, False
        elif name == "ENC_DROP":
            p.enc_fail = True
        elif name == "NAN_CURRENT":
            p.nan_current = True
        elif name == "CAM_DROP":
            p.cam_fail = True
        elif name == "ESTOP":
            p.estop = True
        elif name == "ESTOP_RELEASE":
            p.estop, onset = False, False
        elif name == "LINUX_HANG":
            s.alive = False
        elif name == "SETUP_LINUX_HANG":
            s.alive, onset = False, False
        elif name == "BRIDGE_DOWN":
            s.bridge = False
        elif name == "SETUP_AI_HANG":
            s.ai_alive, onset = False, False
        elif name == "SETUP_BRIDGE_DOWN":
            s.bridge, onset = False, False
        elif name == "FORCE_START":
            s.cmdq.append((t, "START", "FORCED", None))
            s.forced = {"t": t, "accepted": None, "restop_ms": None, "dwell_ms": 0.0, "by": None}
            onset = False
        if onset and s.t_onset is None:
            s.t_onset = t

    def _send(s, t, kind, src, code=None):
        if s.bridge:
            s.cmdq.append((t + s.c["bridge_ms"] / 1000, kind, src, code))

    # ---- Linux
    def _linux_tick(s, t):
        c = s.c
        due = [x for x in s.teleq if x[0] <= t]
        s.teleq = [x for x in s.teleq if x[0] > t]
        if due:
            s.pkt = due[-1][1]
        if s.pkt is None:
            return
        p = s.pkt
        cam = s.plant.camera(t, glitch=s.rng.random() < s.nominal.get("cam_glitch_p", 0.0))
        if cam == "unavailable":
            s.cam_bad_since = t if s.cam_bad_since is None else s.cam_bad_since
            cam_eff = s.cam_good if (t - s.cam_bad_since) * 1000 < c["cam_debounce_ms"] else "unavailable"
        else:
            s.cam_bad_since, s.cam_good, cam_eff = None, cam, cam
        ctx = {"estop": p["estop"], "sensors_valid": fin(p["i"]) and fin(p["T"]) and fin(p["enc"]),
               "encoder_speed_mps": p["enc"], "motor_current_a": p["i"], "motor_temp_c": p["T"], "camera": cam_eff,
               "last_stop_reason": p["reason"], "state": "RUNNING" if p["run"] else "STOPPED"}
        s.ctx, age = ctx, t - p["t"]
        if not p["run"] and s.alert_t is None and s.t_onset is not None and t >= s.t_onset:
            s.alert_t = t + c["tts_ttfa_ms"] / 1000
        d = AR.watchdog_action(ctx, age, p["run"])
        if d.action == "STOP":
            s._send(t, "STOP", "LINUX", reason_code(d.reasons))
            s.log.append((t, f"LINUX auto-STOP {list(d.reasons)}"))
        while s.voice_q and s.voice_q[0][0] <= t:
            tv, tool = s.voice_q.pop(0)
            lat = c["voice_fast_ms"] if tool == "request_stop" else c["voice_slm_ms"]
            s.pending_voice.append((tv + lat / 1000, tool, tv))
        for item in [x for x in s.pending_voice if x[0] <= t]:
            s.pending_voice.remove(item)
            _, tool, tv = item
            dd = AR.decide(tool, ctx, age)
            if tool == "explain_stop":
                s.answers.append((t, diagnose(ctx)))
                s.log.append((t, f"VOICE why -> {diagnose(ctx)}"))
            elif dd.action == "START":
                s._send(t, "START", "LINUX", None)
                s.log.append((t, f"VOICE restart -> ALLOW (said at {tv:.2f}s)"))
            elif dd.action == "STOP":
                s._send(t, "STOP", "LINUX", "OPERATOR")
                s.log.append((t, "VOICE stop -> STOP"))
            else:
                s.log.append((t, f"VOICE {tool} -> {dd.action} {list(dd.reasons)}"))

    def run(s):
        c, dt, n = s.c, s.c["dt"], int(s.dur / s.c["dt"])
        nxt_tele = nxt_tick = nxt_hb = nxt_bump = 0.0
        faults = list(s.faults)
        for k in range(n):
            t = k * dt
            while faults and faults[0][0] <= t:
                _, name, arg = faults.pop(0)
                s._fault(t, name, arg)
            if s.nominal.get("bump_rate") and t >= nxt_bump:
                s.plant.bump = 0.4
                nxt_bump = t + s.rng.expovariate(s.nominal["bump_rate"])
                s._bump_end = t + 0.3
            if s.plant.bump and t >= getattr(s, "_bump_end", 0):
                s.plant.bump = 0.0
            s.plant.step(s.mcu.run)
            if s.wait_overheat and s.plant.T >= AR.TEMP_LIMIT_C:
                s.t_onset, s.wait_overheat = t, False
            i, T, enc = s.plant.meas()
            due = [x for x in s.cmdq if x[0] <= t]
            if due:
                s.cmdq = [x for x in s.cmdq if x[0] > t]
            cmds = [(a, b, cc) for _, a, b, cc in due]
            n_ev = len(s.mcu.events)
            s.mcu.step(t, i, T, enc, s.plant.estop, cmds)
            for ev in s.mcu.events[n_ev:]:
                p = s.plant
                if ev[1] == "START":
                    unsafe = p.obstacle or p.overload or p.estop or p.T >= AR.TEMP_LIMIT_C or p.enc_fail or p.nan_current
                    if ev[2] == "LINUX" and unsafe:
                        s.unsafe_linux_starts += 1
                    if ev[2] == "FORCED" and s.forced:
                        s.forced["accepted"] = True
                s.log.append((t, f"MCU {ev[1]} {ev[2]}"))
                if ev[1] == "STOP" and s.t_onset is not None and t >= s.t_onset and s.stop_event is None:
                    s.stop_event = (t, ev[2])
            if s.forced and s.forced["accepted"] is None and s.mcu.rejected and s.mcu.rejected[-1][1] == "FORCED":
                s.forced["accepted"] = False
            if s.forced and s.forced["accepted"]:
                if s.plant.i >= AR.CURRENT_LIMIT_A:
                    s.forced["dwell_ms"] += dt * 1000
                if s.forced["restop_ms"] is None and not s.mcu.run and t > s.forced["t"] + 0.1:
                    s.forced["restop_ms"] = (t - s.forced["t"]) * 1000
                    s.forced["by"] = s.mcu.reason
            if s.t_onset is not None and t >= s.t_onset and s.t_vstop is None and s.plant.v < c["v_stopped"]:
                s.t_vstop = t
            if k % 50 == 0:
                s.v_at[round(t, 1)] = s.plant.v
            if not s.mcu.run:
                s.stopped_time += dt
            for tag, dtm in (("early", 0.5), ("late", 2.5)):
                if s.stop_event and tag not in s.diag_samples and t >= s.stop_event[0] + dtm and s.ctx:
                    s.diag_samples[tag] = diagnose(s.ctx) if s.alive else "n/a (Linux chết)"
            if t >= nxt_tele:
                if s.bridge:
                    s.teleq.append((t + c["bridge_ms"] / 1000, {"i": i, "T": T, "enc": enc, "estop": s.plant.estop,
                                                                "run": s.mcu.run, "reason": s.mcu.reason, "t": t}))
                nxt_tele += c["tele_ms"] / 1000
            if s.alive:
                if t >= nxt_hb and (s.ai_alive or not c["hb_coupled"]):
                    s._send(t, "HB", "LINUX")
                    nxt_hb += c["hb_ms"] / 1000
                elif t >= nxt_hb:
                    nxt_hb += c["hb_ms"] / 1000
                if t >= nxt_tick:
                    if s.ai_alive:
                        s._linux_tick(t)
                    nxt_tick += c["tick_ms"] / 1000
            if not (s.alive and s.ai_alive):
                s.teleq.clear()
        return s

    def result(s):
        o = s.t_onset
        r = {"t_onset": o}
        if o is not None and s.stop_event:
            r.update(stop_cmd_ms=round((s.stop_event[0] - o) * 1000, 1), stop_by=s.stop_event[1])
        if o is not None and s.t_vstop is not None:
            r["stopped_ms"] = round((s.t_vstop - o) * 1000, 1)
        return r


# ============================================================ kịch bản & bảng
def story(cfg=None):
    """Chuỗi: chạy -> kẹt -> dừng -> hỏi lý do -> restart bị từ chối -> gỡ vật cản -> restart được phép -> lệnh dừng."""
    sim = Sim(cfg, seed=1, duration=24.0,
              faults=[(3.0, "JAM", None), (14.0, "REMOVE_OBSTACLE", None)],
              voice=[(8.0, "explain_stop"), (10.0, "request_restart"), (16.0, "request_restart"), (21.0, "request_stop")]).run()
    r = sim.result()
    starts = [t for t, k, _ in sim.mcu.events if k == "START"]
    stops = [(t, why) for t, k, why in sim.mcu.events if k == "STOP"]
    denied = [m for t, m in sim.log if "restart ->" in m and "NONE" in m]
    steps = []
    steps.append(("Kẹt -> MCU dừng", r.get("stop_cmd_ms", 1e9) <= LIMIT_MS(sim.c)["JAM"],
                  f"{r.get('stop_cmd_ms')} ms, lớp {r.get('stop_by')}; vận tốc về 0 sau {r.get('stopped_ms')} ms"))
    steps.append(("Cảnh báo bằng giọng", sim.alert_t is not None,
                  f"{(sim.alert_t - sim.t_onset) * 1000:.0f} ms sau lỗi (Linux thấy dừng + TTFA)" if sim.alert_t else "không có"))
    why = sim.answers[0][1] if sim.answers else None
    steps.append(("Hỏi lý do dừng", why == "BLOCKAGE", f"trả lời: {why}"))
    steps.append(("Restart khi còn vật cản bị từ chối", bool(denied) and not any(t < 14.0 for t in starts),
                  denied[0] if denied else "không có bản ghi từ chối"))
    ok_restart = [t for t in starts if t > 14.0]
    steps.append(("Gỡ vật cản -> restart được phép", bool(ok_restart),
                  f"START tại {ok_restart[0]:.2f}s (nói lúc 16.0s)" if ok_restart else "không START"))
    v_before_stop = sim.v_at.get(20.5, 0.0)
    steps.append(("Chạy lại bình thường (trước lệnh dừng cuối)", v_before_stop > 0.7, f"v tại 20.5 s = {v_before_stop:.2f} m/s"))
    last_stop = [t for t, why_ in stops if t > 20.9]
    steps.append(("Lệnh dừng bằng giọng", bool(last_stop),
                  f"dừng tại {last_stop[0]:.2f}s (nói lúc 21.0s, trễ giọng {sim.c['voice_fast_ms']} ms)" if last_stop else "không dừng"))
    steps.append(("Không START nguy hiểm", sim.unsafe_linux_starts == 0, f"unsafe START qua Arbiter = {sim.unsafe_linux_starts}"))
    return steps, sim


MATRIX_FAULTS = ["JAM", "OVERLOAD", "OVERHEAT", "ENC_DROP", "NAN_CURRENT", "ESTOP", "CAM_DROP"]
# (biến thể, lỗi chuẩn bị trước lỗi chính, ghi đè cấu hình)
VARIANTS = {"linux_ok": ([], {}),
            "ai_hung (heartbeat gắn vòng an toàn)": ([(-0.1, "SETUP_AI_HANG")], {}),
            "ai_hung (heartbeat TÁCH RỜI)": ([(-0.1, "SETUP_AI_HANG")], {"hb_coupled": False}),
            "linux_hung (chết hẳn)": ([(-0.1, "SETUP_LINUX_HANG")], {}),
            "bridge_down": ([(-0.1, "SETUP_BRIDGE_DOWN")], {})}


def matrix(cfg=None, tf=3.0):
    rows = []
    for fault in MATRIX_FAULTS:
        for var, (pre, over) in VARIANTS.items():
            faults = [(tf + dt_, nm, None) for dt_, nm in pre] + [(tf, fault, None)]
            sim = Sim(dict(cfg or {}, **over), seed=7, duration=tf + 4.5, faults=faults).run()
            r = sim.result()
            lim, note = LIMIT_MS(sim.c)[fault], ""
            if var != "linux_ok" and "TÁCH" not in var:
                lim = max(lim, sim.c["wd_ms"] + 150)   # lỗi chỉ Linux thấy / Linux chết: watchdog phải dừng
            if r.get("stop_cmd_ms") is None:           # lỗi chưa kịp thành hình vì watchdog đã dừng trước (vd quá nhiệt)
                ev = [e for e in sim.mcu.events if e[1] == "STOP" and e[0] >= tf]
                if ev:
                    r["stop_cmd_ms"], r["stop_by"] = round((ev[0][0] - tf) * 1000, 1), ev[0][2]
                    note = "watchdog dừng trước khi lỗi thành hình"
            if "TÁCH" in var and fault == "CAM_DROP":
                note = "camera chỉ Linux thấy; heartbeat vẫn phát => không ai dừng"
            rows.append({"fault": fault, "variant": var, "stop_cmd_ms": r.get("stop_cmd_ms"), "stopped_ms": r.get("stopped_ms"),
                         "stop_by": r.get("stop_by"), "limit_ms": lim,
                         "PASS": r.get("stop_cmd_ms") is not None and r["stop_cmd_ms"] <= lim,
                         "diag_early": sim.diag_samples.get("early", "-"), "diag_late": sim.diag_samples.get("late", "-"),
                         "truth": TRUTH[fault], "note": note})
    for var, pre in (("linux_hung", [(1.0, "LINUX_HANG")]), ("bridge_down", [(1.0, "BRIDGE_DOWN")])):  # mất liên lạc đơn thuần
        sim = Sim(cfg, seed=7, duration=5.0, faults=[(t_, n_, None) for t_, n_, _ in [(a, b, None) for a, b in pre]]).run()
        r, lim = sim.result(), sim.c["wd_ms"] + 150
        rows.append({"fault": "COMM_LOSS", "variant": var, "stop_cmd_ms": r.get("stop_cmd_ms"), "stopped_ms": r.get("stopped_ms"),
                     "stop_by": r.get("stop_by"), "limit_ms": lim, "PASS": r.get("stop_cmd_ms") is not None and r["stop_cmd_ms"] <= lim,
                     "diag_early": "-", "diag_late": "-", "truth": "UNKNOWN", "note": ""})
    return rows


def forced_start(cfg=None, tf=3.0):
    """Giả lập Arbiter BỊ LỖI (START cưỡng bức, bỏ qua Arbiter) khi lỗi vẫn còn: MCU có tự bảo vệ không?"""
    rows = []
    for fault in MATRIX_FAULTS:
        sim = Sim(cfg, seed=7, duration=tf + 6.0, faults=[(tf, fault, None), (tf + 2.5, "FORCE_START", None)]).run()
        f = sim.forced or {}
        rows.append({"fault": fault, "MCU_nhận_START": f.get("accepted"),
                     "dừng_lại_sau_ms": None if f.get("restop_ms") is None else round(f["restop_ms"]), "dừng_bởi": f.get("by"),
                     "dòng_quá_ngưỡng_ms": round(f.get("dwell_ms", 0)) if f.get("accepted") else 0,
                     "kết_luận": ("MCU chặn được" if f.get("accepted") is False else
                                  "MCU nhận rồi dừng lại" if f.get("restop_ms") is not None else "MCU nhận và KHÔNG dừng lại")})
    return rows


def soak(cfg=None, hours=1.0, glitch_p=1e-4):
    """Không có lỗi thật: tải đột biến (dòng) + camera chớp tắt. Đếm dừng nhầm/giờ; tự START lại sau 5 s qua Arbiter."""
    rows = []
    for deb in (0, 300):
        sim = Sim(dict(cfg or {}, cam_debounce_ms=deb), seed=11, duration=hours * 3600,
                  nominal={"bump_rate": 1 / 20, "cam_glitch_p": glitch_p})
        sim.voice_q = []
        # tự khởi động lại sau mỗi lần dừng nhầm: đặt lệnh restart định kỳ, Arbiter quyết định
        sim.voice_q = [(t, "request_restart") for t in range(5, int(hours * 3600), 5)]
        sim.run()
        stops = [e for e in sim.mcu.events if e[1] == "STOP"]
        rows.append({"cam_debounce_ms": deb, "giờ_mô_phỏng": hours, "dừng_nhầm": len(stops),
                     "dừng_nhầm_mỗi_giờ": round(len(stops) / hours, 1),
                     "lý_do": dict(sorted({r: sum(1 for e in stops if e[2] == r) for r in {e[2] for e in stops}}.items())),
                     "thời_gian_dừng_s": round(sim.stopped_time, 1), "unsafe_START_qua_Arbiter": sim.unsafe_linux_starts})
    return rows


def _print(title, rows):
    print(f"\n=== {title}")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        keys = list(rows[0])
        print(" | ".join(keys))
        for r in rows:
            print(" | ".join(str(r.get(k)) for k in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--out", default="results_sim")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    steps, _ = story()
    print("=== KỊCH BẢN VÒNG KÍN")
    for name, ok, detail in steps:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    m = matrix()
    _print("FAULT INJECTION (fault-to-stop, lớp dừng, chẩn đoán)", m)
    f = forced_start()
    _print("START CƯỠNG BỨC (Arbiter bị lỗi) — MCU tự bảo vệ?", f)
    s = soak(hours=a.hours)
    _print(f"SOAK {a.hours:g} giờ KHÔNG lỗi thật — dừng nhầm", s)
    for name, rows in (("matrix", m), ("forced_start", f), ("soak", s)):
        with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\nĐã lưu CSV vào {out}/")


if __name__ == "__main__":
    main()
