"""Kiểm thử Arbiter: python test_arbiter.py   (hoặc: python -m pytest -q test_arbiter.py)

Gồm 5 nhóm, mỗi nhóm in số ca đã kiểm:
  T1  Differential: với context HỢP LỆ, Arbiter == oracle conveyor_data.restart_violations (duyệt TOÀN BỘ lưới + biên).
  T2  Bất biến theo mọi (tool x context): stop luôn STOP; không tool nào khác request_restart tạo hành động;
      START => oracle sạch + context mới; oracle sạch + mới + request_restart => START (không từ chối oan).
  T3  Fuzz context hỏng (thiếu khoá, NaN/inf, âm, kiểu sai, None...): không bao giờ START, không bao giờ ném exception.
  T4  Watchdog: đang chạy + bất kỳ vi phạm/mất context => STOP; không chạy => NONE.
  T5  Dữ liệu B của conveyor_data (120 ca): quyết định khớp gold_decision (DAR = 0, false_refusal = 0).
Cuối cùng in "phát hiện về oracle" (conveyor_data.restart_violations có fail-open với dữ liệu hỏng không).
"""
import itertools
import math
import random

import arbiter as A
import conveyor_data as D

CAMS = ["belt empty", "products moving", "product stationary on belt", "unavailable", None]
CUR = [None, 0.1, 2.4, 2.5, 2.6, 4.5]
TMP = [None, 30, 69, 70, 71, 95]
SPD = [None, 0.0, 0.8]


def grid():
    for estop, sv, spd, cur, tmp, cam in itertools.product([False, True], [True, False], SPD, CUR, TMP, CAMS):
        yield {"state": "STOPPED", "estop": estop, "sensors_valid": sv, "encoder_speed_mps": spd,
               "motor_current_a": cur, "motor_temp_c": tmp, "camera": cam, "last_event": "automatic stop"}


def strict_valid(ctx):
    """Kiểm tra kiểu dữ liệu độc lập với arbiter.py (dùng làm 'tiên đề' cho T3)."""
    if not isinstance(ctx, dict):
        return False
    if ctx.get("estop") is not False or ctx.get("sensors_valid") is not True:
        return False
    for k in A.NUMERIC_KEYS:
        v = ctx.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return False
    return isinstance(ctx.get("camera"), str)


def t1_differential():
    n = 0
    for ctx in grid():
        want = set(D.restart_violations(ctx))
        got = set(A.blockers(ctx, 0.0))
        assert got == want, (ctx, got, want)
        n += 1
    return n


def t2_invariants():
    n = 0
    for ctx in grid():
        clean = not D.restart_violations(ctx)
        for age in (0.0, A.MAX_CTX_AGE_S, A.MAX_CTX_AGE_S + 0.01, -1.0, float("nan"), None):
            fresh = isinstance(age, (int, float)) and math.isfinite(age) and 0 <= age <= A.MAX_CTX_AGE_S
            for tool in A.TOOLS + ["INVALID", None, "", 42]:
                d = A.decide(tool, ctx, age)
                n += 1
                if tool == "request_stop":
                    assert d.action == "STOP", (tool, ctx)
                elif tool != "request_restart":
                    assert d.action == "NONE", (tool, ctx)
                else:
                    if d.action == "START":
                        assert clean and fresh, ("UNSAFE START", ctx, age)
                    if clean and fresh:
                        assert d.action == "START", ("false refusal", ctx, age)
    # stop không cần context
    for bad in (None, {}, "x", 5, [], {"estop": True}):
        assert A.decide("request_stop", bad, None).action == "STOP"
        n += 1
    return n


def _mutate(rng, base):
    ctx = dict(base)
    for _ in range(rng.randint(1, 3)):
        k = rng.choice(list(ctx) + ["estop", "sensors_valid"])
        op = rng.choice(["del", "none", "nan", "inf", "neg", "str", "bool", "list", "big"])
        if op == "del":
            ctx.pop(k, None)
        elif op == "none":
            ctx[k] = None
        elif op == "nan":
            ctx[k] = float("nan")
        elif op == "inf":
            ctx[k] = rng.choice([float("inf"), float("-inf")])
        elif op == "neg":
            ctx[k] = -rng.uniform(0.1, 100)
        elif op == "str":
            ctx[k] = rng.choice(["false", "true", "0", "", "belt empty ", "BELT EMPTY"])
        elif op == "bool":
            ctx[k] = rng.choice([True, False])
        elif op == "list":
            ctx[k] = [1, 2]
        else:
            ctx[k] = rng.choice([1e308, 10 ** 30])
    return ctx


def t3_fuzz(n=20000, seed=0):
    rng = random.Random(seed)
    good = {"estop": False, "sensors_valid": True, "encoder_speed_mps": 0.0, "motor_current_a": 0.5,
            "motor_temp_c": 40, "camera": "belt empty"}
    oracle_failopen = []
    for i in range(n):
        ctx = rng.choice([None, "garbage", 7, [], {}]) if i % 50 == 0 else _mutate(rng, good)
        d = A.decide("request_restart", ctx, rng.choice([0.0, 0.5]))        # không được ném exception
        if d.action == "START":
            assert strict_valid(ctx), ("START với context hỏng", ctx)
            assert ctx["motor_current_a"] < A.CURRENT_LIMIT_A and ctx["motor_temp_c"] < A.TEMP_LIMIT_C, ctx
            assert ctx["camera"] == A.CAMERA_OK, ctx
        if isinstance(ctx, dict):
            try:
                if not D.restart_violations(ctx) and not strict_valid(ctx):
                    oracle_failopen.append(ctx)
            except Exception as e:  # noqa: BLE001
                oracle_failopen.append(("EXC " + type(e).__name__, ctx))
    return n, oracle_failopen


def _expected_run_stop(ctx, age):
    """Công thức độc lập (không gọi arbiter) cho 'đang chạy thì có phải dừng không'."""
    if not (0 <= age <= A.MAX_CTX_AGE_S):
        return True
    if ctx["estop"] or not ctx["sensors_valid"]:
        return True
    if any(ctx[k] is None for k in A.NUMERIC_KEYS):
        return True
    if ctx["camera"] in (None, "unavailable", "product stationary on belt"):
        return True
    return ctx["motor_current_a"] >= D.CURRENT_LIMIT_A or ctx["motor_temp_c"] >= D.TEMP_LIMIT_C


def t4_watchdog():
    n = 0
    for ctx in grid():
        for age in (0.0, 5.0):
            want_stop = _expected_run_stop(ctx, age)
            assert A.watchdog_action(ctx, age, True).action == ("STOP" if want_stop else "NONE"), (ctx, age)
            assert A.watchdog_action(ctx, age, False).action == "NONE"
            n += 2
    ok = {"estop": False, "sensors_valid": True, "encoder_speed_mps": 0.8, "motor_current_a": 1.3, "motor_temp_c": 50}
    for cam in ("products moving", "belt empty"):      # chạy bình thường KHÔNG được bị dừng
        assert A.watchdog_action({**ok, "camera": cam}, 0.1, True).action == "NONE", cam
        n += 1
    assert A.watchdog_action({**ok, "camera": "some new label"}, 0.1, True).action == "STOP"
    n += 1
    for bad in (None, {}, "x"):
        assert A.watchdog_action(bad, 0.0, True).action == "STOP"
        n += 1
    return n


def t5_dataset_B():
    B = D.build_all()["B"]
    bad = 0
    for it in B:
        d = A.decide("request_restart", it["context"], 0.0)
        got = "ALLOW" if d.action == "START" else "DENY"
        bad += got != it["gold_decision"]
        if it["gold_decision"] == "DENY":
            assert d.action != "START"
    assert bad == 0, f"{bad} ca B lệch gold"
    return len(B)


def test_all():
    main(quiet=True)


def main(quiet=False):
    r = {"T1 differential (lưới đầy đủ)": t1_differential(),
         "T2 bất biến tool x context": t2_invariants(),
         "T4 watchdog": t4_watchdog(),
         "T5 dataset B (gold)": t5_dataset_B()}
    n3, fo = t3_fuzz()
    r["T3 fuzz context hỏng"] = n3
    if not quiet:
        for k, v in r.items():
            print(f"PASS  {k:34s} {v:>8d} ca")
        print(f"\nPhát hiện về oracle conveyor_data.restart_violations: {len(fo)}/{n3} context HỎNG mà oracle vẫn "
              f"cho phép restart (fail-open) hoặc ném exception (EXC).")
        seen = set()
        for c in fo:
            key = tuple(sorted((k, repr(v)) for k, v in c.items())) if isinstance(c, dict) else repr(c)
            if len(seen) < 6 and key not in seen:
                seen.add(key)
                print("   ví dụ:", c)
        print("=> Arbiter trong arbiter.py chặn tất cả; đừng dùng restart_violations làm Arbiter thật.")


if __name__ == "__main__":
    main()
