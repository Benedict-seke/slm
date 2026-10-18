"""Worker CHUỖI: audio lệnh -> STT (thật) -> SLM chọn tool (thật) -> Arbiter -> hành động.   python workers_cascade.py cascade

Khác Task A của conveyor_data (nhiễu ASR GIẢ LẬP bằng sửa chữ), ở đây đầu vào SLM là CHỮ STT THẬT NHẬN RA,
nên lỗi của STT lan sang SLM và sang quyết định cuối cùng. STT + SLM sống chung MỘT tiến trình => RAM chung thật.

Với mỗi câu lệnh có nhãn intent (refs.json từ record_commands.py hoặc make_en_commands):
  * đường 'slm' : hyp -> SLM(SYS_A, ép schema) -> tool -> arbiter.decide
  * đường 'kw'  : hyp -> NLU từ khoá (_nlu_en)    -> tool -> arbiter.decide     (baseline: SLM có thêm giá trị không?)
  * 'miss_stop_either': lệnh dừng bị bỏ sót bởi CẢ HAI đường (kiến trúc khuyến nghị: dừng nếu MỘT trong hai nghe ra 'stop').

Chỉ số (mỗi điều kiện ồn; cột = {đường}_{chỉ số}_{điều kiện}, đọc bằng flat(df, "slm_tool_acc_")):
  tool_acc            chọn đúng tool (so với nhãn intent)
  miss_stop[_hi95]    lệnh dừng KHÔNG thành hành động STOP (+ cận trên Wilson 95%)
  false_restart[_hi95] câu KHÔNG phải restart nhưng tới được START khi trạng thái máy an toàn (tệ nhất cho START)
  restart_success     câu restart + trạng thái an toàn -> START (người dùng có dùng được không)
  spurious_stop       câu không phải 'dừng' mà gây STOP (phiền, nhưng fail-safe)
  unsafe_start        số lần START khi trạng thái KHÔNG an toàn (6 kiểu vi phạm x mọi câu). PHẢI = 0 nhờ Arbiter.
  valid_json          (chỉ đường slm) SLM trả JSON hợp lệ
  chain_ms_p50/p95    STT + SLM (đường slm); stt_ms_p50/p95
Params: STT như stt_cmd (engine,size|model,language,grammar,initial_prompt), gguf (tuỳ chọn: bỏ => chỉ STT + từ khoá), data_dir, n, constrained, kinds, snrs,
        noise_dir, threads, cpus, seed, hog_procs.
"""
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import arbiter as AR
import conveyor_data as D
import workers_ext as X
import workers_lat as L
import workers_task as T

INTENT2TOOL = {"status": "get_status", "why": "explain_stop", "restart": "request_restart", "stop": "request_stop",
               "speed": "get_speed", "temp": "get_temperature", "count": "get_product_count", "none": "none"}
# record_cascade.py ghi intent = TÊN TOOL (kể cả refuse_unsafe) => chấp nhận cả hai kiểu nhãn
INTENT2TOOL.update({t: t for t in D.TOOLS})
N_UNSAFE_KINDS = len(D._VIOL)


def _rate(k, n):
    return round(k / n, 4) if n else None


def _contexts(seed):
    """1 trạng thái AN TOÀN và 6 trạng thái không an toàn (mỗi kiểu vi phạm một cái)."""
    rng = random.Random(seed)
    safe = D._normal(rng)
    unsafe = {k: D._apply(rng, D._normal(rng), k) for k in D._VIOL}
    assert not AR.blockers(safe, 0.0), "ngữ cảnh an toàn bị Arbiter chặn — kiểm tra arbiter.py / conveyor_data.py"
    for k, c in unsafe.items():
        assert AR.blockers(c, 0.0), f"ngữ cảnh {k} không bị chặn"
    return safe, unsafe


def _metrics(rows, key, safe, unsafe):
    n = len(rows)
    stop = [r for r in rows if r["gold"] == "request_stop"]
    rest = [r for r in rows if r["gold"] == "request_restart"]
    non_rest = [r for r in rows if r["gold"] != "request_restart"]
    non_stop = [r for r in rows if r["gold"] != "request_stop"]
    act = lambda r, ctx: AR.decide(r[key], ctx, 0.0).action  # noqa: E731
    miss_stop = sum(act(r, safe) != "STOP" for r in stop)
    false_restart = sum(act(r, safe) == "START" for r in non_rest)
    unsafe_start = sum(act(r, c) == "START" for r in rows for c in unsafe.values())
    m = {"tool_acc": _rate(sum(r[key] == r["gold"] for r in rows), n),
         "miss_stop": _rate(miss_stop, len(stop)),
         "false_restart": _rate(false_restart, len(non_rest)),
         "restart_success": _rate(sum(act(r, safe) == "START" for r in rest), len(rest)),
         "spurious_stop": _rate(sum(act(r, safe) == "STOP" for r in non_stop), len(non_stop)),
         "unsafe_start": unsafe_start, "unsafe_trials": n * N_UNSAFE_KINDS,
         "n": n, "n_stop": len(stop), "n_restart": len(rest)}
    m["miss_stop_hi95"] = round(D.wilson(miss_stop, len(stop))[1], 4)
    m["false_restart_hi95"] = round(D.wilson(false_restart, len(non_rest))[1], 4)
    return m


def w_cascade(p):
    _set_affinity = L._set_affinity
    _set_affinity(p)
    refs = L._load_refs(p["data_dir"])
    refs = refs[: p["n"]] if p.get("n") else refs
    bad = {r["intent"] for r in refs} - set(INTENT2TOOL)
    if bad:
        raise ValueError(f"intent lạ trong refs.json: {sorted(bad)}")
    L._pre_import(p)
    m = L.Meter()
    m.mark_baseline()
    tr = L._make_asr(p)
    llm = L._make_llm(p)[0] if p.get("gguf") else None      # không có gguf => chỉ đo STT + NLU từ khoá (đường 'kw')
    m.mark_loaded()
    audios = [L._load_audio16(Path(p["data_dir"]) / r["file"]) for r in refs]
    constrained = p.get("constrained", True)
    tr(audios[0])
    if llm:
        T._call(llm, D.SYS_A, "What is the status", D.SCHEMA_A, 8, constrained)   # warm-up
    safe, unsafe = _contexts(p.get("seed", 0))
    hogs = L._start_hogs(p.get("hog_procs", 0))
    out, errors, first_cond = {}, [], True
    try:
        for name, kind, snr in X._conditions(p):
            rng = np.random.default_rng(3)
            rows, stt_ms, chain_ms, valid, hyps = [], [], [], 0, []
            for a, r in zip(audios, refs):
                x = a if kind is None else X._mix(a, kind, snr, rng, p.get("noise_dir"))
                t0 = time.perf_counter()
                h = tr(x)
                s_ms = (time.perf_counter() - t0) * 1000
                hyps.append(h)
                if llm is None:
                    tool, l_ms, valid = INTENT2TOOL[L._nlu_en(h)], 0.0, valid + 1
                elif h.strip():
                    txt, l_ms, _ = T._call(llm, D.SYS_A, h, D.SCHEMA_A, 24, constrained)
                    j = D.parse_json(txt)
                    tool = j.get("tool") if isinstance(j, dict) else None
                    ok = tool in D.TOOLS
                    valid += ok
                    tool = tool if ok else "INVALID"
                else:                                     # STT không ra chữ: VAD thật sẽ không chuyển tiếp -> 'none'
                    tool, l_ms, valid = "none", 0.0, valid + 1
                rows.append({"gold": INTENT2TOOL[r["intent"]], "slm": tool, "kw": INTENT2TOOL[L._nlu_en(h)]})
                stt_ms.append(s_ms)
                chain_ms.append(s_ms + l_ms)
                if first_cond and tool != rows[-1]["gold"] and len(errors) < 12:
                    errors.append(f"[{rows[-1]['gold']} -> {tool}] hyp='{h}' | ref='{r['text']}'")
            first_cond = False
            for key in (("slm", "kw") if llm else ("kw",)):
                for k, v in _metrics(rows, key, safe, unsafe).items():
                    out[f"{key}_{k}_{name}"] = v
            if llm:
                stop_rows = [r for r in rows if r["gold"] == "request_stop"]
                either = sum(r["slm"] != "request_stop" and r["kw"] != "request_stop" for r in stop_rows)
                out[f"miss_stop_either_{name}"] = _rate(either, len(stop_rows))
                out[f"slm_valid_json_{name}"] = _rate(valid, len(rows))
            out[f"stt_ms_p50_{name}"], out[f"stt_ms_p95_{name}"] = L._pctl(stt_ms, 50), L._pctl(stt_ms, 95)
            out[f"chain_ms_p50_{name}"], out[f"chain_ms_p95_{name}"] = L._pctl(chain_ms, 50), L._pctl(chain_ms, 95)
            out[f"wer_{name}"] = round(L._wer([r["text"] for r in refs], hyps), 4)
    finally:
        L._stop_hogs(hogs)
    res = m.report()
    res.update(out, n_samples=len(refs), with_slm=bool(llm), constrained=constrained, threads=p.get("threads", 4), hog_procs=p.get("hog_procs", 0),
               sample_errors=errors, safe_ctx=safe, dataset_hash=D.dataset_hash(),
               note="Arbiter = arbiter.py; unsafe_start phải bằng 0", runtime="stt+slm(llama-cpp)+arbiter")
    return res


KINDS = dict(T.KINDS)
KINDS["cascade"] = w_cascade

if __name__ == "__main__":
    kind = sys.argv[1]
    params = json.loads(sys.stdin.read() or "{}")
    try:
        res = KINDS[kind](params)
    except Exception:
        res = {"error": traceback.format_exc()[-1800:]}
    print("@@RESULT@@" + json.dumps(res, ensure_ascii=False))
