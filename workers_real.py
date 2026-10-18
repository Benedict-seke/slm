"""Worker đo LỆNH DỪNG bằng GIỌNG NGƯỜI THẬT (Speech Commands) trong tiếng ồn máy:  python workers_real.py stop_kw

data_dir = thư mục do prep_real_data.py speech tạo (refs.json: file, text, intent 'stop'|'none', word).
Với mỗi điều kiện ồn (sạch + {kinds} x {snrs}; noise 'real' lấy từ noise_dir):
  miss_stop[_hi95]   clip 'stop' mà NLU từ khoá KHÔNG ra 'stop'            (cận trên Wilson 95%)
  false_stop[_hi95]  clip KHÁC (go/on/off/yes/no/...) mà NLU ra 'stop'     (dừng nhầm do nghe nhầm)
  recall_<từ>        STT nghe ra đúng từ đó
  lat_ms_p50/p95     thời gian STT mỗi clip
GIỚI HẠN: clip 1 giây, MỘT từ — không thay cho câu lệnh đầy đủ. NLU là từ khoá (_nlu_en), không qua SLM.
Params: như stt_cmd (engine,size|model,language,grammar,initial_prompt), data_dir, kinds, snrs, noise_dir, n, threads, cpus.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import conveyor_data as D
import workers_ext as X
import workers_lat as L

MIN_N = 20   # dưới mức này cận trên 95% quá rộng — cảnh báo


def _rate(k, n):
    return round(k / n, 4) if n else None


def w_stop_kw(p):
    L._set_affinity(p)
    refs = L._load_refs(p["data_dir"])
    refs = refs[: p["n"]] if p.get("n") else refs
    L._pre_import(p)
    m = L.Meter()
    m.mark_baseline()
    tr = L._make_asr(p)
    m.mark_loaded()
    audios = [L._load_audio16(Path(p["data_dir"]) / r["file"]) for r in refs]
    tr(audios[0])
    gold = [r["intent"] for r in refs]
    words = sorted({r["word"] for r in refs})
    n_stop = sum(g == "stop" for g in gold)
    n_other = len(gold) - n_stop
    out, samples, lat_clean = {}, [], []
    for name, kind, snr in X._conditions(p):
        rng = np.random.default_rng(5)
        hyps, preds = [], []
        for a in audios:
            x = a if kind is None else X._mix(a, kind, snr, rng, p.get("noise_dir"))
            t = time.perf_counter()
            h = tr(x)
            if kind is None:
                lat_clean.append((time.perf_counter() - t) * 1000)
            hyps.append(h)
            preds.append(L._nlu_en(h))
        miss = sum(g == "stop" and pr != "stop" for g, pr in zip(gold, preds))
        fals = sum(g != "stop" and pr == "stop" for g, pr in zip(gold, preds))
        out[f"miss_stop_{name}"] = _rate(miss, n_stop)
        out[f"miss_stop_hi95_{name}"] = round(D.wilson(miss, n_stop)[1], 4)
        out[f"false_stop_{name}"] = _rate(fals, n_other)
        out[f"false_stop_hi95_{name}"] = round(D.wilson(fals, n_other)[1], 4)
        for w in words:
            idx = [i for i, r in enumerate(refs) if r["word"] == w]
            ok = sum(L._fold(w) in L._fold(hyps[i]) for i in idx)
            out[f"recall_{w}_{name}"] = _rate(ok, len(idx))
        if kind is None:
            samples = [f"{r['word']} => '{h}'" for r, h in list(zip(refs, hyps))[:8]]
    res = m.report()
    res.update(out, n_samples=len(refs), n_stop=n_stop, n_other=n_other,
               lat_ms_p50=L._pctl(lat_clean, 50), lat_ms_p95=L._pctl(lat_clean, 95), sample_hyps=samples,
               warn_small_n=(n_stop < MIN_N), runtime=p.get("engine", "fw"),
               note="giọng người thật, 1 từ/clip; noise 'machine' tổng hợp hoặc 'real' từ noise_dir")
    return res


KINDS = dict(L.KINDS)
KINDS["stop_kw"] = w_stop_kw

if __name__ == "__main__":
    kind = sys.argv[1]
    params = json.loads(sys.stdin.read() or "{}")
    try:
        res = KINDS[kind](params)
    except Exception:
        res = {"error": traceback.format_exc()[-1800:]}
    print("@@RESULT@@" + json.dumps(res, ensure_ascii=False))
