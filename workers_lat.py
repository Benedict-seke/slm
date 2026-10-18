"""Worker LATENCY + "TTS đọc đúng câu chưa" cho bài băng tải (Arduino UNO Q).

Dùng giống workers_ext.py:  python workers_lat.py <kind>  (params JSON qua stdin, kết quả @@RESULT@@{json}).
Mọi worker cũ (stt_cmd, tts_quality, e2e, cv_latency, ...) vẫn gọi được vì KINDS kế thừa từ workers_ext.

Worker mới:
  tts_correct   TTS có đọc ĐÚNG câu không? (chỉ tiếng sạch, không ồn): round-trip WER/CER, exact-match,
                key-token recall (từ khoá/số bắt buộc phải nghe được), nhiều judge STT; kèm TTFA / thời gian
                tổng hợp xong / độ dài audio.
  slm_latency   SLM GGUF (llama-cpp-python, trong tiến trình): thời gian load, TTFT, tok/s giải mã,
                tổng thời gian trả lời; theo độ dài ngữ cảnh (pad) và có/không tái dùng cache tiền tố;
                chịu tải nền (hog) và giới hạn lõi CPU (cpus).
  e2e_full      Lệnh giọng nói -> STT -> NLU -> (SLM) -> TTS trong MỘT tiến trình: thời gian tới ÂM THANH ĐẦU TIÊN
                cho từng đường (nhanh / có SLM; chờ trọn câu vs phát ngay câu đầu), RAM chung thật.

Quy ước thời gian: mọi mốc tính từ lúc audio lệnh đã có đủ (chưa gồm thời gian nói và độ trễ endpointing của VAD;
xem tham số vad_endpoint_ms).
"""
import json
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import workers_ext as X
from workers_ext import (Meter, _cer, _cpu_temp_c, _load_audio16, _load_refs, _make_asr,  # noqa: F401
                         _pctl, _piper_load, _piper_stream, _pre_import, _start_hogs, _stop_hogs,
                         _strip_accents, _wer)


# ================================================================ tiện ích chung
def _fold(s: str) -> str:
    """Chữ thường, bỏ dấu câu + dấu thanh, chuẩn hoá khoảng trắng; bọc 2 đầu bằng dấu cách để khớp nguyên từ."""
    s = re.sub(r"[^\w\s]", " ", _strip_accents(s))
    return " " + re.sub(r"\s+", " ", s).strip() + " "


def _digits_to_words(s: str, lang: str = "en") -> str:
    """STT hay trả '0.8', '65'; câu tham chiếu viết bằng chữ => đổi chữ số thành chữ (cần `pip install num2words`)."""
    try:
        from num2words import num2words
    except Exception:
        return s

    def rep(m):
        t = m.group(0)
        if lang != "en":
            t = t.replace(",", ".")
        try:
            v = float(t) if "." in t else int(t)
            return " " + num2words(v, lang=lang).replace("-", " ") + " "
        except Exception:
            return m.group(0)

    return re.sub(r"\d+(?:[.,]\d+)?", rep, s)


def _unpack(item):
    """Câu = 'text' hoặc {'say': chữ đưa vào TTS, 'ref': chữ người nghe phải nghe ra (mặc định = say), 'keys': [từ khoá]}."""
    if isinstance(item, str):
        return item, item, []
    return item["say"], item.get("ref", item["say"]), item.get("keys", [])


def _set_affinity(p):
    """cpus=[0,1,2,3] ghim tiến trình (và hog con) vào 4 lõi — mô phỏng 4 lõi A53 của UNO Q (chỉ tương đối)."""
    cpus = p.get("cpus")
    if not cpus:
        return
    try:
        import psutil
        psutil.Process().cpu_affinity(list(cpus))
    except Exception:
        pass


def _first_sentence(text: str):
    """Câu đầu tiên đã HOÀN CHỈNH trong text (kết thúc bằng . ! ? hoặc xuống dòng, theo sau là khoảng trắng)."""
    m = re.search(r"^(.{8,}?[.!?])\s", text.lstrip(), flags=re.S) or re.search(r"^(.{8,}?)\n", text.lstrip(), flags=re.S)
    return m.group(1).strip() if m else None


def _nlu_en(text: str) -> str:
    """NLU từ khoá tiếng Anh (cùng thứ tự ưu tiên với _nlu tiếng Việt trong workers_ext)."""
    s = _fold(text)

    def has(*ws):
        return any(f" {w} " in s for w in ws)

    if has("why", "reason", "caused", "cause"):
        return "why"
    if has("restart", "again", "back on"):
        return "restart"
    if has("temperature", "hot", "overheating"):
        return "temp"
    if has("speed", "how fast"):
        return "speed"
    if has("how many", "count"):
        return "count"
    if has("status", "condition", "running"):
        return "status"
    if has("stop", "halt", "shut"):
        return "stop"
    return "none"


# ================================================================ đo TTS / SLM dùng chung
def _tts_run(v, text):
    """(ttfa_ms, full_ms, audio_float32, sample_rate). ttfa = tới mẫu âm thanh đầu tiên."""
    t0 = time.perf_counter()
    first, chunks, sr = None, [], 22050
    for a, sr in _piper_stream(v, text):
        if first is None:
            first = (time.perf_counter() - t0) * 1000
        chunks.append(a)
    full = (time.perf_counter() - t0) * 1000
    x = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    return (first if first is not None else full), full, x, sr


def _slm_stream(llm, messages, max_tokens):
    for ch in llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.0, stream=True):
        yield (ch["choices"][0].get("delta") or {}).get("content") or ""


def _timed_generate(stream):
    """Chạy một luồng sinh chữ, trả mốc thời gian (ms từ lúc bắt đầu): ttft, hết câu đầu, hết toàn bộ."""
    t0 = time.perf_counter()
    parts, t_first, t_sent = [], None, None
    for piece in stream:
        if not piece:
            continue
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        parts.append(piece)
        if t_sent is None and _first_sentence("".join(parts)):
            t_sent = now
    t_end = time.perf_counter()
    return {"text": "".join(parts),
            "ttft_ms": (t_first - t0) * 1000 if t_first else None,
            "t_sent_ms": ((t_sent or t_end) - t0) * 1000,
            "total_ms": (t_end - t0) * 1000}


_SYS = ("You are the assistant of a conveyor belt. Answer in at most two short sentences, using ONLY the sensor "
        "data given. Never claim something the data does not show.")
_SNAPS = [  # (câu hỏi, snapshot cảm biến) — dữ liệu đưa vào prompt giống Multimodal Context Builder
    ("Why did the conveyor stop?", {"state": "STOPPED", "encoder_speed_mps": 0.05, "motor_current_a": 3.1,
                                    "motor_temp_c": 42, "camera": "product stationary on belt"}),
    ("Is it safe to restart?", {"state": "STOPPED", "encoder_speed_mps": 0.0, "motor_current_a": 0.2,
                                "motor_temp_c": 38, "camera": "belt empty"}),
    ("What is the motor temperature?", {"state": "RUNNING", "encoder_speed_mps": 0.8, "motor_current_a": 1.2,
                                        "motor_temp_c": 65, "camera": "products moving"}),
    ("Why did the conveyor stop?", {"state": "STOPPED", "encoder_speed_mps": 0.0, "motor_current_a": 1.1,
                                    "motor_temp_c": 78, "camera": "belt empty"}),
    ("Why did the conveyor stop?", {"state": "STOPPED", "encoder_speed_mps": None, "motor_current_a": None,
                                    "motor_temp_c": 40, "camera": "unavailable"}),
]
_WHY_IDX = [i for i, (q, _) in enumerate(_SNAPS) if q.startswith("Why")]


def _build_msgs(q, snap, pad_tokens, ntok):
    """pad_tokens > 0 thêm 'lịch sử telemetry' để mô phỏng ngữ cảnh dài (Telemetry Ring Buffer)."""
    log = ""
    if pad_tokens:
        line = "t-{k}s speed=0.80 current=1.2 temp=41\n"
        n = int(np.ceil(pad_tokens / max(1, ntok(line.format(k=10)))))
        log = "Recent telemetry (oldest first):\n" + "".join(line.format(k=n - j) for j in range(n))
    return [{"role": "system", "content": _SYS},
            {"role": "user", "content": f"{log}Sensor snapshot: {json.dumps(snap)}\nQuestion: {q}"}]


def _make_llm(p):
    from llama_cpp import Llama
    llm = Llama(model_path=p["gguf"], n_ctx=p.get("n_ctx", 2048), n_threads=p.get("threads", 4),
                n_gpu_layers=0, verbose=False)
    return llm, (lambda s: len(llm.tokenize(s.encode("utf-8"), add_bos=False)))


# ================================================================ TTS: đọc đúng câu chưa
def w_tts_correct(p):
    """Không ồn. Mỗi câu: Piper đọc -> judge STT nghe lại -> so với câu tham chiếu.
    - rt_wer / rt_cer: round-trip. CHỈ ĐÁNG TIN khi judge mạnh hơn nhiều so với TTS (xem cột judge_floor trong notebook).
    - exact_*: tỉ lệ câu nghe lại KHỚP HOÀN TOÀN (sau chuẩn hoá). key_recall_*: tỉ lệ từ khoá/số nghe được."""
    import soundfile as sf
    items = [_unpack(s) for s in p["sentences"]]
    lang = p.get("language", "en")
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    import piper  # noqa: F401
    m = Meter()
    m.mark_baseline()
    v, onnx_path = _piper_load(p["voice"])
    m.mark_loaded()
    _tts_run(v, "hello")  # warm-up
    ttfa, synth, dur = [], [], []
    for i, (say, _ref, _keys) in enumerate(items):
        a, b, x, sr = _tts_run(v, say)
        ttfa.append(a)
        synth.append(b)
        dur.append(len(x) / sr)
        sf.write(str(out_dir / f"{i}.wav"), x, sr)
    res = m.report()
    res.update(ttfa_ms_p50=_pctl(ttfa, 50), ttfa_ms_p95=_pctl(ttfa, 95), synth_ms_p50=_pctl(synth, 50),
               synth_ms_p95=_pctl(synth, 95), audio_s_mean=round(float(np.mean(dur)), 2),
               rtf=round(sum(synth) / 1000 / sum(dur), 3), n_sentences=len(items), wav_dir=str(out_dir),
               onnx_mb=round(Path(onnx_path).stat().st_size / 1048576, 1) if Path(onnx_path).exists() else None,
               runtime="onnxruntime(piper)")
    del v
    from faster_whisper import WhisperModel
    auds = [_load_audio16(out_dir / f"{i}.wav") for i in range(len(items))]
    refs = [r for _, r, _ in items]
    for j in p.get("judges", ["small.en"]):
        model = WhisperModel(j, device="cpu", compute_type="int8", cpu_threads=p.get("threads", 4),
                             download_root=str(Path("models") / "fw"))
        hyps = []
        for a in auds:
            segs, _ = model.transcribe(a, language=lang, beam_size=5, vad_filter=False,
                                       condition_on_previous_text=False)
            hyps.append(" ".join(s.text for s in segs).strip())
        norm_h = [_digits_to_words(h, lang) for h in hyps]
        tag = j.replace(".", "_")
        em = [_fold(r) == _fold(h) for r, h in zip(refs, norm_h)]
        kr = [sum(_fold(k) in _fold(h) for k in keys) / len(keys) for (_s, _r, keys), h in zip(items, norm_h) if keys]
        res[f"rt_wer_{tag}"] = round(_wer(refs, norm_h), 4)
        res[f"rt_cer_{tag}"] = round(_cer(refs, norm_h), 4)
        res[f"exact_{tag}"] = round(sum(em) / len(em), 4)
        res[f"key_recall_{tag}"] = round(float(np.mean(kr)), 4) if kr else None
        res[f"mismatch_{tag}"] = [f"{r}  =>  {h}" for r, h, e in zip(refs, hyps, em) if not e][:10]
    return res


# ================================================================ SLM latency
def w_slm_latency(p):
    """gguf=<đường dẫn>. Mỗi điều kiện (mode × pad) chạy reps × 5 prompt cảm biến.
    mode 'cold'  = xoá cache trước mỗi câu (prefill đủ cả system prompt) — con số an toàn/xấu nhất.
    mode 'prefix'= giữ cache => system prompt chung được tái dùng (thực tế nếu orchestrator giữ phiên llama).
    pad = số token telemetry thêm vào prompt (0 = chỉ snapshot). TTFT gồm prefill + token đầu."""
    _set_affinity(p)
    m = Meter()
    m.mark_baseline()
    t = time.perf_counter()
    llm, ntok = _make_llm(p)
    load_s = time.perf_counter() - t
    m.mark_loaded()
    mt, reps = p.get("max_tokens", 64), p.get("reps", 5)
    q0, s0 = _SNAPS[0]
    first = _timed_generate(_slm_stream(llm, _build_msgs(q0, s0, 0, ntok), mt))  # lần gọi đầu sau load (nguội)
    llm.reset()
    _timed_generate(_slm_stream(llm, _build_msgs(q0, s0, 0, ntok), mt))           # warm-up, không tính
    hogs = _start_hogs(p.get("hog_procs", 0))
    temp0, res, head = _cpu_temp_c(), {}, None
    try:
        for pad in p.get("pads", [0]):
            for mode in p.get("modes", ["cold", "prefix"]):
                ttft, tps, tot, ntk, ptk, pre = [], [], [], [], [], []
                if mode == "prefix":
                    llm.reset()
                    _timed_generate(_slm_stream(llm, _build_msgs(q0, s0, pad, ntok), mt))
                for _ in range(reps):
                    for q, snap in _SNAPS:
                        msgs = _build_msgs(q, snap, pad, ntok)
                        if mode == "cold":
                            llm.reset()
                        g = _timed_generate(_slm_stream(llm, msgs, mt))
                        if g["ttft_ms"] is None:
                            continue
                        n_out = ntok(g["text"])
                        n_in = sum(ntok(x["content"]) for x in msgs)
                        ttft.append(g["ttft_ms"])
                        tot.append(g["total_ms"])
                        ntk.append(n_out)
                        ptk.append(n_in)
                        pre.append(n_in / (g["ttft_ms"] / 1000))
                        dec_s = (g["total_ms"] - g["ttft_ms"]) / 1000
                        if n_out > 1 and dec_s > 0:
                            tps.append((n_out - 1) / dec_s)
                sfx = f"{mode}_pad{pad}"
                cond = {f"ttft_ms_p50_{sfx}": _pctl(ttft, 50), f"ttft_ms_p95_{sfx}": _pctl(ttft, 95),
                        f"decode_tps_p50_{sfx}": _pctl(tps, 50), f"prefill_tps_approx_{sfx}": _pctl(pre, 50),
                        f"total_ms_p50_{sfx}": _pctl(tot, 50), f"total_ms_p95_{sfx}": _pctl(tot, 95),
                        f"prompt_tokens_{sfx}": _pctl(ptk, 50), f"out_tokens_{sfx}": _pctl(ntk, 50)}
                res.update(cond)
                if head is None:
                    head = {"ttft_ms_p50": cond[f"ttft_ms_p50_{sfx}"], "ttft_ms_p95": cond[f"ttft_ms_p95_{sfx}"],
                            "decode_tps_p50": cond[f"decode_tps_p50_{sfx}"],
                            "total_ms_p50": cond[f"total_ms_p50_{sfx}"], "total_ms_p95": cond[f"total_ms_p95_{sfx}"],
                            "headline_condition": sfx}
    finally:
        _stop_hogs(hogs)
    out = m.report()
    out.update(head or {})
    out.update(res)
    out.update(load_s=round(load_s, 2), first_call_ttft_ms=round(first["ttft_ms"], 0) if first["ttft_ms"] else None,
               threads=p.get("threads", 4), cpus=p.get("cpus"), hog_procs=p.get("hog_procs", 0),
               n_ctx=p.get("n_ctx", 2048), max_tokens=mt, reps=reps, n_prompts=len(_SNAPS),
               temp_start_c=temp0, temp_end_c=_cpu_temp_c(), sample_answer=first["text"][:200],
               runtime="llama-cpp-python")
    return out


# ================================================================ E2E đầy đủ
def w_e2e_full(p):
    """data_dir/refs.json = [{file,text,intent}] (lệnh giọng). Tham số: engine STT như stt_cmd (engine,size|model,language,...),
    voice (Piper), gguf (tuỳ chọn; thiếu thì đường 'why' dùng câu mẫu), nlu 'en'|'vi', responses {intent: câu mẫu},
    slm_intents (mặc định ['why']), n, repeat, hog_procs, cpus, vad_endpoint_ms."""
    _set_affinity(p)
    refs = _load_refs(p["data_dir"])
    refs = (refs[: p["n"]] if p.get("n") else refs) * p.get("repeat", 1)
    _pre_import(p)
    import piper  # noqa: F401
    nlu = X._nlu if p.get("nlu", "en") == "vi" else _nlu_en
    resp, slm_intents, mt = p["responses"], set(p.get("slm_intents", ["why"])), p.get("max_tokens", 64)
    m = Meter()
    m.mark_baseline()
    tr = _make_asr(p)
    v, _ = _piper_load(p["voice"])
    llm, ntok, load_slm = None, None, None
    if p.get("gguf"):
        t = time.perf_counter()
        llm, ntok = _make_llm(p)
        load_slm = round(time.perf_counter() - t, 2)
    m.mark_loaded()
    audios = [_load_audio16(Path(p["data_dir"]) / r["file"]) for r in refs]
    tr(audios[0])
    _tts_run(v, resp.get("status", "ok"))
    if llm:
        q0, s0 = _SNAPS[0]
        _timed_generate(_slm_stream(llm, _build_msgs(q0, s0, 0, ntok), mt))
    hogs = _start_hogs(p.get("hog_procs", 0))
    fast, slm, stop_fa, ok, slow = [], [], [], 0, []
    try:
        for k, (a, r) in enumerate(zip(audios, refs)):
            t0 = time.perf_counter()
            h = tr(a)
            stt_ms = (time.perf_counter() - t0) * 1000
            intent = nlu(h)
            ok += intent == r["intent"]
            slow.append((stt_ms, f"{stt_ms:.0f} ms | ref='{r['text']}' | hyp='{h}'"))
            if llm and intent in slm_intents:
                q, snap = _SNAPS[_WHY_IDX[k % len(_WHY_IDX)]]
                llm.reset()  # prefill đầy đủ mỗi lệnh (đường xấu nhất)
                g = _timed_generate(_slm_stream(llm, _build_msgs(q, snap, 0, ntok), mt))
                txt = g["text"].strip() or resp.get(intent, resp.get("none", "Sorry."))
                f_first, _, _, _ = _tts_run(v, _first_sentence(txt + " ") or txt)
                f_full, _, x, sr = _tts_run(v, txt)
                slm.append({"stt": stt_ms, "ttft": g["ttft_ms"] or 0, "total": g["total_ms"], "ntok": ntok(txt),
                            "wait": stt_ms + g["total_ms"] + f_full, "stream": stt_ms + g["t_sent_ms"] + f_first,
                            "reply_s": len(x) / sr})
            else:
                f, _, x, sr = _tts_run(v, resp.get(intent, resp.get("none", "Sorry, I did not understand.")))
                fast.append({"stt": stt_ms, "tts": f, "first": stt_ms + f, "reply_s": len(x) / sr})
                if r["intent"] == "stop":
                    stop_fa.append(stt_ms + f)
    finally:
        _stop_hogs(hogs)

    def col(rows, key):
        return [r[key] for r in rows]

    vad = p.get("vad_endpoint_ms", 0)
    res = m.report()
    res.update(n_fast=len(fast), n_slm=len(slm), intent_acc=round(ok / len(refs), 4),
               fast_stt_ms_p50=_pctl(col(fast, "stt"), 50), fast_stt_ms_p95=_pctl(col(fast, "stt"), 95),
               fast_tts_ttfa_ms_p50=_pctl(col(fast, "tts"), 50),
               fast_first_audio_ms_p50=_pctl(col(fast, "first"), 50), fast_first_audio_ms_p95=_pctl(col(fast, "first"), 95),
               fast_first_audio_ms_max=_pctl(col(fast, "first"), 100),
               fast_reply_end_ms_p50=_pctl([r["first"] + r["reply_s"] * 1000 for r in fast], 50),
               stop_first_audio_ms_p95=_pctl(stop_fa, 95), n_stop=len(stop_fa),
               slm_stt_ms_p50=_pctl(col(slm, "stt"), 50), slm_ttft_ms_p50=_pctl(col(slm, "ttft"), 50),
               slm_total_ms_p50=_pctl(col(slm, "total"), 50), slm_total_ms_p95=_pctl(col(slm, "total"), 95),
               slm_tokens_p50=_pctl(col(slm, "ntok"), 50),
               slm_first_audio_wait_ms_p50=_pctl(col(slm, "wait"), 50), slm_first_audio_wait_ms_p95=_pctl(col(slm, "wait"), 95),
               slm_first_audio_stream_ms_p50=_pctl(col(slm, "stream"), 50),
               slm_first_audio_stream_ms_p95=_pctl(col(slm, "stream"), 95),
               assumed_vad_endpoint_ms=vad,
               fast_first_audio_incl_vad_ms_p50=(_pctl(col(fast, "first"), 50) or 0) + vad if fast else None,
               slm_first_audio_stream_incl_vad_ms_p50=(_pctl(col(slm, "stream"), 50) or 0) + vad if slm else None,
               slm_load_s=load_slm, hog_procs=p.get("hog_procs", 0), cpus=p.get("cpus"), n_samples=len(refs),
               slowest_stt=[s for _, s in sorted(slow, key=lambda z: -z[0])[:3]],
               runtime="stt+nlu+slm+piper")
    return res


KINDS = dict(X.KINDS)
KINDS.update({k[2:]: v for k, v in list(globals().items())
              if k.startswith("w_") and callable(v) and v.__module__ == __name__})

if __name__ == "__main__":
    kind = sys.argv[1]
    params = json.loads(sys.stdin.read() or "{}")
    try:
        res = KINDS[kind](params)
    except Exception:
        res = {"error": traceback.format_exc()[-1800:]}
    print("@@RESULT@@" + json.dumps(res, ensure_ascii=False))
