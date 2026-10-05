"""Worker MỞ RỘNG cho bài "Offline AI Assistant for Conveyor Belt Manufacturing".

Dùng giống workers.py:  python workers_ext.py <kind>  (params JSON qua stdin, kết quả @@RESULT@@{json})
Tất cả worker cũ (stt_fw, tts_piper, cls_ort...) vẫn gọi được vì được import lại ở đây.

Worker mới:
  stt_noise     WER theo (loại ồn, SNR) — tiếng ồn nhà máy
  stt_cmd       Lệnh giọng nói (tiếng Việt/Anh): WER/CER, độ chính xác intent, an toàn (false-restart, miss-stop), latency/câu
  make_vi_cmds  Sinh bộ lệnh tổng hợp bằng piper (chỉ khi chưa có bản ghi thật)
  tts_quality   TTFA, RTF, peak/clipping, độ im lặng đầu, âm lượng
  tts_judge     Round-trip WER/CER (sạch + trong ồn) bằng STT cố định
  cv_latency    Latency bền vững (pre + infer + post), p50/p95/p99, drift nhiệt, có thể chạy cùng "tải nền"
  cls_corrupt   Top-1 dưới các nhiễu (tối/sáng/mờ/nhiễu/jpeg)
  det_corrupt   mAP YOLO dưới các nhiễu (COCO128)
  cv_blockage   Phát hiện kẹt sản phẩm trên video băng tải TỔNG HỢP: delay, tỉ lệ phát hiện, báo động giả, ms/khung
  e2e           STT -> NLU -> TTS trong MỘT tiến trình: RAM chung + latency lệnh->phản hồi
"""
import json
import os
import subprocess
import sys
import time
import traceback
import unicodedata
from collections import deque
from pathlib import Path

import numpy as np

import workers as W
from workers import Meter, _load_refs, _norm, _wer, _ensure_vosk  # noqa: F401


# ================================================================ tiện ích chung
def _pctl(xs, q):
    return round(float(np.percentile(xs, q)), 1) if len(xs) else None


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _cer(refs, hyps) -> float:
    import jiwer
    return jiwer.cer([_norm(r) for r in refs], [_norm(h) if _norm(h) else "<empty>" for h in hyps])


def _load_audio16(path):
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a.astype(np.float32)


def _noise(kind, n, sr, rng, noise_dir=None):
    """Tạo nhiễu. 'machine' = hum 50Hz + hài + băng rộng điều biên + tiếng va kim loại.
    'real' = lấy ngẫu nhiên 1 file wav trong noise_dir (khuyến nghị: ghi âm xưởng thật)."""
    if kind == "real" and noise_dir and list(Path(noise_dir).glob("*.wav")):
        import soundfile as sf
        f = rng.choice(sorted(Path(noise_dir).glob("*.wav")))
        x, s = sf.read(str(f), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if s != sr:
            import librosa
            x = librosa.resample(x, orig_sr=s, target_sr=sr)
        if len(x) < n:
            x = np.tile(x, int(np.ceil(n / len(x))))
        st = int(rng.integers(0, len(x) - n + 1))
        return x[st:st + n].astype(np.float32)
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        X = np.fft.rfft(rng.standard_normal(n))
        f = np.arange(len(X), dtype=np.float64)
        f[0] = 1
        x = np.fft.irfft(X / np.sqrt(f), n)
    else:  # machine
        t = np.arange(n) / sr
        hum = sum(np.sin(2 * np.pi * 50 * h * t + rng.uniform(0, 6.28)) / h for h in range(1, 9))
        bb = np.convolve(rng.standard_normal(n) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)), np.ones(8) / 8, mode="same")
        clank = np.zeros(n)
        for pos in rng.integers(0, n, size=max(1, int(3 * n / sr))):
            L = min(n - pos, int(0.03 * sr))
            k = np.arange(L)
            clank[pos:pos + L] += np.exp(-k / (0.005 * sr)) * np.sin(2 * np.pi * 2000 * k / sr) * 3
        x = hum * 0.8 + bb + clank
    return np.asarray(x, dtype=np.float32)


def _mix(sig, kind, snr_db, rng, noise_dir=None):
    n = _noise(kind, len(sig), 16000, rng, noise_dir)
    ps, pn = float(np.mean(sig ** 2)) + 1e-12, float(np.mean(n ** 2)) + 1e-12
    out = sig + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    pk = float(np.abs(out).max())
    return (out / pk if pk > 1 else out).astype(np.float32)


# ---------------------------------------------------------------- NLU từ khoá (đường nhanh, không cần SLM)
def _nlu(text: str) -> str:
    s = f" {_norm(_strip_accents(text))} "

    def has(*ws):
        return any(f" {w} " in s for w in ws)

    if has("tai sao", "vi sao", "nguyen nhan"):
        return "why"
    if has("khoi dong lai", "chay lai", "bat lai", "restart"):
        return "restart"
    if has("nhiet do", "nong"):
        return "temp"
    if has("toc do"):
        return "speed"
    if has("dem"):
        return "count"
    if has("trang thai", "tinh trang", "dang chay"):
        return "status"
    if has("dung", "tat", "stop"):
        return "stop"
    return "none"


# ---------------------------------------------------------------- engine STT dùng chung
def _pre_import(p):
    if p.get("engine", "fw") == "vosk":
        import vosk  # noqa: F401
    else:
        import faster_whisper  # noqa: F401


def _make_asr(p):
    eng = p.get("engine", "fw")
    if eng == "vosk":
        import vosk
        vosk.SetLogLevel(-1)
        model = vosk.Model(_ensure_vosk(p["model"]))
        grammar = json.dumps(p["grammar"], ensure_ascii=False) if p.get("grammar") else None

        def tr(a):
            rec = vosk.KaldiRecognizer(model, 16000, grammar) if grammar else vosk.KaldiRecognizer(model, 16000)
            rec.AcceptWaveform((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
            return json.loads(rec.FinalResult())["text"]
        return tr
    from faster_whisper import WhisperModel
    model = WhisperModel(p["size"], device="cpu", compute_type="int8", cpu_threads=p.get("threads", 4),
                         download_root=str(Path("models") / "fw"))
    lang, prompt, beam = p.get("language", "en"), p.get("initial_prompt"), p.get("beam", 1)

    def tr(a):
        segs, _ = model.transcribe(a, language=lang, beam_size=beam, vad_filter=False,
                                   initial_prompt=prompt, condition_on_previous_text=False)
        return " ".join(s.text for s in segs).strip()
    return tr


def _conditions(p):
    conds = [("clean", None, None)]
    for k in p.get("kinds", ["machine"]):
        for snr in p.get("snrs", [20, 10, 5, 0]):
            conds.append((f"{k}_{snr}db", k, snr))
    return conds


# ================================================================ STT
def w_stt_noise(p):
    """WER theo điều kiện nhiễu. data_dir = thư mục LibriSpeech đã prep (refs.json)."""
    refs = _load_refs(p["data_dir"])[: p.get("n", 60)]
    _pre_import(p)
    m = Meter()
    m.mark_baseline()
    tr = _make_asr(p)
    m.mark_loaded()
    audios = [_load_audio16(Path(p["data_dir"]) / r["file"]) for r in refs]
    out = {}
    for name, kind, snr in _conditions(p):
        rng = np.random.default_rng(0)
        hyps = [tr(a if kind is None else _mix(a, kind, snr, rng, p.get("noise_dir"))) for a in audios]
        out[f"wer_{name}"] = round(_wer([r["text"] for r in refs], hyps), 4)
    res = m.report()
    res.update(out, n_samples=len(refs), runtime=p.get("engine", "fw"))
    return res


def w_stt_cmd(p):
    """Lệnh giọng nói: data_dir/refs.json = [{file,text,intent}]. Đo cả đường STT -> NLU từ khoá."""
    refs = _load_refs(p["data_dir"])
    _pre_import(p)
    m = Meter()
    m.mark_baseline()
    tr = _make_asr(p)
    m.mark_loaded()
    audios = [_load_audio16(Path(p["data_dir"]) / r["file"]) for r in refs]
    tr(audios[0])  # warm-up (không tính latency)
    out, lat_clean = {}, []
    for name, kind, snr in _conditions(p):
        rng = np.random.default_rng(1)
        hyps, preds = [], []
        for a, r in zip(audios, refs):
            x = a if kind is None else _mix(a, kind, snr, rng, p.get("noise_dir"))
            t = time.time()
            h = tr(x)
            if kind is None:
                lat_clean.append((time.time() - t) * 1000)
            hyps.append(h)
            preds.append(_nlu(h))
        gold = [r["intent"] for r in refs]
        n_not_restart = max(1, sum(g != "restart" for g in gold))
        n_stop = max(1, sum(g == "stop" for g in gold))
        out[f"wer_{name}"] = round(_wer([r["text"] for r in refs], hyps), 4)
        out[f"cer_{name}"] = round(_cer([r["text"] for r in refs], hyps), 4)
        out[f"intent_acc_{name}"] = round(sum(a == b for a, b in zip(gold, preds)) / len(gold), 4)
        # An toàn: nhận nhầm thành RESTART khi không phải restart; bỏ sót lệnh STOP
        out[f"false_restart_{name}"] = round(sum(pr == "restart" and g != "restart" for g, pr in zip(gold, preds)) / n_not_restart, 4)
        out[f"miss_stop_{name}"] = round(sum(g == "stop" and pr != "stop" for g, pr in zip(gold, preds)) / n_stop, 4)
        if kind is None:
            out["sample_hyps"] = [f"{r['text']}  =>  {h}" for r, h in list(zip(refs, hyps))[:6]]
    res = m.report()
    res.update(out, n_samples=len(refs), lat_ms_p50=_pctl(lat_clean, 50), lat_ms_p95=_pctl(lat_clean, 95),
               audio_s_mean=round(float(np.mean([len(a) / 16000 for a in audios])), 2),
               runtime=p.get("engine", "fw"))
    return res


# ================================================================ TTS
def _piper_load(voice):
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice
    lang, name, quality = voice.split("-")[:3]
    sub = f"{lang[:2]}/{lang}/{name}/{quality}"
    onnx_path = hf_hub_download("rhasspy/piper-voices", f"{sub}/{voice}.onnx", local_dir="models/piper")
    hf_hub_download("rhasspy/piper-voices", f"{sub}/{voice}.onnx.json", local_dir="models/piper")
    return PiperVoice.load(onnx_path), onnx_path


def _piper_stream(v, text):
    """Sinh (audio_float32, sr) theo từng đoạn — để đo time-to-first-audio."""
    if hasattr(v, "synthesize_stream_raw"):  # piper-tts < 1.3
        sr = v.config.sample_rate
        for raw in v.synthesize_stream_raw(text):
            yield np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr
    else:
        for c in v.synthesize(text):
            yield np.asarray(c.audio_float_array, dtype=np.float32), c.sample_rate


def w_make_vi_cmds(p):
    """Sinh bộ lệnh TỔNG HỢP (lạc quan hơn người thật!) khi bạn chưa ghi âm. Tốt hơn: dùng record_commands.py."""
    import soundfile as sf
    out = Path(p["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    v, _ = _piper_load(p["voice"])
    refs, i = [], 0
    for intent, phrases in p["variants"].items():
        for ph in phrases:
            chunks = list(_piper_stream(v, ph))
            audio, sr = np.concatenate([c[0] for c in chunks]), chunks[0][1]
            for sp in p.get("speeds", [0.9, 1.0, 1.1]):
                n_new = int(len(audio) / sp)
                x = np.interp(np.linspace(0, len(audio) - 1, n_new), np.arange(len(audio)), audio).astype(np.float32)
                pad = np.zeros(int(0.3 * sr), dtype=np.float32)
                sf.write(str(out / f"{i}.wav"), np.concatenate([pad, x, pad]), sr)
                refs.append({"file": f"{i}.wav", "text": ph, "intent": intent, "synthetic": True})
                i += 1
    json.dump(refs, open(out / "refs.json", "w", encoding="utf-8"), ensure_ascii=False)
    return {"n": len(refs), "dir": str(out)}


def w_tts_quality(p):
    import soundfile as sf
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    import piper  # noqa: F401
    m = Meter()
    m.mark_baseline()
    v, onnx_path = _piper_load(p["voice"])
    m.mark_loaded()
    list(_piper_stream(v, "xin chào"))  # warm-up
    ttfa, synth_s, audio_s, peaks, clips, rms, lead = [], 0.0, 0.0, [], [], [], []
    for i, s in enumerate(p["sentences"]):
        t0 = time.time()
        first, chunks = None, []
        for a, sr in _piper_stream(v, s):
            if first is None:
                first = (time.time() - t0) * 1000
            chunks.append(a)
        synth_s += time.time() - t0
        x = np.concatenate(chunks)
        audio_s += len(x) / sr
        ttfa.append(first)
        pk = float(np.abs(x).max()) + 1e-9
        peaks.append(20 * np.log10(pk))
        clips.append(float(np.mean(np.abs(x) >= 0.999)))
        rms.append(20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-9))
        idx = np.argmax(np.abs(x) > 0.02 * pk)
        lead.append(idx / sr * 1000)
        sf.write(str(out_dir / f"{i}.wav"), x, sr)
    res = m.report()
    res.update(ttfa_ms_p50=_pctl(ttfa, 50), ttfa_ms_p95=_pctl(ttfa, 95), rtf=round(synth_s / audio_s, 3),
               peak_dbfs_max=round(max(peaks), 1), clip_frac=round(float(np.mean(clips)), 5),
               rms_dbfs_mean=round(float(np.mean(rms)), 1), lead_silence_ms=round(float(np.mean(lead)), 0),
               speech_chars_per_s=round(sum(len(s) for s in p["sentences"]) / audio_s, 1),
               audio_s_total=round(audio_s, 1), wav_dir=str(out_dir),
               onnx_mb=round(Path(onnx_path).stat().st_size / 1048576, 1), runtime="onnxruntime(piper)")
    return res


def w_tts_judge(p):
    """STT cố định nghe lại wav (sạch + trong ồn nhà máy). Không đo RAM (tiến trình riêng)."""
    from faster_whisper import WhisperModel
    model = WhisperModel(p.get("judge", "base.en"), device="cpu", compute_type="int8",
                         download_root=str(Path("models") / "fw"))
    lang = p.get("language", "en")
    audios = [_load_audio16(Path(p["wav_dir"]) / f"{i}.wav") for i in range(len(p["sentences"]))]
    out = {}
    for name, kind, snr in _conditions(p):
        rng = np.random.default_rng(2)
        hyps = []
        for a in audios:
            x = a if kind is None else _mix(a, kind, snr, rng, p.get("noise_dir"))
            segs, _ = model.transcribe(x, language=lang, beam_size=1)
            hyps.append(" ".join(s.text for s in segs))
        out[f"rt_wer_{name}"] = round(_wer(p["sentences"], hyps), 4)
        out[f"rt_cer_{name}"] = round(_cer(p["sentences"], hyps), 4)
    out["judge"] = f"faster-whisper {p.get('judge', 'base.en')} int8"
    return out


# ================================================================ CV
def _yolo_post(out, conf=0.25, iou=0.45):
    a = out[0]
    a = a[0].T if a.shape[1] < a.shape[2] else a[0]
    sc = a[:, 4:].max(1)
    k = sc > conf
    a, sc = a[k], sc[k]
    if len(a) == 0:
        return 0
    b = np.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2, a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], 1)
    order, keep = sc.argsort()[::-1], 0
    while order.size:
        i = order[0]
        keep += 1
        r = order[1:]
        xx1, yy1 = np.maximum(b[i, 0], b[r, 0]), np.maximum(b[i, 1], b[r, 1])
        xx2, yy2 = np.minimum(b[i, 2], b[r, 2]), np.minimum(b[i, 3], b[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ai = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
        ar = (b[r, 2] - b[r, 0]) * (b[r, 3] - b[r, 1])
        order = r[inter / (ai + ar - inter + 1e-9) < iou]
    return keep


def _start_hogs(n):
    return [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(n)]


def _stop_hogs(hs):
    for h in hs:
        h.kill()


def _cpu_temp_c():
    vals = []
    for f in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            vals.append(int(f.read_text()) / 1000)
        except Exception:
            pass
    return round(max(vals), 1) if vals else None


def w_cv_latency(p):
    """Chạy liên tục duration_s giây: pre (resize+norm) + infer + post. kind: yolo|cls."""
    import onnxruntime as ort
    from PIL import Image
    sz = p["imgsz"]
    src = None
    for g in [p.get("image")] + [str(x) for x in list(Path("datasets/coco128/images/train2017").glob("*.jpg"))[:1]] \
            + [str(x) for x in list(Path("data/imagenet").glob("*.jpg"))[:1]]:
        if g and Path(g).exists():
            src = g
            break
    if src:
        frame = Image.open(src).convert("RGB").resize((640, 480))
    else:
        rng = np.random.default_rng(0)
        frame = Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8))
    m = Meter()
    m.mark_baseline()
    so = ort.SessionOptions()
    so.intra_op_num_threads = p.get("threads", 4)
    sess = ort.InferenceSession(p["onnx"], so, providers=["CPUExecutionProvider"])
    m.mark_loaded()
    name = sess.get_inputs()[0].name
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    hogs = _start_hogs(p.get("hog_procs", 0))
    t_pre, t_inf, t_post, tot = [], [], [], []
    temp0, t_end = _cpu_temp_c(), time.time() + p.get("duration_s", 30)
    try:
        while time.time() < t_end:
            a = time.time()
            im = frame.resize((sz, sz))
            x = np.asarray(im, dtype=np.float32) / 255.0
            if p.get("kind", "yolo") == "cls":
                x = (x - mean) / std
            x = x.transpose(2, 0, 1)[None].astype(np.float32)
            b = time.time()
            o = sess.run(None, {name: x})
            c = time.time()
            _ = _yolo_post(o) if p.get("kind", "yolo") == "yolo" else int(o[0].argmax())
            d = time.time()
            t_pre.append((b - a) * 1000), t_inf.append((c - b) * 1000), t_post.append((d - c) * 1000), tot.append((d - a) * 1000)
    finally:
        _stop_hogs(hogs)
    q = max(1, len(tot) // 4)
    res = m.report()
    res.update(frames=len(tot), fps_sustained=round(len(tot) / p.get("duration_s", 30), 2),
               ms_total_p50=_pctl(tot, 50), ms_total_p95=_pctl(tot, 95), ms_total_p99=_pctl(tot, 99),
               ms_total_max=_pctl(tot, 100), ms_pre_p50=_pctl(t_pre, 50), ms_infer_p50=_pctl(t_inf, 50),
               ms_post_p50=_pctl(t_post, 50), drift_ratio=round(float(np.median(tot[-q:]) / np.median(tot[:q])), 3),
               temp_start_c=temp0, temp_end_c=_cpu_temp_c(), threads=p.get("threads", 4),
               hog_procs=p.get("hog_procs", 0), image_src=src or "noise", runtime="onnxruntime")
    return res


def _corrupt(img, name, rng):
    from io import BytesIO
    from PIL import Image, ImageEnhance, ImageFilter
    if name == "clean":
        return img
    if name == "dark":
        return ImageEnhance.Brightness(img).enhance(0.35)
    if name == "bright":
        return ImageEnhance.Brightness(img).enhance(1.8)
    if name == "lowcontrast":
        return ImageEnhance.Contrast(img).enhance(0.4)
    if name == "blur":
        return img.filter(ImageFilter.GaussianBlur(2.5))
    if name == "motion":
        a = np.asarray(img, dtype=np.float32)
        k = 15
        a = sum(np.roll(a, s, axis=1) for s in range(-(k // 2), k // 2 + 1)) / k
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    if name == "noise":
        a = np.asarray(img, dtype=np.float32) + rng.normal(0, 20, (img.height, img.width, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    if name == "jpeg":
        buf = BytesIO()
        img.save(buf, "JPEG", quality=10)
        return Image.open(BytesIO(buf.getvalue())).convert("RGB")
    raise ValueError(name)


CORRUPTIONS = ["clean", "dark", "bright", "lowcontrast", "blur", "motion", "noise", "jpeg"]


def w_cls_corrupt(p):
    import onnxruntime as ort
    from PIL import Image
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    items = json.load(open(Path(p["data_dir"]) / "labels.json", encoding="utf-8"))[: p.get("n", 200)]

    def prep(img):
        w, h = img.size
        s = 256 / min(w, h)
        img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
        w, h = img.size
        l, t = (w - 224) // 2, (h - 224) // 2
        a = np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255.0
        return ((a - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)

    so = ort.SessionOptions()
    so.intra_op_num_threads = p.get("threads", 4)
    sess = ort.InferenceSession(p["onnx"], so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = {}
    for c in p.get("corruptions", CORRUPTIONS):
        rng, ok = np.random.default_rng(0), 0
        for it in items:
            img = _corrupt(Image.open(Path(p["data_dir"]) / it["file"]).convert("RGB"), c, rng)
            ok += int(sess.run(None, {name: prep(img)})[0].argmax()) == it["label"]
        out[f"top1_{c}"] = round(ok / len(items), 4)
    out["n_samples"] = len(items)
    return out


def w_det_corrupt(p):
    """mAP50-95 của YOLO-ONNX trên COCO128 sau khi làm hỏng ảnh (ultralytics sẽ kéo torch -> bỏ qua cột RAM)."""
    import shutil
    import tempfile

    import yaml
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    model = YOLO(p["onnx"], task="detect")
    model.val(data="coco128.yaml", imgsz=p["imgsz"], device="cpu", batch=1, verbose=False, plots=False)  # tải COCO128
    root = Path(SETTINGS["datasets_dir"]) / "coco128"
    cfg = yaml.safe_load(open(Path(sys.modules["ultralytics"].__file__).parent / "cfg/datasets/coco128.yaml",
                              encoding="utf-8"))
    tmp, out = Path(tempfile.mkdtemp()), {}
    for c in p.get("corruptions", CORRUPTIONS):
        rng = np.random.default_rng(0)
        d = tmp / c
        (d / "images/train2017").mkdir(parents=True)
        shutil.copytree(root / "labels/train2017", d / "labels/train2017")
        for f in sorted((root / "images/train2017").glob("*.jpg")):
            _corrupt(Image.open(f).convert("RGB"), c, rng).save(d / "images/train2017" / f.name, quality=95)
        yml = d / "data.yaml"
        yaml.safe_dump({"path": str(d), "train": "images/train2017", "val": "images/train2017", "names": cfg["names"]},
                       open(yml, "w", encoding="utf-8"))
        r = model.val(data=str(yml), imgsz=p["imgsz"], device="cpu", batch=1, verbose=False, plots=False)
        out[f"map50_95_{c}"] = round(float(r.box.map), 4)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


# ---------------------------------------------------------------- phát hiện kẹt (video băng tải tổng hợp)
def _render_conveyor(H, W, fps, seed, n_frames, stuck_frame=None, stop_frame=None, flicker=0.0, noise=0.0,
                     jitter=0, empty=False, speed_scale=1.0):
    """Băng tải nhìn từ trên: texture di chuyển sang phải, sản phẩm tối. Trả về list frame uint8 (H,W)."""
    rng = np.random.default_rng(seed)
    P = W * 2
    tex = rng.standard_normal((H, P)).astype(np.float32)
    c = np.cumsum(np.pad(tex, ((0, 0), (3, 2)), mode="wrap"), axis=1)
    tex = 150 + 20 * (c[:, 5:] - c[:, :-5]) / 5 * 2.2
    v = W / 4.0 * speed_scale  # px/s -> sản phẩm đi hết khung ~4 s
    pw, ph, S = max(8, W // 7), H // 3, W // 2
    cyc = 3 * S
    y0 = H // 2 - ph // 2
    j0 = int(rng.integers(0, cyc))
    frozen, stuck_x0 = {}, {}
    frames = []
    for i in range(n_frames):
        t = i / fps
        eff_t = t if stop_frame is None or i < stop_frame else stop_frame / fps
        off = int(round(v * eff_t))
        f = np.roll(tex, off, axis=1)[:, :W].copy()
        if not empty:
            xs = [((j0 + j * S + v * eff_t) % cyc) - pw for j in range(3)]
            if stuck_frame is not None and i >= stuck_frame:
                if not frozen:  # đóng băng sản phẩm gần giữa khung + các sản phẩm phía sau
                    t_s = stuck_frame / fps
                    x_s = [((j0 + j * S + v * t_s) % cyc) - pw for j in range(3)]
                    vis = [j for j in range(3) if 0 < x_s[j] < W - pw]
                    if vis:
                        tgt = min(vis, key=lambda j: abs(x_s[j] - W * 0.5))
                        for j in vis:
                            if x_s[j] <= x_s[tgt]:
                                frozen[j] = x_s[j]
                    stuck_x0.update({j: x_s[j] for j in range(3)})
                xs = [frozen.get(j, xs[j]) for j in range(3)]
                xs = [x if (j in frozen or x >= stuck_x0[j] - 1e-6) else -1e9 for j, x in enumerate(xs)]
            for x in xs:
                xi = int(round(x))
                a, b = max(0, xi), min(W, xi + pw)
                if b > a:
                    f[y0:y0 + ph, a:b] = 50 + 6 * rng.standard_normal((ph, b - a))
        if flicker:
            f *= 1 + flicker * np.sin(2 * np.pi * 3 * t)
        if noise:
            f += rng.normal(0, noise, f.shape)
        if jitter:
            f = np.roll(f, (int(rng.integers(-jitter, jitter + 1)), int(rng.integers(-jitter, jitter + 1))), axis=(0, 1))
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


class _StuckDetector:
    """Detector nhẹ (numpy thuần): (1) tốc độ belt từ texture dải trên, (2) segment sản phẩm tối ở dải giữa,
    (3) 'kẹt' = có segment đứng yên trong khi belt đang chạy đủ lâu; 'dừng' = belt không chạy."""

    def __init__(self, H, W, fps, dt=0.25, hold_s=1.0):
        self.H, self.W, self.fps = H, W, fps
        self.k = max(1, round(dt * fps))
        self.hold = max(1, int(np.ceil(hold_s * fps)))
        self.exp = W / 4.0 * (self.k / fps)  # dịch chuyển kỳ vọng (thực tế: lấy từ encoder)
        self.tol = max(1.5, 0.25 * self.exp)
        self.smax = int(self.exp * 2) + 2
        self.minw = max(4, W // 14)
        self.r0, self.r1 = int(0.42 * H), max(int(0.42 * H) + 2, int(0.58 * H))
        self.buf = deque(maxlen=self.k + 1)
        self.c_stuck = self.c_stop = 0

    def _segs(self, f):
        thr = 0.55 * float(np.median(f))
        cols = (f[self.r0:self.r1] < thr).mean(0) >= 0.75
        d = np.diff(np.concatenate([[0], cols.astype(np.int8), [0]]))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        return [(a + b) / 2 for a, b in zip(st, en) if b - a >= self.minw and a > 0 and b < self.W]

    def process(self, f):
        f = f.astype(np.float32)
        top = f[: max(2, self.H // 6)].mean(0)
        top = (top - top.mean()) / (top.std() + 1e-6)
        self.buf.append((top, self._segs(f)))
        if len(self.buf) <= self.k:
            return None
        ptop, psegs = self.buf[0]
        best, bs = -9.0, 0
        for s in range(0, self.smax + 1):
            c = float(np.mean(top[s:] * ptop[: self.W - s]))
            if c > best:
                best, bs = c, s
        moving = bs >= 0.5 * self.exp
        stat = any(any(abs(c - pc) <= self.tol for pc in psegs) for c in self.buf[-1][1])
        self.c_stop = 0 if moving else self.c_stop + 1
        self.c_stuck = self.c_stuck + 1 if (moving and stat) else 0
        if self.c_stop >= self.hold:
            return "stopped"
        if self.c_stuck >= self.hold:
            return "stuck"
        return None


def _run_det(frames, det):
    times, alarms = [], []
    for fr in frames:
        t = time.perf_counter()
        a = det.process(fr)
        times.append(time.perf_counter() - t)
        alarms.append(a)
    return alarms, times


def w_cv_blockage(p):
    H, W, fps = p["H"], p["W"], p["fps"]
    seeds, dur = p.get("seeds", 12), p.get("dur_s", 20)
    n, ts = int(fps * dur), int(fps * 8)
    jit = max(1, round(H / 60))
    m = Meter()
    m.mark_baseline()
    det_stuck, det_stop, all_times = [], [], []
    for sd in range(seeds):
        al, tm = _run_det(_render_conveyor(H, W, fps, sd, n, stuck_frame=ts, noise=3), _StuckDetector(H, W, fps))
        all_times += tm
        hit = next((i for i, a in enumerate(al) if a == "stuck" and i >= ts), None)
        early = any(a for a in al[:ts])
        det_stuck.append((hit - ts) / fps if hit is not None and not early else None)
        al, tm = _run_det(_render_conveyor(H, W, fps, 100 + sd, n, stop_frame=ts, noise=3), _StuckDetector(H, W, fps))
        hit = next((i for i, a in enumerate(al) if a in ("stopped", "stuck") and i >= ts), None)
        det_stop.append((hit - ts) / fps if hit is not None else None)
    scen = {"normal": {}, "flicker": {"flicker": 0.3}, "noise": {"noise": 10}, "empty": {"empty": True},
            "jitter": {"jitter": jit}, "slow_belt": {"speed_scale": 0.75},
            "noise_hard": {"noise": 30}, "flicker_hard": {"flicker": 0.5, "noise": 10},
            "jitter_hard": {"jitter": 2 * jit, "noise": 10}}
    fa, sim_s, fa_by = 0, 0.0, {}
    for nm, kw in scen.items():
        c = 0
        for sd in range(seeds):
            al, _ = _run_det(_render_conveyor(H, W, fps, 200 + sd, n, **kw), _StuckDetector(H, W, fps))
            prev = None
            for a in al:
                if a and a != prev:
                    c += 1
                prev = a
            sim_s += n / fps
        fa_by[nm] = c
        fa += c
    hard = []  # kẹt thật trong điều kiện xấu (ồn mạnh + chớp sáng + rung camera)
    for sd in range(seeds):
        al, _ = _run_det(_render_conveyor(H, W, fps, 300 + sd, n, stuck_frame=ts, noise=25, flicker=0.3, jitter=jit),
                         _StuckDetector(H, W, fps))
        hit = next((i for i, a in enumerate(al) if a == "stuck" and i >= ts), None)
        hard.append((hit - ts) / fps if hit is not None else None)
    ok = [d for d in det_stuck if d is not None]
    ok2 = [d for d in det_stop if d is not None]
    okh = [d for d in hard if d is not None]
    res = m.report()
    res.update(res_hw=f"{W}x{H}", fps=fps, ms_per_frame=round(float(np.mean(all_times)) * 1000, 3),
               stuck_detect_rate=round(len(ok) / seeds, 3),
               stuck_delay_s_p50=round(float(np.median(ok)), 2) if ok else None,
               stuck_delay_s_max=round(float(max(ok)), 2) if ok else None,
               stuck_hard_detect_rate=round(len(okh) / seeds, 3),
               stuck_hard_delay_s_p50=round(float(np.median(okh)), 2) if okh else None,
               stop_detect_rate=round(len(ok2) / seeds, 3),
               stop_delay_s_p50=round(float(np.median(ok2)), 2) if ok2 else None,
               false_alarm_events=fa, false_alarm_by_scenario=fa_by, sim_minutes_normal=round(sim_s / 60, 1),
               note="video tổng hợp — chỉ kiểm tra logic/chi phí, KHÔNG thay cho dữ liệu camera thật",
               runtime="numpy")
    return res


# ================================================================ E2E
def w_e2e(p):
    """STT -> NLU -> TTS (time-to-first-audio) trong MỘT tiến trình => RAM chung thật + latency lệnh->phản hồi."""
    refs = _load_refs(p["data_dir"])[: p.get("n", 30)]
    _pre_import(p)
    import piper  # noqa: F401
    m = Meter()
    m.mark_baseline()
    tr = _make_asr(p)
    v, _ = _piper_load(p["voice"])
    m.mark_loaded()
    audios = [_load_audio16(Path(p["data_dir"]) / r["file"]) for r in refs]
    resp = p["responses"]
    tr(audios[0])
    list(_piper_stream(v, resp.get("status", "ok")))
    hogs = _start_hogs(p.get("hog_procs", 0))
    t_stt, t_tts, t_all, ok = [], [], [], 0
    try:
        for a, r in zip(audios, refs):
            t0 = time.time()
            h = tr(a)
            t1 = time.time()
            intent = _nlu(h)
            ok += intent == r["intent"]
            text = resp.get(intent, resp.get("none", "Xin lỗi, tôi không hiểu."))
            for _a, _sr in _piper_stream(v, text):
                break
            t2 = time.time()
            t_stt.append((t1 - t0) * 1000), t_tts.append((t2 - t1) * 1000), t_all.append((t2 - t0) * 1000)
    finally:
        _stop_hogs(hogs)
    slm = p.get("slm_ms", 0)
    res = m.report()
    res.update(stt_ms_p50=_pctl(t_stt, 50), stt_ms_p95=_pctl(t_stt, 95), tts_ttfa_ms_p50=_pctl(t_tts, 50),
               e2e_ms_p50=_pctl(t_all, 50), e2e_ms_p95=_pctl(t_all, 95),
               e2e_with_slm_ms_p50=round(_pctl(t_all, 50) + slm, 1), slm_ms_assumed=slm,
               intent_acc=round(ok / len(refs), 4), hog_procs=p.get("hog_procs", 0), n_samples=len(refs),
               audio_s_mean=round(float(np.mean([len(a) / 16000 for a in audios])), 2), runtime="fw+piper")
    return res


KINDS = dict(W.KINDS)
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
