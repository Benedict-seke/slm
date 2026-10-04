"""Worker chạy trong TIẾN TRÌNH CON riêng cho từng model.

Gọi: python workers.py <kind>   (params JSON qua stdin; kết quả in ra dòng @@RESULT@@{json})

RAM được báo cáo theo 3 mốc, đều đo trong chính tiến trình con:
  ram_runtime_mb    = RSS sau khi import thư viện  - RSS lúc khởi động   (chi phí runtime: onnxruntime, torch...)
  ram_model_load_mb = RSS sau khi nạp model        - RSS trước khi nạp    (chi phí trọng số)
  ram_peak_mb       = RSS đỉnh khi suy luận        - RSS trước khi nạp    (chi phí model + bộ nhớ tạm)
  ram_total_peak_mb = RSS đỉnh                     - RSS lúc khởi động    (≈ tổng một dịch vụ độc lập)
"""
import gc
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path

import psutil

_PROC = psutil.Process(os.getpid())


def rss_mb() -> float:
    return _PROC.memory_info().rss / (1024 * 1024)


RSS_START = rss_mb()


class Meter:
    """Gọi mark_baseline() SAU khi import, TRƯỚC khi nạp model."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.samples = []
        self.base = None
        self.load = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(rss_mb())
            time.sleep(self.interval)

    def mark_baseline(self):
        gc.collect()
        self.base = rss_mb()
        self._t.start()

    def mark_loaded(self):
        self.load = rss_mb()

    def report(self) -> dict:
        self._stop.set()
        self._t.join(timeout=2)
        peak = max(self.samples + [self.base, self.load or 0.0, rss_mb()])
        return {
            "ram_runtime_mb": round(self.base - RSS_START, 1),
            "ram_model_load_mb": round((self.load or self.base) - self.base, 1),
            "ram_peak_mb": round(peak - self.base, 1),
            "ram_total_peak_mb": round(peak - RSS_START, 1),
        }


# ---------------------------------------------------------------- tiện ích
def _norm(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _wer(refs, hyps) -> float:
    import jiwer
    return jiwer.wer([_norm(r) for r in refs], [_norm(h) if _norm(h) else "<empty>" for h in hyps])


def _load_refs(d: str):
    return json.load(open(Path(d) / "refs.json", encoding="utf-8"))


# ---------------------------------------------------------------- selftest
def w_selftest(p):
    """Kiểm tra harness: cấp phát ~p['mb'] MB, delta báo cáo phải xấp xỉ giá trị đó."""
    import numpy as np
    m = Meter()
    m.mark_baseline()
    buf = np.ones(int(p.get("mb", 200)) * 1024 * 1024, dtype=np.uint8)  # ones() để chắc chắn commit trang nhớ
    m.mark_loaded()
    time.sleep(0.3)
    _ = int(buf.sum())
    return m.report()


# ---------------------------------------------------------------- STT
def w_stt_fw(p):
    """faster-whisper (CTranslate2) int8 trên CPU."""
    import numpy as np  # noqa: F401
    import soundfile as sf
    from faster_whisper import WhisperModel

    refs = _load_refs(p["data_dir"])
    m = Meter()
    m.mark_baseline()
    t0 = time.time()
    model = WhisperModel(p["size"], device="cpu", compute_type="int8", cpu_threads=p.get("threads", 4),
                         download_root=str(Path("models") / "fw"))
    load_s = time.time() - t0
    m.mark_loaded()

    hyps, audio_s, t1 = [], 0.0, time.time()
    for r in refs:
        audio, sr = sf.read(str(Path(p["data_dir"]) / r["file"]), dtype="float32")
        audio_s += len(audio) / sr
        segs, _ = model.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        hyps.append(" ".join(s.text for s in segs))
    infer_s = time.time() - t1
    out = m.report()
    out.update(load_s=round(load_s, 1), rtf=round(infer_s / audio_s, 3),
               wer=round(_wer([r["text"] for r in refs], hyps), 4), n_samples=len(refs), runtime="ctranslate2-int8")
    return out


def _ensure_vosk(name: str) -> str:
    import urllib.request
    root = Path("models") / "vosk"
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if not target.exists():
        z = root / f"{name}.zip"
        urllib.request.urlretrieve(f"https://alphacephei.com/vosk/models/{name}.zip", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
        z.unlink()
    return str(target)


def w_stt_vosk(p):
    import numpy as np
    import soundfile as sf
    import vosk

    vosk.SetLogLevel(-1)
    refs = _load_refs(p["data_dir"])
    path = _ensure_vosk(p["model"])
    m = Meter()
    m.mark_baseline()
    t0 = time.time()
    model = vosk.Model(path)
    load_s = time.time() - t0
    m.mark_loaded()

    hyps, audio_s, t1 = [], 0.0, time.time()
    for r in refs:
        audio, sr = sf.read(str(Path(p["data_dir"]) / r["file"]), dtype="float32")
        audio_s += len(audio) / sr
        rec = vosk.KaldiRecognizer(model, sr)
        rec.AcceptWaveform((audio * 32767).astype(np.int16).tobytes())
        hyps.append(json.loads(rec.FinalResult())["text"])
    infer_s = time.time() - t1
    out = m.report()
    out.update(load_s=round(load_s, 1), rtf=round(infer_s / audio_s, 3),
               wer=round(_wer([r["text"] for r in refs], hyps), 4), n_samples=len(refs), runtime="kaldi(vosk)")
    return out


def w_stt_hf(p):
    """Tham chiếu: model HF chạy bằng torch CPU (KHÔNG phải runtime triển khai thật trên UNO Q)."""
    import soundfile as sf
    import torch
    from transformers import pipeline

    refs = _load_refs(p["data_dir"])
    m = Meter()
    m.mark_baseline()
    t0 = time.time()
    asr = pipeline("automatic-speech-recognition", model=p["repo"], device=-1,
                   torch_dtype=torch.float32, trust_remote_code=True)
    load_s = time.time() - t0
    m.mark_loaded()

    hyps, audio_s, t1 = [], 0.0, time.time()
    for r in refs:
        audio, sr = sf.read(str(Path(p["data_dir"]) / r["file"]), dtype="float32")
        audio_s += len(audio) / sr
        hyps.append(asr(audio)["text"])
    infer_s = time.time() - t1
    out = m.report()
    out.update(load_s=round(load_s, 1), rtf=round(infer_s / audio_s, 3),
               wer=round(_wer([r["text"] for r in refs], hyps), 4), n_samples=len(refs), runtime="torch-cpu(ref)")
    return out


# ---------------------------------------------------------------- TTS
def w_tts_piper(p):
    import numpy as np
    import soundfile as sf
    from huggingface_hub import hf_hub_download
    from piper import PiperVoice

    voice = p["voice"]
    lang, name, quality = voice.split("-")[:3]
    sub = f"{lang[:2]}/{lang}/{name}/{quality}"
    onnx_path = hf_hub_download("rhasspy/piper-voices", f"{sub}/{voice}.onnx", local_dir="models/piper")
    hf_hub_download("rhasspy/piper-voices", f"{sub}/{voice}.onnx.json", local_dir="models/piper")

    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    m = Meter()
    m.mark_baseline()
    t0 = time.time()
    v = PiperVoice.load(onnx_path)
    load_s = time.time() - t0
    m.mark_loaded()

    total_audio, t1 = 0.0, time.time()
    for i, s in enumerate(p["sentences"]):
        if hasattr(v, "synthesize_stream_raw"):  # piper-tts < 1.3
            raw = b"".join(v.synthesize_stream_raw(s))
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sr = v.config.sample_rate
        else:
            chunks = list(v.synthesize(s))
            audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
            sr = chunks[0].sample_rate
        total_audio += len(audio) / sr
        sf.write(str(out_dir / f"{i}.wav"), audio, sr)
    synth_s = time.time() - t1
    out = m.report()
    out.update(load_s=round(load_s, 2), rtf=round(synth_s / total_audio, 3), wav_dir=str(out_dir),
               onnx_mb=round(Path(onnx_path).stat().st_size / 1048576, 1), runtime="onnxruntime(piper)")
    return out


def w_tts_espeak(p):
    exe = os.environ.get("ESPEAK_CMD") or shutil.which("espeak-ng") or r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    if not Path(exe).exists() and not shutil.which(exe):
        return {"error": "Chưa cài espeak-ng (https://github.com/espeak-ng/espeak-ng/releases) hoặc đặt biến ESPEAK_CMD."}
    import soundfile as sf
    out_dir = Path(p["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    total_audio, t1 = 0.0, time.time()
    for i, s in enumerate(p["sentences"]):
        wav = out_dir / f"{i}.wav"
        subprocess.run([exe, "-v", "en-us", "-s", "160", "-w", str(wav), s], check=True, capture_output=True)
        info = sf.info(str(wav))
        total_audio += info.frames / info.samplerate
    synth_s = time.time() - t1
    # espeak-ng chạy ở tiến trình ngoài nên không đo RSS được ở đây; thực tế thường < 20MB.
    return {"ram_runtime_mb": None, "ram_model_load_mb": None, "ram_peak_mb": None, "ram_total_peak_mb": 20.0,
            "ram_note": "ước lượng (tiến trình ngoài, thường <20MB) — hãy đo lại bằng Task Manager",
            "rtf": round(synth_s / total_audio, 3), "wav_dir": str(out_dir), "runtime": "espeak-ng"}


def w_wer_wavs(p):
    """Thẩm định TTS: cho STT cố định (faster-whisper base.en) nghe lại các wav -> WER. Không đo RAM."""
    import soundfile as sf
    from faster_whisper import WhisperModel
    model = WhisperModel("base.en", device="cpu", compute_type="int8", download_root=str(Path("models") / "fw"))
    hyps = []
    for i, _ in enumerate(p["sentences"]):
        audio, sr = sf.read(str(Path(p["wav_dir"]) / f"{i}.wav"), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        segs, _ = model.transcribe(audio, language="en", beam_size=1)
        hyps.append(" ".join(s.text for s in segs))
    return {"roundtrip_wer": round(_wer(p["sentences"], hyps), 4), "judge": "faster-whisper base.en int8"}


# ---------------------------------------------------------------- CV
def w_ort_footprint(p):
    """RAM thật của một model ONNX (không import torch)."""
    import numpy as np
    import onnxruntime as ort

    m = Meter()
    m.mark_baseline()
    so = ort.SessionOptions()
    so.intra_op_num_threads = p.get("threads", 4)
    t0 = time.time()
    sess = ort.InferenceSession(p["onnx"], so, providers=["CPUExecutionProvider"])
    load_s = time.time() - t0
    m.mark_loaded()
    x = np.random.rand(*p["shape"]).astype(np.float32)
    name = sess.get_inputs()[0].name
    for _ in range(2):
        sess.run(None, {name: x})
    n = p.get("runs", 20)
    t1 = time.time()
    for _ in range(n):
        sess.run(None, {name: x})
    out = m.report()
    out.update(load_s=round(load_s, 2), ms_per_image=round((time.time() - t1) / n * 1000, 1),
               onnx_mb=round(Path(p["onnx"]).stat().st_size / 1048576, 1), runtime="onnxruntime")
    return out


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def w_cls_ort(p):
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    mean = np.array(_MEAN, dtype=np.float32)
    std = np.array(_STD, dtype=np.float32)
    items = json.load(open(Path(p["data_dir"]) / "labels.json"))

    def prep(img):
        w, h = img.size
        s = 256 / min(w, h)
        img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
        w, h = img.size
        l, t = (w - 224) // 2, (h - 224) // 2
        a = np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255.0
        return ((a - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)

    m = Meter()
    m.mark_baseline()
    so = ort.SessionOptions()
    so.intra_op_num_threads = p.get("threads", 4)
    sess = ort.InferenceSession(p["onnx"], so, providers=["CPUExecutionProvider"])
    m.mark_loaded()
    name = sess.get_inputs()[0].name
    correct, t_inf = 0, 0.0
    for it in items:
        x = prep(Image.open(Path(p["data_dir"]) / it["file"]).convert("RGB"))
        t = time.time()
        pred = int(sess.run(None, {name: x})[0].argmax())
        t_inf += time.time() - t
        correct += pred == it["label"]
    out = m.report()
    out.update(top1_acc=round(correct / len(items), 4), ms_per_image=round(t_inf / len(items) * 1000, 1),
               n_samples=len(items), onnx_mb=round(Path(p["onnx"]).stat().st_size / 1048576, 1), runtime="onnxruntime")
    return out


def w_det_yolo_val(p):
    """mAP trên COCO128 bằng ONNX (ultralytics import torch cho dataloader nên ram_runtime lớn —
    dùng w_ort_footprint để lấy RAM model sạch)."""
    from ultralytics import YOLO
    m = Meter()
    m.mark_baseline()
    model = YOLO(p["onnx"], task="detect")
    m.mark_loaded()
    t0 = time.time()
    r = model.val(data="coco128.yaml", imgsz=p["imgsz"], device="cpu", batch=1, verbose=False, plots=False)
    out = m.report()
    out.update(map50_95=round(float(r.box.map), 4), map50=round(float(r.box.map50), 4),
               val_s=round(time.time() - t0, 1), note="COCO128: chỉ so sánh tương đối trong notebook này")
    return out


def w_ocr_rapid(p):
    import numpy as np
    import jiwer
    from PIL import Image
    from rapidocr_onnxruntime import RapidOCR

    refs = _load_refs(p["data_dir"])
    m = Meter()
    m.mark_baseline()
    t0 = time.time()
    eng = RapidOCR()
    load_s = time.time() - t0
    m.mark_loaded()
    cers, t1 = [], time.time()
    for r in refs:
        res, _ = eng(np.array(Image.open(Path(p["data_dir"]) / r["file"]).convert("RGB")))
        hyp = " ".join(x[1] for x in res) if res else ""
        cers.append(jiwer.cer(r["text"], hyp) if hyp else 1.0)
    out = m.report()
    out.update(load_s=round(load_s, 2), cer=round(sum(cers) / len(cers), 4),
               ms_per_image=round((time.time() - t1) / len(refs) * 1000, 1), n_samples=len(refs),
               runtime="onnxruntime(rapidocr)")
    return out


def w_ocr_tesseract(p):
    cmd = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if not Path(cmd).exists() and not shutil.which("tesseract"):
        return {"error": "Chưa cài Tesseract hoặc sai đường dẫn (đặt biến TESSERACT_CMD)."}
    import jiwer
    import pytesseract
    from PIL import Image
    if Path(cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = cmd
    refs = _load_refs(p["data_dir"])
    cers, t1 = [], time.time()
    for r in refs:
        hyp = pytesseract.image_to_string(Image.open(Path(p["data_dir"]) / r["file"]).convert("RGB"),
                                          config="--psm 7").strip()
        cers.append(jiwer.cer(r["text"], hyp) if hyp else 1.0)
    return {"cer": round(sum(cers) / len(cers), 4), "ms_per_image": round((time.time() - t1) / len(refs) * 1000, 1),
            "n_samples": len(refs), "ram_total_peak_mb": 80.0,
            "ram_note": "ước lượng (tiến trình ngoài) — hãy đo lại", "runtime": "tesseract"}


# ---------------------------------------------------------------- export ONNX
def w_export_timm(p):
    import torch
    import timm
    Path(p["out"]).parent.mkdir(parents=True, exist_ok=True)
    net = timm.create_model(p["timm_name"], pretrained=True).eval()
    x = torch.randn(1, 3, 224, 224)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=17)
    try:
        torch.onnx.export(net, x, p["out"], dynamo=False, **kw)
    except TypeError:  # torch cũ không có tham số dynamo
        torch.onnx.export(net, x, p["out"], **kw)
    return {"out": p["out"], "size_mb": round(Path(p["out"]).stat().st_size / 1048576, 1)}


def w_export_yolo(p):
    from ultralytics import YOLO
    path = YOLO(p["weights"]).export(format="onnx", imgsz=p["imgsz"], simplify=False, opset=17, half=False)
    dst = Path(p["out"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, dst)
    return {"out": str(dst), "size_mb": round(dst.stat().st_size / 1048576, 1)}


KINDS = {k[2:]: v for k, v in list(globals().items()) if k.startswith("w_") and callable(v)}

if __name__ == "__main__":
    kind = sys.argv[1]
    params = json.loads(sys.stdin.read() or "{}")
    try:
        res = KINDS[kind](params)
    except Exception:
        res = {"error": traceback.format_exc()[-1800:]}
    print("@@RESULT@@" + json.dumps(res))
