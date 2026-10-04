"""Hạ tầng dùng chung cho stt_lite / tts_lite / cv_lite / ram_budget.

Mỗi model chạy trong một TIẾN TRÌNH CON (workers.py) để RAM không bị cộng dồn
từ model trước hoặc từ torch/CUDA/dataset của notebook.
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = Path("results_lite")
RESULTS.mkdir(exist_ok=True)

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Please turn off the power supply before opening the cabinet.",
    "Temperature in the lab is currently twenty three degrees Celsius.",
    "Warning: the gas valve on the second bench is still open.",
    "Set the oscilloscope to ten megahertz and press start.",
    "The experiment will begin in approximately five minutes.",
    "All safety equipment must be worn inside this area.",
    "Battery level is low, please connect the charger soon.",
    "The calibration results have been saved to the shared drive.",
    "Remember to log the reading before switching off the device.",
]

OCR_STRINGS = [
    "TEMP: 23.5 C", "VOLTAGE 12.0V", "PRESSURE 1.2 BAR", "SN-884213-A",
    "BATCH 2026-10-03", "RPM 3450", "CH1 ON  CH2 OFF", "pH 7.02",
    "FLOW RATE 1.5 L/MIN", "ERROR CODE E04",
]


# ------------------------------------------------------------ chạy cô lập
def run_isolated(kind: str, params: dict, timeout: int = 3600) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TOKENIZERS_PARALLELISM="false",
               PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    try:
        cp = subprocess.run([sys.executable, str(HERE / "workers.py"), kind], input=json.dumps(params),
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout sau {timeout}s"}
    for line in reversed(cp.stdout.splitlines()):
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"error": (cp.stderr or cp.stdout)[-1800:] or f"tiến trình con thoát mã {cp.returncode}"}


def _path(subdir, name):
    d = RESULTS / subdir
    d.mkdir(exist_ok=True, parents=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return d / f"{safe}.json"


def bench(subdir: str, name: str, kind: str, params: dict, force: bool = False) -> dict:
    """Chạy 1 model cô lập; dùng cache JSON nếu đã có (trừ khi force=True)."""
    p = _path(subdir, name)
    if p.exists() and not force:
        return json.load(open(p, encoding="utf-8"))
    res = run_isolated(kind, params)
    if "error" in res:
        print(f"[ERROR] {name}:\n{res['error']}\n")
        return {"model": name, "error": res["error"][-300:]}
    row = {"model": name, **res}
    json.dump(row, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[done] {name}: " + ", ".join(f"{k}={v}" for k, v in row.items()
                                          if k in ("ram_total_peak_mb", "ram_peak_mb", "wer", "cer", "top1_acc",
                                                   "map50_95", "rtf", "roundtrip_wer", "ms_per_image")))
    return row


def update_row(subdir: str, name: str, **kw):
    p = _path(subdir, name)
    if p.exists():
        row = json.load(open(p, encoding="utf-8"))
        row.update(kw)
        json.dump(row, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def table(subdir: str) -> pd.DataFrame:
    rows = [json.load(open(f, encoding="utf-8")) for f in sorted((RESULTS / subdir).glob("*.json"))]
    return pd.DataFrame(rows)


def plot_tradeoff(df, x, y, title, ylabel, fname):
    import matplotlib.pyplot as plt
    ok = df.dropna(subset=[x, y])
    if ok.empty:
        print("Không có dữ liệu để vẽ.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ok[x], ok[y], s=80)
    for _, r in ok.iterrows():
        ax.annotate(r["model"], (r[x], r[y]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("RAM tổng đỉnh của 1 tiến trình độc lập (MB)  — thấp hơn = tốt hơn")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.show()


# ------------------------------------------------------------ chuẩn bị dữ liệu
def hf_login():
    """Dùng biến môi trường HF_TOKEN (không ghi token vào notebook)."""
    from huggingface_hub import login
    tok = os.environ.get("HF_TOKEN")
    if tok:
        login(token=tok)
    else:
        print("Chưa có HF_TOKEN — nếu cần dataset gated (ImageNet), chạy `huggingface-cli login` trước.")


def prep_librispeech(n: int = 200, outdir: str = "data/librispeech") -> str:
    import io
    import soundfile as sf
    from datasets import Audio, load_dataset
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "refs.json").exists() and len(json.load(open(out / "refs.json"))) >= n:
        return str(out)
    ds = load_dataset("openslr/librispeech_asr", data_dir="clean", split="test", streaming=True,
                      revision="refs/convert/parquet").cast_column("audio", Audio(decode=False))
    refs = []
    for i, ex in enumerate(ds.take(n)):
        arr, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        sf.write(str(out / f"{i}.wav"), arr, sr)
        refs.append({"file": f"{i}.wav", "text": ex["text"]})
    json.dump(refs, open(out / "refs.json", "w", encoding="utf-8"))
    return str(out)


def prep_imagenet(n: int = 500, outdir: str = "data/imagenet") -> str:
    from datasets import load_dataset
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "labels.json").exists() and len(json.load(open(out / "labels.json"))) >= n:
        return str(out)
    hf_login()
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    items = []
    for i, ex in enumerate(ds.take(n)):
        ex["image"].convert("RGB").save(out / f"{i}.jpg", quality=95)
        items.append({"file": f"{i}.jpg", "label": int(ex["label"])})
    json.dump(items, open(out / "labels.json", "w"))
    return str(out)


def render_ocr_set(outdir: str = "data/ocr", n: int = 40, seed: int = 0) -> str:
    """Ảnh chữ tổng hợp CÓ nhiễu: xoay nhẹ, làm mờ, nhiễu hạt, đổi cỡ chữ/độ tương phản."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "refs.json").exists():
        return str(out)
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    fonts = [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\arial.ttf", "DejaVuSansMono.ttf", "DejaVuSans.ttf"]
    refs = []
    for i in range(n):
        text = OCR_STRINGS[i % len(OCR_STRINGS)]
        size = rng.randint(24, 36)
        font = None
        for fp in fonts:
            try:
                font = ImageFont.truetype(fp, size)
                break
            except OSError:
                pass
        if font is None:
            font = ImageFont.load_default(size=size)
        img = Image.new("RGB", (460, 110), (rng.randint(225, 255),) * 3)
        ImageDraw.Draw(img).text((12, 30), text, fill=(rng.randint(0, 60),) * 3, font=font)
        img = img.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC, fillcolor=(240, 240, 240))
        if rng.random() < 0.6:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
        arr = np.asarray(img).astype(np.float32) + nrng.normal(0, rng.uniform(2, 12), (img.height, img.width, 3))
        Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).save(out / f"{i}.png")
        refs.append({"file": f"{i}.png", "text": text})
    json.dump(refs, open(out / "refs.json", "w", encoding="utf-8"))
    return str(out)
