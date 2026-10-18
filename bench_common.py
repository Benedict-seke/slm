"""Hạ tầng dùng chung cho stt_lite / tts_lite / cv_lite / ram_budget / latency / task benchmark.

Mỗi model chạy trong một TIẾN TRÌNH CON (workers*.py) để RAM không bị cộng dồn
từ model trước hoặc từ torch/CUDA/dataset của notebook.

CHANGELOG so với bản cũ (các lỗi đã sửa):
 1. bench() cache theo HASH THAM SỐ: đổi tham số/worker -> tự chạy lại thay vì lặng lẽ trả kết quả cũ.
    (File cache cũ không có hash vẫn được chấp nhận để không mất kết quả đã chạy.)
 2. Timeout/crash không còn để lại tiến trình mồ côi: diệt cả CÂY tiến trình con, và dọn các "hog" CPU
    (`python -c "while True: pass"`) bị bỏ lại — trước đây chúng quay vô hạn và làm méo các lần đo sau.
    Đồng thời stdout/stderr đi qua file tạm: hog mồ côi giữ pipe từng khiến subprocess.run TREO tới hết timeout.
 3. Ghi JSON nguyên tử (tmp + replace) và bỏ qua file cache hỏng thay vì làm sập notebook.
 4. run_script(script, ...) dùng chung: bench_ext / bench_lat không cần chép lại run_isolated (bản chép lại
    từng bỏ qua mọi sửa lỗi ở đây).
 5. plot_tradeoff nhận xlabel (nhãn trục x trước đây luôn ghi 'RAM...' dù x là gì).
 6. render_ocr_set tôn trọng n/seed (trước đây bỏ qua nếu refs.json đã tồn tại dù n khác).
 7. Mọi open() có encoding và đóng file (Windows mặc định cp1252).
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
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


# ------------------------------------------------------------ JSON an toàn
def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ------------------------------------------------------------ chạy cô lập
def _kill_tree(proc):
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        for c in parent.children(recursive=True):
            try:
                c.kill()
            except psutil.Error:
                pass
        parent.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _reap_hogs():
    """Diệt các hog CPU mồ côi do worker chết trước khi kịp dọn (chữ ký chính xác: -c 'while True: pass')."""
    try:
        import psutil
    except ImportError:
        return
    for pr in psutil.process_iter(["cmdline"]):
        try:
            cl = pr.info.get("cmdline") or []
            if len(cl) >= 3 and cl[-2] == "-c" and cl[-1] == "while True: pass":
                pr.kill()
        except Exception:
            pass


def run_script(script: str, kind: str, params: dict, timeout: int = 3600) -> dict:
    """Chạy `python <script> <kind>` (params JSON qua stdin) trong tiến trình con; đọc dòng @@RESULT@@.
    stdout/stderr ghi vào file tạm (không dùng pipe): nếu worker chết để lại hog mồ côi đang giữ pipe thì
    communicate() sẽ treo tới hết timeout (bản cũ có thể treo cả giờ); với file tạm thì không bao giờ treo."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TOKENIZERS_PARALLELISM="false",
               PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
        try:
            proc = subprocess.Popen([sys.executable, str(HERE / script), kind], stdin=subprocess.PIPE,
                                    stdout=fo, stderr=fe, env=env)
        except Exception as e:  # noqa: BLE001
            return {"error": f"không khởi chạy được {script}: {e}"}
        try:
            proc.stdin.write(json.dumps(params).encode("utf-8"))
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            _reap_hogs()
            return {"error": f"timeout sau {timeout}s"}
        fo.seek(0)
        fe.seek(0)
        out = fo.read().decode("utf-8", "replace")
        err = fe.read().decode("utf-8", "replace")
    for line in reversed(out.splitlines()):
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    _reap_hogs()
    return {"error": (err or out)[-1800:] or f"tiến trình con thoát mã {proc.returncode}"}


def run_isolated(kind: str, params: dict, timeout: int = 3600) -> dict:
    return run_script("workers.py", kind, params, timeout)


def _path(subdir, name):
    d = RESULTS / subdir
    d.mkdir(exist_ok=True, parents=True)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return d / f"{safe}.json"


def _phash(kind, params) -> str:
    blob = json.dumps([kind, params], sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:10]


def bench(subdir: str, name: str, kind: str, params: dict, force: bool = False) -> dict:
    """Chạy 1 model cô lập; dùng cache JSON nếu đã có VÀ tham số không đổi (trừ khi force=True)."""
    p = _path(subdir, name)
    h = _phash(kind, params)
    if p.exists() and not force:
        try:
            row = _read_json(p)
            if row.get("model") not in (None, name):
                # hai tên khác nhau cho cùng tên file (ký tự đặc biệt bị đổi thành '_') -> không được dùng nhầm cache
                print(f"[warn] {p.name} đang thuộc '{row.get('model')}', trùng tên file với '{name}' "
                      f"— chạy lại và ghi đè; hãy đặt tên cấu hình khác nhau ở ký tự chữ/số.")
            elif row.get("_phash") in (None, h):  # None = file cũ chưa có hash -> chấp nhận
                return row
            else:
                print(f"[stale] {name}: tham số đã đổi so với lần chạy trước -> chạy lại")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] cache hỏng {p.name} ({e}) -> chạy lại")
    res = run_isolated(kind, params)
    if "error" in res:
        print(f"[ERROR] {name}:\n{res['error']}\n")
        return {"model": name, "error": res["error"][-300:]}
    row = {"model": name, **res, "_phash": h}
    _write_json(p, row)
    print(f"[done] {name}: " + ", ".join(f"{k}={v}" for k, v in row.items()
                                          if k in ("ram_total_peak_mb", "ram_peak_mb", "wer", "cer", "top1_acc",
                                                   "map50_95", "rtf", "roundtrip_wer", "ms_per_image")))
    return row


def update_row(subdir: str, name: str, **kw):
    p = _path(subdir, name)
    if p.exists():
        row = _read_json(p)
        row.update(kw)
        _write_json(p, row)


def table(subdir: str) -> pd.DataFrame:
    rows = []
    for f in sorted((RESULTS / subdir).glob("*.json")):
        try:
            rows.append(_read_json(f))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] bỏ qua file hỏng {f.name}: {e}")
    df = pd.DataFrame(rows)
    return df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")


def plot_tradeoff(df, x, y, title, ylabel, fname,
                  xlabel="RAM tổng đỉnh của 1 tiến trình độc lập (MB)  — thấp hơn = tốt hơn"):
    import matplotlib.pyplot as plt
    ok = df.dropna(subset=[x, y])
    if ok.empty:
        print("Không có dữ liệu để vẽ.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ok[x], ok[y], s=80)
    for _, r in ok.iterrows():
        ax.annotate(r["model"], (r[x], r[y]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel(xlabel)
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
    if (out / "refs.json").exists() and len(_read_json(out / "refs.json")) >= n:
        return str(out)
    ds = load_dataset("openslr/librispeech_asr", data_dir="clean", split="test", streaming=True,
                      revision="refs/convert/parquet").cast_column("audio", Audio(decode=False))
    refs = []
    for i, ex in enumerate(ds.take(n)):
        arr, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        sf.write(str(out / f"{i}.wav"), arr, sr)
        refs.append({"file": f"{i}.wav", "text": ex["text"]})
    with open(out / "refs.json", "w", encoding="utf-8") as f:
        json.dump(refs, f)
    return str(out)


def prep_imagenet(n: int = 500, outdir: str = "data/imagenet") -> str:
    from datasets import load_dataset
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "labels.json").exists() and len(_read_json(out / "labels.json")) >= n:
        return str(out)
    hf_login()
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    items = []
    for i, ex in enumerate(ds.take(n)):
        ex["image"].convert("RGB").save(out / f"{i}.jpg", quality=95)
        items.append({"file": f"{i}.jpg", "label": int(ex["label"])})
    with open(out / "labels.json", "w", encoding="utf-8") as f:
        json.dump(items, f)
    return str(out)


def render_ocr_set(outdir: str = "data/ocr", n: int = 40, seed: int = 0) -> str:
    """Ảnh chữ tổng hợp CÓ nhiễu: xoay nhẹ, làm mờ, nhiễu hạt, đổi cỡ chữ/độ tương phản."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "meta.json"
    if (out / "refs.json").exists() and meta.exists():
        m = _read_json(meta)
        if m.get("n") == n and m.get("seed") == seed:
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
    with open(out / "refs.json", "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False)
    _write_json(meta, {"n": n, "seed": seed})
    return str(out)
