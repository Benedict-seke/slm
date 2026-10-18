"""Chuẩn bị dữ liệu CÔNG KHAI (không cần tự thu):
  speech : Google Speech Commands v0.02 (CC BY 4.0) -> data/speech_cmds/{refs.json,*.wav}   (từ stop/go/on/off/... giọng người thật)
  noise  : tiếng máy thật từ DCASE 2020 Task 2 dev (ToyConveyor / valve ...; CC BY-NC-SA 4.0) hoặc MIMII -> data/noise/*.wav

Cách dùng:
  python prep_real_data.py speech --src speech_commands_test_set_v0.02.tar.gz --out data/speech_cmds --n 100
  python prep_real_data.py speech --download            # thử tải test set (URL và dung lượng CHƯA xác minh — nếu lỗi, tải tay)
  python prep_real_data.py noise  --src dev_data_ToyConveyor.zip --out data/noise --n 24
  python prep_real_data.py noise  --src dev_data_valve.zip --out data/noise --n 12 --append

Nguồn tải (đã kiểm trang):
  Speech Commands : test set v0.02 (thư mục con theo từ: stop, go, on, off, yes, no, up, down, left, right, _silence_, _unknown_)
  Noise           : https://zenodo.org/records/3678171  (dev_data_ToyConveyor.zip 1.9 GB, dev_data_valve.zip 1.0 GB, ...)
                    LƯU Ý: MIMII gốc (zenodo 3384388) mỗi file 6.9-10.9 GB — đừng tải nếu không cần.
"""
import argparse
import io
import json
import random
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

TEST_URL = "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"   # chưa xác minh
TARGET_WORDS = ["stop", "go", "on", "off", "yes", "no", "up", "down", "left", "right"]


def _to_16k_mono(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x.astype(np.float32)


def prep_speech(src, out="data/speech_cmds", n_per_word=100, words=TARGET_WORDS, seed=0, pad_s=0.25):
    """src: file .tar.gz của test set hoặc thư mục đã giải nén. Mỗi từ lấy tối đa n_per_word clip, đệm im lặng pad_s hai đầu.
    refs.json: [{file, text (=từ), intent ('stop' nếu từ là stop, ngược lại 'none'), word, synthetic: False, source}]"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    items = {w: [] for w in words}                       # w -> [(name, bytes)]
    src = Path(src)
    if src.is_dir():
        for w in words:
            for f in sorted((src / w).glob("*.wav")) if (src / w).is_dir() else []:
                items[w].append((f.name, f.read_bytes()))
    else:
        with tarfile.open(src) as tf:
            for m in tf.getmembers():
                parts = Path(m.name).parts
                if m.isfile() and m.name.endswith(".wav") and len(parts) >= 2 and parts[-2] in items:
                    items[parts[-2]].append((parts[-1], tf.extractfile(m).read()))
    refs = []
    for w in words:
        rng.shuffle(items[w])
        for name, raw in items[w][:n_per_word]:
            x, sr = sf.read(io.BytesIO(raw), dtype="float32")
            x = _to_16k_mono(x, sr)
            pad = np.zeros(int(pad_s * 16000), np.float32)
            fn = f"{w}_{len(refs)}.wav"
            sf.write(str(out / fn), np.concatenate([pad, x, pad]), 16000)
            refs.append({"file": fn, "text": w, "intent": "stop" if w == "stop" else "none", "word": w,
                         "synthetic": False, "source": "speech_commands_v0.02"})
    if not refs:
        raise SystemExit(f"Không tìm thấy clip nào trong {src} — cần các thư mục con {words[:3]}...")
    with open(out / "refs.json", "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False)
    cnt = {w: sum(r["word"] == w for r in refs) for w in words}
    print(f"Đã lưu {len(refs)} clip vào {out}: {cnt}")
    return str(out)


def download_speech_test(dest="speech_commands_test_set_v0.02.tar.gz"):
    import urllib.request
    print("Tải", TEST_URL)
    urllib.request.urlretrieve(TEST_URL, dest)
    return dest


def prep_noise(src, out="data/noise", n=24, seed=0, pattern="normal", append=False):
    """Lấy n clip nền máy từ zip (đọc từng file, không giải nén cả zip) hoặc thư mục. pattern: chỉ lấy tên có chuỗi này
    ('normal' = máy chạy bình thường, nền xưởng vẫn có). Đổi sang mono 16 kHz. Cảnh báo giấy phép CC BY-NC-SA: ghi nguồn."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    src = Path(src)
    start = len(list(out.glob("*.wav"))) if append else 0
    if not append:
        for f in out.glob("*.wav"):
            f.unlink()
    tag = src.stem.replace("dev_data_", "")
    if src.is_dir():
        names = [p for p in src.rglob("*.wav") if pattern in p.name]
        read = lambda p: sf.read(str(p), dtype="float32")   # noqa: E731
    else:
        zf = zipfile.ZipFile(src)
        names = [m for m in zf.namelist() if m.endswith(".wav") and pattern in Path(m).name]
        read = lambda m: sf.read(io.BytesIO(zf.read(m)), dtype="float32")   # noqa: E731
    if not names:
        raise SystemExit(f"Không có file .wav nào chứa '{pattern}' trong {src}")
    rng.shuffle(names)
    for k, nm in enumerate(names[:n]):
        x, sr = read(nm)
        sf.write(str(out / f"{tag}_{start + k:03d}.wav"), _to_16k_mono(x, sr), 16000)
    print(f"Đã lưu {min(n, len(names))} clip nền ({tag}) vào {out} (tổng {len(list(out.glob('*.wav')))} file)")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["speech", "noise"])
    ap.add_argument("--src")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int)
    ap.add_argument("--pattern", default="normal")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.what == "speech":
        src = download_speech_test() if a.download else a.src
        prep_speech(src, a.out or "data/speech_cmds", a.n or 100)
    else:
        prep_noise(a.src, a.out or "data/noise", a.n or 24, pattern=a.pattern, append=a.append)
