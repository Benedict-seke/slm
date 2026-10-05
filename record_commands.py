"""Ghi âm bộ lệnh THẬT cho benchmark STT (khuyến nghị: 3+ người, có/không bật máy ồn nền).

    pip install sounddevice soundfile
    python record_commands.py --speaker hieu --reps 2

Mỗi lệnh: Enter để bắt đầu, ghi 4 giây (16 kHz mono). Kết quả nối thêm vào data/vi_commands/refs.json
(cùng định dạng với bộ tổng hợp, synthetic=False) nên eval_project.ipynb dùng được ngay.
"""
import argparse
import json
import random
from pathlib import Path

from bench_ext import VI_VARIANTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--out", default="data/vi_commands_real")
    a = ap.parse_args()
    import sounddevice as sd
    import soundfile as sf

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rp = out / "refs.json"
    refs = json.load(open(rp, encoding="utf-8")) if rp.exists() else []
    items = [(i, p) for i, ps in VI_VARIANTS.items() for p in ps] * a.reps
    random.shuffle(items)
    for k, (intent, text) in enumerate(items, 1):
        input(f"[{k}/{len(items)}] Nói: «{text}»   (Enter để ghi)")
        x = sd.rec(int(a.secs * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        fn = f"{a.speaker}_{len(refs)}.wav"
        sf.write(str(out / fn), x, 16000)
        refs.append({"file": fn, "text": text, "intent": intent, "speaker": a.speaker, "synthetic": False})
        json.dump(refs, open(rp, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Xong: {len(refs)} bản ghi trong {out}. Dùng DATA_VI = '{out}' trong notebook.")


if __name__ == "__main__":
    main()
