"""Ghi âm bộ lệnh THẬT cho benchmark STT / latency (khuyến nghị: 3+ người, có/không bật máy ồn nền).

    pip install sounddevice soundfile
    python record_commands.py --speaker hieu --reps 2              # tiếng Anh (mặc định)
    python record_commands.py --speaker hieu --reps 2 --lang vi    # tiếng Việt (tuỳ chọn)

Mỗi lệnh: Enter để bắt đầu, ghi --secs giây (16 kHz mono). Kết quả NỐI THÊM vào <out>/refs.json
(cùng định dạng với bộ tổng hợp, synthetic=False): out mặc định là data/en_commands_real hoặc data/vi_commands_real.

Dùng bản ghi thật: truyền data_dir=<out> cho stt_cmd / e2e_full, hoặc copy refs.json + wav vào thư mục bộ tổng hợp.
(Lưu ý: bench_lat.make_en_commands() / bench_ext.prep_vi_commands() KHÔNG tự đọc thư mục *_real — phải truyền data_dir.)
"""
import argparse
import json
import random
from pathlib import Path

try:  # bench_lat có EN_VARIANTS; bench_ext chỉ có VI_VARIANTS
    from bench_lat import EN_VARIANTS, VI_VARIANTS
except ImportError:
    from bench_ext import VI_VARIANTS
    EN_VARIANTS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--lang", choices=["en", "vi"], default="en")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--noise", default="quiet", help="ghi chú điều kiện nền (quiet / fan / machine ...), lưu vào refs.json")
    a = ap.parse_args()
    variants = EN_VARIANTS if a.lang == "en" else VI_VARIANTS
    if variants is None:
        raise SystemExit("Không tìm thấy EN_VARIANTS — đặt bench_lat.py cạnh file này, hoặc dùng --lang vi.")
    import sounddevice as sd
    import soundfile as sf

    out = Path(a.out or f"data/{a.lang}_commands_real")
    out.mkdir(parents=True, exist_ok=True)
    rp = out / "refs.json"
    refs = json.load(open(rp, encoding="utf-8")) if rp.exists() else []
    items = [(i, p) for i, ps in variants.items() for p in ps] * a.reps
    random.shuffle(items)
    for k, (intent, text) in enumerate(items, 1):
        input(f"[{k}/{len(items)}] Nói: «{text}»   (Enter để ghi)")
        x = sd.rec(int(a.secs * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        fn = f"{a.speaker}_{a.noise}_{len(refs)}.wav"
        sf.write(str(out / fn), x, 16000)
        refs.append({"file": fn, "text": text, "intent": intent, "speaker": a.speaker, "noise": a.noise,
                     "synthetic": False})
        with open(rp, "w", encoding="utf-8") as f:   # ghi sau mỗi câu: Ctrl+C giữa chừng không mất dữ liệu
            json.dump(refs, f, ensure_ascii=False)
    print(f"Xong: {len(refs)} bản ghi trong {out}. Dùng data_dir = '{out}' trong notebook.")


if __name__ == "__main__":
    main()
