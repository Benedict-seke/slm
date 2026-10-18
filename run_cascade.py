"""Chạy test chuỗi và in bảng so sánh 3 đường (keyword / slm / hybrid).

  # 0) (chỉ để chạy thử khi CHƯA có ghi âm thật) sinh bộ tổng hợp tiếng Anh bằng Piper:
  python run_cascade.py --make-synthetic
  # 1) chạy:
  python run_cascade.py --data data/en_cascade_real --gguf model_cache/qwen2.5-0.5b-instruct-q4_k_m.gguf --stt small.en \
         --kinds machine --snrs 20 10 5
  # có noise thật (MIMII đã đổi sang wav 16 kHz mono trong data/noise):  --kinds machine real --noise-dir data/noise
Kết quả: results_lite/cascade/<tên>.json  +  results_lite/cascade/<tên>_rows.csv
"""
import argparse
import sys

sys.path.insert(0, ".")
import pandas as pd  # noqa: E402

from bench_cascade import bc, bx, run_isolated  # noqa: E402


def make_synthetic(out="data/en_cascade_synth", voice="en_US-lessac-medium"):
    try:
        from cascade_core import pick_variants
    except ImportError:
        sys.exit("Thiếu file cascade_core.py để sinh dữ liệu tổng hợp. Hãy dùng bộ data có sẵn như data/en_commands.")
    r = run_isolated("make_vi_cmds", {"out_dir": out, "voice": voice, "variants": pick_variants(4)})
    assert "error" not in r, r["error"]
    print(f"Đã sinh {r['n']} clip TỔNG HỢP tại {out} — chỉ để chạy thử; số đo STT sẽ lạc quan hơn người thật.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-synthetic", action="store_true")
    ap.add_argument("--data", default="data/en_cascade_real")
    ap.add_argument("--gguf")
    ap.add_argument("--stt", default="small.en", help="tiny.en | base.en | small.en | vosk")
    ap.add_argument("--kinds", nargs="*", default=["machine"])
    ap.add_argument("--snrs", nargs="*", type=int, default=[20, 10, 5])
    ap.add_argument("--noise-dir", default="data/noise")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hog", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.make_synthetic:
        return make_synthetic()
    if not a.gguf:
        sys.exit("Thiếu --gguf")
    stt = ({"engine": "vosk", "model": "vosk-model-small-en-us-0.15"} if a.stt == "vosk"
           else {"engine": "fw", "size": a.stt, "language": "en"})
    params = {**stt, "gguf": a.gguf, "data_dir": a.data, "kinds": a.kinds, "snrs": a.snrs, "noise_dir": a.noise_dir,
              "n": a.n, "threads": a.threads, "hog_procs": a.hog}
    name = a.name or f"{a.stt}+{a.gguf.replace(chr(92), '/').split('/')[-1]}"
    row = bx("cascade", name, "cascade", params, force=a.force)
    if "error" in row:
        sys.exit(1)
    conds = sorted({k.split("tool_acc_", 1)[1] for k in row if "_tool_acc_" in k},
                   key=lambda c: (c != "clean", c))
    recs = []
    for path in ("slm", "kw"):
        for c in conds:
            if f"{path}_tool_acc_{c}" in row:
                r = {"path": path, "cond": c, "wer": row.get(f"wer_{c}")}
                for m in ("tool_acc", "miss_stop", "miss_stop_hi95", "false_restart", "false_restart_hi95",
                          "restart_success", "spurious_stop", "unsafe_start"):
                    r[m] = row.get(f"{path}_{m}_{c}")
                if path == "slm":
                    r["miss_stop_either"] = row.get(f"miss_stop_either_{c}")
                recs.append(r)
    df = pd.DataFrame(recs)
    (bc.RESULTS / "cascade").mkdir(parents=True, exist_ok=True)
    df.to_csv(bc.RESULTS / "cascade" / f"{bc._path('cascade', name).stem}_rows.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    print(f"\nRAM STT+SLM chung: {row.get('ram_total_peak_mb')} MB")
    print("`unsafe_start` PHẢI bằng 0 (nhờ Arbiter). `false_restart` > 0 nghĩa là Arbiter không cứu được lỗi nghe nhầm ý định.")
    for e in (row.get("sample_errors") or [])[:6]:
        print("   ", e)


if __name__ == "__main__":
    main()
