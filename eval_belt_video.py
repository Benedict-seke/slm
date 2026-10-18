"""Chạy detector kẹt/dừng (workers_ext._StuckDetector) trên VIDEO BĂNG TẢI THẬT bất kỳ (vd bộ iron-ore của Mendeley).

    pip install opencv-python-headless
    python eval_belt_video.py --video belt.mp4 --fps 10 --size 160 120
    python eval_belt_video.py --video belt.mp4 --freeze-at 12        # giả lập băng tải DỪNG tại giây 12 (giữ nguyên khung)
    python eval_belt_video.py --video belt.mp4 --belt-px-per-s 40    # nếu biết tốc độ (px/s trên khung đã thu nhỏ)

Nó đo gì:
  * video chạy bình thường  -> số báo động 'stuck'/'stopped' mỗi phút (= BÁO ĐỘNG GIẢ, vì video không có kẹt)
  * --freeze-at T           -> trễ phát hiện 'stopped' sau khi giữ khung từ giây T
Nó KHÔNG đo: kẹt thật (không có nhãn kẹt công khai). Detector giả định: vật tối trên nền sáng ở dải giữa khung, băng chạy
sang PHẢI (script tự thử lật ngang), kết cấu mặt băng nhìn thấy ở dải trên cùng. Nếu video không đúng giả định, báo động
sai/không báo là KẾT QUẢ hợp lệ (detector chưa dùng được cho cảnh đó) — hãy ghi vào báo cáo thay vì chỉnh cho đẹp.
Tốc độ băng (px/s) tự ước lượng từ 5 giây đầu (giả định băng đang chạy), hoặc đưa --belt-px-per-s.
"""
import argparse
import json

import numpy as np


def load_frames(path, size=(160, 120), fps=10.0, start_s=0.0, max_s=None):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(1, int(round(src_fps / fps)))
    eff_fps = src_fps / step
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / src_fps
        i += 1
        if t < start_s:
            continue
        if max_s is not None and t > start_s + max_s:
            break
        if (i - 1) % step:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(g, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames, eff_fps, src_fps


def _best_shift(a, b, smax):
    """a = khung sau, b = khung trước (dải trên). Trả (dịch sang phải tốt nhất, tương quan)."""
    best, bs = -9.0, 0
    for s in range(0, smax + 1):
        c = float(np.mean(a[s:] * b[: len(b) - s]))
        if c > best:
            best, bs = c, s
    return bs, best


def calibrate(frames, fps, seconds=5.0, dt=0.25):
    """Ước lượng tốc độ băng (px/s) và hướng (flip) từ đoạn đầu video, giả định băng đang chạy."""
    H, W = frames[0].shape
    k = max(1, round(dt * fps))
    n = min(len(frames) - k, int(seconds * fps))
    res = {}
    for flip in (False, True):
        shifts, cors = [], []
        for i in range(0, max(1, n), max(1, k)):
            def band(f):
                f = f[:, ::-1] if flip else f
                t = f[: max(2, H // 6)].astype(np.float32).mean(0)
                return (t - t.mean()) / (t.std() + 1e-6)
            s, c = _best_shift(band(frames[i + k]), band(frames[i]), W // 4)
            shifts.append(s)
            cors.append(c)
        res[flip] = (float(np.median(shifts)) / (k / fps), float(np.mean(cors)))
    flip = res[True][1] > res[False][1]
    return res[flip][0], flip, {"corr_noflip": round(res[False][1], 3), "corr_flip": round(res[True][1], 3)}


def make_detector(H, W, fps, px_per_s, hold_s=1.0, min_prod_px=None, band=(0.42, 0.58)):
    from workers_ext import _StuckDetector
    d = _StuckDetector(H, W, fps, hold_s=hold_s)
    d.exp = px_per_s * (d.k / fps)
    d.tol = max(1.5, 0.25 * d.exp)
    d.smax = int(d.exp * 2) + 2
    if min_prod_px:
        d.minw = int(min_prod_px)
    d.r0, d.r1 = int(band[0] * H), max(int(band[0] * H) + 2, int(band[1] * H))
    return d


def run(frames, fps, px_per_s, flip, freeze_at=None, **kw):
    H, W = frames[0].shape
    det = make_detector(H, W, fps, px_per_s, **kw)
    frozen_i = int(freeze_at * fps) if freeze_at is not None else None
    events, prev, last = [], None, None
    for i, f in enumerate(frames):
        if frozen_i is not None and i >= frozen_i:
            f = last
        else:
            last = f
        a = det.process(f[:, ::-1] if flip else f)
        if a and a != prev:
            events.append((round(i / fps, 2), a))
        prev = a
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--size", type=int, nargs=2, default=[160, 120])
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--max-s", type=float, default=None)
    ap.add_argument("--belt-px-per-s", type=float, default=None)
    ap.add_argument("--flip", action="store_true", help="ép lật ngang (mặc định: tự chọn)")
    ap.add_argument("--freeze-at", type=float, default=None)
    ap.add_argument("--hold-s", type=float, default=1.0)
    ap.add_argument("--min-product-px", type=int, default=None)
    a = ap.parse_args()
    frames, fps, src_fps = load_frames(a.video, tuple(a.size), a.fps, a.start, a.max_s)
    if len(frames) < int(8 * fps):
        raise SystemExit(f"Video quá ngắn ({len(frames) / fps:.1f}s sau khi lấy mẫu); cần ≥ 8 s")
    px, flip, info = calibrate(frames, fps)
    px = a.belt_px_per_s or px
    flip = a.flip or flip
    ev = run(frames, fps, px, flip, a.freeze_at, hold_s=a.hold_s, min_prod_px=a.min_product_px)
    dur = len(frames) / fps
    pre = [e for e in ev if a.freeze_at is None or e[0] < a.freeze_at]
    post = [e for e in ev if a.freeze_at is not None and e[0] >= a.freeze_at]
    rep = {"video": a.video, "dur_s": round(dur, 1), "fps_used": round(fps, 2), "src_fps": round(src_fps, 1),
           "belt_px_per_s": round(px, 2), "flip": flip, "calib": info,
           "false_alarms": len(pre), "false_alarms_per_min": round(len(pre) / (dur / 60), 2) if a.freeze_at is None else
           round(len(pre) / (a.freeze_at / 60), 2), "events_before_freeze": pre[:10]}
    if a.freeze_at is not None:
        det = next((t for t, k in post if k in ("stopped", "stuck")), None)
        rep.update(freeze_at=a.freeze_at, stop_detect_delay_s=None if det is None else round(det - a.freeze_at, 2))
    if px < 1.0:
        rep["warning"] = "tốc độ băng ước lượng ~0: video có thể không có băng chạy ở 5 s đầu, hoặc kết cấu dải trên quá nghèo"
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
