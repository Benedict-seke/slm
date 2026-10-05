"""Wrapper cho bộ benchmark mở rộng (bài băng tải). Dùng lại bench()/table()/update_row() của bench_common,
nhưng chạy worker trong workers_ext.py (vẫn gọi được mọi worker cũ)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import bench_common as bc
from bench_common import *  # noqa: F401,F403

HERE = Path(__file__).resolve().parent


def run_isolated(kind: str, params: dict, timeout: int = 3600) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TOKENIZERS_PARALLELISM="false",
               PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    try:
        cp = subprocess.run([sys.executable, str(HERE / "workers_ext.py"), kind], input=json.dumps(params),
                            capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout sau {timeout}s"}
    for line in reversed(cp.stdout.splitlines()):
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"error": (cp.stderr or cp.stdout)[-1800:] or f"tiến trình con thoát mã {cp.returncode}"}


bc.run_isolated = run_isolated  # bench() trong bench_common sẽ dùng bản này


def bx(subdir, name, kind, params, force=False):
    """bench() + in toàn bộ chỉ số vô hướng (không chỉ vài khoá mặc định)."""
    row = bench(subdir, name, kind, params, force)
    if "error" not in row:
        print("    ", {k: v for k, v in row.items()
                       if k != "model" and not k.startswith("ram_") and not isinstance(v, (list, dict))})
    return row


# ------------------------------------------------------------------ dữ liệu cho kịch bản băng tải
# Tiếng Việt tự nhiên cho từng ý định (dùng để sinh tổng hợp / làm kịch bản ghi âm).
VI_VARIANTS = {
    "status":  ["Trạng thái băng tải thế nào", "Băng tải đang chạy thế nào", "Cho tôi biết tình trạng băng tải"],
    "why":     ["Tại sao băng tải dừng", "Vì sao băng tải ngừng lại", "Nguyên nhân băng tải dừng là gì"],
    "restart": ["Khởi động lại băng tải", "Chạy lại băng tải đi", "Bật lại băng tải"],
    "stop":    ["Dừng băng tải", "Tắt băng tải ngay", "Dừng khẩn cấp"],
    "speed":   ["Tốc độ băng tải là bao nhiêu", "Băng tải đang chạy tốc độ bao nhiêu"],
    "temp":    ["Nhiệt độ động cơ bao nhiêu", "Động cơ có nóng không"],
    "count":   ["Hôm nay đếm được bao nhiêu sản phẩm", "Đã đếm bao nhiêu sản phẩm rồi"],
    "none":    ["Hôm nay thời tiết thế nào", "Mấy giờ rồi", "Bật đèn phòng họp"],   # ngoài miền: không được kích hoạt gì
}

VI_RESPONSES = {
    "status": "Băng tải đang chạy bình thường.",
    "why": "Băng tải đã dừng vì phát hiện sản phẩm bị kẹt.",
    "restart": "Đang kiểm tra an toàn trước khi khởi động lại.",
    "stop": "Đã dừng băng tải.",
    "speed": "Tốc độ băng tải là không phẩy tám mét trên giây.",
    "temp": "Nhiệt độ động cơ bình thường.",
    "count": "Hôm nay đã đếm được một trăm hai mươi sản phẩm.",
    "none": "Xin lỗi, tôi không hiểu lệnh này.",
}

# Câu cảnh báo / trạng thái cho TTS (viết số bằng chữ — nên chuẩn hoá số TRƯỚC khi đưa vào TTS).
VI_TTS_SENTENCES = [
    "Cảnh báo, băng tải bị kẹt, đã dừng động cơ.",
    "Dòng điện động cơ vượt ngưỡng an toàn.",
    "Nhiệt độ động cơ quá cao, vui lòng kiểm tra.",
    "Băng tải đang chạy bình thường.",
    "Không thể khởi động lại, vẫn còn vật cản trên băng tải.",
    "Băng tải đã khởi động lại.",
    "Tốc độ băng tải là không phẩy tám mét trên giây.",
    "Dòng điện động cơ là ba phẩy hai ampe.",
    "Nhiệt độ động cơ là sáu mươi lăm độ.",
    "Đã phát hiện sản phẩm lỗi, vui lòng kiểm tra băng tải.",
]


def prep_vi_commands(outdir="data/vi_commands", voice="vi_VN-vais1000-medium", force=False) -> str:
    """Nếu đã có data/vi_commands/refs.json (bản ghi thật từ record_commands.py) thì dùng luôn;
    nếu chưa, sinh bộ TỔNG HỢP bằng piper (số đo sẽ lạc quan hơn thực tế)."""
    out = Path(outdir)
    if (out / "refs.json").exists() and not force:
        refs = json.load(open(out / "refs.json", encoding="utf-8"))
        kind = "TỔNG HỢP" if all(r.get("synthetic") for r in refs) else "THẬT/hỗn hợp"
        print(f"Dùng {len(refs)} lệnh có sẵn ({kind}) tại {out}")
        return str(out)
    r = run_isolated("make_vi_cmds", {"out_dir": str(out), "voice": voice, "variants": VI_VARIANTS})
    assert "error" not in r, r["error"]
    print(f"Đã sinh {r['n']} lệnh TỔNG HỢP — hãy ghi âm thật bằng record_commands.py để số đo đáng tin.")
    return str(out)


def flat(df, prefix, cols_like=("wer", "cer", "intent_acc", "false_restart", "miss_stop", "top1", "map50_95", "rt_wer", "rt_cer")):
    """Trả về bảng (model × điều kiện) cho các cột bắt đầu bằng prefix, vd flat(df,'wer_')."""
    import pandas as pd
    cs = [c for c in df.columns if c.startswith(prefix)]
    t = df.set_index("model")[cs]
    t.columns = [c[len(prefix):] for c in cs]
    return t.astype(float) if len(t) else pd.DataFrame()


def plot_curves(df, prefix, title, ylabel, fname, kind="machine"):
    """Vẽ metric theo SNR (cột dạng {prefix}clean, {prefix}{kind}_{snr}db)."""
    import matplotlib.pyplot as plt
    t = flat(df, prefix)
    if t.empty:
        print("Không có dữ liệu.")
        return
    cols = ["clean"] + sorted([c for c in t.columns if c.startswith(kind)], key=lambda c: -int(c.split("_")[-1][:-2]))
    cols = [c for c in cols if c in t.columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, row in t.iterrows():
        ax.plot(range(len(cols)), [row[c] for c in cols], marker="o", label=model)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace(f"{kind}_", "").replace("db", " dB") for c in cols])
    ax.set_xlabel(f"Điều kiện (ồn '{kind}', SNR giảm dần →)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.show()
