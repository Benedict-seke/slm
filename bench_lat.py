"""Wrapper cho bộ đo LATENCY + TTS-đúng-câu (workers_lat.py). Dùng lại bench()/table()/update_row()/bx() của
bench_common / bench_ext, nhưng chạy worker trong workers_lat.py (vẫn gọi được mọi worker cũ)."""
from pathlib import Path

import bench_common as bc
import bench_ext as be
from bench_ext import *  # noqa: F401,F403  (bench, bx, table, flat, VI_*, ...)

HERE = Path(__file__).resolve().parent
_TASK_KINDS = {"slm_task"}   # worker trong workers_task.py; các kind còn lại -> workers_lat.py (kế thừa cả workers_ext)


def run_isolated(kind: str, params: dict, timeout: int = 3600) -> dict:
    script = "workers_task.py" if kind in _TASK_KINDS else "workers_lat.py"
    return bc.run_script(script, kind, params, timeout)


bc.run_isolated = run_isolated  # bench() trong bench_common sẽ dùng bản này
be.run_isolated = run_isolated

# ------------------------------------------------------------------ dữ liệu tiếng Anh (chính) cho kịch bản băng tải
EN_VARIANTS = {
    "status":  ["What is the conveyor status", "Is the conveyor running", "Tell me the belt condition"],
    "why":     ["Why did the conveyor stop", "What caused the belt to stop", "Tell me the reason it stopped"],
    "restart": ["Restart the conveyor", "Start the belt again", "Turn the conveyor back on"],
    "stop":    ["Stop the conveyor", "Shut down the belt now", "Emergency stop"],
    "speed":   ["How fast is the belt moving", "What is the belt speed"],
    "temp":    ["What is the motor temperature", "Is the motor too hot"],
    "count":   ["How many products have passed", "Count the products today"],
    "none":    ["What is the weather today", "What time is it", "Turn on the meeting room light"],
}

EN_RESPONSES = {
    "status": "The conveyor is running normally.",
    "why": "The conveyor stopped because a product is blocking the belt.",
    "restart": "Checking safety conditions before restarting.",
    "stop": "The conveyor has been stopped.",
    "speed": "The belt speed is zero point eight meters per second.",
    "temp": "The motor temperature is normal.",
    "count": "One hundred twenty products have been counted today.",
    "none": "Sorry, I did not understand that command.",
}

# TTS: {'say': chữ đưa vào Piper, 'ref': chữ người nghe phải nghe ra, 'keys': từ khoá bắt buộc nghe được}.
# 3 câu có số xuất hiện 2 lần: dạng chữ số thô ("0.8") và dạng đã chuẩn hoá bằng chữ — để biết có cần chuẩn hoá số trước TTS.
EN_TTS_SENTENCES = [
    {"say": "Warning, the conveyor is jammed and the motor has been stopped.", "keys": ["jammed", "motor", "stopped"]},
    {"say": "Motor current is above the safe limit.", "keys": ["current", "safe", "limit"]},
    {"say": "Motor temperature is too high, please check it.", "keys": ["temperature", "high"]},
    {"say": "The conveyor is running normally.", "keys": ["running", "normally"]},
    {"say": "Cannot restart, an obstacle is still on the belt.", "keys": ["restart", "obstacle"]},
    {"say": "The conveyor has restarted.", "keys": ["restarted"]},
    {"say": "A defective product was detected, please check the belt.", "keys": ["defective", "detected"]},
    {"say": "Belt speed is 0.8 meters per second.", "ref": "Belt speed is zero point eight meters per second.",
     "keys": ["zero", "point", "eight"]},
    {"say": "Belt speed is zero point eight meters per second.", "keys": ["zero", "point", "eight"]},
    {"say": "Motor current is 3.2 amps.", "ref": "Motor current is three point two amps.",
     "keys": ["three", "point", "two", "amps"]},
    {"say": "Motor current is three point two amps.", "keys": ["three", "point", "two", "amps"]},
    {"say": "Motor temperature is 65 degrees Celsius.", "ref": "Motor temperature is sixty five degrees Celsius.",
     "keys": ["sixty", "five", "celsius"]},
    {"say": "Motor temperature is sixty five degrees Celsius.", "keys": ["sixty", "five", "celsius"]},
]


def make_en_commands(outdir="data/en_commands", voice="en_US-lessac-medium", force=False) -> str:
    """Sinh bộ lệnh TIẾNG ANH TỔNG HỢP bằng Piper (đủ dùng cho đo LATENCY; không dùng để kết luận độ chính xác STT)."""
    out = Path(outdir)
    if (out / "refs.json").exists() and not force:
        print(f"Dùng bộ lệnh có sẵn tại {out}")
        return str(out)
    r = run_isolated("make_vi_cmds", {"out_dir": str(out), "voice": voice, "variants": EN_VARIANTS})
    assert "error" not in r, r["error"]
    print(f"Đã sinh {r['n']} lệnh tổng hợp tại {out}")
    return str(out)
