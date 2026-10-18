"""Wrapper cho dữ liệu công khai: from bench_real import *    (bao gồm mọi thứ của bench_lat + cascade).

Định tuyến: 'stop_kw' -> workers_real.py; 'cascade' -> workers_cascade.py; 'slm_task' -> workers_task.py; còn lại -> workers_lat.py.
"""
import bench_common as bc
import bench_ext as be
import bench_lat as bl
from bench_lat import *  # noqa: F401,F403

_ROUTE = {"stop_kw": "workers_real.py", "cascade": "workers_cascade.py"}


def run_isolated(kind: str, params: dict, timeout: int = 3600) -> dict:
    script = _ROUTE.get(kind) or ("workers_task.py" if kind in bl._TASK_KINDS else "workers_lat.py")
    return bc.run_script(script, kind, params, timeout)


bc.run_isolated = run_isolated
be.run_isolated = run_isolated
bl.run_isolated = run_isolated

from prep_real_data import prep_noise, prep_speech  # noqa: E402,F401


def verdict_stop(df, cond="machine_10db", max_miss_hi=0.10, max_false_hi=0.05):
    """Đạt/không đạt cho lệnh dừng bằng giọng thật. NGƯỠNG TỰ ĐẶT; dùng cận trên 95% nên cần đủ clip 'stop' (≥ ~60)."""
    import pandas as pd
    g = lambda r, k, d: r.get(f"{k}_{cond}") if r.get(f"{k}_{cond}") == r.get(f"{k}_{cond}") else d  # noqa: E731
    rows = []
    for _, r in df.iterrows():
        row = {"model": r["model"], "ram_mb": r.get("ram_total_peak_mb"), "n_stop": r.get("n_stop"),
               "miss_stop": g(r, "miss_stop", 1), "miss_stop_hi95": g(r, "miss_stop_hi95", 1),
               "false_stop_hi95": g(r, "false_stop_hi95", 1), "lat_p95_ms": r.get("lat_ms_p95")}
        row["PASS"] = row["miss_stop_hi95"] <= max_miss_hi and row["false_stop_hi95"] <= max_false_hi
        rows.append(row)
    return pd.DataFrame(rows)
