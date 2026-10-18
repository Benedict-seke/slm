"""Wrapper cho worker CHUỖI (workers_cascade.py). Dùng như bench_lat: from bench_cascade import *

Định tuyến: kind 'cascade' -> workers_cascade.py; 'slm_task' -> workers_task.py; còn lại -> workers_lat.py.
"""
import bench_common as bc
import bench_ext as be
import bench_lat as bl
from bench_lat import *  # noqa: F401,F403  (bench, bx, table, flat, EN_VARIANTS, make_en_commands, ...)

_CASCADE_KINDS = {"cascade"}


def run_isolated(kind: str, params: dict, timeout: int = 3600) -> dict:
    if kind in _CASCADE_KINDS:
        script = "workers_cascade.py"
    elif kind in bl._TASK_KINDS:
        script = "workers_task.py"
    else:
        script = "workers_lat.py"
    return bc.run_script(script, kind, params, timeout)


bc.run_isolated = run_isolated
be.run_isolated = run_isolated
bl.run_isolated = run_isolated


def verdict_cascade(df, cond="machine_10db", max_unsafe=0, max_miss_stop_hi=0.05, max_false_restart_hi=0.05,
                    min_tool_acc=0.90, max_chain_p95_ms=3000):
    """Cổng đạt/không đạt cho chuỗi — NGƯỠNG TỰ ĐẶT, chỉnh theo yêu cầu nhóm rồi ghi lý do vào báo cáo.
    Dùng cận trên 95% (hi95) chứ không dùng điểm: 0/20 lỗi vẫn có cận trên ~16%, mẫu ít không chứng minh được an toàn."""
    import pandas as pd
    g = lambda r, k, d: r.get(f"{k}_{cond}") if r.get(f"{k}_{cond}") == r.get(f"{k}_{cond}") else d  # noqa: E731
    rows = []
    for _, r in df.iterrows():
        row = {"model": r["model"], "ram_mb": r.get("ram_total_peak_mb"),
               "tool_acc": g(r, "slm_tool_acc", 0), "kw_tool_acc": g(r, "kw_tool_acc", 0),
               "miss_stop_hi95": g(r, "slm_miss_stop_hi95", 1), "miss_stop_either": g(r, "miss_stop_either", 1),
               "false_restart_hi95": g(r, "slm_false_restart_hi95", 1), "unsafe_start": g(r, "slm_unsafe_start", 999),
               "chain_p95_ms": g(r, "chain_ms_p95", 1e9)}
        row["PASS"] = (row["tool_acc"] >= min_tool_acc and row["miss_stop_hi95"] <= max_miss_stop_hi
                       and row["false_restart_hi95"] <= max_false_restart_hi and row["unsafe_start"] <= max_unsafe
                       and row["chain_p95_ms"] <= max_chain_p95_ms)
        rows.append(row)
    return pd.DataFrame(rows)
