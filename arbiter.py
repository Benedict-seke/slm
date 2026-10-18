"""Safety & Invariant Arbiter — bản tham chiếu (stdlib thuần, KHÔNG import conveyor_data).

Vai trò theo sơ đồ: SLM chỉ ĐỀ XUẤT một tool; Arbiter quyết định cho phép hay không; STM32 thực thi.
Nguyên tắc thiết kế (fail-closed):
  * request_stop LUÔN được chấp nhận, kể cả khi không có context.
  * START chỉ khi: context hợp lệ về kiểu dữ liệu + mới (age <= MAX_CTX_AGE_S) + không vi phạm điều kiện nào.
  * Thiếu khoá, None, NaN/inf, số âm, bool/chuỗi ở ô số, estop không phải đúng False -> coi là KHÔNG an toàn.
  * Tool khác request_restart / request_stop không bao giờ tạo hành động điều khiển.
  * watchdog_action(): đang chạy mà có bất kỳ running_blocker nào (kể cả context cũ/mất) -> STOP.
    (running_blockers KHÁC blockers: khi chạy, camera 'products moving' là bình thường.)

Các hằng số dưới đây là GIẢ ĐỊNH — hãy chỉnh theo phần cứng của nhóm và ghi lý do vào báo cáo.
"""
import math
from dataclasses import dataclass

CURRENT_LIMIT_A = 2.5
TEMP_LIMIT_C = 70.0
MAX_CTX_AGE_S = 1.0          # context cũ hơn mức này coi như mất dữ liệu
CAMERA_OK = "belt empty"
NUMERIC_KEYS = ("encoder_speed_mps", "motor_current_a", "motor_temp_c")

TOOLS = ["get_status", "get_speed", "get_temperature", "get_product_count", "explain_stop",
         "request_restart", "request_stop", "refuse_unsafe", "none"]


@dataclass(frozen=True)
class Decision:
    action: str            # "STOP" | "START" | "NONE"
    reasons: tuple = ()


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def blockers(ctx, age_s) -> list:
    """Danh sách lý do chặn START (và, khi đang chạy, buộc STOP). Rỗng = an toàn."""
    if not isinstance(ctx, dict):
        return ["NO_CONTEXT"]
    out = []
    if ctx.get("estop") is not False:
        out.append("ESTOP")
    if ctx.get("sensors_valid") is not True or not all(_num(ctx.get(k)) for k in NUMERIC_KEYS):
        out.append("SENSOR_FAULT")
    cam = ctx.get("camera")
    if cam is None or cam == "unavailable":
        out.append("CAMERA_UNAVAILABLE")
    elif cam != CAMERA_OK:
        out.append("BLOCKAGE")
    c, t = ctx.get("motor_current_a"), ctx.get("motor_temp_c")
    if _num(c) and c >= CURRENT_LIMIT_A:
        out.append("OVERCURRENT")
    if _num(t) and t >= TEMP_LIMIT_C:
        out.append("OVERTEMP")
    if not _num(age_s) or age_s < 0 or age_s > MAX_CTX_AGE_S:
        out.append("STALE_CONTEXT")
    return out


CAMERA_STATES_OK_RUNNING = ("belt empty", "products moving")
STOP_ON_CAMERA_LOSS = True   # chính sách: mất camera khi đang chạy -> dừng (chỉnh nếu dây chuyền chấp nhận chạy tiếp)


def running_blockers(ctx, age_s) -> list:
    """Điều kiện buộc DỪNG khi băng tải ĐANG CHẠY. Khác blockers() (dành cho restart):
    khi chạy, camera 'products moving' là bình thường; chỉ 'product stationary on belt' mới là kẹt."""
    if not isinstance(ctx, dict):
        return ["NO_CONTEXT"]
    out = []
    if ctx.get("estop") is not False:
        out.append("ESTOP")
    if ctx.get("sensors_valid") is not True or not all(_num(ctx.get(k)) for k in NUMERIC_KEYS):
        out.append("SENSOR_FAULT")
    cam = ctx.get("camera")
    if cam == "product stationary on belt":
        out.append("BLOCKAGE")
    elif cam in (None, "unavailable"):
        if STOP_ON_CAMERA_LOSS:
            out.append("CAMERA_UNAVAILABLE")
    elif cam not in CAMERA_STATES_OK_RUNNING:
        out.append("CAMERA_UNKNOWN")           # nhãn lạ: fail-closed
    c, t = ctx.get("motor_current_a"), ctx.get("motor_temp_c")
    if _num(c) and c >= CURRENT_LIMIT_A:
        out.append("OVERCURRENT")
    if _num(t) and t >= TEMP_LIMIT_C:
        out.append("OVERTEMP")
    if not _num(age_s) or age_s < 0 or age_s > MAX_CTX_AGE_S:
        out.append("STALE_CONTEXT")
    return out


def decide(tool, ctx, age_s=0.0) -> Decision:
    """Quyết định cho MỘT đề xuất của SLM/lệnh giọng nói."""
    if tool == "request_stop":
        return Decision("STOP", ("OPERATOR_REQUEST",))
    if tool == "request_restart":
        b = blockers(ctx, age_s)
        return Decision("NONE", tuple(b)) if b else Decision("START", ("OK",))
    return Decision("NONE", ())          # đọc dữ liệu / refuse_unsafe / none / tool lạ: không điều khiển


def watchdog_action(ctx, age_s, running: bool) -> Decision:
    """Chạy định kỳ, KHÔNG phụ thuộc SLM. Đang chạy + có blocker -> STOP."""
    if not running:
        return Decision("NONE", ())
    b = running_blockers(ctx, age_s)
    return Decision("STOP", tuple(b)) if b else Decision("NONE", ())
