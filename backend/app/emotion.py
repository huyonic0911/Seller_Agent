"""Suy luận cảm xúc từ TEXT câu trả lời đã sinh (không đụng prompt contract).

Chạy sau khi LLM đã trả lời → gắn nhãn cảm xúc để frontend đổi biểu cảm avatar.
Cố ý là pure function, heuristic theo từ khóa tiếng Việt: không phụ thuộc settings,
không gọi model, hoạt động với mọi provider (qwen / anthropic / finetuned / offline).

Tập cảm xúc cố định — KHỚP với bảng map bên frontend (live2d.ts / Live2DStage.tsx):
    neutral | happy | excited | thinking | apologetic | friendly
"""
from __future__ import annotations

import re

# Các nhãn hợp lệ (frontend map từ đây sang expression/motion).
EMOTIONS = ("neutral", "happy", "excited", "thinking", "apologetic", "friendly")
DEFAULT_EMOTION = "neutral"

# Từ khóa (đã hạ chữ thường) → cảm xúc. Kiểm theo thứ tự ưu tiên bên dưới.
_APOLOGETIC = (
    "xin lỗi", "hết hàng", "hết mất", "cháy hàng", "tạm hết", "chưa có hàng",
    "không có", "chưa về", "lỗi", "rất tiếc", "tiếc quá", "hết size", "sold out",
)
_EXCITED = (
    "khuyến mãi", "giảm giá", "sale", "ưu đãi", "flash sale", "deal", "hot",
    "sập giá", "giá sốc", "mua ngay", "nhanh tay", "số lượng có hạn", "freeship",
    "miễn phí", "quà tặng", "tặng kèm", "chốt đơn liền", "🎉", "🔥", "😍", "🤩",
)
_HAPPY = (
    "còn hàng", "sẵn hàng", "cảm ơn", "cám ơn", "tuyệt vời", "rất đẹp", "xinh",
    "chất lượng", "yêu thích", "hài lòng", "đảm bảo", "chính hãng", "😊", "😁", "❤️",
)
_THINKING = (
    "để shop kiểm tra", "chờ shop", "đợi shop", "shop xem lại", "để mình xem",
    "khoảng", "có thể", "tùy", "hình như", "shop chưa chắc",
)
_FRIENDLY = (
    "dạ", "cả nhà", "mình ơi", "nha", "nhé", "ạ", "iu", "mua giúp shop", "yêu",
)


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


def infer_emotion(text: str | None) -> str:
    """Trả về một nhãn trong EMOTIONS dựa trên nội dung câu trả lời.

    Thứ tự ưu tiên: apologetic > excited > thinking > happy > friendly > neutral.
    (Ưu tiên các trạng thái "mạnh"/đặc thù trước, rồi mới đến lịch sự chung.)
    """
    if not text:
        return DEFAULT_EMOTION
    t = text.lower()

    if _has_any(t, _APOLOGETIC):
        return "apologetic"

    # Nhiều dấu chấm than → hào hứng, kể cả khi thiếu từ khóa.
    exclam = t.count("!")
    if _has_any(t, _EXCITED) or exclam >= 2:
        return "excited"

    if _has_any(t, _THINKING):
        return "thinking"

    if _has_any(t, _HAPPY):
        return "happy"

    # Câu hỏi ngược lại khách → hơi "thinking" (đang cần thêm thông tin).
    if t.rstrip().endswith("?") or re.search(r"\b(gì|nào|bao nhiêu|khi nào)\b", t):
        return "thinking"

    if _has_any(t, _FRIENDLY):
        return "friendly"

    return DEFAULT_EMOTION
