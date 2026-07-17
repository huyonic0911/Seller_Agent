"""Prompt contract dùng chung cho cả serving (runtime) và training.

Đây là NGUỒN SỰ THẬT DUY NHẤT cho cách dựng system/user prompt của agent bán hàng.
`app/llm.py` (serving) và `training/llm/prompt_contract.py` (sinh data + eval) đều
import từ đây, để prompt lúc train KHỚP TUYỆT ĐỐI với prompt lúc serve — nếu lệch
là mất phần lớn lợi ích fine-tune.

Nguyên tắc: chỉ dựng chuỗi thuần (pure functions), không phụ thuộc settings hay
store, để training script import được mà không kéo theo cả app.
"""
from __future__ import annotations

# Các mảnh cố định của prompt. Đổi ở đây là đổi đồng thời cả train lẫn serve.
REFERENCE_HEADER = "DỮ LIỆU THAM CHIẾU (chỉ dùng đúng những gì có ở đây):"
FINAL_INSTRUCTION = (
    "Chỉ trả về câu trả lời cuối cùng để đọc trên live, không kèm giải thích."
)


def build_system_text(persona: str, catalog_text: str) -> str:
    """Dựng system prompt = persona + khối dữ liệu tham chiếu + hậu tố.

    `persona` đã được format sẵn (điền {shop_name}). `catalog_text` là chuỗi danh
    mục do ProductStore.catalog_text() sinh ra (hoặc catalog tổng hợp cùng schema
    khi sinh synthetic data).
    """
    return (
        persona
        + "\n\n"
        + REFERENCE_HEADER
        + "\n"
        + catalog_text
        + "\n\n"
        + FINAL_INSTRUCTION
    )


def build_user_text(comment: str, author: str | None) -> str:
    """Dựng user turn từ comment của khách (kèm tên nếu có)."""
    return comment if not author else f"Khách '{author}' hỏi: {comment}"
