"""Cầu nối prompt-contract cho pipeline training.

Import lại builder từ `app/prompt.py` (nguồn sự thật duy nhất) + tiện ích dựng
catalog_text từ dict sản phẩm tùy ý (để sinh catalog tổng hợp lúc gen synthetic,
không cần file products.json). Việc dùng lại `ProductStore.catalog_text()` đảm bảo
khối "DANH MỤC" lúc train khớp y hệt lúc serve.

Chạy các script training từ thư mục `backend/` (đã tự chèn vào sys.path).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Cho phép `from app...` khi chạy `python -m training.llm.*` từ backend/.
_BACKEND = Path(__file__).resolve().parents[2]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.config import DEFAULT_PERSONA  # noqa: E402
from app.products import ProductStore  # noqa: E402
from app.prompt import build_system_text, build_user_text  # noqa: E402

__all__ = [
    "DEFAULT_PERSONA",
    "build_system_text",
    "build_user_text",
    "catalog_text_from",
    "system_text_for",
    "shop_name_of",
]


def shop_name_of(shop: dict[str, Any] | None) -> str:
    return (shop or {}).get("name", "shop mình")


def catalog_text_from(shop: dict[str, Any] | None, products: list[dict[str, Any]]) -> str:
    """Dựng chuỗi catalog bằng đúng logic của ProductStore, từ dict trong bộ nhớ."""
    store = ProductStore.__new__(ProductStore)  # bỏ qua __init__ (không đọc file)
    store.shop = shop or {}
    store.products = products
    return store.catalog_text()


def system_text_for(
    shop: dict[str, Any] | None,
    products: list[dict[str, Any]],
    persona: str = DEFAULT_PERSONA,
) -> str:
    """System prompt hoàn chỉnh cho 1 catalog (persona đã điền {shop_name})."""
    persona_filled = persona.format(shop_name=shop_name_of(shop))
    return build_system_text(persona_filled, catalog_text_from(shop, products))
