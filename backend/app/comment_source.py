"""Nguồn comment cho agent.

Interface `CommentSource` để Phase 2 cắm connector TikTok Live thật mà không đổi
lõi. MVP dùng `SimulatedCommentSource`: comment được đẩy vào từ trang admin hoặc
bộ sinh tự động.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Comment:
    text: str
    author: str = "khách"
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CommentSource:
    """Interface chung: các nguồn comment đẩy Comment vào hàng đợi."""

    def __init__(self) -> None:
        self.queue: "asyncio.Queue[Comment]" = asyncio.Queue()

    async def push(self, comment: Comment) -> None:
        await self.queue.put(comment)

    async def get(self) -> Comment:
        return await self.queue.get()


class SimulatedCommentSource(CommentSource):
    """Nguồn comment giả lập cho MVP (nhập tay từ AdminPanel)."""

    SAMPLES = [
        "Áo thun còn size L màu đen không shop?",
        "Quần jean bao nhiêu tiền vậy ạ?",
        "Túi tote còn hàng không shop ơi?",
        "Giày sneaker có size 39 không?",
        "Ship về Đà Nẵng mất mấy ngày ạ?",
        "Cho mình xin thông tin đặt hàng với",
    ]
