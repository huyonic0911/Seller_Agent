"""Ghi log tương tác comment→reply để thu thập dữ liệu fine-tune vòng sau.

Thiết kế: fire-and-forget, KHÔNG chặn đường trả lời live. Mỗi record ghi 1 dòng
JSONL, file xoay theo ngày. Lỗi ghi log không bao giờ được làm chết pipeline.

Dữ liệu này là "flywheel": tương tác thật của khách (và reply được admin sửa tay
= gold label) sẽ được lọc, trộn với synthetic và đưa vào train round 2+.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("seller_agent.interaction_log")


class InteractionLogger:
    def __init__(self, enabled: bool, log_dir: Path | str) -> None:
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self._dir_ready = False

    def log(self, record: dict[str, Any]) -> None:
        """Lên lịch ghi 1 record (không await, không chặn caller)."""
        if not self.enabled:
            return
        record.setdefault("ts_logged", datetime.now(timezone.utc).isoformat())
        try:
            asyncio.get_running_loop().create_task(self._write(record))
        except RuntimeError:
            # Không có event loop đang chạy — ghi thẳng (đường hiếm).
            try:
                self._append(record)
            except Exception:
                logger.debug("Ghi interaction log thất bại (no loop).")

    def log_feedback(
        self,
        comment: str,
        reply: str,
        verdict: str,
        edited_reply: str | None = None,
        author: str | None = None,
    ) -> None:
        """Ghi feedback của admin cho 1 reply.

        verdict: "good" | "bad" | "edited". `edited_reply` (nếu có) là reply admin
        sửa tay — đây là gold label giá trị nhất cho vòng train sau.
        """
        self.log(
            {
                "kind": "feedback",
                "comment": comment,
                "author": author,
                "reply": reply,
                "verdict": verdict,
                "edited_reply": edited_reply,
            }
        )

    async def _write(self, record: dict[str, Any]) -> None:
        try:
            await asyncio.to_thread(self._append, record)
        except Exception:
            logger.debug("Ghi interaction log thất bại.")

    def _append(self, record: dict[str, Any]) -> None:
        if not self._dir_ready:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._dir_ready = True
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self.log_dir / f"interactions-{day}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
