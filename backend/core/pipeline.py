"""Answer pipeline: đọc comment từ hàng đợi → Claude → TTS → broadcast tới màn live.

`StreamHub` quản lý các client WebSocket đang xem live (frontend). `AnswerPipeline`
là worker asyncio xử lý tuần tự từng comment (mô phỏng người bán trả lời từng câu).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from fastapi import WebSocket

from core.comment_source import Comment, CommentSource
from core.config import settings
from core.emotion import infer_emotion
from core.interaction_log import InteractionLogger
from modules.llm.brain import SellerBrain
from modules.vision.lipsync import audio_to_visemes
from modules.voice.tts import TTSEngine

logger = logging.getLogger("seller_agent.pipeline")


class StreamHub:
    """Quản lý và broadcast tới các client đang xem live."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        data = json.dumps(message, ensure_ascii=False)
        async with self._lock:
            targets = list(self._clients)
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)


class AnswerPipeline:
    def __init__(
        self,
        source: CommentSource,
        brain: SellerBrain,
        tts: TTSEngine,
        hub: StreamHub,
    ) -> None:
        self.source = source
        self.brain = brain
        self.tts = tts
        self.hub = hub
        self._task: asyncio.Task | None = None
        self._interactions = InteractionLogger(
            settings.interaction_log, settings.interaction_log_dir
        )

    def log_feedback(
        self,
        comment: str,
        reply: str,
        verdict: str,
        edited_reply: str | None = None,
        author: str | None = None,
    ) -> None:
        self._interactions.log_feedback(comment, reply, verdict, edited_reply, author)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while True:
            comment = await self.source.get()
            try:
                await self._handle(comment)
            except Exception:  # không để 1 lỗi làm chết worker
                logger.exception("Lỗi khi xử lý comment: %s", comment.text)
                await self.hub.broadcast(
                    {
                        "type": "error",
                        "comment": comment.text,
                        "author": comment.author,
                        "message": "Xin lỗi, có lỗi khi xử lý câu hỏi này.",
                        "emotion": "apologetic",
                    }
                )

    async def _handle(self, comment: Comment) -> None:
        # 1. Báo cho màn live biết đang có câu hỏi được xử lý
        await self.hub.broadcast(
            {"type": "comment", "author": comment.author, "text": comment.text, "ts": comment.ts}
        )

        # 2. Gọi Claude sinh câu trả lời
        reply = await self.brain.answer(comment.text, author=comment.author)
        if not reply:
            reply = "Dạ cả nhà chờ shop chút xíu nha!"

        # 3. Voice (module voice) → audio; Vision (module vision) → viseme nhép miệng
        audio_b64 = ""
        visemes: list[dict] = []
        try:
            audio = await self.tts.synthesize(reply)
            if audio:
                audio_b64 = base64.b64encode(audio).decode("ascii")
                visemes = audio_to_visemes(audio)  # khung độ-mở-miệng (WAV); mp3 -> []
        except Exception:
            logger.exception("Lỗi TTS")

        # 4. Suy luận cảm xúc từ text để avatar đổi biểu cảm (không đụng prompt contract).
        emotion = infer_emotion(reply)

        # 5. Đẩy câu trả lời + audio + viseme + cảm xúc về màn live để đọc, nhép miệng & biểu cảm
        await self.hub.broadcast(
            {
                "type": "reply",
                "author": comment.author,
                "comment": comment.text,
                "text": reply,
                "audio": audio_b64,
                "audio_format": self.tts.audio_format,
                "visemes": visemes,
                "emotion": emotion,
            }
        )

        # 6. Ghi log tương tác (fire-and-forget) để thu data fine-tune vòng sau.
        self._interactions.log(
            {
                "comment": comment.text,
                "author": comment.author,
                "ts": comment.ts,
                "reply": reply,
                "emotion": emotion,
                "provider": self.brain.provider,
                "model": settings.model,
            }
        )
