"""FastAPI app: WebSocket cho comment (vào) và stream phản hồi (ra) + REST tiện ích."""
from __future__ import annotations

import contextlib
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.comment_source import Comment, SimulatedCommentSource
from core.config import settings
from core.pipeline import AnswerPipeline, StreamHub
from modules.llm.brain import SellerBrain
from modules.llm.rag import ProductStore
from modules.vision.avatar import get_avatar_config
from modules.voice.tts import TTSEngine

logging.basicConfig(level=logging.INFO)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    store = ProductStore(settings.products_path)
    source = SimulatedCommentSource()
    brain = SellerBrain(store)
    tts = TTSEngine()
    hub = StreamHub()
    pipeline = AnswerPipeline(source, brain, tts, hub)
    pipeline.start()

    app.state.store = store
    app.state.source = source
    app.state.hub = hub
    app.state.pipeline = pipeline
    try:
        yield
    finally:
        await pipeline.stop()
        with contextlib.suppress(Exception):
            await brain.aclose()


app = FastAPI(title="Seller Agent Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "provider": settings.llm_provider,
        "model": settings.model,
        "openai_base_url": settings.openai_base_url if settings.llm_provider in {"qwen", "openai"} else None,
        "tts_engine": settings.tts_engine,
        "tts_enabled": settings.tts_enabled,
        "products": len(app.state.store.all()),
    }


@app.get("/products")
async def products():
    return {"shop": app.state.store.shop, "products": app.state.store.all()}


@app.get("/samples")
async def samples():
    return {"samples": SimulatedCommentSource.SAMPLES}


@app.get("/avatar")
async def avatar():
    """Cấu hình avatar cho frontend (module vision): kiểu, model, tham số miệng."""
    return get_avatar_config().to_dict()


@app.websocket("/ws/comments")
async def ws_comments(ws: WebSocket):
    """Nhận comment giả lập từ AdminPanel: {"text": "...", "author": "..."}."""
    await ws.accept()
    source: SimulatedCommentSource = app.state.source
    try:
        while True:
            data = await ws.receive_json()
            text = (data or {}).get("text", "").strip()
            if not text:
                continue
            author = (data or {}).get("author") or "khách"
            await source.push(Comment(text=text, author=author))
            await ws.send_json({"type": "ack", "text": text})
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """Màn hình live: nhận comment/reply + audio để đọc & nhép miệng."""
    hub: StreamHub = app.state.hub
    await hub.connect(ws)
    try:
        while True:
            # Giữ kết nối mở; không cần dữ liệu vào từ phía màn live.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.disconnect(ws)
