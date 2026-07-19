"""Embedder self-host bge-m3 (BAAI/bge-m3) — dùng cho RAG.

Nạp model qua sentence-transformers (dense 1024 chiều, đã chuẩn hóa để dùng cosine).
Nạp lười ở lần encode đầu. Mặc định chạy CPU (tránh tranh VRAM với viXTTS); đổi
EMBED_DEVICE=cuda nếu muốn nhanh hơn và còn VRAM.
"""
from __future__ import annotations

import logging

from core.config import settings

logger = logging.getLogger("seller_agent.embedder")


class Embedder:
    def __init__(self) -> None:
        self.model_name = settings.embed_model
        self.device = settings.embed_device
        self._model = None

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Nạp embedding model %s (device=%s)...", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def ensure_loaded(self) -> None:
        """Nạp model ngay bây giờ (gọi lúc start server để không nạp lười khi query)."""
        if self._model is None:
            self._load()

    def encode(self, texts: list[str]):
        """Trả về numpy array (n, dim), đã normalize (cosine = dot)."""
        if self._model is None:
            self._load()
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=16
        )

    @property
    def dim(self) -> int:
        if self._model is None:
            self._load()
        fn = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        return fn()
