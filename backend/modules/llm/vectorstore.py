"""Qdrant vector DB wrapper cho RAG.

Có QDRANT_URL -> nối tới Qdrant server (Docker). Rỗng -> chế độ nhúng local
(QdrantClient(path=...)) không cần server. Điểm lưu payload = dict sản phẩm.
"""
from __future__ import annotations

import logging

from core.config import settings

logger = logging.getLogger("seller_agent.vectorstore")


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.collection = settings.qdrant_collection
        self.dim = dim
        self._client = self._connect()

    def _connect(self):
        from qdrant_client import QdrantClient

        if settings.qdrant_url:
            logger.info("Nối Qdrant server: %s", settings.qdrant_url)
            return QdrantClient(url=settings.qdrant_url)
        logger.info("Qdrant chế độ nhúng local: %s", settings.qdrant_path)
        return QdrantClient(path=settings.qdrant_path)

    def recreate(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

    def upsert(self, ids: list[int], vectors, payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=i, vector=v.tolist(), payload=pl)
            for i, v, pl in zip(ids, vectors, payloads)
        ]
        self._client.upsert(collection_name=self.collection, points=points)

    def search(self, vector, top_k: int) -> list[dict]:
        res = self._client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [pt.payload for pt in res.points]

    def count(self) -> int:
        return self._client.count(collection_name=self.collection).count

    def exists(self) -> bool:
        try:
            return self._client.collection_exists(self.collection)
        except Exception:
            return False
