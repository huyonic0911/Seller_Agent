"""Retriever RAG: câu hỏi khách -> top-k sản phẩm liên quan (bge-m3 + Qdrant).

Dùng trong SellerBrain để chỉ đưa sản phẩm LIÊN QUAN vào prompt thay vì cả danh mục.
"""
from __future__ import annotations

import logging

from core.config import settings
from modules.llm.embedder import Embedder
from modules.llm.rag import ProductStore
from modules.llm.vectorstore import VectorStore

logger = logging.getLogger("seller_agent.retriever")


class Retriever:
    def __init__(self, store: ProductStore) -> None:
        self.store = store
        self._embedder = Embedder()
        self._vs: VectorStore | None = None

    def _ensure(self) -> None:
        if self._vs is None:
            self._vs = VectorStore(self._embedder.dim)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        self._ensure()
        top_k = top_k or settings.rag_top_k
        qv = self._embedder.encode([query])[0]
        return self._vs.search(qv, top_k)

    def context(self, query: str, top_k: int | None = None) -> tuple[str, list[dict]]:
        """Trả về (đoạn context để nhét vào prompt, danh sách sản phẩm)."""
        products = self.search(query, top_k)
        lines = ["- " + self.store.product_line(p) for p in products]
        return "\n".join(lines), products
