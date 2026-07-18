"""Xây index Qdrant cho RAG: đọc danh mục -> embed bge-m3 -> upsert vào Qdrant.

Chạy từ backend/:
  uv run python -m modules.llm.ingest
  # hoặc chỉ định file: PRODUCTS_PATH=data/samsung_phones.json uv run python -m modules.llm.ingest
"""
from __future__ import annotations

from core.config import settings
from modules.llm.embedder import Embedder
from modules.llm.rag import ProductStore
from modules.llm.vectorstore import VectorStore


def main() -> None:
    store = ProductStore(settings.products_path)
    products = store.all()
    print(f"[ingest] {len(products)} sản phẩm từ {settings.products_path}")

    embedder = Embedder()
    texts = [store.product_text(p) for p in products]
    print(f"[ingest] embedding bằng {settings.embed_model} (device={settings.embed_device})...")
    vectors = embedder.encode(texts)
    print(f"[ingest] xong embedding, dim={embedder.dim}")

    vs = VectorStore(embedder.dim)
    vs.recreate()
    vs.upsert(list(range(len(products))), vectors, products)
    print(f"[ingest] ✅ đã index {vs.count()} điểm vào collection '{settings.qdrant_collection}'")

    # Kiểm chứng nhanh vài truy vấn
    for q in ["điện thoại gập màn hình lớn", "máy pin trâu giá rẻ", "flagship chụp ảnh đẹp có bút"]:
        hits = vs.search(embedder.encode([q])[0], 3)
        names = [h.get("ten") for h in hits]
        print(f"[ingest] '{q}' -> {names}")


if __name__ == "__main__":
    main()
