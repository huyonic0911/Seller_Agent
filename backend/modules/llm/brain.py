"""Bộ não của agent — sinh câu trả lời cho comment của khách.

Hỗ trợ nhiều nhà cung cấp LLM, chọn qua LLM_PROVIDER:
  - "qwen"      : Qwen self-host qua Ollama (hoặc bất kỳ endpoint tương thích OpenAI:
                  vLLM, LM Studio, llama.cpp server...). Mặc định.
  - "anthropic" : Claude qua Anthropic SDK (cần ANTHROPIC_API_KEY).
  - "offline"   : trả lời theo template tra cứu danh mục (không cần model) — để demo.

Nếu backend được chọn lỗi (vd chưa bật Ollama), tự động fallback sang chế độ offline
để UI + TTS + lip-sync vẫn demo được.
"""
from __future__ import annotations

import asyncio
import logging

from core.config import settings
from core.prompt import build_system_text, build_user_text
from modules.llm.rag import ProductStore

logger = logging.getLogger("seller_agent.llm")


class SellerBrain:
    def __init__(self, store: ProductStore):
        self.store = store
        self.provider = settings.llm_provider
        self._client = None  # lazy
        self._warned_fallback = False
        self._warned_rag = False

        # RAG: retriever bge-m3 + Qdrant (nạp lười ở lần dùng đầu)
        self._retriever = None
        if settings.rag_enabled:
            try:
                from modules.llm.retriever import Retriever

                self._retriever = Retriever(store)
            except Exception as exc:  # thiếu thư viện RAG -> bỏ, dùng full catalog
                logger.warning("Không khởi tạo được RAG (%s) → dùng full catalog.", exc)

        if self.provider == "anthropic" and not settings.anthropic_api_key:
            logger.warning("provider=anthropic nhưng thiếu ANTHROPIC_API_KEY → dùng offline.")
            self.provider = "offline"
        elif self.provider not in {"qwen", "openai", "finetuned", "anthropic", "offline"}:
            logger.warning("LLM_PROVIDER không hợp lệ: %s → dùng qwen.", self.provider)
            self.provider = "qwen"

        logger.info(
            "LLM provider=%s model=%s rag=%s",
            self.provider, settings.model, "on" if self._retriever else "off",
        )

    def warmup(self) -> None:
        """Nạp sẵn model RAG (bge-m3) lúc start server thay vì nạp lười ở query đầu.

        Chạy đồng bộ (CPU-heavy ~vài giây) — gọi trong thread ở lifespan để không chặn.
        """
        if self._retriever is None:
            return
        try:
            self._retriever.warmup()
        except Exception as exc:
            logger.warning("Warmup RAG lỗi (%s) → sẽ nạp lười khi cần.", exc)

    # ---- Dựng phần tham chiếu (RAG top-k hoặc full catalog) --------------
    def _build_reference(self, query: str) -> str:
        if self._retriever is not None:
            try:
                ctx, prods = self._retriever.context(query)
                if ctx:
                    return self.store.shop_text() + "\n\nSẢN PHẨM LIÊN QUAN:\n" + ctx
            except Exception as exc:
                if not self._warned_rag:
                    logger.warning("RAG lỗi (%s) → dùng full catalog.", exc)
                    self._warned_rag = True
        return self.store.catalog_text()

    # ---- Dựng prompt (dùng chung mọi provider + training) ----------------
    # Ủy quyền core/prompt.py để prompt lúc serve KHỚP với lúc train; RAG chỉ thay
    # phần dữ liệu tham chiếu (top-k) cho khối catalog, không đổi cấu trúc prompt.
    def _system_text(self, reference: str) -> str:
        persona = settings.persona.format(shop_name=self.store.shop_name())
        return build_system_text(persona, reference)

    def _user_text(self, comment: str, author: str | None) -> str:
        return build_user_text(comment, author)

    # ---- Entry point ------------------------------------------------------
    async def answer(self, comment: str, author: str | None = None) -> str:
        if self.provider == "offline":
            return self._offline_answer(comment)

        # Truy hồi (embedding) nặng CPU → chạy trong thread để không chặn event loop
        reference = await asyncio.to_thread(self._build_reference, comment)
        system = self._system_text(reference)
        user = self._user_text(comment, author)
        try:
            if self.provider == "anthropic":
                text = await self._answer_anthropic(system, user)
            else:  # qwen / openai — endpoint tương thích OpenAI
                text = await self._answer_openai(system, user)
            return text.strip() or self._offline_answer(comment)
        except Exception as exc:
            if not self._warned_fallback:
                logger.warning("Backend LLM lỗi (%s) → fallback offline. Chi tiết: %s", self.provider, exc)
                self._warned_fallback = True
            else:
                logger.debug("Backend LLM lỗi: %s", exc)
            return self._offline_answer(comment)

    # ---- Qwen / OpenAI-compatible (Ollama, vLLM, ...) --------------------
    async def _answer_openai(self, system: str, user: str) -> str:
        from openai import AsyncOpenAI

        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key or "not-needed",
            )
        resp = await self._client.chat.completions.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    # ---- Anthropic / Claude ----------------------------------------------
    async def _answer_anthropic(self, system: str, user: str) -> str:
        from anthropic import AsyncAnthropic

        if self._client is None:
            self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        async with self._client.messages.stream(
            model=settings.model,
            max_tokens=settings.max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": settings.effort},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        ) as stream:
            message = await stream.get_final_message()
        parts = [b.text for b in message.content if b.type == "text"]
        return " ".join(p.strip() for p in parts if p.strip())

    # ---- Chế độ offline (không cần model) --------------------------------
    def _offline_answer(self, comment: str) -> str:
        matches = self.store.search(comment, limit=1)
        if not matches:
            return "Dạ cả nhà cho shop xin thêm thông tin sản phẩm mình quan tâm nha!"
        p = matches[0]
        ten = p.get("ten")
        gia = self.store._fmt_price(p.get("gia_km") or p.get("gia"))
        ton = p.get("ton_kho")
        if ton == 0:
            return f"Dạ {ten} hiện đang hết hàng cả nhà ơi, shop sẽ báo ngay khi có hàng lại nha!"
        km = " đang khuyến mãi" if p.get("gia_km") else ""
        return f"Dạ {ten}{km} còn hàng nha cả nhà, giá chỉ {gia} thôi ạ. Mình chốt đơn shop gói gửi liền nha!"

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
