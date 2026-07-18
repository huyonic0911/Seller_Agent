"""Sinh dữ liệu synthetic (comment→reply) grounded bằng Claude teacher.

Mỗi mẫu được sinh dựa trên MỘT catalog cụ thể (tổng hợp hoặc từ products.json),
teacher thấy đúng khối reference mà student sẽ thấy lúc train/serve. Output là
JSONL "raw record" — build_dataset.py sẽ lọc & chuyển sang ChatML.

Cách chạy (từ backend/):
    export ANTHROPIC_API_KEY=...
    python -m training.llm.gen_synthetic --n 4000 --out training/data/raw/synthetic_raw.jsonl

Raw record schema (1 dòng/mẫu):
    {
      "shop": {...}, "products": [...],          # catalog dùng cho mẫu
      "intent": "gia_km", "hard_negative": false, "codeswitch": true,
      "conversation": [
        {"role": "user", "author": "Linh", "text": "..."},
        {"role": "assistant", "text": "..."}
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

from .prompt_contract import DEFAULT_PERSONA, catalog_text_from, shop_name_of
from .taxonomy import CODESWITCH_TERMS, INTENTS, build_synthetic_catalog

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("gen_synthetic")

_TEACHER_SYSTEM = """\
Bạn là công cụ sinh DỮ LIỆU HUẤN LUYỆN cho một trợ lý bán hàng livestream tiếng Việt.
Nhiệm vụ: tạo các đoạn hội thoại (comment của khách → câu trả lời mẫu của người bán).

Người bán phải tuân thủ ĐÚNG persona sau:
<persona>
{persona}
</persona>

QUY TẮC BẮT BUỘC cho câu trả lời mẫu (assistant):
- Chỉ dùng thông tin có trong KHỐI DỮ LIỆU dưới đây. TUYỆT ĐỐI không bịa giá, tồn kho, size, màu, chính sách.
- Ngắn gọn 1-3 câu, giọng livestream tự nhiên, không markdown, không emoji tên khách.
- Nếu thông tin KHÔNG có trong dữ liệu (sản phẩm/size/màu không tồn tại) hoặc hàng đã hết:
  nói thật là không có/hết, rồi gợi ý sản phẩm/thời gian phù hợp. KHÔNG bịa ra để chiều khách.
- Chốt đơn hoặc mời để lại thông tin khi hợp lý.

KHỐI DỮ LIỆU THAM CHIẾU (giống hệt cái người bán sẽ thấy lúc chạy thật):
{catalog}

ĐẦU RA: chỉ in JSON hợp lệ (không giải thích, không markdown fence), dạng:
{{"examples": [{{"conversation": [{{"role":"user","author":"<tên>","text":"<comment>"}}, {{"role":"assistant","text":"<reply>"}}]}}]}}
"""


def _build_user_instruction(
    intent: dict[str, Any], k: int, codeswitch: bool, multi_turn: bool
) -> str:
    lines = [
        f"Sinh {k} ví dụ ĐA DẠNG cho tình huống: {intent['desc']}",
        "Yêu cầu đa dạng: đổi tên khách, cách hỏi, độ dài, giọng (vội, hỏi kỹ, nghi ngờ), có typo nhẹ.",
        "Tên khách: đa dạng tên Việt; thỉnh thoảng để tên có emoji/ký tự lạ để người bán KHÔNG đọc lại tên đó.",
    ]
    if intent["hard_negative"]:
        lines.append(
            "ĐÂY LÀ CASE KHÓ (hard-negative): câu hỏi mà dữ liệu KHÔNG đáp ứng được. "
            "Reply mẫu PHẢI từ chối/nói thật + gợi ý cái đang có, tuyệt đối không bịa."
        )
    if multi_turn:
        lines.append(
            "Mỗi hội thoại gồm 2-4 lượt (user/assistant xen kẽ), kết thúc bằng khách chốt đơn "
            "và người bán xin thông tin giao hàng."
        )
    if codeswitch:
        lines.append(
            "Chêm tự nhiên vài từ tiếng Anh (ví dụ: "
            + ", ".join(random.sample(CODESWITCH_TERMS, k=4))
            + ") trong comment và/hoặc reply, nhưng tiếng Việt vẫn là ngôn ngữ nền."
        )
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _plan_counts(total: int) -> list[tuple[dict[str, Any], int]]:
    """Chia tổng số mẫu cho các intent theo trọng số."""
    wsum = sum(i["weight"] for i in INTENTS)
    plan = []
    acc = 0
    for i in INTENTS[:-1]:
        c = round(total * i["weight"] / wsum)
        plan.append((i, c))
        acc += c
    plan.append((INTENTS[-1], max(0, total - acc)))  # phần dư cho intent cuối
    return plan


def _make_client():
    from anthropic import Anthropic

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        logger.error("Thiếu ANTHROPIC_API_KEY.")
        sys.exit(1)
    return Anthropic(api_key=key)


def _gen_batch(
    client, model: str, catalog: dict[str, Any], intent: dict[str, Any],
    k: int, codeswitch: bool, multi_turn: bool, max_tokens: int,
) -> list[dict[str, Any]]:
    persona = DEFAULT_PERSONA.format(shop_name=shop_name_of(catalog["shop"]))
    catalog_text = catalog_text_from(catalog["shop"], catalog["products"])
    system = _TEACHER_SYSTEM.format(persona=persona, catalog=catalog_text)
    user = _build_user_instruction(intent, k, codeswitch, multi_turn)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.9,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    parsed = _extract_json(raw)
    if not parsed or "examples" not in parsed:
        logger.warning("Batch parse lỗi (intent=%s) — bỏ qua.", intent["name"])
        return []
    out = []
    for ex in parsed["examples"]:
        conv = ex.get("conversation")
        if not conv:
            continue
        out.append({
            "shop": catalog["shop"],
            "products": catalog["products"],
            "intent": intent["name"],
            "hard_negative": intent["hard_negative"],
            "codeswitch": codeswitch,
            "conversation": conv,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="tổng số mẫu muốn sinh")
    ap.add_argument("--out", default="training/data/raw/synthetic_raw.jsonl")
    ap.add_argument("--model", default="claude-sonnet-5", help="teacher model")
    ap.add_argument("--per-call", type=int, default=8, help="số mẫu mỗi lần gọi API")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--codeswitch-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--products", default=None,
                    help="đường dẫn products.json thật để dùng làm 1 phần catalog")
    ap.add_argument("--real-ratio", type=float, default=0.25,
                    help="tỉ lệ batch dùng catalog thật (nếu có --products)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    random.seed(args.seed)

    real_catalog = None
    if args.products and Path(args.products).exists():
        data = json.loads(Path(args.products).read_text(encoding="utf-8"))
        real_catalog = {"shop": data.get("shop", {}), "products": data.get("products", [])}

    client = _make_client()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for intent, count in _plan_counts(args.n):
            remaining = count
            while remaining > 0:
                k = min(args.per_call, remaining)
                catalog = (
                    real_catalog
                    if (real_catalog and rng.random() < args.real_ratio)
                    else build_synthetic_catalog(rng)
                )
                codeswitch = (not intent["hard_negative"]) and rng.random() < args.codeswitch_ratio
                multi_turn = intent["name"] == "multi_turn_chot"
                try:
                    batch = _gen_batch(
                        client, args.model, catalog, intent, k,
                        codeswitch, multi_turn, args.max_tokens,
                    )
                except Exception as exc:
                    logger.warning("Lỗi gọi teacher (intent=%s): %s", intent["name"], exc)
                    batch = []
                for rec in batch:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += len(batch)
                remaining -= k
                logger.info("intent=%s +%d (tổng %d)", intent["name"], len(batch), written)

    logger.info("Xong. Ghi %d mẫu → %s", written, out_path)


if __name__ == "__main__":
    main()
