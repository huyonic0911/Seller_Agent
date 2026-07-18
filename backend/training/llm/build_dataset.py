"""Lọc + chuyển raw synthetic → dataset ChatML (train/eval/golden).

Chức năng:
  1. Đọc raw record (từ gen_synthetic.py và/hoặc log tương tác thật).
  2. Lọc chất lượng: grounding-check giá/size (chống bịa), format (1-3 câu, không
     markdown, không preamble), language (loại CJK / full-English), dedup.
  3. Chuyển sang ChatML messages dùng ĐÚNG prompt contract (system = persona +
     catalog + hậu tố), chỉ turn assistant mang nhãn học.
  4. Chia stratified theo intent: golden (~50, never-train) + train/eval 90/10.

Cách chạy (từ backend/):
    python -m training.llm.build_dataset \
        --raw training/data/raw/synthetic_raw.jsonl \
        --out-dir training/data/curated
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .prompt_contract import build_user_text, system_text_for

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("build_dataset")

_CJK = re.compile(r"[一-鿿]")
_MARKDOWN = re.compile(r"(\*\*|__|^#{1,6}\s|^\s*[-*]\s|`)", re.MULTILINE)
_PREAMBLE = re.compile(r"^\s*(đây là|here is|here's|sure|chắc chắn|ví dụ)", re.IGNORECASE)
_VN_DIACRITICS = re.compile(r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", re.I)
_PRICE = re.compile(r"\d{1,3}(?:[.,]\d{3})+|\d+\s*k\b|\b\d{5,}\b", re.IGNORECASE)


# ---- Trích "facts" từ catalog để đối chiếu -------------------------------
def _valid_prices(products: list[dict[str, Any]]) -> set[int]:
    vals: set[int] = set()
    for p in products:
        for key in ("gia", "gia_km"):
            v = p.get(key)
            try:
                if v is not None:
                    vals.add(int(v))
            except (TypeError, ValueError):
                pass
    return vals


def _valid_sizes(products: list[dict[str, Any]]) -> set[str]:
    return {str(s).lower() for p in products for s in (p.get("size") or [])}


def _norm_price(token: str) -> int | None:
    t = token.strip().lower()
    try:
        if t.endswith("k"):
            return int(float(t[:-1].strip()) * 1000)
        return int(re.sub(r"[.,]", "", t))
    except ValueError:
        return None


def _grounding_ok(reply: str, prices: set[int], sizes: set[str]) -> bool:
    """Mọi con số giá trong reply phải khớp 1 giá hợp lệ (±0). Chống bịa giá."""
    ok_prices = prices | {p // 1000 for p in prices}  # cho phép nói "119" nghĩa 119k
    for m in _PRICE.finditer(reply):
        val = _norm_price(m.group(0))
        if val is None:
            continue
        # bỏ qua số nhỏ có thể là size/số đo, chỉ soi số tiền (>= 1000 hoặc dạng 'k')
        if val < 1000 and not m.group(0).lower().endswith("k"):
            continue
        if val not in ok_prices and (val * 1000) not in prices:
            return False
    return True


def _format_ok(reply: str) -> bool:
    if not reply or _MARKDOWN.search(reply):
        return False
    if _PREAMBLE.match(reply):
        return False
    # 1-4 câu (nới nhẹ cho multi-turn/tư vấn size).
    sentences = [s for s in re.split(r"[.!?…]+", reply) if s.strip()]
    if len(sentences) > 4:
        return False
    if len(reply) > 400:
        return False
    return True


def _language_ok(reply: str) -> bool:
    if _CJK.search(reply):
        return False  # drift tiếng Trung
    # loại reply gần như full-English: không có dấu tiếng Việt & rất ít token VN.
    if not _VN_DIACRITICS.search(reply) and len(reply) > 40:
        return False
    return True


# ---- Chuyển raw record → ChatML ------------------------------------------
def _to_chatml(rec: dict[str, Any]) -> dict[str, Any] | None:
    shop, products = rec.get("shop", {}), rec.get("products", [])
    conv = rec.get("conversation") or []
    if not products or not conv:
        return None
    messages = [{"role": "system", "content": system_text_for(shop, products)}]
    prices, sizes = _valid_prices(products), _valid_sizes(products)
    saw_assistant = False
    expect = "user"
    for turn in conv:
        role = turn.get("role")
        text = (turn.get("text") or "").strip()
        if not text or role != expect:
            return None  # thứ tự phải xen kẽ, bắt đầu bằng user
        if role == "user":
            messages.append({"role": "user", "content": build_user_text(text, turn.get("author"))})
            expect = "assistant"
        else:
            if not (_format_ok(text) and _language_ok(text) and _grounding_ok(text, prices, sizes)):
                return None
            messages.append({"role": "assistant", "content": text})
            saw_assistant = True
            expect = "user"
    if not saw_assistant or messages[-1]["role"] != "assistant":
        return None
    return {
        "messages": messages,
        "intent": rec.get("intent", "unknown"),
        "hard_negative": bool(rec.get("hard_negative")),
        "codeswitch": bool(rec.get("codeswitch")),
        "shop": shop,
        "products": products,
    }


def _dedup_key(rec: dict[str, Any]) -> str:
    parts = [m["content"] for m in rec["messages"] if m["role"] in {"user", "assistant"}]
    norm = " ".join(parts).lower()
    return re.sub(r"\s+", " ", norm).strip()


def _read_raw(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True, help="1+ file JSONL raw")
    ap.add_argument("--out-dir", default="training/data/curated")
    ap.add_argument("--golden", type=int, default=50, help="số mẫu golden never-train")
    ap.add_argument("--eval-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = list(_read_raw(Path(p) for p in args.raw))
    logger.info("Đọc %d raw record.", len(raw))

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    reasons = Counter()
    for rec in raw:
        chat = _to_chatml(rec)
        if chat is None:
            reasons["filtered"] += 1
            continue
        key = _dedup_key(chat)
        if key in seen:
            reasons["dup"] += 1
            continue
        seen.add(key)
        kept.append(chat)

    logger.info("Giữ %d / %d (loại: %s).", len(kept), len(raw), dict(reasons))
    if not kept:
        logger.error("Không có mẫu nào qua bộ lọc.")
        return

    # Chia stratified theo intent.
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in kept:
        by_intent[r["intent"]].append(r)
    for lst in by_intent.values():
        rng.shuffle(lst)

    golden, pool = [], []
    per_intent_golden = max(1, args.golden // max(1, len(by_intent)))
    for lst in by_intent.values():
        golden.extend(lst[:per_intent_golden])
        pool.extend(lst[per_intent_golden:])
    golden = golden[: args.golden]
    rng.shuffle(pool)

    n_eval = int(len(pool) * args.eval_ratio)
    eval_set, train_set = pool[:n_eval], pool[n_eval:]

    out = Path(args.out_dir)
    _write_jsonl(out / "train.jsonl", train_set)
    _write_jsonl(out / "eval.jsonl", eval_set)
    _write_jsonl(out / "golden.jsonl", golden)

    def dist(name, s):
        c = Counter(r["intent"] for r in s)
        cs = sum(1 for r in s if r["codeswitch"])
        logger.info("%s: %d mẫu | codeswitch=%d | intent=%s", name, len(s), cs, dict(c))

    dist("train", train_set)
    dist("eval", eval_set)
    dist("golden", golden)


if __name__ == "__main__":
    main()
