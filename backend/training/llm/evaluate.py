"""Đánh giá model bán hàng: metric tự động + LLM-as-judge (Claude), so baseline.

Sinh reply từ model-under-test (endpoint OpenAI-compatible: Ollama/vLLM) cho từng
mẫu golden/eval, rồi chấm:
  - Tự động: grounding (không bịa giá/size), format (1-3 câu, không markdown),
    language (không drift Trung/Anh).
  - LLM-judge (Claude): persona 1-5, fluency 1-5, faithfulness 1-5, chốt đơn hợp
    cảnh, và refusal đúng (cho case hard-negative).
Chỉ nên PROMOTE adapter khi thắng scorecard của baseline (vd qwen2.5:7b gốc).

Chạy (từ backend/):
    export ANTHROPIC_API_KEY=...   # cho judge
    python -m training.llm.evaluate \
        --model seller-qwen3:8b --base-url http://localhost:11434/v1 \
        --golden training/data/curated/golden.jsonl --out training/data/scorecard.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from .build_dataset import (
    _format_ok,
    _grounding_ok,
    _language_ok,
    _valid_prices,
    _valid_sizes,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate")

_JUDGE_SYSTEM = """\
Bạn là giám khảo chấm câu trả lời của một trợ lý bán hàng livestream tiếng Việt.
Chấm DỰA TRÊN dữ liệu tham chiếu được cung cấp. Trả về JSON hợp lệ, KHÔNG giải thích:
{"persona": 1-5, "fluency": 1-5, "faithfulness": 1-5, "order_closing": true/false, "refusal_correct": true/false/null}
- persona: giọng thân thiện livestream, xưng shop/mình, ngắn gọn, không markdown.
- fluency: tiếng Việt tự nhiên; nếu có chêm tiếng Anh thì có tự nhiên không.
- faithfulness: chỉ dùng đúng dữ liệu, không bịa giá/tồn/size/màu/chính sách.
- order_closing: có khéo chốt đơn/mời để lại thông tin khi hợp cảnh không.
- refusal_correct: CHỈ áp dụng khi câu hỏi hỏi thứ KHÔNG có trong dữ liệu/hết hàng
  → true nếu model từ chối/nói thật đúng, false nếu bịa. Nếu không phải case đó, trả null.
"""


def _gen_openai(client, model: str, messages: list[dict[str, Any]],
                temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else None


def _judge(anthropic_client, judge_model: str, system_ctx: str, comment: str,
           gold: str, reply: str) -> dict[str, Any]:
    user = (
        f"DỮ LIỆU THAM CHIẾU:\n{system_ctx}\n\n"
        f"CÂU HỎI KHÁCH: {comment}\n\n"
        f"REPLY MẪU (tham khảo): {gold}\n\n"
        f"REPLY CẦN CHẤM: {reply}"
    )
    msg = anthropic_client.messages.create(
        model=judge_model,
        max_tokens=300,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return _extract_json(raw) or {}


def _last_user_comment(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def evaluate_model(client, model, records, anthropic_client, judge_model,
                   temperature, max_tokens, do_judge) -> dict[str, Any]:
    auto = {"grounding": [], "format": [], "language": []}
    judged = {"persona": [], "fluency": [], "faithfulness": [], "order_closing": []}
    refusal = {"total": 0, "correct": 0}
    rows = []

    for rec in records:
        messages = rec["messages"]
        # Bỏ assistant cuối (gold) khỏi input; giữ system + các lượt trước.
        input_msgs = messages[:-1] if messages[-1]["role"] == "assistant" else messages
        gold = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
        products = rec.get("products", [])
        prices, sizes = _valid_prices(products), _valid_sizes(products)
        system_ctx = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        comment = _last_user_comment(input_msgs)

        try:
            reply = _gen_openai(client, model, input_msgs, temperature, max_tokens)
        except Exception as exc:
            logger.warning("Sinh reply lỗi: %s", exc)
            continue

        g = _grounding_ok(reply, prices, sizes)
        fo = _format_ok(reply)
        lo = _language_ok(reply)
        auto["grounding"].append(1.0 if g else 0.0)
        auto["format"].append(1.0 if fo else 0.0)
        auto["language"].append(1.0 if lo else 0.0)

        jscore = {}
        if do_judge and anthropic_client is not None:
            try:
                jscore = _judge(anthropic_client, judge_model, system_ctx, comment, gold, reply)
            except Exception as exc:
                logger.warning("Judge lỗi: %s", exc)
        for k in ("persona", "fluency", "faithfulness"):
            if isinstance(jscore.get(k), (int, float)):
                judged[k].append(float(jscore[k]))
        if isinstance(jscore.get("order_closing"), bool):
            judged["order_closing"].append(1.0 if jscore["order_closing"] else 0.0)
        if rec.get("hard_negative"):
            refusal["total"] += 1
            rc = jscore.get("refusal_correct")
            if rc is True or (rc is None and g):  # fallback: không bịa giá coi như đạt
                refusal["correct"] += 1

        rows.append({"comment": comment, "gold": gold, "reply": reply,
                     "grounding": g, "intent": rec.get("intent")})

    return {
        "model": model,
        "n": len(rows),
        "auto": {k: _mean(v) for k, v in auto.items()},
        "judge": {k: _mean(v) for k, v in judged.items()},
        "refusal_accuracy": round(refusal["correct"] / refusal["total"], 3) if refusal["total"] else None,
        "rows": rows,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _scorecard_md(results: list[dict[str, Any]]) -> str:
    lines = ["# Scorecard — Seller Agent LLM", ""]
    hdr = ["model", "n", "grounding", "format", "language",
           "persona", "fluency", "faithfulness", "order_closing", "refusal_acc"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in results:
        a, j = r["auto"], r["judge"]
        lines.append("| " + " | ".join(str(x) for x in [
            r["model"], r["n"], a.get("grounding"), a.get("format"), a.get("language"),
            j.get("persona"), j.get("fluency"), j.get("faithfulness"),
            j.get("order_closing"), r["refusal_accuracy"],
        ]) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model-under-test (tên trên Ollama/vLLM)")
    ap.add_argument("--baseline", default=None, help="model baseline để so (vd qwen2.5:7b)")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--golden", default="training/data/curated/golden.jsonl")
    ap.add_argument("--out", default="training/data/scorecard.md")
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    ap.add_argument("--no-judge", action="store_true", help="chỉ chạy metric tự động")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = _read_jsonl(Path(args.golden))
    if args.limit:
        records = records[: args.limit]
    logger.info("Đánh giá trên %d mẫu.", len(records))

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    anthropic_client = None
    do_judge = not args.no_judge
    if do_judge:
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning("Thiếu ANTHROPIC_API_KEY → bỏ LLM-judge.")
            do_judge = False
        else:
            from anthropic import Anthropic
            anthropic_client = Anthropic()

    results = [evaluate_model(client, args.model, records, anthropic_client,
                              args.judge_model, args.temperature, args.max_tokens, do_judge)]
    if args.baseline:
        results.append(evaluate_model(client, args.baseline, records, anthropic_client,
                                      args.judge_model, args.temperature, args.max_tokens, do_judge))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(_scorecard_md(results), encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "rows"} for r in results],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Ghi scorecard → %s", args.out)
    print(_scorecard_md(results))


if __name__ == "__main__":
    main()
