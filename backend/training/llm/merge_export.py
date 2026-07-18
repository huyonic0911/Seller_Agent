"""Merge adapter LoRA → GGUF + Modelfile để phục vụ qua Ollama.

Sau khi có Modelfile:
    ollama create seller-qwen3:8b -f <out_dir>/Modelfile
    # rồi trong backend/.env:  LLM_PROVIDER=finetuned  LLM_MODEL=seller-qwen3:8b

Chạy trên máy GPU (môi trường requirements-llm-train.txt):
    python -m training.llm.merge_export \
        --adapter training/data/adapters/qwen3-8b-seller-r1 \
        --out training/data/adapters/qwen3-8b-seller-r1-gguf \
        --quant q4_k_m

Lựa chọn khác (throughput cao): bỏ bước GGUF, dùng --merged-16bit rồi serve base
merged bằng vLLM (OpenAI-compatible) — chỉ cần trỏ OPENAI_BASE_URL sang vLLM.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("merge_export")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="thư mục adapter đã train")
    ap.add_argument("--out", required=True, help="thư mục xuất GGUF/merged")
    ap.add_argument("--quant", default="q4_k_m", help="q4_k_m | q5_k_m | q8_0 | f16")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--merged-16bit", action="store_true",
                    help="chỉ xuất bản merged 16-bit (cho vLLM), bỏ GGUF")
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        dtype=None,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.merged_16bit:
        model.save_pretrained_merged(str(out), tokenizer, save_method="merged_16bit")
        logger.info("Đã xuất merged 16-bit → %s (dùng cho vLLM).", out)
        return

    # GGUF + Modelfile cho Ollama.
    model.save_pretrained_gguf(str(out), tokenizer, quantization_method=args.quant)
    modelfile = out / "Modelfile"
    logger.info("Đã xuất GGUF (%s) → %s", args.quant, out)
    if modelfile.exists():
        logger.info("Tạo model Ollama:  ollama create seller-qwen3:8b -f %s", modelfile)
    else:
        logger.warning("Không thấy Modelfile tự sinh — kiểm tra output của Unsloth.")


if __name__ == "__main__":
    main()
