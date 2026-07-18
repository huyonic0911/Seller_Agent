"""Fine-tune QLoRA/LoRA cho Qwen3 bằng Unsloth (1×A6000).

Chạy trên máy có GPU + môi trường requirements-llm-train.txt:
    python -m training.llm.train_qlora --config training/llm/configs/qwen3_8b_qlora.yaml

Chỉ train trên turn assistant (mask system+user) qua train_on_responses_only để
model học TRẢ LỜI, không học chép lại catalog. Prompt trong dataset đã theo đúng
prompt contract (build_dataset.py) nên khớp tuyệt đối với lúc serve.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_qlora")


def _load_config(path: str | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "base_model": "unsloth/Qwen3-8B",
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        "train_file": "training/data/curated/train.jsonl",
        "eval_file": "training/data/curated/eval.jsonl",
        "output_dir": "training/data/adapters/qwen3-8b-seller-r1",
        "epochs": 3,
        "learning_rate": 2e-4,
        "per_device_batch_size": 2,
        "grad_accum": 8,
        "warmup_ratio": 0.03,
        "lr_scheduler": "cosine",
        "optim": "adamw_8bit",
        "neftune_noise_alpha": 5,
        "logging_steps": 10,
        "eval_steps": 50,
        "save_steps": 100,
        "seed": 42,
    }
    if path:
        import yaml
        cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/llm/configs/qwen3_8b_qlora.yaml")
    ap.add_argument("--base-model", default=None, help="override base_model")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = _load_config(args.config)
    if args.base_model:
        cfg["base_model"] = args.base_model
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    logger.info("Config: %s", cfg)

    # Import nặng đặt trong main để file import được ở máy không GPU.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    ds = load_dataset(
        "json",
        data_files={"train": cfg["train_file"], "eval": cfg["eval_file"]},
    )

    def _format(batch):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch["messages"]
        ]
        return {"text": texts}

    ds = ds.map(_format, batched=True, remove_columns=ds["train"].column_names)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_length"],
            per_device_train_batch_size=cfg["per_device_batch_size"],
            gradient_accumulation_steps=cfg["grad_accum"],
            warmup_ratio=cfg["warmup_ratio"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            bf16=True,
            logging_steps=cfg["logging_steps"],
            optim=cfg["optim"],
            lr_scheduler_type=cfg["lr_scheduler"],
            eval_strategy="steps",
            eval_steps=cfg["eval_steps"],
            save_steps=cfg["save_steps"],
            neftune_noise_alpha=cfg["neftune_noise_alpha"],
            seed=cfg["seed"],
            output_dir=cfg["output_dir"],
            report_to="none",
        ),
    )

    # Mask loss ở system+user; chỉ học phần assistant. Marker theo chat template Qwen3.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()

    out = Path(cfg["output_dir"])
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Đã lưu adapter → %s", out)


if __name__ == "__main__":
    main()
