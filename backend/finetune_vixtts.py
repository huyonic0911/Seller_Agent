#!/usr/bin/env python3
"""Fine-tune viXTTS (GPT) trên dataset đã chuẩn bị bằng prepare_dataset.py.

Điều kiện: đã có dataset/ (metadata_train.csv, metadata_eval.csv, wavs/) và
models/viXTTS/ (model.pth, config.json, vocab.json). Script tự tải thêm
dvae.pth + mel_stats.pth từ base coqui/XTTS-v2 nếu thiếu.

Chạy:
  cd backend
  uv run python finetune_vixtts.py --epochs 12 --batch-size 2

⚠️ VRAM: XTTS GPT train khá nặng. 8GB dễ OOM. Nếu OOM:
   - giảm --batch-size 1, --max-wav 200000
   - hoặc train trên GPU cloud ≥16GB rồi tải checkpoint về (inference vẫn chạy 8GB).
"""
from __future__ import annotations

import argparse
import os
import sys

MODEL_DIR = "models/viXTTS"
BASE_REPO = "coqui/XTTS-v2"  # nguồn dvae.pth + mel_stats.pth


def ensure_base_files() -> tuple[str, str, str, str]:
    """Đảm bảo có đủ file để train; tải dvae.pth/mel_stats.pth nếu thiếu."""
    from huggingface_hub import hf_hub_download

    xtts_ckpt = os.path.join(MODEL_DIR, "model.pth")
    config_file = os.path.join(MODEL_DIR, "config.json")
    vocab = os.path.join(MODEL_DIR, "vocab.json")
    for f in (xtts_ckpt, config_file, vocab):
        if not os.path.isfile(f):
            sys.exit(f"❌ Thiếu {f}. Tải model viXTTS trước (snapshot_download capleaf/viXTTS).")

    dvae = os.path.join(MODEL_DIR, "dvae.pth")
    mel = os.path.join(MODEL_DIR, "mel_stats.pth")
    if not os.path.isfile(dvae):
        print("[ft] tải dvae.pth từ", BASE_REPO)
        hf_hub_download(BASE_REPO, "dvae.pth", local_dir=MODEL_DIR)
    if not os.path.isfile(mel):
        print("[ft] tải mel_stats.pth từ", BASE_REPO)
        hf_hub_download(BASE_REPO, "mel_stats.pth", local_dir=MODEL_DIR)
    return xtts_ckpt, vocab, dvae, mel


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune viXTTS")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="runs/vixtts_ft")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-wav", type=int, default=255995, help="Độ dài wav tối đa (mẫu); giảm nếu OOM")
    ap.add_argument("--max-text", type=int, default=200)
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.dataset, "metadata_train.csv")):
        sys.exit(f"❌ Chưa có dataset ở '{args.dataset}'. Chạy prepare_dataset.py trước.")

    from trainer import Trainer, TrainerArgs

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import (
        GPTArgs,
        GPTTrainer,
        GPTTrainerConfig,
        XttsAudioConfig,
    )

    xtts_ckpt, vocab, dvae, mel = ensure_base_files()
    os.makedirs(args.out, exist_ok=True)

    dataset_cfg = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="vixtts_ft",
        path=args.dataset,
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_eval.csv",
        language=args.language,
    )

    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6s
        min_conditioning_length=66150,   # 3s
        debug_loading_failures=False,
        max_wav_length=args.max_wav,
        max_text_length=args.max_text,
        mel_norm_file=mel,
        dvae_checkpoint=dvae,
        xtts_checkpoint=xtts_ckpt,
        tokenizer_file=vocab,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_cfg = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    config = GPTTrainerConfig(
        output_path=args.out,
        model_args=model_args,
        run_name="vixtts_ft",
        project_name="seller_agent",
        audio=audio_cfg,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=25,
        plot_step=100,
        save_step=500,
        save_n_checkpoints=2,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5, "last_epoch": -1},
        epochs=args.epochs,
        test_sentences=[
            {
                "text": "Xin chào cả nhà, hôm nay shop có rất nhiều ưu đãi hấp dẫn nha.",
                "speaker_wav": "voices/female_1_ref.wav",
                "language": args.language,
            }
        ],
    )

    model = GPTTrainer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        [dataset_cfg], eval_split=True, eval_split_max_size=256, eval_split_size=0.05
    )
    print(f"[ft] train={len(train_samples)} eval={len(eval_samples)} | epochs={args.epochs} "
          f"batch={args.batch_size} grad_accum={args.grad_accum}")

    trainer = Trainer(
        TrainerArgs(grad_accum_steps=args.grad_accum),
        config,
        output_path=args.out,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    print(f"[ft] ✅ Xong. Checkpoint trong {args.out}/ (thư mục run mới nhất).")
    print("[ft] Ghép thành model dùng được: xem hướng dẫn ở cuối README mục fine-tune.")


if __name__ == "__main__":
    main()
