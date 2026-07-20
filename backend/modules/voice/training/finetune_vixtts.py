#!/usr/bin/env python3
"""Fine-tune viXTTS (GPT) trên dataset đã chuẩn bị bằng prepare_dataset.py.

Điều kiện: đã có dataset/ (metadata_train.csv, metadata_eval.csv, wavs/) và
models/viXTTS/ (model.pth, config.json, vocab.json). Script tự tải thêm
dvae.pth + mel_stats.pth từ base coqui/XTTS-v2 nếu thiếu.

Kiểm thử code (không cần GPU lớn):
  # forward-only, xác minh data→tokenize(vi)→model→loss, vừa VRAM 8GB (nên chạy CPU cho chắc device):
  CUDA_VISIBLE_DEVICES="" uv run python modules/voice/training/finetune_vixtts.py --dry-run --limit-samples 8
  # chạy thử 1-batch có optimizer trên CPU (chậm, cần ~10GB RAM):
  CUDA_VISIBLE_DEVICES="" uv run python modules/voice/training/finetune_vixtts.py --debug

Chạy thật (GPU lớn, vd A6000 48GB) — chạy từ thư mục backend/:
  uv run python modules/voice/training/finetune_vixtts.py --epochs 30 --batch-size 4
  # sau khi train xong -> export thành model dùng được:
  uv run python modules/voice/training/export_vixtts.py --run runs/vixtts_ft --out models/viXTTS-ft

VRAM tham khảo (batch 4, fp32): ~18–22GB. A6000/48GB dư sức, có thể tăng --batch-size.
Mốc VRAM theo GPU:
  - ≥24GB : --batch-size 4..8 thoải mái
  - 12–16GB: --batch-size 2..3, cân nhắc --max-wav 220000
  - 8GB    : full fine-tune KHÔNG ổn (OOM ở optimizer state) — dùng GPU lớn hơn.
Lưu ý: giữ --workers 0 (worker con của DataLoader không có bản vá tokenizer 'vi').
"""
from __future__ import annotations

import argparse
import os
import sys

MODEL_DIR = "models/viXTTS"
BASE_REPO = "coqui/XTTS-v2"  # nguồn dvae.pth + mel_stats.pth


def patch_vietnamese_tokenizer() -> None:
    """coqui-tts gốc không hỗ trợ 'vi' → thêm nhánh vi (bắt buộc cho cả TRAIN)."""
    import re

    from TTS.tts.layers.xtts import tokenizer as xtok

    if getattr(xtok.VoiceBpeTokenizer, "_vi_patched", False):
        return
    try:
        from num2words import num2words

        def _vi_num(m):
            try:
                return " " + num2words(int(m.group(0)), lang="vi") + " "
            except Exception:
                return m.group(0)
    except Exception:
        _vi_num = None

    _orig = xtok.VoiceBpeTokenizer.preprocess_text

    def preprocess_text(self, txt, lang):
        if lang == "vi":
            txt = txt.replace('"', "").lower()
            if _vi_num is not None:
                txt = re.sub(r"\d+", _vi_num, txt)
            return re.sub(r"\s+", " ", txt).strip()
        return _orig(self, txt, lang)

    xtok.VoiceBpeTokenizer.preprocess_text = preprocess_text
    xtok.VoiceBpeTokenizer._vi_patched = True


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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-wav", type=int, default=255995, help="Độ dài wav tối đa (mẫu); giảm nếu OOM")
    ap.add_argument("--max-text", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0,
                    help="Số DataLoader worker. GIỮ 0: worker con không có bản vá tokenizer 'vi' → lỗi.")
    ap.add_argument("--limit-samples", type=int, default=0,
                    help="Giới hạn số mẫu train (0 = tất cả) — để chạy thử nhanh.")
    ap.add_argument("--debug", action="store_true",
                    help="Chạy kiểm thử code 1-batch: 1 epoch, ít mẫu, save ngay, bỏ synth test. "
                         "Nên kèm CUDA_VISIBLE_DEVICES='' để chạy CPU, tránh OOM trên GPU 8GB.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Chỉ nạp model + 1 batch + forward ra loss (KHÔNG optimizer/backward) → "
                         "vừa VRAM 8GB, xác minh đường data→model→loss trước khi lên GPU lớn.")
    args = ap.parse_args()

    if args.debug:
        args.epochs = 1
        if args.limit_samples <= 0:
            args.limit_samples = 6

    if not os.path.isfile(os.path.join(args.dataset, "metadata_train.csv")):
        sys.exit(f"❌ Chưa có dataset ở '{args.dataset}'. Chạy prepare_dataset.py trước.")

    from trainer import Trainer, TrainerArgs

    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig

    patch_vietnamese_tokenizer()  # BẮT BUỘC: cho phép tokenize tiếng Việt khi train

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
        debug_loading_failures=True,
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
        num_loader_workers=args.workers,
        num_eval_loader_workers=args.workers,
        eval_split_max_size=256,
        run_eval=not args.debug,          # debug: bỏ eval cho nhanh
        print_step=1 if args.debug else 25,
        plot_step=100,
        save_step=1 if args.debug else 500,  # debug: lưu checkpoint ngay để verify export
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
        test_sentences=[] if args.debug else [
            {
                "text": "Xin chào cả nhà, hôm nay shop có rất nhiều ưu đãi hấp dẫn nha.",
                "speaker_wav": "voices/female_1_ref.wav",
                "language": args.language,
            }
        ],
    )

    model = GPTTrainer.init_from_config(config)
    # XTTS gốc KHÔNG liệt kê 'vi' trong config.languages → synthesize() (được test_run gọi để
    # đọc thử test_sentences khi eval) sẽ assert fail. Tokenizer đã được vá 'vi' riêng ở trên,
    # nên chỉ cần khai báo 'vi' là ngôn ngữ hợp lệ để synth test tiếng Việt chạy được.
    for _cfg in (getattr(model, "config", None), getattr(getattr(model, "xtts", None), "config", None)):
        _langs = getattr(_cfg, "languages", None)
        if isinstance(_langs, list) and "vi" not in _langs:
            _langs.append("vi")
    train_samples, eval_samples = load_tts_samples(
        [dataset_cfg], eval_split=True, eval_split_max_size=256, eval_split_size=0.05
    )
    if args.limit_samples > 0:
        train_samples = train_samples[: args.limit_samples]
        eval_samples = eval_samples[: max(2, args.limit_samples // 3)]
    print(f"[ft] train={len(train_samples)} eval={len(eval_samples)} | epochs={args.epochs} "
          f"batch={args.batch_size} grad_accum={args.grad_accum} debug={args.debug}")

    if args.dry_run:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ft] DRY-RUN: forward-only trên {dev} (không optimizer/backward)")
        model = model.to(dev)
        model.eval()
        loader = model.get_data_loader(
            config, {}, is_eval=False, samples=train_samples, verbose=True, num_gpus=1, rank=0
        )
        batch = next(iter(loader))
        # format_batch_on_device chạy dvae (đã ở dev) trên batch → phải đưa batch lên dev trước,
        # nếu không sẽ lệch device (Trainer thật tự làm bước này).
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch = model.format_batch_on_device(batch)
        with torch.no_grad():
            _, loss_dict = model.train_step(batch, criterion=None)
        losses = {k: round(float(v), 4) for k, v in loss_dict.items()
                  if hasattr(v, "item") or isinstance(v, (int, float))}
        print(f"[ft] ✅ DRY-RUN OK — forward chạy được, loss: {losses}")
        print("[ft] Đường data→tokenize(vi)→model→loss OK. Optimizer/backward là code coqui chuẩn, "
              "sẽ chạy trên GPU lớn. Bê lên A6000 và bỏ --dry-run.")
        return

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
