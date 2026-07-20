#!/usr/bin/env python3
"""Xuất model viXTTS đã fine-tune thành thư mục dùng được cho inference.

Lấy best_model.pth từ run train, bỏ optimizer state (nhẹ hơn ~3x), copy kèm
config.json + vocab.json, rồi tự kiểm chứng bằng 1 câu tiếng Việt.

Xxong: đặt VIXTTS_MODEL_DIR=<out> trong .env là agent dùng giọng đã fine-tune.

Chạy:
  cd backend
  uv run python export_vixtts.py --run runs/vixtts_ft --out models/viXTTS-ft
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

BASE_DIR = "models/viXTTS"  # nguồn config.json + vocab.json
# BASE_DIR = "runs/vixtts_ft/vixtts_ft-July-20-2026_03+22PM-2f9822c"

def patch_vietnamese_tokenizer() -> None:
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


def find_checkpoint(run_dir: str) -> str:
    cands = glob.glob(os.path.join(run_dir, "**", "best_model.pth"), recursive=True)
    if not cands:
        cands = glob.glob(os.path.join(run_dir, "**", "*.pth"), recursive=True)
        cands = [c for c in cands if os.path.basename(c).startswith(("best_model", "checkpoint"))]
    if not cands:
        sys.exit(f"❌ Không thấy checkpoint (.pth) trong {run_dir}")
    return max(cands, key=os.path.getmtime)  # mới nhất


def main() -> None:
    ap = argparse.ArgumentParser(description="Export viXTTS fine-tuned")
    ap.add_argument("--run", default="runs/vixtts_ft", help="Thư mục run train")
    ap.add_argument("--ckpt", default="", help="Chỉ định checkpoint cụ thể (bỏ qua --run)")
    ap.add_argument("--base", default=BASE_DIR, help="Thư mục model gốc (lấy config/vocab)")
    ap.add_argument("--out", default="models/viXTTS-ft", help="Thư mục model xuất ra")
    ap.add_argument("--ref", default="voices/female_1_ref.wav", help="Mẫu giọng để test")
    ap.add_argument("--no-test", action="store_true", help="Bỏ bước kiểm chứng")
    args = ap.parse_args()

    import torch

    ckpt_path = args.ckpt or find_checkpoint(args.run)
    print(f"[export] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    os.makedirs(args.out, exist_ok=True)
    # 1) model.pth: chỉ giữ trọng số (bỏ optimizer/scaler)
    torch.save({"model": state}, os.path.join(args.out, "model.pth"))
    # 2) config.json + vocab.json từ model gốc
    for fn in ("config.json", "vocab.json"):
        src = os.path.join(args.base, fn)
        if not os.path.isfile(src):
            sys.exit(f"❌ Thiếu {src} ở model gốc.")
        shutil.copy2(src, os.path.join(args.out, fn))
    # 2b) Thêm 'vi' vào config.languages: XTTS gốc không liệt kê 'vi' nên synthesize()
    #     sẽ assert fail nếu runtime dùng đường synthesize thay vì inference. Tokenizer 'vi'
    #     được vá lúc chạy; đây chỉ khai báo ngôn ngữ hợp lệ để không bị chặn.
    import json
    cfg_out = os.path.join(args.out, "config.json")
    with open(cfg_out, encoding="utf-8") as f:
        cfg_json = json.load(f)
    langs = cfg_json.get("languages")
    if isinstance(langs, list) and "vi" not in langs:
        langs.append("vi")
        with open(cfg_out, "w", encoding="utf-8") as f:
            json.dump(cfg_json, f, ensure_ascii=False, indent=4)
        print("[export] + thêm 'vi' vào config.languages")
    print(f"[export] ✅ đã tạo {args.out}/ (model.pth + config.json + vocab.json)")

    if args.no_test:
        return

    # 3) Kiểm chứng: nạp model xuất ra + đọc thử 1 câu
    print("[export] kiểm chứng: nạp model + tổng hợp thử...")
    patch_vietnamese_tokenizer()
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(os.path.join(args.out, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=args.out, use_deepspeed=False)
    if torch.cuda.is_available():
        model.cuda()
    if not os.path.isfile(args.ref):
        print(f"[export] ⚠️ không thấy mẫu giọng {args.ref} — bỏ qua test tổng hợp.")
        return
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[args.ref])
    out = model.inference(
        "Xin chào cả nhà, đây là giọng đã được tinh chỉnh.",
        "vi",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,
    )
    import numpy as np
    import soundfile as sf

    test_path = os.path.join(args.out, "export_test.wav")
    sf.write(test_path, np.asarray(out["wav"], dtype="float32"), 24000)
    print(f"[export] ✅ model chạy được. Nghe thử: {test_path}")
    print(f"[export] 👉 Đặt trong .env: VIXTTS_MODEL_DIR={args.out}")


if __name__ == "__main__":
    main()
