#!/usr/bin/env python3
"""Chuẩn bị dataset fine-tune viXTTS từ 1 file audio dài (vd data/female_1.wav).

Bước:
  1. Transcribe bằng faster-whisper (tiếng Việt, word-level timestamps, VAD lọc im lặng).
  2. Ghép từ thành câu 3–11s tại ranh giới câu / chỗ ngắt hơi.
  3. Cắt audio -> wav mono 22050 Hz.
  4. Xuất dataset/wavs/*.wav + metadata_train.csv + metadata_eval.csv
     (định dạng coqui: audio_file|text|speaker_name).

Chạy:
  cd backend
  uv run python prepare_dataset.py --src data/female_1.wav
  # tùy chọn: --whisper-model medium  --whisper-device cuda  --min 3 --max 11
"""
from __future__ import annotations

import argparse
import os
import random
import sys


def log(m: str) -> None:
    print(f"[prep] {m}", flush=True)


def chunk_words(words, min_s: float, max_s: float, pause: float):
    """Gom [(start,end,word)] thành các đoạn 3–11s, ưu tiên ngắt ở dấu câu / khoảng lặng."""
    chunks: list[list] = []
    cur: list = []
    for idx, (s, e, w) in enumerate(words):
        cur.append((s, e, w))
        dur = cur[-1][1] - cur[0][0]
        text = "".join(x[2] for x in cur).strip()
        ends_punct = text.endswith((".", "!", "?", "…", ",", ":", ";"))
        gap_next = (words[idx + 1][0] - e) if idx + 1 < len(words) else 999.0
        if dur >= max_s or (dur >= min_s and (ends_punct or gap_next > pause)):
            chunks.append(cur)
            cur = []
    if cur and (cur[-1][1] - cur[0][0]) >= min_s:
        chunks.append(cur)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser(description="Chuẩn bị dataset fine-tune viXTTS")
    ap.add_argument("--src", default="data/female_1.wav", help="File audio nguồn")
    ap.add_argument("--out", default="dataset", help="Thư mục dataset đầu ra")
    ap.add_argument("--speaker", default="female_1", help="Tên speaker")
    ap.add_argument("--sr", type=int, default=22050, help="Sample rate wav đầu ra")
    ap.add_argument("--min", type=float, default=3.0, help="Độ dài đoạn tối thiểu (s)")
    ap.add_argument("--max", type=float, default=11.0, help="Độ dài đoạn tối đa (s)")
    ap.add_argument("--pause", type=float, default=0.4, help="Khoảng lặng để ngắt đoạn (s)")
    ap.add_argument("--pad", type=float, default=0.10, help="Đệm 2 đầu mỗi đoạn (s)")
    ap.add_argument("--eval-ratio", type=float, default=0.05, help="Tỉ lệ eval")
    ap.add_argument("--whisper-model", default="large-v3", help="large-v3 | medium | small")
    ap.add_argument("--whisper-device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--min-chars", type=int, default=6, help="Bỏ đoạn có transcript quá ngắn")
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        sys.exit(f"❌ Không thấy file nguồn: {args.src}")

    import librosa
    import numpy as np
    import soundfile as sf

    # 1) Nạp audio mono ở sr đích (để cắt), + transcribe từ file gốc
    log(f"nạp audio: {args.src}")
    audio, _ = librosa.load(args.src, sr=args.sr, mono=True)
    total_s = len(audio) / args.sr
    log(f"thời lượng: {total_s / 60:.1f} phút @ {args.sr}Hz mono")

    # 2) Whisper transcribe
    from faster_whisper import WhisperModel

    def build_model(device: str):
        compute = "float16" if device == "cuda" else "int8"
        log(f"nạp Whisper '{args.whisper_model}' (device={device}, compute={compute})...")
        return WhisperModel(args.whisper_model, device=device, compute_type=compute)

    devices = ["cuda", "cpu"] if args.whisper_device == "auto" else [args.whisper_device]
    model = None
    for dev in devices:
        try:
            model = build_model(dev)
            break
        except Exception as e:
            log(f"  device={dev} không dùng được ({str(e)[:80]}), thử tiếp...")
    if model is None:
        sys.exit("❌ Không khởi tạo được Whisper.")

    log("đang transcribe (có thể vài phút)...")
    seg_iter, info = model.transcribe(
        args.src, language="vi", word_timestamps=True, vad_filter=True
    )
    words: list = []
    for seg in seg_iter:
        if seg.words:
            for w in seg.words:
                if w.word and w.word.strip():
                    words.append((w.start, w.end, w.word))
        elif seg.text and seg.text.strip():
            words.append((seg.start, seg.end, " " + seg.text.strip()))
    log(f"tổng số từ nhận dạng: {len(words)}")
    if not words:
        sys.exit("❌ Không nhận dạng được nội dung — kiểm tra lại file audio.")

    # 3) Gom thành đoạn
    chunks = chunk_words(words, args.min, args.max, args.pause)
    log(f"gom thành {len(chunks)} đoạn thô")

    # 4) Cắt audio + ghi wav + metadata
    wav_dir = os.path.join(args.out, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    rows: list[tuple[str, str]] = []
    kept = 0
    for i, ch in enumerate(chunks):
        start = max(0.0, ch[0][0] - args.pad)
        end = min(total_s, ch[-1][1] + args.pad)
        text = "".join(x[2] for x in ch).strip()
        text = " ".join(text.split())
        if len(text) < args.min_chars:
            continue
        a = int(start * args.sr)
        b = int(end * args.sr)
        clip = audio[a:b]
        if len(clip) < int(0.5 * args.sr):  # <0.5s thì bỏ
            continue
        # chuẩn hóa biên độ nhẹ
        peak = float(np.max(np.abs(clip))) or 1.0
        clip = (clip / peak) * 0.95
        name = f"utt_{kept:04d}.wav"
        sf.write(os.path.join(wav_dir, name), clip, args.sr)
        rows.append((f"wavs/{name}", text))
        kept += 1

    if kept < 8:
        log(f"⚠️ Chỉ có {kept} đoạn — hơi ít cho fine-tune, cân nhắc thêm audio.")

    # 5) Split train/eval
    random.seed(42)
    random.shuffle(rows)
    n_eval = max(2, int(len(rows) * args.eval_ratio))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

    def write_csv(path: str, data: list[tuple[str, str]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("audio_file|text|speaker_name\n")
            for af, tx in data:
                f.write(f"{af}|{tx}|{args.speaker}\n")

    write_csv(os.path.join(args.out, "metadata_train.csv"), train_rows)
    write_csv(os.path.join(args.out, "metadata_eval.csv"), eval_rows)

    total_dur = sum(
        sf.info(os.path.join(args.out, af)).duration for af, _ in rows
    )
    log("──────────────────────────────────────────")
    log(f"✅ Dataset: {args.out}/")
    log(f"   clip giữ lại : {kept}  (train {len(train_rows)} / eval {len(eval_rows)})")
    log(f"   tổng thời lượng giọng: {total_dur / 60:.1f} phút")
    log(f"   metadata_train.csv / metadata_eval.csv (audio_file|text|speaker_name)")
    log("👉 Kiểm tra vài dòng metadata + nghe thử 1-2 wav trước khi train.")


if __name__ == "__main__":
    main()
