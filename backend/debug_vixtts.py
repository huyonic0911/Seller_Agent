#!/usr/bin/env python3
"""Debug viXTTS — clone giọng từ 1 file WAV rồi đọc thử 1 câu, xuất ra WAV.

Chạy TRÊN SERVER có GPU (RTX 5060) đã cài xong:
  - torch bản CUDA 12.8   (uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128)
  - coqui-tts + soundfile  (uv pip install -r requirements-vixtts.txt)
  - model capleaf/viXTTS   (snapshot_download -> models/viXTTS)

Dùng:
  cd backend
  .venv/bin/python debug_vixtts.py
  # tùy chọn:
  .venv/bin/python debug_vixtts.py --text "Câu khác" --ref-start 20 --ref-dur 25 --device cpu

Kết quả:
  - voices/female_1_ref.wav   : đoạn reference đã cắt+mono (tái dùng cho app qua VIXTTS_SPEAKER_WAV)
  - debug_vixtts_out.wav       : audio tổng hợp để nghe thử
"""
from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_TEXT = (
    "Xin chào mọi người, hôm nay chúng ta sẽ cùng tìm hiểu về ứng dụng claude cho design"
)
DEFAULT_TEXT = (
"""
# Bài thuyết trình quảng cáo màn hình ASUS 210Hz

**Xin chào quý thầy cô và các bạn!**

Hôm nay, mình xin giới thiệu đến mọi người một sản phẩm dành cho những ai yêu thích chơi game và làm việc với hiệu suất cao – **màn hình ASUS 210Hz**.

Trong thế giới công nghệ hiện đại, một chiếc màn hình không chỉ cần hiển thị đẹp mà còn phải mang đến trải nghiệm mượt mà và ổn định. ASUS 210Hz được thiết kế để đáp ứng những yêu cầu đó.

Điểm nổi bật đầu tiên là **tần số quét 210Hz**, giúp hình ảnh chuyển động cực kỳ mượt mà, giảm hiện tượng giật, xé hình và mang lại lợi thế rõ rệt trong các tựa game FPS, MOBA hay đua xe tốc độ cao.

Bên cạnh đó, màn hình sở hữu **độ phân giải sắc nét**, màu sắc trung thực cùng góc nhìn rộng, giúp người dùng có trải nghiệm tuyệt vời khi học tập, làm việc, chỉnh sửa ảnh, xem phim hay giải trí.

ASUS còn tích hợp nhiều công nghệ hiện đại như **Adaptive Sync**, giúp đồng bộ khung hình để giảm hiện tượng xé hình, cùng với **Low Blue Light** và **Flicker-Free**, giúp bảo vệ mắt khi sử dụng trong thời gian dài.

Không chỉ mạnh về hiệu năng, màn hình ASUS còn có thiết kế hiện đại với viền mỏng, chân đế chắc chắn và kiểu dáng sang trọng, phù hợp với mọi không gian từ góc học tập đến phòng chơi game.

Tóm lại, nếu bạn đang tìm kiếm một chiếc màn hình có tốc độ hiển thị nhanh, hình ảnh đẹp, độ bền cao và đến từ một thương hiệu uy tín, thì **ASUS 210Hz** là một lựa chọn rất đáng cân nhắc.

**Xin cảm ơn mọi người đã lắng nghe!**

"""
)
DEFAULT_SRC = "data/female_1.wav"
DEFAULT_REF_OUT = "voices/female_1_ref.wav"
DEFAULT_OUT = "debug_vixtts_out.wav"
DEFAULT_MODEL_DIR = os.getenv("VIXTTS_MODEL_DIR", "models/viXTTS")


def log(msg: str) -> None:
    print(f"[debug-vixtts] {msg}", flush=True)


def patch_vietnamese_tokenizer() -> None:
    """Thêm hỗ trợ 'vi' cho tokenizer XTTS (coqui-tts gốc không có).

    Dùng cleaner tối giản + đọc số bằng num2words('vi') — tránh multilingual_cleaners
    (vốn tra các dict theo lang, không có 'vi' nên sẽ lỗi).
    """
    import re

    from TTS.tts.layers.xtts import tokenizer as xtok

    try:
        from num2words import num2words

        def _vi_num(m: "re.Match") -> str:
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


def strip_markdown(text: str) -> str:
    """Bỏ ký tự markdown để TTS không đọc '#', '*'..."""
    import re

    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # [text](url) -> text
    text = re.sub(r"[*_#>]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_text(text: str, limit: int = 200) -> list[str]:
    """Tách thành các đoạn <= limit ký tự tại ranh giới câu (XTTS giới hạn ~400 token/lần)."""
    import re

    chunks: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        cur = ""
        for s in re.split(r"(?<=[.!?…:])\s+", line):
            while len(s) > limit:  # câu quá dài -> cắt cứng
                head, s = s[:limit], s[limit:]
                chunks.append((f"{cur} {head}".strip()) if cur else head)
                cur = ""
            if len(cur) + len(s) + 1 <= limit:
                cur = f"{cur} {s}".strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
    return [c for c in chunks if c.strip()]


def prepare_reference(src: str, start: float, dur: float, out_path: str) -> str:
    """Cắt 1 đoạn [start, start+dur] và trộn về mono, lưu ra out_path."""
    import numpy as np
    import soundfile as sf

    if not os.path.isfile(src):
        sys.exit(f"❌ Không thấy file nguồn: {src}")

    info = sf.info(src)
    sr = info.samplerate
    total_sec = info.frames / sr
    log(f"nguồn: {src} | {sr}Hz | {info.channels} kênh | {total_sec:.1f}s")

    start = max(0.0, start)
    if start >= total_sec:
        start = 0.0
    with sf.SoundFile(src) as f:
        f.seek(int(start * sr))
        n = int(dur * sr) if dur and dur > 0 else -1
        data = f.read(n, dtype="float32")

    if data.ndim == 2:  # stereo -> mono
        data = data.mean(axis=1)
    if data.size == 0:
        sys.exit("❌ Đoạn reference rỗng — chỉnh lại --ref-start/--ref-dur.")

    # chuẩn hóa biên độ nhẹ để tránh quá to/nhỏ
    peak = float(np.max(np.abs(data))) or 1.0
    data = (data / peak) * 0.95

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, data, sr)
    log(f"reference đã cắt: {out_path} | {len(data) / sr:.1f}s mono")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug clone giọng viXTTS")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="Câu cần đọc")
    ap.add_argument("--src", default=DEFAULT_SRC, help="File WAV giọng gốc")
    ap.add_argument("--ref-out", default=DEFAULT_REF_OUT, help="File reference sau khi cắt")
    ap.add_argument("--ref-start", type=float, default=0.0, help="Giây bắt đầu cắt reference")
    ap.add_argument("--ref-dur", type=float, default=30.0, help="Độ dài reference (giây), 0=cả file")
    ap.add_argument("--out", default=DEFAULT_OUT, help="File WAV kết quả")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Thư mục model viXTTS")
    ap.add_argument("--language", default=os.getenv("VIXTTS_LANGUAGE", "vi"))
    ap.add_argument("--device", default=os.getenv("VIXTTS_DEVICE", "cuda"), choices=["cuda", "cpu"])
    ap.add_argument("--temperature", type=float, default=float(os.getenv("VIXTTS_TEMPERATURE", "0.7")))
    ap.add_argument("--no-vinorm", action="store_true", help="Tắt chuẩn hóa số/viết tắt")
    args = ap.parse_args()

    if not os.path.isdir(args.model_dir):
        sys.exit(
            f"❌ Chưa có model viXTTS ở '{args.model_dir}'. Tải bằng:\n"
            "   .venv/bin/python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('capleaf/viXTTS', local_dir='models/viXTTS')\""
        )

    # 1) Chuẩn bị reference (cắt + mono)
    ref = prepare_reference(args.src, args.ref_start, args.ref_dur, args.ref_out)

    # 2) Chuẩn hóa text (tùy chọn)
    text = args.text
    if not args.no_vinorm:
        try:
            from vinorm import TTSnorm

            text = TTSnorm(text, unknown=False, lower=False, rule=True)
            log(f"text sau chuẩn hóa: {text}")
        except Exception as e:
            log(f"(bỏ qua vinorm: {e})")

    # 3) Nạp model
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    patch_vietnamese_tokenizer()  # cho phép tokenizer đọc tiếng Việt

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    log(f"torch {torch.__version__} | CUDA available={torch.cuda.is_available()} | dùng={'cuda' if use_cuda else 'cpu'}")
    if args.device == "cuda" and not torch.cuda.is_available():
        log("⚠️ Yêu cầu cuda nhưng không thấy GPU — chạy CPU (sẽ chậm).")

    t0 = time.time()
    config = XttsConfig()
    config.load_json(os.path.join(args.model_dir, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=args.model_dir, use_deepspeed=False)
    if use_cuda:
        model.cuda()
    log(f"nạp model xong ({time.time() - t0:.1f}s)")

    # 4) Tính latent giọng từ reference
    t0 = time.time()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[ref])
    log(f"tính latent giọng xong ({time.time() - t0:.1f}s)")

    # 5) Tổng hợp — bỏ markdown, tách đoạn (<400 token/lần), ghép + chèn khoảng nghỉ
    import numpy as np
    import soundfile as sf

    pieces = split_text(strip_markdown(text), limit=200)
    log(f"tách thành {len(pieces)} đoạn")
    gap = np.zeros(int(0.15 * 24000), dtype="float32")  # 0.15s nghỉ giữa đoạn

    t0 = time.time()
    wavs: list[np.ndarray] = []
    for i, piece in enumerate(pieces, 1):
        out = model.inference(
            piece,
            args.language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=args.temperature,
        )
        wavs.append(np.asarray(out["wav"], dtype="float32"))
        wavs.append(gap)
        log(f"  đoạn {i}/{len(pieces)} ({len(piece)} ký tự) ✓")
    infer_sec = time.time() - t0

    wav = np.concatenate(wavs) if wavs else np.zeros(1, dtype="float32")
    sf.write(args.out, wav, 24000)
    audio_sec = len(wav) / 24000
    log(f"tổng hợp xong ({infer_sec:.2f}s cho {audio_sec:.1f}s audio, RTF={infer_sec / max(audio_sec, 1e-6):.2f})")
    log(f"✅ Đã ghi: {os.path.abspath(args.out)}")
    log(f"👉 Nghe thử: mở file, hoặc: aplay {args.out}")
    log(f"👉 Dùng cho app: đặt VIXTTS_SPEAKER_WAV={args.ref_out} trong .env")


if __name__ == "__main__":
    main()
