#!/usr/bin/env python3
"""Test một checkpoint viXTTS (sau fine-tune) — clone giọng từ reference rồi đọc thử, xuất WAV.

Nạp THẲNG checkpoint từ thư mục run (không cần export trước): lấy config.json + vocab.json
từ model gốc (--base) và trọng số từ checkpoint (--ckpt / tự tìm trong --run).

Chạy từ backend/ (venv đã cài torch cu128 + coqui-tts):
  # test checkpoint MỚI NHẤT trong runs/vixtts_ft, dùng reference đã chuẩn bị:
  .venv-vixtts/bin/python modules/voice/training/debug_vixtts.py

  # chỉ định checkpoint + câu khác + so sánh với model gốc:
  .venv-vixtts/bin/python modules/voice/training/debug_vixtts.py \
      --ckpt runs/vixtts_ft/<run>/best_model.pth --text "Câu cần đọc"
  .venv-vixtts/bin/python modules/voice/training/debug_vixtts.py --base-only   # nghe giọng model gốc

  # test một thư mục model đã export (config+vocab+model.pth trong 1 chỗ):
  .venv-vixtts/bin/python modules/voice/training/debug_vixtts.py --model-dir models/viXTTS-ft

Kết quả: file WAV (mặc định debug_vixtts_out.wav) để nghe thử.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# DEFAULT_TEXT = (
# """
# Để sử dụng Claude hiệu quả cho công việc thiết kế, bạn cần biết cách khai thác tối đa khả năng ngôn ngữ và tư duy logic của trợ lý AI này. Trước hết, hãy dùng Claude làm người bạn đồng hành trong giai đoạn lên ý tưởng bằng cách yêu cầu nó gợi ý các phong cách thị giác, bảng màu hoặc chủ đề phù hợp với dự án. Bạn có thể mô tả đối tượng khách hàng và nhờ Claude xây dựng sơ đồ trang web hoặc đề xuất bố cục giao diện người dùng một cách khoa học. Tiếp theo, Claude là công cụ đắc lực để biên soạn nội dung hiển thị trên thiết kế, giúp bạn viết câu định vị, tiêu đề hoặc văn bản giữ chỗ thay vì dùng chữ "Lorem Ipsum" vô nghĩa. Đặc biệt, nếu bạn sử dụng các công cụ tạo ảnh như Midjourney hay Stable Diffusion, hãy đưa ra ý tưởng và nhờ Claude tối ưu hóa thành những câu lệnh tiếng Anh chi tiết, chuẩn xác để có kết quả hình ảnh tốt nhất. Cuối cùng, bạn hãy tải ảnh phác thảo hoặc thiết kế sơ bộ lên để Claude nhận xét, đánh giá và gợi ý cách cải thiện trải nghiệm người dùng trước khi bàn giao sản phẩm.
# """
# )
DEFAULT_TEXT = (
    "Bất ngờ chưa"
)
DEFAULT_RUN = "models/viXTTS-ft"          # nơi chứa checkpoint sau fine-tune
DEFAULT_BASE = os.getenv("VIXTTS_MODEL_DIR", "models/viXTTS")  # nguồn config.json + vocab.json
DEFAULT_REF = "voices/female_1_ref.wav"  # reference đã chuẩn bị (prepare_dataset tự tạo)
DEFAULT_SRC = "data/voices/data_02.wav"  # fallback: cắt reference từ đây nếu chưa có --ref
DEFAULT_OUT = "debug_vixtts_out.wav"


def log(msg: str) -> None:
    print(f"[debug-vixtts] {msg}", flush=True)


def patch_vietnamese_tokenizer() -> None:
    """Thêm hỗ trợ 'vi' cho tokenizer XTTS (coqui-tts gốc không có)."""
    import re

    from TTS.tts.layers.xtts import tokenizer as xtok

    if getattr(xtok.VoiceBpeTokenizer, "_vi_patched", False):
        return
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
    xtok.VoiceBpeTokenizer._vi_patched = True


def strip_markdown(text: str) -> str:
    """Bỏ ký tự markdown để TTS không đọc '#', '*'..."""
    import re

    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # [text](url) -> text
    text = re.sub(r"[*_#>]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_text(text: str, max_chars: int = 240) -> list[str]:
    """Tách văn bản thành TỪNG CÂU tại ranh giới '.', '!', '?', '…' — mỗi câu là 1 đoạn
    để tổng hợp riêng rồi ghép lại. Không gói nhiều câu / cắt giữa câu (nguyên nhân gây
    'vấp' khi merge). Chỉ câu quá dài (> max_chars, dễ vượt giới hạn token XTTS) mới tách
    thêm ở dấu phẩy; cắt cứng là biện pháp cuối (hiếm khi tới)."""
    import re

    # gộp mọi khoảng trắng/xuống dòng thành 1 space (câu có thể vắt qua nhiều dòng)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # mỗi câu = chuỗi tới (và gồm) dấu kết câu; phần đuôi không có dấu cũng thành 1 câu
    sentences = [s.strip() for s in re.findall(r"[^.!?…]+[.!?…]+|[^.!?…]+$", text) if s.strip()]

    def split_long(sent: str) -> list[str]:
        if len(sent) <= max_chars:
            return [sent]
        parts, cur = [], ""
        for seg in re.split(r"(?<=,)\s+", sent):  # tách ở dấu phẩy, gom lại <= max_chars
            if len(cur) + len(seg) + 1 <= max_chars:
                cur = f"{cur} {seg}".strip()
            else:
                if cur:
                    parts.append(cur)
                cur = seg
        if cur:
            parts.append(cur)
        out: list[str] = []
        for p in parts:  # nếu 1 mảnh vẫn quá dài (câu không có phẩy) -> cắt cứng
            while len(p) > max_chars:
                out.append(p[:max_chars].strip())
                p = p[max_chars:].strip()
            if p:
                out.append(p)
        return out

    chunks: list[str] = []
    for sent in sentences:
        chunks.extend(split_long(sent))

    # Mỗi đoạn PHẢI kết thúc bằng dấu câu: XTTS là mô hình tự hồi quy, dùng dấu kết câu ('.')
    # làm TÍN HIỆU DỪNG khi sinh audio. Thiếu nó → model sinh lố ở cuối → vấp/lắp bắp/đọc sai
    # từ cuối. Dấu '.' KHÔNG bị đọc thành tiếng, chỉ là tín hiệu dừng. Vì vậy: giữ dấu kết câu
    # có sẵn, và với mảnh tách ở dấu phẩy (câu dài) thì bỏ ',' cuối rồi thêm '.'.
    out: list[str] = []
    for c in chunks:
        c = re.sub(r"\s+", " ", c).strip().rstrip(",").strip()
        if not c:
            continue
        if c[-1] not in ".!?…":
            c += "."
        out.append(c)
    return out


def prepare_reference(src: str, start: float, dur: float, out_path: str) -> str:
    """Cắt 1 đoạn [start, start+dur] và trộn về mono, lưu ra out_path."""
    import numpy as np
    import soundfile as sf

    if not os.path.isfile(src):
        sys.exit(f"❌ Không thấy file nguồn để cắt reference: {src} (dùng --ref nếu đã có sẵn).")

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

    peak = float(np.max(np.abs(data))) or 1.0
    data = (data / peak) * 0.95

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, data, sr)
    log(f"reference đã cắt: {out_path} | {len(data) / sr:.1f}s mono")
    return out_path


def find_checkpoint(run_dir: str) -> str:
    """Tìm checkpoint mới nhất trong run_dir (ưu tiên best_model.pth)."""
    cands = glob.glob(os.path.join(run_dir, "**", "best_model.pth"), recursive=True)
    if not cands:
        cands = [
            c for c in glob.glob(os.path.join(run_dir, "**", "*.pth"), recursive=True)
            if os.path.basename(c).startswith(("best_model", "checkpoint"))
        ]
    if not cands:
        sys.exit(f"❌ Không thấy checkpoint (.pth) trong {run_dir}. Fine-tune trước hoặc dùng --model-dir.")
    return max(cands, key=os.path.getmtime)


def main() -> None:
    ap = argparse.ArgumentParser(description="Test checkpoint viXTTS sau fine-tune")
    # --- nguồn model (chọn 1; ưu tiên: --model-dir > --ckpt > --run) ---
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help="Thư mục run train — tự tìm checkpoint mới nhất (mặc định).")
    ap.add_argument("--ckpt", default="", help="Chỉ định file checkpoint .pth cụ thể.")
    ap.add_argument("--model-dir", default="",
                    help="Thư mục model hoàn chỉnh (config+vocab+model.pth), vd model đã export.")
    ap.add_argument("--base", default=DEFAULT_BASE, help="Nguồn config.json + vocab.json khi nạp checkpoint rời.")
    ap.add_argument("--base-only", action="store_true", help="Bỏ qua checkpoint fine-tune, nghe giọng model GỐC.")
    # --- reference giọng ---
    ap.add_argument("--ref", default=DEFAULT_REF, help="File reference có sẵn (dùng thẳng nếu tồn tại).")
    ap.add_argument("--src", default=DEFAULT_SRC, help="File audio gốc để cắt reference (khi --ref chưa có).")
    ap.add_argument("--ref-start", type=float, default=0.0, help="Giây bắt đầu cắt reference")
    ap.add_argument("--ref-dur", type=float, default=8.0, help="Độ dài reference (giây), 0=cả file")
    # --- văn bản + tổng hợp ---
    ap.add_argument("--text", default=DEFAULT_TEXT, help="Câu cần đọc")
    ap.add_argument("--out", default=DEFAULT_OUT, help="File WAV kết quả")
    ap.add_argument("--language", default=os.getenv("VIXTTS_LANGUAGE", "vi"))
    ap.add_argument("--device", default=os.getenv("VIXTTS_DEVICE", "cuda"), choices=["cuda", "cpu"])
    ap.add_argument("--temperature", type=float, default=float(os.getenv("VIXTTS_TEMPERATURE", "0.7")))
    ap.add_argument("--max-chars", type=int, default=240,
                    help="Ngưỡng ký tự để tách câu quá dài (ở dấu phẩy). Câu <= ngưỡng giữ nguyên.")
    ap.add_argument("--gap", type=float, default=0.15, help="Khoảng nghỉ (giây) chèn giữa các câu.")
    args = ap.parse_args()

    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    patch_vietnamese_tokenizer()  # cho phép tokenizer đọc tiếng Việt

    # 1) Xác định nguồn model: config/vocab + checkpoint
    def is_model_dir(d: str) -> bool:
        """Thư mục model hoàn chỉnh: có đủ config.json + vocab.json + model.pth (vd đã export)."""
        return all(os.path.isfile(os.path.join(d, f)) for f in ("config.json", "vocab.json", "model.pth"))

    ckpt_path: str | None = None
    model_dir = args.model_dir
    # --run (hoặc default) trỏ tới thư mục model hoàn chỉnh -> tự coi như --model-dir,
    # khỏi báo "không thấy checkpoint" khi runs/ chỉ còn model đã export.
    if not model_dir and not args.ckpt and not args.base_only and os.path.isdir(args.run) and is_model_dir(args.run):
        model_dir = args.run
    if model_dir:                          # thư mục hoàn chỉnh (config+vocab+model.pth)
        base_dir = model_dir
        config_json = os.path.join(base_dir, "config.json")
        vocab_path = os.path.join(base_dir, "vocab.json")
        checkpoint_dir = base_dir          # model.pth nằm cùng chỗ
        src_desc = f"model-dir {base_dir}"
    else:                                   # config/vocab từ --base, trọng số từ checkpoint rời
        base_dir = args.base
        config_json = os.path.join(base_dir, "config.json")
        vocab_path = os.path.join(base_dir, "vocab.json")
        checkpoint_dir = None
        if args.base_only:
            ckpt_path = os.path.join(base_dir, "model.pth")  # model gốc
            src_desc = f"BASE (gốc) {ckpt_path}"
        else:
            ckpt_path = args.ckpt or find_checkpoint(args.run)
            src_desc = f"checkpoint {ckpt_path}"

    for f in (config_json, vocab_path):
        if not os.path.isfile(f):
            sys.exit(f"❌ Thiếu {f}. Kiểm tra --base/--model-dir (cần config.json + vocab.json).")
    log(f"nguồn model: {src_desc}")
    log(f"config/vocab: {base_dir}")

    # 2) Reference: dùng --ref nếu có sẵn, không thì cắt từ --src
    if args.ref and os.path.isfile(args.ref):
        ref = args.ref
        import soundfile as sf
        log(f"reference (dùng sẵn): {ref} | {sf.info(ref).duration:.1f}s")
    else:
        log(f"chưa có {args.ref} — cắt reference từ {args.src}")
        ref = prepare_reference(args.src, args.ref_start, args.ref_dur, args.ref or DEFAULT_REF)

    # 3) Nạp model
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    log(f"torch {torch.__version__} | CUDA={torch.cuda.is_available()} | dùng={'cuda' if use_cuda else 'cpu'}")
    if args.device == "cuda" and not torch.cuda.is_available():
        log("⚠️ Yêu cầu cuda nhưng không thấy GPU — chạy CPU (sẽ chậm).")

    t0 = time.time()
    config = XttsConfig()
    config.load_json(config_json)
    if isinstance(getattr(config, "languages", None), list) and "vi" not in config.languages:
        config.languages.append("vi")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_path=ckpt_path,
        vocab_path=vocab_path,
        use_deepspeed=False,
    )
    if use_cuda:
        model.cuda()
    log(f"nạp model xong ({time.time() - t0:.1f}s)")

    # 4) Tính latent giọng từ reference
    t0 = time.time()
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[ref])
    log(f"tính latent giọng xong ({time.time() - t0:.1f}s)")

    # 5) Tổng hợp — bỏ markdown, tách đoạn, ghép + chèn khoảng nghỉ
    import numpy as np
    import soundfile as sf

    pieces = split_text(strip_markdown(args.text), max_chars=args.max_chars)
    log(f"tách thành {len(pieces)} câu")
    gap = np.zeros(int(args.gap * 24000), dtype="float32")  # khoảng nghỉ giữa câu

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
    log(f"✅ Đã ghi: {os.path.abspath(args.out)}  (nguồn: {src_desc})")
    log(f"👉 Nghe thử: aplay {args.out}")


if __name__ == "__main__":
    main()
