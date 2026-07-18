"""TTS — chuyển văn bản thành giọng nói tiếng Việt.

Hỗ trợ 2 engine, chọn qua TTS_ENGINE:
  - "piper" : TTS LOCAL, offline (rhasspy/piper) — khuyến nghị cho server tự host.
              Không phụ thuộc mạng, ổn định. Cần binary piper + file voice .onnx.
  - "edge"  : edge-tts (Microsoft, cloud) — tiện nhưng hay lỗi NoAudioReceived nếu
              mạng chặn hoặc lệch giờ hệ thống.

`synthesize(text) -> bytes` trả về audio (Piper: WAV, Edge: MP3). Trình duyệt
decode được cả hai qua Web Audio API.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile

from core.config import settings

logger = logging.getLogger("seller_agent.tts")


class _EdgeBackend:
    def __init__(self) -> None:
        self.voice = settings.tts_voice
        self.rate = settings.tts_rate

    async def synthesize(self, text: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)


class _PiperBackend:
    """Gọi binary piper qua subprocess: đọc text từ stdin, ghi WAV ra file tạm."""

    def __init__(self) -> None:
        self.bin = settings.piper_bin
        self.model = settings.piper_model
        self.length_scale = settings.piper_length_scale

    def _resolve_bin(self) -> str:
        """Trả về đường dẫn binary piper chạy được, hoặc raise lỗi rõ ràng."""
        import shutil

        # Nếu là tên trần (không có '/') → tìm trong PATH; nếu là đường dẫn → kiểm tra file.
        if os.sep in self.bin or self.bin.startswith("."):
            path = os.path.abspath(self.bin)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Không tìm thấy binary piper tại '{path}'. "
                    "Kiểm tra PIPER_BIN trong .env (nên dùng đường dẫn tuyệt đối)."
                )
            if not os.access(path, os.X_OK):
                raise PermissionError(f"Binary piper không có quyền chạy: {path}. Chạy: chmod +x {path}")
            return path
        found = shutil.which(self.bin)
        if not found:
            raise FileNotFoundError(
                f"Không tìm thấy '{self.bin}' trong PATH. Đặt PIPER_BIN = đường dẫn tuyệt đối "
                "tới binary piper (vd /home/ban/.../backend/piper/piper) trong .env."
            )
        return found

    async def synthesize(self, text: str) -> bytes:
        if not self.model or not os.path.exists(self.model):
            raise FileNotFoundError(
                f"Không tìm thấy Piper voice model: '{self.model}'. "
                "Tải file .onnx (+ .onnx.json) và đặt PIPER_MODEL trong .env."
            )
        piper_bin = self._resolve_bin()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            proc = await asyncio.create_subprocess_exec(
                piper_bin,
                "--model",
                self.model,
                "--length_scale",
                str(self.length_scale),
                "--output_file",
                out_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate(text.encode("utf-8"))
            if proc.returncode != 0:
                raise RuntimeError(
                    f"piper lỗi (code {proc.returncode}): {stderr.decode('utf-8', 'ignore')[:300]}"
                )
            with open(out_path, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass


class _EspeakBackend:
    """espeak-ng (local, offline). Chất lượng máy móc nhưng cực dễ cài, chạy chắc.

    Cài: sudo apt install espeak-ng
    """

    def __init__(self) -> None:
        self.bin = settings.espeak_bin
        self.voice = settings.espeak_voice
        self.speed = settings.espeak_speed

    async def synthesize(self, text: str) -> bytes:
        import shutil

        if not shutil.which(self.bin) and not os.path.isfile(self.bin):
            raise FileNotFoundError(
                f"Không tìm thấy '{self.bin}'. Cài: sudo apt install espeak-ng"
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            proc = await asyncio.create_subprocess_exec(
                self.bin, "-v", self.voice, "-s", str(self.speed), "-w", out_path, "--stdin",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate(text.encode("utf-8"))
            if proc.returncode != 0:
                raise RuntimeError(f"espeak-ng lỗi: {stderr.decode('utf-8', 'ignore')[:300]}")
            with open(out_path, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass


class _ViXTTSBackend:
    """viXTTS — voice cloning tiếng Việt (XTTS-v2 fine-tune) chạy local GPU.

    Clone giọng zero-shot từ 1 mẫu audio (settings.vixtts_speaker_wav). Model + coqui-tts
    nạp lười ở lần synth đầu (tránh crash lúc khởi động nếu chưa cài xong).

    ⚠️ XTTS dùng Coqui Public Model License — hạn chế dùng thương mại; tự cân nhắc.
    """

    _MAX_CHARS = 200  # XTTS giới hạn độ dài/câu → tách nhỏ rồi ghép

    def __init__(self) -> None:
        self.model_dir = settings.vixtts_model_dir
        self.speaker_wav = settings.vixtts_speaker_wav
        self.language = settings.vixtts_language
        self.device = settings.vixtts_device
        self.temperature = settings.vixtts_temperature
        self._model = None
        self._gpt_cond_latent = None
        self._speaker_embedding = None

    @staticmethod
    def _patch_vi_tokenizer() -> None:
        """coqui-tts gốc không hỗ trợ 'vi' → thêm cleaner tối giản + đọc số num2words('vi')."""
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

    def _load(self) -> None:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        self._patch_vi_tokenizer()

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Không thấy thư mục model viXTTS: '{self.model_dir}'. "
                "Tải model capleaf/viXTTS theo README và đặt VIXTTS_MODEL_DIR."
            )
        if not os.path.isfile(self.speaker_wav):
            raise FileNotFoundError(
                f"Không thấy mẫu giọng: '{self.speaker_wav}'. "
                "Đặt file WAV giọng nữ mục tiêu và trỏ VIXTTS_SPEAKER_WAV."
            )
        config = XttsConfig()
        config.load_json(os.path.join(self.model_dir, "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=self.model_dir, use_deepspeed=False)
        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        if use_cuda:
            model.cuda()
        logger.info("viXTTS đã nạp (device=%s)", "cuda" if use_cuda else "cpu")
        # Tính latent giọng 1 lần (dùng lại cho mọi câu → nhanh hơn)
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[self.speaker_wav]
        )
        self._model = model
        self._gpt_cond_latent = gpt_cond_latent
        self._speaker_embedding = speaker_embedding

    @staticmethod
    def _split(text: str, limit: int) -> list[str]:
        import re

        text = text.strip()
        if len(text) <= limit:
            return [text]
        # tách theo câu, gộp lại cho tới ~limit ký tự
        sentences = re.split(r"(?<=[.!?…])\s+", text)
        chunks, cur = [], ""
        for s in sentences:
            if len(cur) + len(s) + 1 <= limit:
                cur = f"{cur} {s}".strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
        return chunks or [text[:limit]]

    def _normalize(self, text: str) -> str:
        # Bỏ markdown để không đọc '#'/'*'; số được xử lý bởi bản vá tokenizer (num2words)
        import re

        text = re.sub(r"`{1,3}", "", text)
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # [text](url) -> text
        text = re.sub(r"[*_#>]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _infer_sync(self, text: str) -> bytes:
        import io

        import numpy as np
        import soundfile as sf

        if self._model is None:
            self._load()

        text = self._normalize(text)
        wavs: list[np.ndarray] = []
        for chunk in self._split(text, self._MAX_CHARS):
            if not chunk.strip():
                continue
            out = self._model.inference(
                chunk,
                self.language,
                self._gpt_cond_latent,
                self._speaker_embedding,
                temperature=self.temperature,
            )
            wavs.append(np.asarray(out["wav"], dtype="float32"))
        if not wavs:
            return b""
        wav = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
        buf = io.BytesIO()
        sf.write(buf, wav, 24000, format="WAV")
        return buf.getvalue()

    async def synthesize(self, text: str) -> bytes:
        # Inference nặng → chạy trong thread để không chặn event loop
        return await asyncio.to_thread(self._infer_sync, text)


class TTSEngine:
    """Dispatcher: chọn backend theo settings.tts_engine."""

    def __init__(self) -> None:
        self.engine = settings.tts_engine
        self.audio_format = "mp3" if self.engine == "edge" else "wav"
        if self.engine == "piper":
            self._backend = _PiperBackend()
        elif self.engine == "espeak":
            self._backend = _EspeakBackend()
        elif self.engine == "vixtts":
            self._backend = _ViXTTSBackend()
        else:
            self._backend = _EdgeBackend()
        logger.info("TTS engine=%s (format=%s)", self.engine, self.audio_format)

    async def synthesize(self, text: str) -> bytes:
        if not settings.tts_enabled or not text.strip():
            return b""
        return await self._backend.synthesize(text)
