"""Cấu hình tập trung cho backend.

Các giá trị đọc từ biến môi trường (có thể đặt trong file .env). Persona ở đây
chính là nơi "train tính cách" cho agent — chỉnh giọng điệu, phong cách bán hàng.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    # Tùy chọn: nạp .env nếu có python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv là tùy chọn
    pass


BASE_DIR = Path(__file__).resolve().parent.parent  # .../backend


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# Persona mặc định của người bán hàng live. Đây là phần "tính cách" của agent.
DEFAULT_PERSONA = """\
Bạn là một người bán hàng live stream chuyên nghiệp, thân thiện của {shop_name}.
Phong cách nói chuyện:
- Nhiệt tình, gần gũi, xưng "shop"/"mình" và gọi khách là "cả nhà"/"mình ơi".
- Câu trả lời NGẮN GỌN (1-3 câu), tự nhiên như đang nói trực tiếp trên live, không dùng markdown.
- Trả lời đúng trọng tâm câu hỏi của khách, dựa CHÍNH XÁC vào dữ liệu sản phẩm được cung cấp
  (giá, tồn kho, size, màu, khuyến mãi, chính sách). Tuyệt đối không bịa thông tin.
- Nếu sản phẩm hết hàng thì nói thật và gợi ý sản phẩm/thời gian về hàng.
- Luôn khéo léo chốt đơn hoặc mời khách để lại thông tin khi phù hợp.
- Nếu câu hỏi không liên quan sản phẩm, trả lời ngắn gọn, lịch sự rồi kéo về chủ đề bán hàng.
- KHÔNG đọc lại tên khách hàng dạng emoji hay ký tự lạ.
"""

# Model mặc định theo từng nhà cung cấp
_DEFAULT_MODEL = {
    "qwen": "qwen2.5:7b",         # self-host qua Ollama (RTX 5060 8GB); hạ xuống qwen2.5:3b nếu chật
    "openai": "qwen2.5:7b",       # bí danh: bất kỳ endpoint tương thích OpenAI
    "finetuned": "seller-qwen3:8b",  # model đã fine-tune (Ollama-GGUF hoặc vLLM); endpoint OpenAI-compatible
    "anthropic": "claude-opus-4-8",
    "offline": "offline",
}


@dataclass
class Settings:
    # Nhà cung cấp LLM: "qwen" (Ollama/OpenAI-compatible) | "anthropic" | "offline"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "qwen").lower())

    # --- Qwen / endpoint tương thích OpenAI (Ollama, vLLM, LM Studio, llama.cpp...) ---
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    )
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "ollama"))

    # --- Anthropic (tùy chọn) ---
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # Chung
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    effort: str = field(default_factory=lambda: os.getenv("LLM_EFFORT", "low"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "512")))

    # TTS — engine: "piper" (local, khuyến nghị) | "edge" (cloud Microsoft)
    tts_engine: str = field(default_factory=lambda: os.getenv("TTS_ENGINE", "edge").lower())
    tts_enabled: bool = field(default_factory=lambda: _get_bool("TTS_ENABLED", True))
    # edge-tts
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "vi-VN-HoaiMyNeural"))
    tts_rate: str = field(default_factory=lambda: os.getenv("TTS_RATE", "+8%"))
    # piper (local)
    piper_bin: str = field(default_factory=lambda: os.getenv("PIPER_BIN", "piper"))
    piper_model: str = field(default_factory=lambda: os.getenv("PIPER_MODEL", ""))
    piper_length_scale: float = field(
        default_factory=lambda: float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))
    )
    # espeak-ng (local, dễ cài nhất)
    espeak_bin: str = field(default_factory=lambda: os.getenv("ESPEAK_BIN", "espeak-ng"))
    espeak_voice: str = field(default_factory=lambda: os.getenv("ESPEAK_VOICE", "vi"))
    espeak_speed: int = field(default_factory=lambda: int(os.getenv("ESPEAK_SPEED", "160")))
    # viXTTS (voice cloning, local GPU) — clone giọng từ 1 mẫu audio
    vixtts_model_dir: str = field(default_factory=lambda: os.getenv("VIXTTS_MODEL_DIR", "models/viXTTS"))
    vixtts_speaker_wav: str = field(default_factory=lambda: os.getenv("VIXTTS_SPEAKER_WAV", "voices/my_voice.wav"))
    vixtts_language: str = field(default_factory=lambda: os.getenv("VIXTTS_LANGUAGE", "vi"))
    vixtts_device: str = field(default_factory=lambda: os.getenv("VIXTTS_DEVICE", "cuda"))
    vixtts_temperature: float = field(
        default_factory=lambda: float(os.getenv("VIXTTS_TEMPERATURE", "0.7"))
    )

    # --- Vision (avatar) ---
    # Kiểu avatar: "svg" (mặc định frontend) | "live2d" (2D) | "3d" (chuẩn bị)
    avatar_type: str = field(default_factory=lambda: os.getenv("AVATAR_TYPE", "live2d").lower())
    # Định danh/đường dẫn model cho frontend nạp (Live2D .model3.json hoặc model 3D)
    avatar_model: str = field(default_factory=lambda: os.getenv("AVATAR_MODEL", ""))
    # Backend có sinh sẵn khung nhép miệng (viseme) từ audio để gửi kèm không
    viseme_enabled: bool = field(default_factory=lambda: _get_bool("VISEME_ENABLED", True))
    viseme_fps: int = field(default_factory=lambda: int(os.getenv("VISEME_FPS", "30")))

    # Dữ liệu
    products_path: Path = field(
        default_factory=lambda: Path(os.getenv("PRODUCTS_PATH", str(BASE_DIR / "data" / "samsung_phones.json")))
    )

    # --- RAG (module llm): bge-m3 embedding + Qdrant vector DB ---
    rag_enabled: bool = field(default_factory=lambda: _get_bool("RAG_ENABLED", True))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-m3"))
    embed_device: str = field(default_factory=lambda: os.getenv("EMBED_DEVICE", "cpu"))  # cpu|cuda
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "4")))
    # Qdrant: có QDRANT_URL -> nối server; rỗng -> chế độ nhúng local theo QDRANT_PATH
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    qdrant_path: str = field(default_factory=lambda: os.getenv("QDRANT_PATH", "qdrant_data"))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "products"))

    # Ghi log tương tác (comment→reply) để thu thập dữ liệu fine-tune vòng sau.
    # Bật INTERACTION_LOG=1 để lưu JSONL vào INTERACTION_LOG_DIR (mặc định logs/interactions).
    interaction_log: bool = field(default_factory=lambda: _get_bool("INTERACTION_LOG", False))
    interaction_log_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("INTERACTION_LOG_DIR", str(BASE_DIR / "logs" / "interactions"))
        )
    )

    # Persona
    persona: str = field(default_factory=lambda: os.getenv("PERSONA", DEFAULT_PERSONA))

    # CORS (frontend dev server)
    allowed_origins: list[str] = field(
        default_factory=lambda: os.getenv(
            "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
    )

    def __post_init__(self) -> None:
        if not self.model:
            self.model = _DEFAULT_MODEL.get(self.llm_provider, "qwen2.5:7b")


settings = Settings()
