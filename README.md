# Seller Agent — AI mô phỏng live stream bán hàng (2D Virtual Seller)

Trợ lý ảo 2D mô phỏng người bán hàng trên live: **đọc comment của khách theo thời
gian thực** và **trả lời như người thật bằng giọng nói tiếng Việt**, dựa trên danh
mục sản phẩm. Avatar hiển thị dạng nhân vật (SVG mặc định hoặc **Live2D**) và
**nhép miệng đồng bộ** khi đọc.

> Đây là **MVP (Phase 1)**: nguồn comment là **giả lập/nhập tay** để chứng minh
> concept lõi AI trước; tích hợp TikTok Live thật ở Phase 2. Xem lộ trình cuối file.

## Kiến trúc

```
[Trang admin: gõ comment] ──ws /ws/comments──▶  FastAPI backend
                                                  ├─ Comment queue (asyncio)
[Màn live: React] ◀── ws /ws/stream ─────────────┤  worker: comment → LLM → TTS
   - Avatar (SVG / Live2D)                        ├─ ProductStore (JSON/Excel)
   - Bong bóng chat                               ├─ LLM: Qwen self-host (Ollama) /
   - Phát audio + lip-sync                        │        Claude / offline template
                                                  └─ edge-tts (giọng Việt)
```

**Bộ não LLM** chọn qua `LLM_PROVIDER`: `qwen` (self-host Qwen qua Ollama — mặc định),
`anthropic` (Claude), hoặc `offline` (template, không cần model). Backend dùng client
tương thích OpenAI nên cắm được mọi endpoint kiểu OpenAI (Ollama, vLLM, LM Studio,
llama.cpp server...).

Chi tiết quyết định kỹ thuật: `backend/app/*` có chú thích, và file kế hoạch trong
`~/.claude/plans/`.

## Yêu cầu

- Python 3.11+ (đã test 3.14) — dùng [`uv`](https://github.com/astral-sh/uv) hoặc `pip`.
- Node.js 18+ (đã test 22).
- **GPU + Ollama** để chạy Qwen self-host (khuyến nghị) — xem mục 0. Nếu chưa dựng
  được, backend tự chạy **chế độ offline** (template) để demo UI/giọng nói.

## 0. Chuẩn bị LLM Qwen self-host (Ollama trên RTX 5060 — 8GB)

RTX 5060 là GPU **Blackwell (sm_120)** — hãy cài **Ollama bản mới nhất** (đã hỗ trợ
Blackwell + CUDA 12.8). Driver NVIDIA nên ≥ 570.

```bash
# Cài Ollama (Linux). Windows/macOS: tải app tại https://ollama.com/download
curl -fsSL https://ollama.com/install.sh | sh

# Kéo model Qwen (Q4, ~4.7GB — vừa 8GB VRAM). Chậm/OOM thì dùng qwen2.5:3b.
ollama pull qwen2.5:7b

# Ollama tự chạy nền và expose API OpenAI-compatible tại http://localhost:11434/v1
# Kiểm tra nhanh:
ollama run qwen2.5:7b "Chào bạn"       # thử tiếng Việt
curl http://localhost:11434/v1/models  # thấy model đã pull
```

Gợi ý model theo VRAM:

| VRAM | Model khuyến nghị | Ghi chú |
|------|-------------------|---------|
| 8GB (5060)     | `qwen2.5:7b` (Q4) | Vừa đủ, context ~4k. Chật thì `qwen2.5:3b` (nhanh hơn) |
| 16GB (5060 Ti) | `qwen2.5:7b` / `qwen2.5:14b` (Q4) | Context dài hơn, chất lượng cao hơn |

> Backend nối tới Ollama qua `OPENAI_BASE_URL` (mặc định `http://localhost:11434/v1`)
> và `LLM_MODEL` (mặc định `qwen2.5:7b`) trong `.env`. Có thể thay bằng vLLM / LM Studio
> / llama.cpp server — chỉ cần endpoint kiểu OpenAI.

## 1. Chạy Backend

```bash
cd backend
cp .env.example .env          # mặc định LLM_PROVIDER=qwen trỏ tới Ollama local

# Dùng uv (khuyến nghị):
uv venv .venv
uv pip install --python .venv -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000

# hoặc dùng pip thường:
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
# uvicorn app.main:app --reload --port 8000
```

Kiểm tra: mở http://localhost:8000/health → thấy `provider`, `model`, số sản phẩm, giọng TTS.

## 2. Chạy Frontend

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

Trên trang web:
1. Bấm **"🔊 Bật âm thanh"** (bắt buộc — chính sách autoplay của trình duyệt).
2. Gõ comment ở khung dưới **hoặc** bấm các nút mẫu để gửi.
3. Avatar sẽ đọc câu trả lời và **nhép miệng** theo giọng.

## 2b. TTS local với Piper (khuyến nghị cho self-host)

`edge-tts` gọi cloud Microsoft nên hay lỗi `NoAudioReceived`. Piper chạy **offline**,
ổn định. Cài 1 lần:

```bash
cd backend

# 1) Tải binary piper (Linux x86_64). Xem release khác tại github.com/rhasspy/piper/releases
curl -L -o piper.tar.gz \
  https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz
tar -xzf piper.tar.gz          # tạo thư mục ./piper/ chứa binary 'piper'

# 2) Tải voice tiếng Việt (.onnx + .onnx.json) từ rhasspy/piper-voices
mkdir -p voices
curl -L -o voices/vi_VN-vais1000-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx
curl -L -o voices/vi_VN-vais1000-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json

# 3) Test nhanh (đọc stdin → WAV)
echo "Xin chào cả nhà, shop có nhiều ưu đãi nha" | ./piper/piper \
  --model voices/vi_VN-vais1000-medium.onnx --output_file /tmp/test.wav
```

Trong `.env` đặt:
```
TTS_ENGINE=piper
PIPER_BIN=./piper/piper                 # hoặc đường dẫn tuyệt đối tới binary
PIPER_MODEL=voices/vi_VN-vais1000-medium.onnx
```

> Voice khác: `vi_VN-25hours_single-medium` (giọng nữ khác). Duyệt tại
> https://huggingface.co/rhasspy/piper-voices/tree/main/vi/vi_VN
> Muốn dùng lại edge-tts: đặt `TTS_ENGINE=edge`.

## 2c. TTS voice cloning giọng riêng với viXTTS (GPU)

Clone **giọng nữ của bạn** — chỉ cần 1 mẫu audio, không cần train. Chạy local GPU
(RTX 5060 dư sức khi LLM đã ở server khác).

**Chuẩn bị mẫu giọng** (`voices/my_voice.wav`) — đây là giọng sẽ được clone:
- **6–30 giây** (dài hơn/nhiều mẫu → giống hơn), **1 người nói**, giọng nữ tự nhiên.
- Phòng yên tĩnh, không nhạc nền, không vọng. Mono, 22.05kHz hoặc 24kHz.
- Nói tự nhiên như đang bán hàng (không đọc đều đều).
- Đây là **giọng của bạn** hoặc người **đã đồng ý** cho dùng.

**Cài đặt (một lần) — combo đã kiểm chứng trên RTX 5060:**
```bash
cd backend

# 1) PyTorch CUDA 12.8 cho Blackwell (sm_120) — CÀI TRƯỚC
uv pip install --python .venv torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2) coqui-tts + transformers (ghim <5) + torchcodec + num2words
uv pip install --python .venv -r requirements-vixtts.txt

# 3) FFmpeg hệ thống (torchcodec cần thư viện libav*)
sudo apt install -y ffmpeg

# 4) Tải model viXTTS (~1.8GB) vào models/viXTTS
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('capleaf/viXTTS', local_dir='models/viXTTS')"

# 5) Đặt mẫu giọng (WAV giọng nữ mục tiêu)
mkdir -p voices && cp /đường/dẫn/giọng_nữ.wav voices/my_voice.wav
```

Trong `.env`:
```
TTS_ENGINE=vixtts
VIXTTS_MODEL_DIR=models/viXTTS
VIXTTS_SPEAKER_WAV=voices/my_voice.wav
VIXTTS_DEVICE=cuda
```

**Test nhanh clone giọng** (script debug, tự cắt reference + đọc thử 1 câu → `debug_vixtts_out.wav`):
```bash
uv run python debug_vixtts.py                       # dùng data/female_1.wav mặc định
uv run python debug_vixtts.py --src /path/giong.wav --ref-start 60 --ref-dur 25
```

Khởi động lại backend. Lần đọc **đầu tiên** hơi lâu (nạp model ~10–17s), câu sau ~1s
(RTF ~0.2 trên 5060). `nvidia-smi` sẽ thấy python dùng ~4–6GB.

> Tiếng Việt (`vi`) được **tự động vá** vào tokenizer XTTS trong code (coqui-tts gốc
> không có), kèm đọc số bằng `num2words` — không cần vinorm.

> ⚠️ **Bản quyền:** XTTS dùng *Coqui Public Model License* — **hạn chế dùng thương mại**.
> Cân nhắc kỹ nếu triển khai bán hàng thật (hoặc dùng Piper/espeak cho môi trường thương mại).
>
> **Đọc số chưa tự nhiên?** đã cài `vinorm` (trong requirements-vixtts) để chuẩn hóa
> "119k" → "một trăm mười chín nghìn". Muốn giống hơn nữa: thu mẫu giọng dài hơn,
> hoặc fine-tune viXTTS trên dataset của bạn (bước nâng cao).

## 3. Tùy chỉnh

- **Tính cách agent (persona):** sửa `PERSONA` trong `.env` hoặc `DEFAULT_PERSONA`
  trong `backend/app/config.py`.
- **Danh mục sản phẩm:** sửa `backend/data/products.json` (hoặc trỏ `PRODUCTS_PATH`
  tới file `.xlsx` — cột: `id, ten, gia, gia_km, mau, size, ton_kho, mo_ta, faq`).
- **Giọng nói:** `TTS_ENGINE` = `vixtts` (clone giọng riêng, GPU — mục 2c) /
  `piper` (local nhanh — 2b) / `espeak` (dễ nhất) / `edge` (cloud). Mỗi engine có
  biến riêng: viXTTS (`VIXTTS_SPEAKER_WAV`...), Piper (`PIPER_MODEL`...),
  edge (`TTS_VOICE`, `TTS_RATE`).
- **Model / provider:** `LLM_PROVIDER` (`qwen`/`anthropic`/`offline`), `LLM_MODEL`
  (mặc định `qwen2.5:7b`), `OPENAI_BASE_URL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`.
- **Live2D thật:** xem `frontend/public/models/README.md`.

## Cấu trúc thư mục

```
backend/
  app/
    main.py            # FastAPI + WebSocket /ws/comments, /ws/stream
    config.py          # cấu hình + persona ("train tính cách")
    products.py        # ProductStore: load JSON/Excel, catalog_text(), search()
    llm.py             # SellerBrain: Qwen (OpenAI-compat) / Claude / offline template
    tts.py             # TTSEngine: edge-tts → MP3 bytes
    pipeline.py        # worker: queue → llm → tts → broadcast; StreamHub
    comment_source.py  # CommentSource + SimulatedCommentSource (interface cho TikTok)
  data/products.json   # danh mục mẫu
frontend/
  src/
    App.tsx
    components/        # Live2DStage (SVG/Live2D), ChatOverlay, AdminPanel
    hooks/useWebSocket.ts
    lib/lipsync.ts     # Web Audio → độ mở miệng 0..1
    lib/live2d.ts      # nạp model Cubism (tùy chọn)
```

## Lộ trình sau MVP

- **Phase 2 — TikTok Live thật:** thay `SimulatedCommentSource` bằng connector
  `TikTokLive` (đọc comment realtime). Thêm lọc/gộp/ưu tiên câu hỏi, chống spam.
- **Phase 3 — Học từ video live cũ:** transcribe (Whisper) → học phong cách nói &
  kịch bản bán hàng cho persona; trích ngoại hình người bán để tạo/chọn avatar;
  (tùy chọn) voice-cloning giọng người bán.
- **Nâng cấp RAG** khi danh mục lớn; **kết nối API shop/CRM** để giá/tồn kho realtime.
