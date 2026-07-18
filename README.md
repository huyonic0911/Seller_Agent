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

Chi tiết quyết định kỹ thuật: `backend/core/*` và `backend/modules/*` có chú thích, và file kế hoạch trong
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

## 0b. RAG: bge-m3 (self-host) + Qdrant (module llm)

Agent trả lời dựa trên **tìm kiếm ngữ nghĩa** danh mục (embedding `BAAI/bge-m3` +
vector DB Qdrant) — chỉ nhét sản phẩm LIÊN QUAN vào prompt thay vì cả danh mục.
Data mẫu: `data/samsung_phones.json` (20 điện thoại Samsung).

```bash
cd backend

# 1) Cài deps RAG (sentence-transformers + qdrant-client)
uv pip install --python .venv -r requirements-rag.txt

# 2) Dựng Qdrant server bằng Docker (cổng 6333)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant

# 3) Index danh mục: embed bge-m3 (tải ~2.2GB lần đầu) -> upsert Qdrant
uv run python -m modules.llm.ingest
#   -> "✅ đã index 20 điểm" + vài truy vấn thử
```

Cấu hình `.env` (đã có mặc định hợp lý):
```
RAG_ENABLED=true
EMBED_MODEL=BAAI/bge-m3
EMBED_DEVICE=cpu               # cpu (an toàn VRAM) | cuda (nhanh hơn)
QDRANT_URL=http://localhost:6333   # rỗng -> chế độ nhúng local (không cần Docker)
RAG_TOP_K=4
```

> - Không có Docker? Bỏ trống `QDRANT_URL` → qdrant-client chạy **chế độ nhúng local**
>   (lưu ở `QDRANT_PATH=qdrant_data`), vẫn ingest/truy vấn được, không cần server.
> - Đổi danh mục: sửa `data/samsung_phones.json` (hoặc trỏ `PRODUCTS_PATH`) rồi **chạy lại ingest**.
> - RAG lỗi/ chưa ingest → SellerBrain tự lùi về nhét full catalog vào prompt (không chết).
> - `bge-m3` chạy self-host trong tiến trình; đổi `EMBED_DEVICE=cuda` nếu còn VRAM.

## 1. Chạy Backend

```bash
cd backend
cp .env.example .env          # mặc định LLM_PROVIDER=qwen trỏ tới Ollama local

# Dùng uv (khuyến nghị):
uv venv .venv
uv pip install --python .venv -r requirements.txt
.venv/bin/uvicorn core.server:app --reload --port 8000

# hoặc dùng pip thường:
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
# uvicorn core.server:app --reload --port 8000
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
uv run python modules/voice/training/debug_vixtts.py                       # dùng data/female_1.wav mặc định
uv run python modules/voice/training/debug_vixtts.py --src /path/giong.wav --ref-start 60 --ref-dur 25
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

## 2d. Fine-tune viXTTS giọng riêng (nâng cao — dùng cả file audio dài làm data)

Khi clone zero-shot chưa đủ giống, fine-tune trên toàn bộ audio (đã cắt câu + transcript).

Chạy trên **GPU lớn** (A6000/48GB — full fine-tune cần ~18–22GB ở batch 4). 5060 8GB
KHÔNG đủ để train (OOM ở optimizer state); train nơi khác rồi tải checkpoint về —
**inference vẫn chạy tốt trên 5060**.

```bash
cd backend

# 1) Chuẩn bị dataset: Whisper transcribe + cắt câu 3–11s + mono 22050 + metadata
uv run python modules/voice/training/prepare_dataset.py --src data/female_1.wav
#   -> dataset/wavs/*.wav + metadata_train.csv + metadata_eval.csv (142 clip ~15.6 phút)

# 2) Train (tự tải dvae.pth + mel_stats.pth từ coqui/XTTS-v2)
uv run python modules/voice/training/finetune_vixtts.py --epochs 30 --batch-size 4
#   -> checkpoint trong runs/vixtts_ft/<run>/  (theo dõi test_sentences + eval loss để tránh overfit)

# 3) Export checkpoint -> model dùng được + tự kiểm chứng 1 câu
uv run python modules/voice/training/export_vixtts.py --run runs/vixtts_ft --out models/viXTTS-ft
```

Rồi trong `.env` đổi 1 dòng để agent dùng giọng fine-tune:
```
VIXTTS_MODEL_DIR=models/viXTTS-ft
```

> - `--workers 0` **bắt buộc**: worker con của DataLoader không có bản vá tokenizer `vi` → lỗi (không liên quan VRAM).
> - Mốc `--batch-size`: ≥24GB → 4..8; 12–16GB → 2..3 (+`--max-wav 220000`); 8GB → không đủ.
> - Train trên cloud/A6000: upload `dataset/` + `models/viXTTS/`, chạy đúng 3 lệnh trên,
>   tải `models/viXTTS-ft/` về máy 5060 để chạy.

## 3. Tùy chỉnh

- **Tính cách agent (persona):** sửa `PERSONA` trong `.env` hoặc `DEFAULT_PERSONA`
  trong `backend/core/config.py`.
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

Backend tách theo 3 module chính + lớp `core` điều phối. Chạy từ `backend/`
(entrypoint `core.server:app`). Asset (data/, models/, voices/, dataset/, runs/)
để ở `backend/` root, dùng chung.

```
backend/
  core/                    # NỀN TẢNG chung (điều phối 3 module)
    server.py              # FastAPI + WebSocket /ws/comments, /ws/stream, REST /avatar...
    config.py              # cấu hình + persona ("train tính cách") + avatar/viseme
    pipeline.py            # worker: comment → LLM → Voice(TTS) → Vision(viseme) → broadcast
    comment_source.py      # CommentSource + SimulatedCommentSource (interface cho TikTok)
  modules/
    llm/                   # ── Module LLM (+ RAG) ──
      brain.py             # SellerBrain: Qwen / Claude / offline; dùng RAG dựng context
      rag.py               # ProductStore: load JSON/Excel, product_text(), catalog_text()
      embedder.py          # bge-m3 self-host (sentence-transformers, 1024d)
      vectorstore.py       # Qdrant (server url hoặc nhúng local)
      retriever.py         # câu hỏi -> top-k SP liên quan
      ingest.py            # `python -m modules.llm.ingest`: embed -> Qdrant
    voice/                 # ── Module Voice Cloning ──
      tts.py               # TTSEngine: vixtts / piper / espeak / edge
      training/            # prepare_dataset · finetune_vixtts · export_vixtts · debug_vixtts
    vision/                # ── Module Vision (avatar 2D/3D) ──
      avatar.py            # cấu hình avatar (kiểu, model, tham số miệng) cho frontend
      lipsync.py           # sinh viseme/độ-mở-miệng từ audio (chuẩn bị cho 3D)
  data/samsung_phones.json # danh mục RAG (20 SP)  ·  data/female_1.wav (giọng nguồn)
  models/ voices/ dataset/ runs/ qdrant_storage/   # model, giọng, dataset, vector DB (không commit)
frontend/                  # CLIENT render vision (Live2D/SVG)
  src/
    components/Live2DStage.tsx   # render SVG/Live2D + nhép miệng
    hooks/useWebSocket.ts        # nhận reply + audio + visemes
    lib/lipsync.ts               # (fallback) Web Audio → độ mở miệng nếu không có visemes
```

## Lộ trình sau MVP

- **Phase 2 — TikTok Live thật:** thay `SimulatedCommentSource` bằng connector
  `TikTokLive` (đọc comment realtime). Thêm lọc/gộp/ưu tiên câu hỏi, chống spam.
- **Phase 3 — Học từ video live cũ:** transcribe (Whisper) → học phong cách nói &
  kịch bản bán hàng cho persona; trích ngoại hình người bán để tạo/chọn avatar;
  (tùy chọn) voice-cloning giọng người bán.
- **Nâng cấp RAG** khi danh mục lớn; **kết nối API shop/CRM** để giá/tồn kho realtime.
