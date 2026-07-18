# Hướng dẫn Fine-tune LLM cho Seller Agent

Tài liệu này hướng dẫn chạy **từ A→Z**: từ chuẩn bị môi trường → sinh dữ liệu →
train QLoRA (Qwen3-8B) → đánh giá → deploy vào backend.

- **Mục tiêu:** biến base Qwen3 ~8-9B thành trợ lý bán hàng livestream tiếng Việt —
  đúng giọng persona, chêm tiếng Anh tự nhiên, **bám catalog và không bịa**, khéo chốt đơn.
- **Phần cứng:** máy này có **4× RTX A6000 (48GB)**. Fine-tune chỉ cần **1 GPU**
  (Unsloth bản OSS train single-GPU) — ta ghim 1 card bằng `CUDA_VISIBLE_DEVICES`.
- **Nguyên tắc cốt lõi:** fine-tune *hành vi*, **KHÔNG** nhồi catalog vào weights.
  Giá/tồn kho vẫn được inject vào prompt lúc chạy → đổi kho không cần train lại.

> Sơ đồ tổng thể:
> `gen_synthetic` (Claude sinh data) → `build_dataset` (lọc + ChatML) →
> `train_qlora` (Unsloth) → `evaluate` (so baseline) → `merge_export` (GGUF) →
> `ollama create` → đổi `.env` là xong.

---

## 0. Yêu cầu trước khi bắt đầu

| Thứ cần | Ghi chú |
|---|---|
| GPU NVIDIA + CUDA driver | máy đã có 4× A6000 |
| Python 3.12 + venv | dự án đã có sẵn `.venv` |
| `ANTHROPIC_API_KEY` | để Claude sinh data + làm giám khảo (judge) |
| Ollama | để serve model sau khi fine-tune (đã dùng cho qwen2.5:7b) |

Tất cả lệnh chạy **từ thư mục `backend/`**.

---

## 1. Cài môi trường training

Môi trường train **tách riêng** khỏi runtime (không làm nặng server). Nên cài vào
`.venv` sẵn có, hoặc tạo venv riêng nếu muốn cách ly hẳn.

```bash
cd backend

# (khuyến nghị) cài torch khớp CUDA của máy TRƯỚC — ví dụ CUDA 12.1:
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121

# rồi cài phần còn lại
.venv/bin/pip install -r requirements-llm-train.txt

# khai báo key cho teacher/judge
export ANTHROPIC_API_KEY=sk-ant-...
```

Kiểm tra nhanh:
```bash
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count(), 'GPU')"
.venv/bin/python -c "import unsloth; print('unsloth OK')"
```

---

## 2. Sinh dữ liệu

### 2A. KHÔNG có API key → sinh MẪU offline (để debug pipeline)

Nếu chưa có `ANTHROPIC_API_KEY`, dùng bộ sinh **offline bằng template** (không gọi
API) để có ngay dataset chạy thử toàn bộ pipeline train→export→eval:

```bash
.venv/bin/python -m training.llm.gen_offline \
    --n 1200 --products data/products.json \
    --out training/data/raw/offline_raw.jsonl
```

Data này grounded đúng catalog (qua được bộ lọc) và đủ 12 intent + code-switch,
**đủ để kiểm tra code fine-tune hoạt động**, nhưng đa dạng/tự nhiên kém hơn teacher
thật — khi có key hãy chuyển sang 2B để chất lượng model tốt hơn. Sang thẳng
[mục 3](#3-lọc--build-dataset-chatml) với file `offline_raw.jsonl`.

### 2B. Có API key → sinh bằng Claude teacher (chất lượng cao)

Sinh cặp `comment → reply` **grounded** theo catalog (dùng cả catalog thật
`data/products.json` lẫn catalog tổng hợp để đa dạng). Đã tự chia theo intent +
hard-negative (chống bịa) + ~20% code-switch VN+EN.

```bash
export ANTHROPIC_API_KEY=sk-ant-...

.venv/bin/python -m training.llm.gen_synthetic \
    --n 4000 \
    --products data/products.json \
    --out training/data/raw/synthetic_raw.jsonl
```

Tham số hữu ích:
- `--n` : tổng số mẫu (bắt đầu 4000–8000; QLoRA cần ít data chất lượng hơn là nhiều).
- `--model` : teacher (mặc định `claude-sonnet-5` — mạnh & rẻ; dùng `claude-opus-4-8` nếu muốn chất hơn).
- `--per-call` : số mẫu/lần gọi API (mặc định 8).
- `--codeswitch-ratio` : tỉ lệ chêm tiếng Anh (mặc định 0.2).
- `--real-ratio` : tỉ lệ dùng catalog thật khi có `--products` (mặc định 0.25).

> 💡 Nên chạy thử `--n 40` trước để kiểm tra key/định dạng, rồi mới chạy full.

---

## 3. Lọc & build dataset (ChatML)

Lọc grounding (không bịa giá/size), format (1–3 câu, không markdown), language
(loại drift tiếng Trung / full-English), dedup → chia `train/eval/golden`
stratified theo intent.

```bash
.venv/bin/python -m training.llm.build_dataset \
    --raw training/data/raw/synthetic_raw.jsonl \
    --out-dir training/data/curated
```

Kết quả:
- `training/data/curated/train.jsonl` — để train.
- `training/data/curated/eval.jsonl` — theo dõi eval loss.
- `training/data/curated/golden.jsonl` — bộ chấm điểm, **không bao giờ train**.

Log sẽ in số mẫu giữ/loại và phân bố intent. Nếu tỉ lệ loại quá cao → xem lại data teacher.

Trộn thêm **dữ liệu thật** (vòng 2+): truyền nhiều file vào `--raw`
(`--raw synthetic_raw.jsonl real_converted.jsonl`). Xem [mục 7](#7-flywheel--thu-data-thật).

---

## 4. Train QLoRA (Qwen3-8B)

Ghim 1 GPU (ví dụ card 0) rồi train theo config:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m training.llm.train_qlora \
    --config training/llm/configs/qwen3_8b_qlora.yaml
```

- Adapter lưu ở `training/data/adapters/qwen3-8b-seller-r1/`.
- Chỉ học phần **assistant** (mask system+user) → model học *trả lời*, không chép catalog.
- Chỉnh siêu tham số trong [training/llm/configs/qwen3_8b_qlora.yaml](../training/llm/configs/qwen3_8b_qlora.yaml):
  epochs, learning_rate, `lora_r`, `max_seq_length`, `base_model`...

**Đổi base model:** muốn dùng bản 9B thì sửa `base_model` trong YAML (vd
`Qwen/Qwen3.5-9B`) hoặc `--base-model ...` — pipeline không đổi.

**QLoRA vs LoRA:** mặc định `load_in_4bit: true` (QLoRA, dư VRAM). A6000 48GB thừa
sức chạy LoRA bf16 (`load_in_4bit: false`) nếu QLoRA chưa đạt giọng.

Theo dõi VRAM khi train: `watch -n2 nvidia-smi`. Với 8B QLoRA thường < 24GB.

---

## 5. Đánh giá (chỉ promote khi thắng baseline)

Trước tiên serve **cả model mới lẫn baseline** để so. Nếu chưa export GGUF, có thể
eval sau bước 6. Cách nhanh nhất là export xong rồi eval:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # cho LLM-judge

.venv/bin/python -m training.llm.evaluate \
    --model seller-qwen3:8b \
    --baseline qwen2.5:7b \
    --base-url http://localhost:11434/v1 \
    --golden training/data/curated/golden.jsonl \
    --out training/data/scorecard.md
```

Đọc `training/data/scorecard.md`:
- **grounding / format / language** (tự động, 0→1).
- **persona / fluency / faithfulness** (LLM-judge, 1→5), **order_closing**, **refusal_acc**.

👉 Chỉ dùng model mới nếu nó **thắng baseline** trên faithfulness + persona + refusal.

> **Không có API key?** Thêm `--no-judge` để chỉ chạy metric tự động (grounding/
> format/language) — không cần `ANTHROPIC_API_KEY`, đủ để kiểm tra pipeline eval.

---

## 6. Export GGUF & deploy vào backend

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m training.llm.merge_export \
    --adapter training/data/adapters/qwen3-8b-seller-r1 \
    --out training/data/adapters/qwen3-8b-seller-r1-gguf \
    --quant q4_k_m

# tạo model trên Ollama
ollama create seller-qwen3:8b -f training/data/adapters/qwen3-8b-seller-r1-gguf/Modelfile
```

Rồi trỏ backend sang model mới trong `backend/.env`:
```dotenv
LLM_PROVIDER=finetuned
LLM_MODEL=seller-qwen3:8b
OPENAI_BASE_URL=http://localhost:11434/v1
```

Khởi động lại backend là xong — **không cần đổi code**.

**Phương án throughput cao (vLLM):** thay vì GGUF, export merged 16-bit
(`merge_export ... --merged-16bit`), serve bằng vLLM (OpenAI-compatible), rồi chỉ
đổi `OPENAI_BASE_URL` sang host vLLM. Prefix-caching của vLLM giúp giảm latency vì
khối catalog trong system prompt lặp lại nhiều.

---

## 7. Flywheel — thu data thật

Bật ghi log tương tác để vòng train sau tốt hơn (dùng giọng khách thật):

```dotenv
# backend/.env
INTERACTION_LOG=true
# INTERACTION_LOG_DIR=logs/interactions
```

- Mỗi comment→reply được ghi JSONL vào `logs/interactions/interactions-YYYYMMDD.jsonl`
  (fire-and-forget, không làm chậm live).
- Admin chấm/sửa reply qua API để tạo **gold label** (giá trị nhất):
  ```bash
  curl -X POST http://localhost:8000/feedback -H 'Content-Type: application/json' \
    -d '{"comment":"áo còn size L ko","reply":"...","verdict":"edited","edited_reply":"Dạ áo còn size L màu đen nha cả nhà!"}'
  ```
  `verdict` = `good` | `bad` | `edited`.

Vòng 2: chuyển log về raw-record schema (kèm catalog của thời điểm đó), trộn với
synthetic ở bước 3, train lại, rồi so scorecard. (Tùy chọn: thêm DPO/ORPO với
chosen = reply grounded, rejected = reply bịa, để siết hành vi từ chối.)

---

## Chạy toàn bộ trong 1 phát (tham khảo)

```bash
cd backend
export ANTHROPIC_API_KEY=sk-ant-...

.venv/bin/python -m training.llm.gen_synthetic --n 4000 --products data/products.json \
    --out training/data/raw/synthetic_raw.jsonl
.venv/bin/python -m training.llm.build_dataset --raw training/data/raw/synthetic_raw.jsonl \
    --out-dir training/data/curated
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m training.llm.train_qlora \
    --config training/llm/configs/qwen3_8b_qlora.yaml
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m training.llm.merge_export \
    --adapter training/data/adapters/qwen3-8b-seller-r1 \
    --out training/data/adapters/qwen3-8b-seller-r1-gguf --quant q4_k_m
ollama create seller-qwen3:8b -f training/data/adapters/qwen3-8b-seller-r1-gguf/Modelfile
.venv/bin/python -m training.llm.evaluate --model seller-qwen3:8b --baseline qwen2.5:7b \
    --golden training/data/curated/golden.jsonl --out training/data/scorecard.md
```

---

## Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `Thiếu ANTHROPIC_API_KEY` | `export ANTHROPIC_API_KEY=...` trước khi chạy gen/evaluate |
| `gen_synthetic` giữ ít mẫu / parse lỗi | giảm `--per-call`, hoặc đổi `--model` sang teacher mạnh hơn |
| `build_dataset` loại quá nhiều | teacher đang bịa giá/markdown → xem lại prompt teacher; check vài dòng raw |
| OOM khi train | giữ `load_in_4bit: true`, giảm `per_device_batch_size`/`max_seq_length` |
| Ollama không thấy model | kiểm tra `ollama list`; chạy lại `ollama create ... -f Modelfile` |
| Model trả lời tiếng Trung/Anh lẫn | tăng slice code-switch chuẩn + data VN; giảm `temperature` khi serve |
| Backend vẫn dùng model cũ | đã set `LLM_PROVIDER=finetuned` + `LLM_MODEL` chưa? restart backend |

Chi tiết kiến trúc & lý do thiết kế: xem [../plans/finetune-llm-seller-agent.md](../plans/finetune-llm-seller-agent.md)
và [../training/README.md](../training/README.md).
