# Fine-tune LLM cho Seller Agent

Pipeline fine-tune một base **Qwen3 ~8-9B** thành trợ lý bán hàng livestream tiếng
Việt: đúng giọng persona, chêm tiếng Anh tự nhiên, **bám catalog trong prompt và
từ chối bịa**, khéo chốt đơn. Train trên **1×A6000 (48GB)** bằng QLoRA (Unsloth).

> **Nguyên tắc:** fine-tune *hành vi*, KHÔNG nhồi catalog vào weights. Giá/tồn kho
> vẫn inject vào system prompt lúc chạy (như hiện tại) — nên fine-tune chỉ là *swap
> provider*, không đại tu. Prompt lúc train khớp tuyệt đối với lúc serve nhờ
> [app/prompt.py](../app/prompt.py) (nguồn sự thật duy nhất, dùng chung cả 2 phía).

## Cài đặt (máy GPU riêng)

```bash
# Cài torch theo đúng CUDA trước, rồi:
pip install -r ../requirements-llm-train.txt
export ANTHROPIC_API_KEY=...   # teacher sinh data + judge
```

Chạy mọi lệnh từ thư mục `backend/` (để `import app.*` hoạt động).

## Quy trình

### 1. Sinh dữ liệu synthetic (teacher = Claude)
```bash
python -m training.llm.gen_synthetic \
    --n 4000 --products data/products.json \
    --out training/data/raw/synthetic_raw.jsonl
```
Chia theo intent (giá/tồn/size, chính sách, tư vấn size, so sánh, chốt đơn,
off-topic) + **hard-negative** (hỏi ngoài catalog / hết hàng → phải từ chối) +
~20% code-switch VN+EN. Xem [taxonomy.py](llm/taxonomy.py).

### 2. Lọc + build dataset ChatML
```bash
python -m training.llm.build_dataset \
    --raw training/data/raw/synthetic_raw.jsonl \
    --out-dir training/data/curated
```
Lọc grounding (không bịa giá/size), format (1-3 câu, không markdown), language
(không drift Trung/Anh), dedup → `train.jsonl` / `eval.jsonl` / `golden.jsonl`
(stratified theo intent). Có thể trộn thêm log thật: `--raw ... logs/interactions/*.jsonl`
(sau khi chuyển log về raw record schema).

### 3. Train QLoRA
```bash
python -m training.llm.train_qlora --config training/llm/configs/qwen3_8b_qlora.yaml
```
Chỉ học turn assistant (mask system+user). Adapter lưu ở `training/data/adapters/...`.

### 4. Đánh giá (so baseline — chỉ promote khi thắng)
```bash
python -m training.llm.evaluate \
    --model seller-qwen3:8b --baseline qwen2.5:7b \
    --base-url http://localhost:11434/v1 \
    --golden training/data/curated/golden.jsonl \
    --out training/data/scorecard.md
```
Metric tự động (grounding/format/language) + LLM-judge (persona/fluency/
faithfulness/chốt đơn/refusal).

### 5. Export & serve
```bash
python -m training.llm.merge_export \
    --adapter training/data/adapters/qwen3-8b-seller-r1 \
    --out training/data/adapters/qwen3-8b-seller-r1-gguf --quant q4_k_m
ollama create seller-qwen3:8b -f training/data/adapters/qwen3-8b-seller-r1-gguf/Modelfile
```
Rồi trong `backend/.env`: `LLM_PROVIDER=finetuned`, `LLM_MODEL=seller-qwen3:8b`.
Không đổi code app.

### 6. Flywheel
Bật `INTERACTION_LOG=true` để thu comment→reply thật + `POST /feedback` cho reply
admin sửa tay (gold label). Vòng sau: trộn real + synthetic → train lại → so scorecard.
