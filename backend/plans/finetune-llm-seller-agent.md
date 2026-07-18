# Kế hoạch: Fine-tune LLM ~8-9B cho Seller Agent (livestream bán hàng TV)

## Context (vì sao làm việc này)

Backend hiện tại ([app/llm.py](app/llm.py) `SellerBrain`) dùng **Qwen2.5:7b qua Ollama** (mặc định) làm "bộ não" trả lời comment khách trong livestream bán hàng. Model gốc chỉ *bám theo* persona + catalog trong system prompt một cách lỏng lẻo: đôi khi bịa thông tin, giọng chưa "chuẩn seller live" tiếng Việt, và code-switch (chêm tiếng Anh: size, freeship, sale, order...) chưa tự nhiên.

Mục tiêu: fine-tune một base **Qwen3-family ~8-9B** để model *giỏi cái nghề* — giọng seller live tiếng Việt, đúng persona (`DEFAULT_PERSONA`), code-switch VN+EN tự nhiên, **bám chặt catalog trong prompt và từ chối bịa**, và khéo chốt đơn — rồi cắm vào backend hiện tại với thay đổi code tối thiểu.

**Quyết định của user:** kế hoạch tổng thể · base Qwen3 ~8-9B · **chưa có data (xây từ đầu)** · train trên **1×A6000 (48GB)**.

## Nguyên tắc kiến trúc cốt lõi (định hình toàn bộ plan)

> **Fine-tune HÀNH VI, giữ SỰ THẬT ở trong prompt (in-context), KHÔNG nhồi catalog vào weights.**

Giá (`gia`, `gia_km`), tồn kho (`ton_kho`), khuyến mãi thay đổi hàng ngày. Nếu nhồi catalog vào weights → model đọc dữ liệu cũ + phải train lại mỗi lần đổi kho. Thay vào đó: catalog vẫn được inject vào system prompt y như [app/llm.py:39-47](app/llm.py#L39) (`_system_text()` → `ProductStore.catalog_text()`), model chỉ học *cách trả lời chỉ dựa trên khối dữ liệu được đưa*. Hệ quả: fine-tune trở thành **swap provider gọn**, không phải đại tu, và giữ nguyên đường RAG có sẵn.

## Cơ sở lựa chọn (đã verify)

- **Base model: Qwen3-8B (bản Instruct/chat)** — tiếng Việt mạnh, ChatML native, QLoRA thoải mái trên 48GB. Qwen3 hỗ trợ 119 ngôn ngữ, Vietnamese tốt. (Nếu user chốt bản 9B mới hơn — vd Qwen3.5-9B — chỉ đổi checkpoint, pipeline không đổi.)
- **Framework: Unsloth** — nhanh nhất cho single-GPU (2-5x so với HF), VRAM thấp, native Qwen3, export thẳng merged 16-bit + **GGUF + Ollama Modelfile** một lệnh → khớp stack Ollama hiện tại. (LLaMA-Factory là phương án thay thế nếu muốn config YAML/UI.)
- **Method: QLoRA (4-bit NF4)** — dù 48GB đủ cho LoRA bf16, QLoRA để dư headroom cho system prompt dài (persona + full catalog) và catalog phình về sau. Nâng lên LoRA bf16 nếu QLoRA underfit giọng.

## Giai đoạn thực hiện

### Stage 0 — Nền tảng & "đóng băng" prompt contract (~0.5 ngày)
- **Refactor `_system_text()` / `_user_text()`** ([app/llm.py:39-50](app/llm.py#L39)) thành builder dùng chung để cả training và serving import **cùng một** chuỗi prompt (persona + khối `DỮ LIỆU THAM CHIẾU` + hậu tố "Chỉ trả về câu trả lời cuối cùng..."). Nếu prompt lúc train khác lúc serve → mất phần lớn lợi ích fine-tune.
- Xác nhận môi trường A6000 tách khỏi máy serving 8GB (RTX 5060).

### Stage 1 — Chiến lược dữ liệu (từ con số 0)

**1a. Sinh synthetic bằng teacher model (nguồn cold-start chính):**
- Dùng **Claude** (tái sử dụng đường `AsyncAnthropic` + `ANTHROPIC_API_KEY` đã có ở [app/llm.py:94](app/llm.py#L94)) sinh cặp `comment → reply` **grounded theo schema thật** của [data/products.json](data/products.json) (`id, ten, gia, gia_km, mau, size, ton_kho, mo_ta, faq`, + `policies`).
- Mỗi mẫu: sample 1-N sản phẩm (hoặc sinh catalog giả cùng schema để đa dạng vượt 4 SKU), dựng khối catalog bằng builder đã freeze → teacher **thấy đúng khối reference mà student sẽ thấy**.
- **Taxonomy intent (đặt quota/loại):** giá/KM · tồn kho/size/màu · mô tả/FAQ · ship/đổi trả/thanh toán · tư vấn size · so sánh 2 SP · chốt đơn · xin thông tin · off-topic kéo về · và **quan trọng nhất: câu hỏi ngoài catalog / bait-bịa** (hỏi SP không có, hỏi giá SP không list, hỏi hàng đã hết `ton_kho=0`) → gold reply phải **từ chối/nói thật + gợi ý cái đang có**. Đây là tín hiệu chống bịa (hard negatives).
- **Code-switch (chêm tiếng Anh):** ~15-25% mẫu chèn EN tự nhiên (size, freeship, sale, order, oversize, unisex...), giữ tiếng Việt là ngôn ngữ nền.
- **Persona:** nhồi `DEFAULT_PERSONA` verbatim vào system của teacher; reply ngắn 1-3 câu, không markdown, "cả nhà/mình", có chốt đơn khi hợp lý, không đọc tên emoji.
- **Đa lượt:** trộn single-turn (khớp pipeline hiện tại) + 2-4 lượt (khách hỏi tiếp rồi chốt) để sẵn sàng khi thêm history.
- **Tỷ lệ round 1 (~4k-8k mẫu):** ~60% QA grounded thẳng · 20% code-switch · 15% hard-negative/từ chối · 5% multi-turn chốt đơn. Prompt-cache khối catalog phía teacher để giảm chi phí.

**1b. Logging dữ liệu thật (cắm ngay, thu hoạch sau):**
- Thêm ghi log **non-blocking** trong `AnswerPipeline._handle()` ([app/pipeline.py:98](app/pipeline.py#L98)) — điểm có sẵn `comment.text/author/ts`, `reply`, provider/model. Ghi JSONL vào `logs/interactions/`, fire-and-forget + try/except (không thêm latency, không làm chết worker).
- Thêm **feedback hook** nhẹ: admin flag reply good/bad hoặc **sửa tay** (reply đã sửa = gold label giá trị cao nhất) qua protocol WebSocket. Đây là flywheel cho round 2+.
- Chưa cần DB — JSONL khớp phong cách file-based của project (giống `requirements-vixtts.txt`). SQLite là nâng cấp sau nếu làm UI review.

**1c. Format & lọc chất lượng:**
- **Format:** ChatML conversational, JSONL 1 hội thoại/dòng (`{"messages":[system,user,assistant]}`), map 1:1 với chat template Qwen3 và messages ở [app/llm.py:86](app/llm.py#L86). **Chỉ train trên turn assistant** (mask loss system+user) — để model học *trả lời*, không học chép lại catalog.
- **Split:** 90/10, chia theo scenario/product-set (không random), giữ eval set stratified theo intent (~300-500), + **golden set ~50 mẫu** never-train cho LLM-judge.
- **Lọc trước khi train:** (a) grounding check tự động — trích giá/size/màu trong reply, assert xuất hiện trong khối catalog nguồn (loại mẫu teacher bịa); (b) length/format (1-3 câu, không markdown, cắt "Here is..."); (c) dedup near-duplicate; (d) language-ID loại reply lỡ full-English/tiếng Trung (Qwen hay drift Trung); (e) tùy chọn LLM chấm 1-5, bỏ <4.

### Stage 2 — Training
- QLoRA 4-bit NF4, Unsloth. **rank r=16-32, alpha=32-64, dropout 0.05**, target toàn bộ q,k,v,o,gate,up,down.
- **Hyperparam khởi điểm:** seq len 4096 (đủ persona+catalog+reply) · effective batch 16-32 (grad-accum) · LR 2e-4 cosine warmup ~3% · 2-3 epoch (theo dõi eval loss để tránh overfit) · bf16 · gradient checkpointing · NEFTune ~5.
- **Code-switch:** không cần chỉnh tokenizer (Qwen3 xử lý VN+EN sẵn) — đòn bẩy là *data slice* + *eval metric*.
- **Chống bịa:** chính = hard-negative slice + grounding filter (Stage 1). Tùy chọn **round 2 DPO/ORPO** nhỏ (chosen = reply grounded, rejected = reply bịa giá/kho) — rẻ trên 48GB, làm sắc hành vi từ chối. Làm SFT trước, thêm DPO nếu điểm faithfulness chững lại.

### Stage 3 — Đánh giá
Chạy trên eval stratified + golden 50, kết hợp tự động + LLM-as-judge (dùng Claude qua client sẵn có):
- **Faithfulness/grounding (quan trọng nhất):** tự động trích mọi claim (giá/kho/size/màu/policy) rồi verify với khối catalog → tỷ lệ hallucination.
- **Refusal đúng:** trên slice hard-negative, đo tần suất model từ chối/kéo về đúng thay vì bịa.
- **Persona/tone:** judge rubric 1-5 (xưng shop/mình/cả nhà, 1-3 câu, không markdown, không đọc tên emoji) — phần đo được bằng regex.
- **Fluency VN & code-switch tự nhiên** + language-ID chống drift Trung/Anh.
- **Chốt đơn:** judge xem có CTA/chốt mềm khi hợp cảnh (và KHÔNG hard-sell khi off-topic/refusal).
- **Baseline regression:** chạy cùng harness trên `qwen2.5:7b` gốc + Claude → chứng minh fine-tune thắng. Dùng win-rate pairwise (nhạy hơn điểm tuyệt đối). Chỉ **promote adapter khi thắng scorecard baseline**. Đo thêm tokens/s + p95 latency (pipeline live nhạy latency).

### Stage 4 — Deploy / tích hợp (thay đổi code tối thiểu)
- **Phương án khuyến nghị: merge LoRA → GGUF → Ollama.** Unsloth export GGUF Q4 + Modelfile → `ollama create seller-qwen3:8b` → chỉ set `LLM_MODEL=seller-qwen3:8b` trong `.env`, **không đổi code** ([app/llm.py:74](app/llm.py#L74) `_answer_openai` đã nói chuyện `/v1` của Ollama). Lưu ý GGUF Q4 8B ~5-6GB — validate trên máy 8GB hoặc serve từ A6000 vì system prompt catalog dài ngốn context.
- **Phương án throughput/chất lượng cao: vLLM** load base bf16 + adapter, expose OpenAI-compatible `/v1`, prefix-caching amortize khối catalog lặp lại (thắng latency). Chỉ đổi `OPENAI_BASE_URL` + `LLM_MODEL`.
- **Provider abstraction:** thêm nhãn provider `seller`/`finetuned` vào set validate [app/llm.py:32](app/llm.py#L32) + `_DEFAULT_MODEL` [app/config.py:46](app/config.py#L46) (chỉ để rõ ràng/observability; code request không đổi). Giữ `offline` làm fallback, giữ `anthropic` cho A/B + sinh data teacher.
- **Catalog:** giữ in-context; khi catalog phình to → kích hoạt `ProductStore.search()` (đã để sẵn cho RAG) inject top-K — hành vi grounding đã học chuyển thẳng sang.

### Stage 5 — Flywheel (lặp)
Ship adapter round-1 (giữ qwen2.5/anthropic làm comparator) → thu log + mẫu admin sửa (1b) → lọc + trộn real+synthetic → retrain (SFT, rồi tùy chọn DPO) → chạy lại scorecard → chỉ promote khi thắng → lặp. Version dataset + adapter theo tag (date/round) để rollback.

## File mới / file đụng vào

**Thư mục mới (tách training khỏi app `/`, theo convention `*_vixtts.py` sẵn có):**
```
backend/
  training/llm/
    prompt_contract.py     # import/mirror builder _system_text/_user_text
    gen_synthetic.py       # Claude → cặp comment/reply grounded (JSONL)
    build_dataset.py       # lọc, ground-check, dedup, split, export ChatML
    train_qlora.py         # Unsloth QLoRA/LoRA entrypoint
    merge_export.py        # merge adapter → GGUF/Modelfile (Ollama) / vLLM
    evaluate.py            # scorecard tự động + LLM-judge
    configs/               # config hyperparam theo run
  training/data/{raw,curated,adapters}/
  requirements-llm-train.txt   # unsloth/peft/trl/transformers/datasets (A6000)
  logs/interactions/           # JSONL runtime từ pipeline
```

**File hiện tại (đều sửa nhẹ):**
- [app/llm.py](app/llm.py) — tách `_system_text()`/`_user_text()` thành builder import được; thêm nhãn provider `finetuned` vào set validate. Không đổi logic request.
- [app/pipeline.py](app/pipeline.py) — thêm logging interaction non-blocking trong `_handle()` (+ feedback flag trong protocol WS). Đây là thay đổi runtime đáng kể duy nhất.
- [app/config.py](app/config.py) — thêm model fine-tuned vào `_DEFAULT_MODEL`; thêm setting `INTERACTION_LOG_DIR` + feedback.
- `.env.example` — ghi chú model mới + cách trỏ Ollama-GGUF vs vLLM.
- `requirements.txt` — **không đổi** (runtime đã có `openai`, `anthropic`); dep train nằm riêng ở `requirements-llm-train.txt`.

## Verification (kiểm chứng end-to-end)
1. **Prompt contract:** unit test so sánh string builder dùng chung khớp byte-by-byte với output cũ của `_system_text()`/`_user_text()`.
2. **Data:** chạy `build_dataset.py` → kiểm grounding-check pass rate, phân bố intent, tỷ lệ code-switch, dedup; xem 20-30 mẫu bằng mắt.
3. **Train:** train QLoRA 1 run nhỏ → eval loss giảm, không NaN, VRAM < 48GB.
4. **Eval:** chạy `evaluate.py` trên golden set, so scorecard adapter vs `qwen2.5:7b` gốc → faithfulness + persona + refusal phải thắng baseline.
5. **Serve:** `ollama create seller-qwen3:8b` → set `LLM_MODEL` → chạy backend, bắn vài comment mẫu (kể cả câu bait-bịa & câu code-switch) qua `/ws/comments`, xác nhận reply grounded + đúng giọng + latency p95 chấp nhận được cho pipeline live.
