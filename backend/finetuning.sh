#!/usr/bin/env bash
# =============================================================================
# finetuning.sh — chạy toàn bộ pipeline fine-tune LLM cho Seller Agent
#
# Cách dùng (chạy từ thư mục backend/):
#   bash finetuning.sh smoke     # test nhanh: tập nhỏ, 1 epoch (kiểm tra code chạy)
#   bash finetuning.sh full      # chạy thật: data đầy đủ, train theo config chính
#   bash finetuning.sh data      # chỉ sinh + build dataset (không train)
#   bash finetuning.sh deps      # chỉ cài dependencies vào .venv-train
#
# Không cần ANTHROPIC_API_KEY: mặc định sinh data OFFLINE bằng template.
# Nếu CÓ key và muốn data chất lượng cao: đặt USE_CLAUDE=1 (xem biến bên dưới).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"   # về thư mục backend/

# ----------------------- Cấu hình (sửa nếu cần) ------------------------------
PY="${PY:-.venv-train/bin/python}"          # interpreter môi trường training (Python 3.12)
GPU="${GPU:-0}"                             # ghim 1 GPU (máy có 4× A6000: 0,1,2,3)
N="${N:-1200}"                              # số mẫu data (chế độ full)
USE_CLAUDE="${USE_CLAUDE:-0}"               # 1 = sinh bằng Claude teacher (cần ANTHROPIC_API_KEY)
OLLAMA_MODEL="${OLLAMA_MODEL:-seller-qwen3:8b}"
BASELINE="${BASELINE:-qwen2.5:7b}"          # model gốc để so scorecard

RAW=training/data/raw/offline_raw.jsonl
CURATED=training/data/curated
ADAPTER_FULL=training/data/adapters/qwen3-8b-seller-r1
ADAPTER_SMOKE=training/data/adapters/smoke-test
GGUF_OUT=training/data/adapters/qwen3-8b-seller-r1-gguf

MODE="${1:-smoke}"
echo ">>> MODE=$MODE | PY=$PY | GPU=$GPU"

# ----------------------- Hàm tiện ích ----------------------------------------
install_deps() {
  echo ">>> [deps] Cài torch cu121 + requirements-llm-train.txt vào .venv-train"
  command -v uv >/dev/null || { echo "Cần 'uv' (uv venv --python 3.12 .venv-train)"; exit 1; }
  [ -d .venv-train ] || uv venv --python 3.12 .venv-train
  UV_HTTP_TIMEOUT=1200 uv pip install --python "$PY" \
      torch --index-url https://download.pytorch.org/whl/cu121
  UV_HTTP_TIMEOUT=1200 uv pip install --python "$PY" -r requirements-llm-train.txt
  "$PY" -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
}

gen_data() {
  if [ "$USE_CLAUDE" = "1" ]; then
    echo ">>> [data] Sinh bằng Claude teacher (n=$N) — cần ANTHROPIC_API_KEY"
    RAW=training/data/raw/synthetic_raw.jsonl
    "$PY" -m training.llm.gen_synthetic --n "$N" --products data/products.json --out "$RAW"
  else
    echo ">>> [data] Sinh OFFLINE bằng template (n=$N) — không cần API"
    "$PY" -m training.llm.gen_offline --n "$N" --products data/products.json --out "$RAW"
  fi
  echo ">>> [data] Build dataset ChatML → $CURATED"
  "$PY" -m training.llm.build_dataset --raw "$RAW" --out-dir "$CURATED"
}

make_smoke_split() {
  mkdir -p training/data/smoke
  head -80 "$CURATED/train.jsonl" > training/data/smoke/train.jsonl
  head -20 "$CURATED/eval.jsonl"  > training/data/smoke/eval.jsonl
  echo ">>> [smoke] Tập nhỏ: $(wc -l < training/data/smoke/train.jsonl) train / $(wc -l < training/data/smoke/eval.jsonl) eval"
}

# ----------------------- Điều phối theo MODE ---------------------------------
case "$MODE" in
  deps)
    install_deps
    ;;

  data)
    gen_data
    ;;

  smoke)
    gen_data
    make_smoke_split
    echo ">>> [smoke] Train 1 epoch trên tập nhỏ (GPU $GPU)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m training.llm.train_qlora \
        --config training/llm/configs/smoke_test.yaml
    echo ">>> [smoke] OK — adapter tại $ADAPTER_SMOKE"
    echo ">>> Nếu chạy tới đây không lỗi nghĩa là pipeline train HOẠT ĐỘNG."
    ;;

  full)
    gen_data
    echo ">>> [full] Train theo config chính (GPU $GPU)"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m training.llm.train_qlora \
        --config training/llm/configs/qwen3_8b_qlora.yaml
    echo ">>> [full] Export GGUF → $GGUF_OUT"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m training.llm.merge_export \
        --adapter "$ADAPTER_FULL" --out "$GGUF_OUT" --quant q4_k_m
    echo ">>> [full] Tạo model Ollama: $OLLAMA_MODEL"
    ollama create "$OLLAMA_MODEL" -f "$GGUF_OUT/Modelfile" || \
        echo "(!) Bỏ qua bước ollama create — chạy tay nếu cần."
    echo ">>> [full] Đánh giá vs baseline ($BASELINE) — thêm --no-judge nếu không có key"
    JUDGE=""
    [ -z "${ANTHROPIC_API_KEY:-}" ] && JUDGE="--no-judge"
    "$PY" -m training.llm.evaluate --model "$OLLAMA_MODEL" --baseline "$BASELINE" \
        --golden "$CURATED/golden.jsonl" --out training/data/scorecard.md $JUDGE
    echo ">>> [full] Xong. Xem training/data/scorecard.md"
    echo ">>> Để dùng model: đặt trong .env  LLM_PROVIDER=finetuned  LLM_MODEL=$OLLAMA_MODEL"
    ;;

  *)
    echo "Không rõ MODE '$MODE'. Dùng: deps | data | smoke | full"; exit 1
    ;;
esac
