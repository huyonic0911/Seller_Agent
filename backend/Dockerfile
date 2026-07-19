# Backend Seller Agent — GPU (viXTTS trên CUDA 12.8) + RAG (bge-m3 CPU).
# Ảnh dựa trên python 3.12; CUDA runtime do wheel torch cu128 mang theo, GPU host
# được nvidia-container-toolkit tiêm vào lúc chạy (compose: deploy.devices nvidia).
#
# Combo phụ thuộc đã kiểm chứng trên RTX 5060 (Blackwell/sm_120) — PIN CHÍNH XÁC,
# đừng nới lỏng (5.x transformers gỡ isin_mps_friendly; torch phải cu128 cho sm_120).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf-cache \
    COQUI_TOS_AGREED=1

# Thư viện hệ thống: FFmpeg (torchcodec/coqui audio IO), libsndfile (soundfile).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv — trình cài phụ thuộc nhanh.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1) PyTorch CUDA 12.8 (Blackwell/sm_120).
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.11.0 torchaudio==2.11.0

# 2) Phụ thuộc ứng dụng (base + viXTTS + RAG). Copy requirements trước để cache layer.
COPY requirements.txt requirements-vixtts.txt requirements-rag.txt ./
RUN uv pip install --system \
      -r requirements.txt \
      -r requirements-vixtts.txt \
      -r requirements-rag.txt \
      torchcodec==0.14.0 \
      transformers==4.57.6 \
      coqui-tts==0.27.5 \
      sentence-transformers==5.6.0 \
      qdrant-client==1.18.0 \
      numpy==2.4.6

# 2b) torchcodec 0.14 là build CUDA 13 (cần libcudart.so.13 + libnvrtc.so.13) nhưng
#     torch cu128 chỉ mang .so.12. Lấy đúng 2 lib CUDA-13 từ redist NVIDIA để torchcodec
#     nạp được (ta chỉ dùng CPU decode → không cần cả toolkit CUDA 13).
RUN apt-get update && apt-get install -y --no-install-recommends xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/cuda13/lib && cd /tmp \
    && B=https://developer.download.nvidia.com/compute/cuda/redist \
    && curl -sL $B/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.0.48-archive.tar.xz -o cudart.tar.xz \
    && curl -sL $B/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-13.0.48-archive.tar.xz -o nvrtc.tar.xz \
    && tar xf cudart.tar.xz && tar xf nvrtc.tar.xz \
    && cp -a cuda_cudart-*/lib/libcudart.so.13* /opt/cuda13/lib/ \
    && cp -a cuda_nvrtc-*/lib/libnvrtc.so.13* /opt/cuda13/lib/ \
    && rm -rf /tmp/*.tar.xz /tmp/cuda_cudart-* /tmp/cuda_nvrtc-*
ENV LD_LIBRARY_PATH=/opt/cuda13/lib:/usr/local/lib/python3.12/site-packages/torch/lib

# 3) Mã nguồn (model/voices/data lớn được mount lúc chạy, xem docker-compose).
COPY . .

EXPOSE 8000
CMD ["uvicorn", "core.server:app", "--host", "0.0.0.0", "--port", "8000"]
