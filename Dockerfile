# RunPod Serverless Dockerfile for Gemma 4 31B inference via vLLM
# Build context must include handler.py and requirements.txt

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

WORKDIR /app

# Install system Python and git (needed by Hugging Face transformers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip/wheel to avoid resolver issues
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install the CUDA 12.8 variant of PyTorch first so vLLM links against
# libcudart.so.12 instead of libcudart.so.13.
RUN pip install --no-cache-dir \
    torch==2.9.0 \
    torchvision==0.24.0 \
    torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy handler
COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=dealignai/Gemma-4-31B-JANG_4M-CRACK
ENV GPU_MEMORY_UTILIZATION=0.90
ENV MAX_MODEL_LEN=8192
ENV MAX_NUM_SEQS=128

CMD ["python3", "-u", "/app/handler.py"]
