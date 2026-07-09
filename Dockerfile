# RunPod Serverless Dockerfile for Gemma 4 inference via vLLM
# Uses the official vLLM Gemma4 image which contains the necessary
# compressed-tensors / AWQ fixes for Gemma 4 quantized checkpoints.
# Build context must include handler.py.

FROM vllm/vllm-openai:gemma4-cu130

WORKDIR /app

# Install the RunPod serverless runtime and HTTP client.
RUN pip install --no-cache-dir runpod requests

# Copy handler
COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=cyankiwi/gemma-4-31B-it-AWQ-4bit
ENV GPU_MEMORY_UTILIZATION=0.90
ENV MAX_MODEL_LEN=8192
ENV MAX_NUM_SEQS=128

CMD ["python3", "-u", "/app/handler.py"]
