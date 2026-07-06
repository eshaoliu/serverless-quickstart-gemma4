# Use vLLM as the inference engine.
FROM vllm/vllm-openai:latest

USER root

# Force Docker to rebuild this layer whenever the GitHub repo has a new commit.
ADD https://api.github.com/repos/eshaoliu/serverless-quickstart/commits?sha=main&per_page=1 /tmp/latest-commit.json

WORKDIR /app

# Install Python dependencies.
# --ignore-installed avoids conflicts with Debian-managed packages (e.g. cryptography).
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed -r requirements.txt

# Copy the RunPod handler
COPY handler.py .

# Build-time verification (non-fatal): confirm handler contains vLLM markers.
RUN grep -E "vllm|VLLM_PORT" /app/handler.py && \
    echo "handler.py is vLLM version" || \
    echo "WARNING: handler.py vLLM marker not found"

# Use the RunPod cached model instead of baking weights into the image.
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=HauhauCS/Gemma4-31B-QAT-Uncensored-HauhauCS-Balanced-MTP
ENV MODEL_FILE=""
ENV VLLM_PORT=8000
ENV TENSOR_PARALLEL_SIZE=1
ENV TRUST_REMOTE_CODE=true
ENV GPU_MEMORY_UTILIZATION=0.95
ENV MAX_MODEL_LEN=32768
ENV MAX_NUM_SEQS=128
# NOTE: set HF_TOKEN via RunPod endpoint env vars if the model is gated.

# Clear any inherited ENTRYPOINT so CMD is interpreted as a plain command.
ENTRYPOINT []

CMD ["python3", "-u", "handler.py"]
