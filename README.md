# serverless-quickstart-gemma4

RunPod Serverless template for Gemma 4 31B (or any vLLM-compatible model).

## Usage

1. Update `Dockerfile`:
   - `MODEL_NAME`: set to the Hugging Face model ID you want to serve.
   - `MODEL_FILE`: if serving a single GGUF, set the filename here.
   - Adjust `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, etc. as needed.

2. Update `benchmark.py`:
   - Replace `ENDPOINT_URL` with your RunPod endpoint URL.

3. Build and push the Docker image:
   ```bash
   docker build -t your-repo/serverless-quickstart-gemma4 .
   docker push your-repo/serverless-quickstart-gemma4
   ```

4. Create a RunPod Serverless endpoint using the pushed image.

## Files

- `handler.py` — starts a local vLLM OpenAI-compatible API and proxies RunPod requests.
- `Dockerfile` — container definition and default environment.
- `benchmark.py` — concurrent benchmark / load test for the endpoint.
- `requirements.txt` — Python dependencies.
- `test_input.json` — sample RunPod request payload.
