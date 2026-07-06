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

## Flash Deploy (alternative to Docker)

`flash_app.py` deploys `dealignai/Gemma-4-31B-JANG_4M-CRACK` via [RunPod Flash](https://docs.runpod.io/flash/quickstart) without building a Docker image.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

flash login   # or export RUNPOD_API_KEY=...
python flash_app.py
```

The first run provisions a GPU worker, installs `vllm/transformers/torch`, downloads the model, and loads it. Subsequent runs within `idle_timeout` reuse the warm worker and cached vLLM engine.

### Notes for this model

- Gemma 4 31B is a large model; pick a GPU with enough VRAM (A100, A6000, or similar).
- The default `MAX_MODEL_LEN` is `8192`. Raise or lower it via environment variable before running.
- The script uses `GpuGroup.ANY` for fast provisioning. If you want to pin specific GPUs, edit the `gpu=` argument in `flash_app.py`.

## Files

- `handler.py` — starts a local vLLM OpenAI-compatible API and proxies RunPod requests.
- `Dockerfile` — container definition and default environment.
- `benchmark.py` — concurrent benchmark / load test for the endpoint.
- `flash_app.py` — RunPod Flash script for `dealignai/Gemma-4-31B-JANG_4M-CRACK`.
- `requirements.txt` — Python dependencies.
- `test_input.json` — sample RunPod request payload.
