import glob
import os
import re
import subprocess
import time

import requests
import runpod

# Model resolution priority:
# 1. MODEL_PATH, if set and points to a model file or directory.
# 2. A RunPod cached Hugging Face model (MODEL_NAME).
# 3. A single model discovered under /mnt/models.
MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "cyankiwi/gemma-4-31B-it-AWQ-4bit",
)
MODEL_FILE = os.environ.get("MODEL_FILE", "")
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "true").lower() in ("1", "true", "yes")
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "128"))

_vllm_process = None
_model_ready = False

print("handler.py: vLLM branch", flush=True)
print(
    f"MODEL_PATH={MODEL_PATH!r} MODEL_NAME={MODEL_NAME!r} "
    f"MODEL_FILE={MODEL_FILE!r} VLLM_PORT={VLLM_PORT} "
    f"GPU_MEMORY_UTILIZATION={GPU_MEMORY_UTILIZATION} MAX_MODEL_LEN={MAX_MODEL_LEN} "
    f"MAX_NUM_SEQS={MAX_NUM_SEQS}",
    flush=True,
)


def strip_thinking(content: str) -> str:
    """Remove reasoning/thinking blocks from model output.

    Some merged/uncensored models emit a chain-of-thought preamble before the
    actual response. This heuristic strips common markers and returns the
    final answer text.
    """
    if not isinstance(content, str):
        return content

    # 1. Remove explicit <think>...</think> tags (QwQ-style).
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)

    # 2. If a "Final Answer" marker exists, keep only what follows it.
    for marker in ("Final Answer:", "最终回答：", "最终答案："):
        if marker in content:
            content = content.split(marker, 1)[1]
            return content.strip()

    # 3. Heuristic: detect "thinking process" preamble and return the last
    # substantial paragraph that does not look like a reasoning bullet.
    if re.search(r"thinking process|我的思考过程|思考过程", content, re.IGNORECASE):
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for paragraph in reversed(paragraphs):
            # Skip bullets, numbered steps, and meta-reasoning lines.
            if not re.match(
                r"^(\d+\.|[-•*]|thinking|analyze|draft|wait[,\s]|so\s|let\s)",
                paragraph,
                re.IGNORECASE,
            ):
                return paragraph
        if paragraphs:
            return paragraphs[-1]

    return content.strip()


def _resolve_snapshot_path(model_id: str) -> str:
    """Resolve the local snapshot path for a RunPod cached Hugging Face model."""
    if "/" not in model_id:
        raise ValueError(f"MODEL_ID '{model_id}' must be in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    if os.path.isdir(snapshots_dir):
        versions = [
            d
            for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if versions:
            versions.sort()
            return os.path.join(snapshots_dir, versions[0])

    raise RuntimeError(f"Cached model not found: {model_id}")


def _pick_model_from_dir(directory: str) -> str | None:
    """Return a usable model path from *directory* or its subdirs.

    For vLLM we normally need a directory containing config.json + safetensors.
    Falls back to a single GGUF file if no Transformers-format model is found.
    """
    config_paths = glob.glob(os.path.join(directory, "**/config.json"), recursive=True)
    config_paths = [p for p in config_paths if os.path.isfile(p)]
    if config_paths:
        config_paths.sort(key=lambda p: len(p.split(os.sep)))
        return os.path.dirname(config_paths[0])

    ggufs = glob.glob(os.path.join(directory, "**", "*.gguf"), recursive=True)
    ggufs = [f for f in ggufs if os.path.isfile(f)]
    if not ggufs:
        return None

    if MODEL_FILE:
        for f in ggufs:
            if os.path.basename(f) == MODEL_FILE:
                return f
        print(
            f"MODEL_FILE {MODEL_FILE} not found under {directory}; "
            f"using other GGUF(s): {ggufs}",
            flush=True,
        )

    if len(ggufs) == 1:
        return ggufs[0]

    print(
        f"Warning: found multiple GGUFs under {directory}: {ggufs}. "
        "Set MODEL_FILE to choose one explicitly.",
        flush=True,
    )
    return ggufs[0]


def _find_model() -> str | None:
    """Return the model path to use.

    Priority:
    1. MODEL_PATH if it exists.
    2. The RunPod cached model for MODEL_NAME.
    3. The only model under /mnt/models (recursively).
    4. None.
    """
    if MODEL_PATH:
        if os.path.isfile(MODEL_PATH):
            return MODEL_PATH
        if os.path.isdir(MODEL_PATH):
            print(f"MODEL_PATH is a directory; looking for model inside: {MODEL_PATH}", flush=True)
            model_path = _pick_model_from_dir(MODEL_PATH)
            if model_path:
                return model_path

    try:
        snapshot_dir = _resolve_snapshot_path(MODEL_NAME)
        print(f"Resolved cached model snapshot: {snapshot_dir}", flush=True)
        model_path = _pick_model_from_dir(snapshot_dir)
        if model_path:
            return model_path
    except Exception as exc:
        print(f"Could not resolve cached model {MODEL_NAME}: {exc}", flush=True)

    config_paths = glob.glob("/mnt/models/**/config.json", recursive=True)
    config_paths = [p for p in config_paths if os.path.isfile(p)]
    if len(config_paths) == 1:
        print(f"Using discovered model dir: {os.path.dirname(config_paths[0])}", flush=True)
        return os.path.dirname(config_paths[0])

    ggufs = glob.glob("/mnt/models/**/*.gguf", recursive=True)
    ggufs = [f for f in ggufs if os.path.isfile(f)]
    if len(ggufs) == 1:
        print(f"Using discovered GGUF: {ggufs[0]}", flush=True)
        return ggufs[0]
    if len(ggufs) > 1:
        print(
            f"Warning: found multiple GGUFs under /mnt/models: {ggufs}. "
            f"Set MODEL_PATH explicitly to choose one.",
            flush=True,
        )
    return None


def _wait_for_vllm(timeout: int = 300) -> bool:
    for _ in range(timeout):
        try:
            response = requests.get(f"{VLLM_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _start_vllm():
    """Start the local vLLM OpenAI-compatible server."""
    global _vllm_process, _model_ready

    model_path = _find_model()
    if not model_path:
        raise RuntimeError(
            "No model is available. Options:\n"
            "  1. Set MODEL_PATH to a local model dir or file.\n"
            "  2. Configure a RunPod cached model (MODEL_NAME).\n"
            "  3. Mount a Network Volume at /mnt/models with a model."
        )

    print(f"Starting vLLM server for model: {model_path}", flush=True)
    print(
        f"vLLM options: port={VLLM_PORT}, tp={TENSOR_PARALLEL_SIZE}, "
        f"trust_remote_code={TRUST_REMOTE_CODE}, "
        f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}, "
        f"max_model_len={MAX_MODEL_LEN}, max_num_seqs={MAX_NUM_SEQS}",
        flush=True,
    )

    cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        MODEL_NAME,
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--port",
        str(VLLM_PORT),
        "--host",
        "127.0.0.1",
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--quantization",
        "compressed-tensors",
        "--limit-mm-per-prompt",
        "image=0,audio=0",
    ]
    if TRUST_REMOTE_CODE:
        cmd.append("--trust-remote-code")

    log_file = open("/tmp/vllm.log", "w", buffering=1)
    _vllm_process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if not _wait_for_vllm(300):
        raise RuntimeError(
            "vLLM server did not start within 300 seconds. "
            "Check /tmp/vllm.log for details."
        )

    _model_ready = True
    print("vLLM server is ready.", flush=True)


def handler(event):
    """RunPod Serverless handler that proxies requests to vLLM."""
    global _vllm_process, _model_ready
    if _vllm_process is None:
        _start_vllm()

    if not _model_ready:
        return {
            "error": (
                "Model is not loaded. Set MODEL_PATH, configure a RunPod cached model, "
                "or mount a Network Volume at /mnt/models."
            )
        }

    input_data = event.get("input", {})

    messages = input_data.get("messages", [])
    if not messages and input_data.get("prompt"):
        messages = [{"role": "user", "content": input_data["prompt"]}]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "temperature": input_data.get("temperature", 0.7),
        "top_p": input_data.get("top_p", 1.0),
        "max_tokens": input_data.get("max_tokens", 512),
    }

    response = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"vLLM request failed: {exc}", flush=True)
        print(f"vLLM request payload: {payload}", flush=True)
        print(f"vLLM response body: {response.text}", flush=True)
        raise

    data = response.json()

    message = data.get("choices", [{}])[0].get("message", {})
    raw_content = message.get("content", "")
    cleaned_content = strip_thinking(raw_content)
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": message.get("role", "assistant"),
                    "content": cleaned_content,
                },
                "finish_reason": "stop",
            }
        ],
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
