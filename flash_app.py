#!/usr/bin/env python3
"""RunPod Flash deploy script for dealignai/Gemma-4-31B-JANG_4M-CRACK.

Usage:
    export RUNPOD_API_KEY=your_key
    source .venv/bin/activate
    python flash_app.py

The first run provisions a GPU worker, installs dependencies, downloads the
model weights, and loads them into vLLM. Subsequent runs within `idle_timeout`
reuse the warm worker and cached model instance.
"""

import asyncio
import os
from typing import Any

from runpod_flash import Endpoint, GpuGroup

# Worker-local singletons. Flash keeps the worker alive for `idle_timeout`
# seconds between requests, so the loaded vLLM engine can be reused.
_llm: Any = None
_tokenizer: Any = None


def _load_model():
    """Load the Gemma 4 31B model with vLLM once per worker."""
    global _llm, _tokenizer
    if _llm is not None:
        return _llm, _tokenizer

    # These imports happen inside the function so they execute on the remote GPU worker.
    from transformers import AutoTokenizer
    from vllm import LLM

    model_id = os.environ.get("MODEL_NAME", "dealignai/Gemma-4-31B-JANG_4M-CRACK")

    print(f"Loading model: {model_id}", flush=True)

    _llm = LLM(
        model=model_id,
        tensor_parallel_size=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")),
        gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95")),
        max_model_len=int(os.environ.get("MAX_MODEL_LEN", "8192")),
        trust_remote_code=True,
        dtype="auto",
    )
    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Model loaded.", flush=True)
    return _llm, _tokenizer


@Endpoint(
    name="gemma4-31b-jang-crack",
    # Use any available GPU for faster provisioning. For a 31B-class model you
    # generally want an A100, A6000, or similarly large VRAM card. You can also
    # pass a list of specific GpuType constants, e.g.
    # gpu=[GpuType.NVIDIA_A100, GpuType.NVIDIA_RTX_A6000],
    gpu=GpuGroup.ANY,
    workers=1,
    idle_timeout=600,  # keep worker warm for 10 minutes
    dependencies=[
        "vllm",
        "transformers",
        "torch",
        "accelerate",
    ],
)
def chat_completion(
    messages,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    stop: list[str] | None = None,
):
    """Run a chat-completion request against the deployed model.

    Args:
        messages: List of {"role": "system|user|assistant", "content": "..."} dicts.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum new tokens to generate.
        stop: Optional list of stop strings.

    Returns:
        OpenAI-compatible chat completion response dict.
    """
    from vllm import SamplingParams

    llm, tokenizer = _load_model()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop or [],
    )

    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text

    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "stop",
            }
        ],
        "model": "dealignai/Gemma-4-31B-JANG_4M-CRACK",
    }


async def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about machine learning."},
    ]

    print("Sending chat completion request to Flash endpoint...")
    result = await chat_completion(messages, max_tokens=256)

    print("\n--- Response ---")
    print(result["choices"][0]["message"]["content"])
    print("\n--- Full payload ---")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
