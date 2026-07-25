#!/usr/bin/env python3
"""
modal_glm_server.py -- the v0.1 SOFTWARE full-model backend, on Modal GPU cloud.

Serves the REAL GLM family (GLM-4.5-Air-FP8, the same model whose U(K) / MTP
accept-rate we measured) via vLLM's OpenAI-compatible server on 2x H100, exposed
as a Modal web endpoint. The WPU host server (host/wpu_modal_backend.py) proxies
chat completions to this endpoint, so a standard OpenAI client -> WPU host ->
this Modal vLLM -> REAL GLM text.

Honest scope: this is SOFTWARE on cloud GPUs, NOT the WPU accelerator. It proves
the product EXPERIENCE with the real model family now; the accelerator replaces
this backend behind the same API once silicon exists (docs/PRODUCT_SPEC.md, Stage 3).

Cost: 2x H100 while a container is warm; scales to ZERO when idle (scaledown), so
it only bills during use + a short idle window. FP8 weights (~110 GB) cached in a
Modal volume, so only the first cold start pays the download.

Deploy (needs your Modal auth, as the measurement jobs used):
    modal deploy tools/modal_glm_server.py
    # -> prints a URL like https://<you>--wpu-glm-server-serve.modal.run
Then point the WPU host at it:
    python3 host/wpu_server.py --backend modal \
        --modal-url https://<you>--wpu-glm-server-serve.modal.run/v1
(Or `modal serve tools/modal_glm_server.py` for an ephemeral dev URL.)
"""
import modal

MODEL = "zai-org/GLM-4.5-Air-FP8"       # real GLM family; swap for a GLM-5.2 repo when available
N_GPU = 2                               # 2x H100 holds the ~110 GB FP8 weights
API_KEY = "wpu-local"                  # simple shared key; the host sends it

app = modal.App("wpu-glm-server")
hf_cache = modal.Volume.from_name("wpu-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(          # vLLM JIT-compiles kernels -> needs nvcc
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install("vllm>=0.10.1", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf",
          "VLLM_LOGGING_LEVEL": "INFO", "CUDA_HOME": "/usr/local/cuda"})
)

VLLM_PORT = 8000


@app.function(
    image=image,
    gpu=f"H100:{N_GPU}",
    volumes={"/hf": hf_cache},
    timeout=30 * 60,
    scaledown_window=5 * 60,            # keep warm 5 min after last request, then -> 0
    min_containers=0,                   # scale to zero when idle (no idle billing)
)
@modal.concurrent(max_inputs=32)        # one server handles many concurrent requests
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * 60)
def serve():
    """Launch vLLM's own OpenAI-compatible API server. The WPU host proxies to it,
       so `/v1/chat/completions` here IS the real GLM completion endpoint."""
    import subprocess
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--tensor-parallel-size", str(N_GPU),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.92",
        "--trust-remote-code",
        "--served-model-name", "wpu-glm",
        "--api-key", API_KEY,
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        # GLM-4.5's own MTP module as the self-speculative draft (same structure as
        # our spec_chain RTL) -- the real product uses this too:
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}',
    ]
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def info():
    print("Deploy:  modal deploy tools/modal_glm_server.py")
    print("Dev URL: modal serve  tools/modal_glm_server.py")
    print(f"Model:   {MODEL} on {N_GPU}x H100, scales to zero when idle.")
    print(f"API key: {API_KEY} (the WPU host sends this as the bearer token).")
