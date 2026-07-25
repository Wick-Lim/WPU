#!/usr/bin/env python3
"""
modal_glm_accept.py -- Job B: GLM-4.5-Air MTP ACCEPT RATE r, measured (Modal).

The last unknown in the residency-box band (~76-95 tok/s [EST],
docs/R3_APPLIANCE_SPEC.md sec 9): the adaptive spec-chain's gain curve needs r.
vLLM runs GLM-4.5's own MTP module as the draft (method "mtp" -- the same
self-speculative structure as our spec_chain_top RTL), so its acceptance
counters ARE the r measurement for the GLM family.

Sweep: num_speculative_tokens k = 1..5 (our adaptive range), same 12 prompts
as the routing trace (chat/code/math), ~256 new tokens each. For each k we
read vLLM's spec-decode counters (num_drafts / num_draft_tokens /
num_accepted_tokens / per-position acceptance) and derive:
  - alpha_pos[i] : per-position acceptance (r estimate per draft position)
  - A_eff(k)     : mean accepted+1 tokens per engine step (the roofline A)
Results -> /out/GLM-4.5-Air/accept_sweep.json in the wpu-traces volume.

Run:  modal run tools/modal_glm_accept.py::accept_sweep
Cost: 2x H100 x ~45-70 min [~$8-12]; FP8 weights (~110GB) cached in the volume.
"""
import json
import modal

MODEL = "zai-org/GLM-4.5-Air-FP8"
KS = [1, 2, 3, 4, 5]
MAX_NEW = 256

app = modal.App("wpu-glm-accept")
vol = modal.Volume.from_name("wpu-traces", create_if_missing=True)
hf_cache = modal.Volume.from_name("wpu-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(          # vLLM JIT-compiles kernels -> needs nvcc
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install("vllm>=0.10.1", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf",
          "VLLM_LOGGING_LEVEL": "INFO", "CUDA_HOME": "/usr/local/cuda"})
)

PROMPTS = [
    "Tell me about the history of the transistor and why it mattered.",
    "My sourdough starter smells like acetone. What is happening and how do I fix it?",
    "Summarize the plot of Hamlet in a few paragraphs, then discuss its themes.",
    "What should I consider when choosing between renting and buying a home?",
    "Write a Python function that parses an ELF file header and prints the section table.",
    "Implement an LRU cache in C with O(1) get and put, then explain the data structures.",
    "Given this SQL schema for orders and customers, write a query for the top 10 customers by revenue per quarter.",
    "Write a Verilog module for a parameterizable gray-code FIFO pointer synchronizer.",
    "Prove that the square root of 2 is irrational, step by step.",
    "A train leaves at 3pm at 60 km/h; another at 4pm at 90 km/h on the same track. When does the second catch the first? Show the algebra.",
    "Compute the integral of x^2 * e^(-x) from 0 to infinity, showing each integration by parts.",
    "Explain and derive the closed form of the Fibonacci sequence.",
]


@app.function(
    image=image,
    gpu="H100:2",
    timeout=4 * 3600,
    volumes={"/out": vol, "/hf": hf_cache},
)
def accept_sweep():
    import gc, os, re
    import torch
    from vllm import LLM, SamplingParams

    os.makedirs("/out/GLM-4.5-Air", exist_ok=True)
    results = {}

    for k in KS:
        print(f"===== num_speculative_tokens = {k} =====", flush=True)
        llm = LLM(
            model=MODEL,
            tensor_parallel_size=2,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            speculative_config={"method": "mtp", "num_speculative_tokens": k},
            disable_log_stats=False,
        )
        msgs = [[{"role": "user", "content": p}] for p in PROMPTS]
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
        outs = llm.chat(msgs, sp)
        gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

        # ---- spec-decode counters (vLLM V1 metrics API, fallback: none) ----
        rec = {"k": k, "gen_tokens": gen_tokens}
        try:
            metrics = llm.get_metrics()
            for m in metrics:
                name = getattr(m, "name", "")
                if "spec_decode" in name:
                    val = getattr(m, "value", None)
                    if val is None:
                        val = getattr(m, "values", None)
                    rec[name] = val
            print("metrics:", {kk: vv for kk, vv in rec.items() if "spec" in kk}, flush=True)
        except Exception as e:
            rec["metrics_error"] = repr(e)
            print("get_metrics failed:", e, flush=True)

        # derive A_eff and per-position acceptance where counters exist
        nd = rec.get("vllm:spec_decode_num_drafts")
        na = rec.get("vllm:spec_decode_num_accepted_tokens")
        if isinstance(nd, (int, float)) and nd and isinstance(na, (int, float)):
            rec["accepted_per_draft"] = na / nd
            rec["A_eff"] = 1.0 + na / nd
        pp = rec.get("vllm:spec_decode_num_accepted_tokens_per_pos")
        if isinstance(pp, (list, tuple)) and nd:
            rec["alpha_per_pos"] = [x / nd for x in pp]
        results[str(k)] = rec

        with open("/out/GLM-4.5-Air/accept_sweep.json", "w") as f:
            json.dump(results, f, indent=2)
        vol.commit()

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps(results, indent=2))
    return results
