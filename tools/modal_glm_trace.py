#!/usr/bin/env python3
"""
modal_glm_trace.py -- GLM-4.5-Air routed-expert trace on Modal (H100).

Job A of the GLM measurement plan (docs/R3_APPLIANCE_SPEC.md §9-10): upgrade
the h/U inputs from the OLMoE proxy to a GLM-FAMILY measurement. Loads
zai-org/GLM-4.5-Air (106B-A12B, same fine-grained-MoE + sigmoid-gating family
as GLM-5.2) in 4-bit (bnb NF4) on one H100-80GB, hooks the MoE gates to record
the EXACT selected expert ids per (token, layer), decodes the same 3 workloads
x 4 prompts x 192 tokens as tools/moe_trace_hf.py, and writes trace_*.npz to a
Modal Volume (download with `modal volume get wpu-traces ...`).

Capture strategy (robust across transformers versions):
  1. try output_router_logits=True (if the Glm4Moe impl supports it);
  2. else register forward hooks on every module whose forward RETURNS the
     (topk_idx, topk_weight) pair (DeepSeek/GLM MoE gate convention) -- the
     exact indices the model used, bias/grouping included.

Run:  modal run tools/modal_glm_trace.py::trace_routing
Cost: ~30-50 min on 1x H100 [~$3-6], model weights cached in the volume.
"""
import json
import modal

MODEL = "zai-org/GLM-4.5-Air"
MAX_NEW = 192

app = modal.App("wpu-glm-trace")
vol = modal.Volume.from_name("wpu-traces", create_if_missing=True)
hf_cache = modal.Volume.from_name("wpu-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "transformers>=4.54,<5", "accelerate",
        "bitsandbytes>=0.43", "safetensors", "numpy<2", "sentencepiece", "tiktoken",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
)

WORKLOADS = {
    "chat": [
        "Tell me about the history of the transistor and why it mattered.",
        "My sourdough starter smells like acetone. What is happening and how do I fix it?",
        "Summarize the plot of Hamlet in a few paragraphs, then discuss its themes.",
        "What should I consider when choosing between renting and buying a home?",
    ],
    "code": [
        "Write a Python function that parses an ELF file header and prints the section table.",
        "Implement an LRU cache in C with O(1) get and put, then explain the data structures.",
        "Given this SQL schema for orders and customers, write a query for the top 10 customers by revenue per quarter.",
        "Write a Verilog module for a parameterizable gray-code FIFO pointer synchronizer.",
    ],
    "math": [
        "Prove that the square root of 2 is irrational, step by step.",
        "A train leaves at 3pm at 60 km/h; another at 4pm at 90 km/h on the same track. When does the second catch the first? Show the algebra.",
        "Compute the integral of x^2 * e^(-x) from 0 to infinity, showing each integration by parts.",
        "Explain and derive the closed form of the Fibonacci sequence.",
    ],
}


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * 3600,
    volumes={"/out": vol, "/hf": hf_cache},
)
def trace_routing():
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="cuda:0",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()
    cfg = model.config
    n_exp = None
    for k in ("n_routed_experts", "num_experts", "num_local_experts", "n_experts"):
        n_exp = n_exp or getattr(cfg, k, None)
    topk = None
    for k in ("num_experts_per_tok", "moe_topk", "num_experts_per_token", "top_k"):
        topk = topk or getattr(cfg, k, None)
    print(f"model={MODEL} experts={n_exp} topk={topk} layers={cfg.num_hidden_layers}")

    # ---- gate hooks: capture the exact (topk_idx, ...) the MoE used ----
    captured = []           # per forward step: list of [topk] int arrays (layer order)
    step_buf = []

    def mk_hook(name):
        def hook(mod, args, out):
            idx = None
            if isinstance(out, tuple) and len(out) >= 1 and torch.is_tensor(out[0]):
                t = out[0]
                if not torch.is_floating_point(t) and t.dim() >= 1 and t.shape[-1] == topk:
                    idx = t
            if idx is not None:
                step_buf.append(idx.reshape(-1, topk)[-1].detach().to("cpu", torch.int16).numpy())
        return hook

    hooks = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__.lower()
        if ("gate" in cls or "router" in cls) and "gateup" not in cls:
            hooks.append(mod.register_forward_hook(mk_hook(name)))
    print(f"hooked {len(hooks)} gate/router modules")

    import os
    os.makedirs("/out/GLM-4.5-Air", exist_ok=True)

    for wl, prompts in WORKLOADS.items():
        all_ids, bounds = [], []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
            past, cur = None, ids
            bounds.append(len(all_ids))
            with torch.no_grad():
                for _ in range(MAX_NEW):
                    step_buf.clear()
                    out = model(cur, past_key_values=past, use_cache=True, return_dict=True)
                    past = out.past_key_values
                    nxt = out.logits[0, -1].argmax()
                    if step_buf:
                        all_ids.append(np.stack(step_buf))   # [n_moe_layers, topk]
                    if nxt.item() == tok.eos_token_id:
                        break
                    cur = nxt.view(1, 1)
            print(f"  [{wl}] prompt done, tokens so far: {len(all_ids)}", flush=True)
        arr = np.stack(all_ids)
        meta = dict(model=MODEL, n_experts=int(n_exp), topk=int(topk),
                    n_layers=int(arr.shape[1]), workload=wl,
                    request_bounds=bounds, prompts=prompts, capture="gate-hooks")
        np.savez_compressed(f"/out/GLM-4.5-Air/trace_{wl}.npz",
                            ids=arr, meta=json.dumps(meta))
        print(f"  [{wl}] saved {arr.shape}")
        vol.commit()
    for h in hooks:
        h.remove()
    return "done"
