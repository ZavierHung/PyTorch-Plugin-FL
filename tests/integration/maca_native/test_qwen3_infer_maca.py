"""
Qwen3 inference on native MetaX PyTorch (torch 2.8.0+metax / cuda:0).

Does NOT use torch_fl or flagos. Device is cuda:0 (MACA cu-bridge exposes MetaX C500).

Usage:
    pytest tests/integration/maca_native/test_qwen3_infer_maca.py -v -s \\
        --model /home/hongzw/Qwen3-0.6B --max-new-tokens 16 --warmup-rounds 1
"""

import time

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="module")
def ctx(request):
    model_path = request.config.getoption("--model")
    max_new_tokens = request.config.getoption("--max-new-tokens")
    warmup_rounds = request.config.getoption("--warmup-rounds")

    assert torch.cuda.is_available(), "MACA cuda runtime not available"
    device = "cuda:0"
    print(
        f"\ntorch={torch.__version__}  device={device}  "
        f"gpu={torch.cuda.get_device_name(0)}  "
        f"cuda_count={torch.cuda.device_count()}"
    )

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cpu"
    )
    model = model.to(device)
    model.eval()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers[0].self_attn.config._attn_implementation = "eager"
    print(
        f"Model device: {next(model.parameters()).device}  "
        f"load time: {time.time() - t0:.2f}s"
    )

    text = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Give me a short introduction to large language model.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    print(f"Input tokens: {input_len}")

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    for i in range(warmup_rounds):
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**gen_kwargs)
        torch.cuda.synchronize()
        print(f"Warmup {i + 1}: {time.perf_counter() - w0:.3f}s")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "inputs": inputs,
        "input_len": input_len,
        "gen_kwargs": gen_kwargs,
        "max_new_tokens": max_new_tokens,
    }


def run_inference(ctx):
    model, gen_kwargs, input_len = ctx["model"], ctx["gen_kwargs"], ctx["input_len"]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**gen_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_tokens = output.shape[1] - input_len
    return output, elapsed, new_tokens


class TestQwen3InferenceMaca:
    def test_benchmark_round(self, ctx):
        output, elapsed, new_tokens = run_inference(ctx)
        tps = new_tokens / elapsed
        print(
            f"\n  Benchmark: {elapsed:.3f}s, {new_tokens} tokens, {tps:.2f} tok/s"
        )
        assert new_tokens == ctx["max_new_tokens"]
        assert tps > 0

    def test_output_non_empty(self, ctx):
        output, elapsed, new_tokens = run_inference(ctx)
        decoded = ctx["tokenizer"].decode(
            output[0][ctx["input_len"] :], skip_special_tokens=True
        )
        print(f"\n  Output ({new_tokens} tok, {new_tokens/elapsed:.2f} tok/s):\n{decoded[:200]}")
        assert len(decoded.strip()) > 0
