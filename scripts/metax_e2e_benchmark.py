#!/usr/bin/env python3
"""Standalone MetaX e2e benchmark (no profiler hooks)."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time


def _patch_triton_math_shim():
    try:
        import triton.language.extra.libdevice as ld
        import triton.language.math as tlm

        for name in ("pow", "erf", "exp", "tanh", "rsqrt", "exp2"):
            if not hasattr(tlm, name) and hasattr(ld, name):
                setattr(tlm, name, getattr(ld, name))
    except Exception:
        pass


_patch_triton_math_shim()

import torch
import torch_fl
from transformers import AutoModelForCausalLM, AutoTokenizer


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def summarize_times(round_times: list[float], tokens: int) -> dict:
    sorted_times = sorted(round_times)
    tps = [tokens / t for t in sorted_times]
    median_time = sorted_times[len(sorted_times) // 2]
    return {
        "round_times_s": round_times,
        "round_tps": [tokens / t for t in round_times],
        "median_time_s": median_time,
        "median_tps": tokens / median_time,
        "mean_time_s": statistics.mean(round_times),
        "mean_tps": statistics.mean(tps),
        "stdev_time_s": statistics.stdev(round_times) if len(round_times) > 1 else 0.0,
        "stdev_tps": statistics.stdev(tps) if len(tps) > 1 else 0.0,
        "min_time_s": sorted_times[0],
        "max_time_s": sorted_times[-1],
        "p25_time_s": _percentile(sorted_times, 0.25),
        "p75_time_s": _percentile(sorted_times, 0.75),
        "spread_pct": (sorted_times[-1] - sorted_times[0]) / median_time * 100
        if median_time
        else 0.0,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/hongzw/Qwen3-0.6B")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument(
        "--discard-first",
        type=int,
        default=0,
        help="Drop first N timed rounds after warmup (extra burn-in)",
    )
    p.add_argument("--json-out", default="", help="Write machine-readable summary JSON")
    return p.parse_args()


def main():
    args = parse_args()
    device = "flagos:0"

    print(f"Device: {device}")
    print(f"Device count: {torch_fl.flagos.device_count()}")
    print(f"FlagGems enabled: {torch_fl.is_flaggems_enabled()}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu"
    )
    model = model.to(device)
    model.eval()
    model.model.layers[0].self_attn.config._attn_implementation = "eager"

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
    input_len = inputs["input_ids"].shape[1]
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.tokens,
        min_new_tokens=args.tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    warmup_s: list[float] = []
    for i in range(args.warmup_rounds):
        torch_fl.flagos.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**gen_kwargs)
        torch_fl.flagos.synchronize()
        elapsed = time.perf_counter() - t0
        warmup_s.append(elapsed)
        print(f"Warmup {i + 1}: {elapsed:.3f}s")

    round_times = []
    for i in range(args.rounds):
        torch_fl.flagos.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**gen_kwargs)
        torch_fl.flagos.synchronize()
        elapsed = time.perf_counter() - t0
        new_tokens = output.shape[1] - input_len
        round_times.append(elapsed)
        print(
            f"Round {i + 1}: {elapsed:.3f}s, {new_tokens} tokens, {new_tokens / elapsed:.2f} tok/s"
        )

    if args.discard_first > 0:
        dropped = round_times[: args.discard_first]
        round_times = round_times[args.discard_first :]
        print(
            f"\nDiscarded first {args.discard_first} timed round(s): "
            f"{[f'{x:.3f}s' for x in dropped]}"
        )
        if not round_times:
            raise SystemExit("No timed rounds left after --discard-first")

    stats = summarize_times(round_times, args.tokens)
    print(f"\nMedian: {stats['median_time_s']:.3f}s ({stats['median_tps']:.2f} tok/s)")
    print(
        f"Mean:   {stats['mean_time_s']:.3f}s ({stats['mean_tps']:.2f} tok/s)  "
        f"stdev: {stats['stdev_time_s']:.3f}s"
    )
    print(
        f"Range:  {stats['min_time_s']:.3f}s – {stats['max_time_s']:.3f}s  "
        f"spread: {stats['spread_pct']:.1f}%"
    )
    print(
        f"IQR:    p25={stats['p25_time_s']:.3f}s  p75={stats['p75_time_s']:.3f}s"
    )

    if args.json_out:
        payload = {
            "model": args.model,
            "tokens": args.tokens,
            "warmup_rounds": args.warmup_rounds,
            "timed_rounds": args.rounds,
            "discard_first": args.discard_first,
            "flagos_config": os.environ.get("FLAGOS_BACKEND_CONFIG", ""),
            "torch_version": torch.__version__,
            "flaggems_enabled": torch_fl.is_flaggems_enabled(),
            "warmup_s": warmup_s,
            **stats,
        }
        out = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
