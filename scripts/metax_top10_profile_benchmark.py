#!/usr/bin/env python3
"""Per-round Top-10 op profiler for Qwen3 inference (low-noise worker)."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict

TOP10_OPS = [
    "aten.cat.default",
    "aten.mm.default",
    "aten.mul.Tensor",
    "aten._to_copy.default",
    "aten.add.Tensor",
    "aten.mean.dim",
    "aten.pow.Tensor_Scalar",
    "aten.rsqrt.default",
    "aten.bmm.default",
    "aten.clone.default",
]


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
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, AutoTokenizer


class OpProfiler(TorchDispatchMode):
    def __init__(self, sync):
        self.sync = sync
        self.op_counts: dict[str, int] = defaultdict(int)
        self.op_times: dict[str, float] = defaultdict(float)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = str(func)
        self.op_counts[op_name] += 1
        self.sync()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        self.sync()
        self.op_times[op_name] += time.perf_counter() - t0
        return result


def _resolve_backend(name: str) -> tuple[str, object, dict]:
    if name == "native":
        device = "cuda:0"
        sync = torch.cuda.synchronize
        meta = {
            "backend": "native_maca_torch",
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "flagos_config": "",
        }
        return device, sync, meta

    if name == "torch_fl":
        import torch_fl

        device = "flagos:0"
        sync = torch_fl.flagos.synchronize
        meta = {
            "backend": "torch_fl",
            "device": device,
            "flaggems_enabled": torch_fl.is_flaggems_enabled(),
            "flagos_config": os.environ.get("FLAGOS_BACKEND_CONFIG", ""),
        }
        return device, sync, meta

    raise SystemExit(f"Unknown --backend {name!r}; use native or torch_fl")


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


def summarize_series(values: list[float]) -> dict:
    s = sorted(values)
    med = s[len(s) // 2]
    return {
        "median_s": med,
        "median_ms": med * 1000,
        "mean_s": statistics.mean(values),
        "stdev_s": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_s": s[0],
        "max_s": s[-1],
        "p25_s": _percentile(s, 0.25),
        "p75_s": _percentile(s, 0.75),
        "spread_pct": (s[-1] - s[0]) / med * 100 if med else 0.0,
        "samples": values,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend",
        choices=("torch_fl", "native"),
        default="torch_fl",
        help="torch_fl=flagos:0; native=MACA cuda:0 without torch_fl",
    )
    p.add_argument("--model", default="/home/hongzw/Qwen3-0.6B")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--warmup-rounds", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--discard-first", type=int, default=1)
    p.add_argument("--json-out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    device, sync, stack_meta = _resolve_backend(args.backend)

    print(f"Backend: {stack_meta['backend']}")
    print(f"Device: {device}")
    if args.backend == "torch_fl":
        print(f"FlagGems enabled: {stack_meta['flaggems_enabled']}")
        print(f"Config: {stack_meta['flagos_config']}")
    else:
        print(f"GPU: {stack_meta.get('gpu_name', '')}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu"
    ).to(device)
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
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**gen_kwargs)
        sync()
        elapsed = time.perf_counter() - t0
        warmup_s.append(elapsed)
        print(f"Warmup {i + 1}: {elapsed:.3f}s")

    per_op_rounds: dict[str, list[float]] = {op: [] for op in TOP10_OPS}
    top10_sum_rounds: list[float] = []
    wall_rounds: list[float] = []

    for i in range(args.rounds):
        profiler = OpProfiler(sync)
        sync()
        t0 = time.perf_counter()
        with profiler:
            with torch.no_grad():
                model.generate(**gen_kwargs)
        sync()
        wall = time.perf_counter() - t0
        wall_rounds.append(wall)

        top10_sum = 0.0
        for op in TOP10_OPS:
            t = profiler.op_times.get(op, 0.0)
            per_op_rounds[op].append(t)
            top10_sum += t
        top10_sum_rounds.append(top10_sum)

        print(
            f"Round {i + 1}: wall={wall:.3f}s top10_sum={top10_sum * 1000:.1f}ms "
            f"mm={profiler.op_times.get('aten.mm.default', 0) * 1000:.1f}ms"
        )

    if args.discard_first > 0:
        wall_rounds = wall_rounds[args.discard_first :]
        top10_sum_rounds = top10_sum_rounds[args.discard_first :]
        for op in TOP10_OPS:
            per_op_rounds[op] = per_op_rounds[op][args.discard_first :]

    op_stats = {op: summarize_series(vals) for op, vals in per_op_rounds.items()}
    top10_stats = summarize_series(top10_sum_rounds)
    wall_stats = summarize_series(wall_rounds)

    print(f"\nTop10 sum median: {top10_stats['median_ms']:.1f}ms "
          f"(spread {top10_stats['spread_pct']:.1f}%)")
    print(f"Wall median: {wall_stats['median_s']:.3f}s "
          f"({args.tokens / wall_stats['median_s']:.2f} tok/s)")

    print("\n=== Top 10 ops (median total time per round) ===")
    ranked = sorted(op_stats.items(), key=lambda x: x[1]["median_s"], reverse=True)
    for op, st in ranked:
        calls = "?"
        print(
            f"{op:40s} {st['median_ms']:7.1f}ms  "
            f"spread {st['spread_pct']:5.1f}%  stdev {st['stdev_s'] * 1000:5.1f}ms"
        )

    payload = {
        "model": args.model,
        "tokens": args.tokens,
        "backend": stack_meta["backend"],
        "device": device,
        "warmup_rounds": args.warmup_rounds,
        "warmup_s": warmup_s,
        "timed_rounds": args.rounds,
        "discard_first": args.discard_first,
        "torch_version": torch.__version__,
        **stack_meta,
        "top10_ops": TOP10_OPS,
        "wall": wall_stats,
        "top10_sum": top10_stats,
        "ops": {
            op: {
                **st,
                "calls_per_round": None,
            }
            for op, st in op_stats.items()
        },
    }
    out = os.path.abspath(args.json_out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
