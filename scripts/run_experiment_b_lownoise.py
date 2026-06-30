#!/usr/bin/env python3
"""Low-noise Experiment B: native MACA vs pure v2 vs hybrid v2 e2e."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "perf"))
from compare_maca_vs_torch_fl_qwen3 import collect_env_meta, run_backend  # noqa: E402

BACKENDS = [
    (
        "native_maca",
        None,
        False,
        "cuda:0 · 无 torch_fl",
    ),
    (
        "pure_metax_v2",
        str(REPO / "torch_fl" / "backends_metax_v2.conf"),
        False,
        "flagos:0 · Top10/非Top10 见 backends_metax_v2.conf",
    ),
    (
        "hybrid_metax_v2",
        str(REPO / "torch_fl" / "backends_metax_flagos_py_v2.conf"),
        True,
        "flagos:0 · Top10 metax_v2 + silu/cos 等 flagos_python",
    ),
]


def spread_pct(times: list[float]) -> float:
    s = sorted(times)
    med = s[len(s) // 2]
    return (s[-1] - s[0]) / med * 100 if med else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/hongzw/Qwen3-0.6B")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--warmup-rounds", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--discard-first", type=int, default=1)
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument(
        "--out-dir",
        default=str(REPO / "profile_traces" / "experiment_b_lownoise"),
    )
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []

    for name, conf, hybrid, note in BACKENDS:
        print(f"\n=== {name} ===")
        if name == "native_maca":
            data = run_backend(
                "native_maca",
                args.model,
                args.tokens,
                args.warmup_rounds,
                args.rounds,
            )
        else:
            data = run_backend(
                "torch_fl",
                args.model,
                args.tokens,
                args.warmup_rounds,
                args.rounds,
                torch_fl_conf=conf,
                enable_flaggems_py=hybrid,
            )
        times = data["round_times_s"]
        if args.discard_first:
            times = times[args.discard_first :]
        tps = [args.tokens / t for t in times]
        med_t = sorted(times)[len(times) // 2]
        entry = {
            "name": name,
            "device": data["device"],
            "conf": conf or "",
            "hybrid": hybrid,
            "note": note,
            "warmup_rounds": args.warmup_rounds,
            "timed_rounds": args.rounds,
            "discard_first": args.discard_first,
            "round_times_s": times,
            "round_tps": tps,
            "median_time_s": med_t,
            "median_tps": args.tokens / med_t,
            "mean_tps": statistics.mean(tps),
            "stdev_tps": statistics.stdev(tps) if len(tps) > 1 else 0.0,
            "spread_pct": spread_pct(times),
        }
        results.append(entry)
        (out / f"{name}.json").write_text(json.dumps(entry, indent=2))
        print(
            f"  median: {entry['median_time_s']:.3f}s  "
            f"{entry['median_tps']:.2f} tok/s  spread {entry['spread_pct']:.1f}%"
        )
        time.sleep(args.cooldown)

    payload = {
        "meta": {
            **collect_env_meta(),
            "model": args.model,
            "tokens": args.tokens,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    (out / "experiment_b_lownoise.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nJSON: {out / 'experiment_b_lownoise.json'}")


if __name__ == "__main__":
    main()
