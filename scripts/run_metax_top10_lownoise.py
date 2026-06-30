#!/usr/bin/env python3
"""
Low-noise Top-10 operator profile comparison (Experiment A core metric).

Each variant runs in a fresh subprocess with per-round OpProfiler (not accumulated).
Reports median / stdev / spread per Top-10 op across timed rounds.

Usage:
  python scripts/run_metax_top10_lownoise.py --mode pure
  python scripts/run_metax_top10_lownoise.py --mode all --trials 2 --shuffle
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

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

VARIANTS: dict[str, list[tuple[str, str, str]]] = {
    "pure": [
        ("pure_baseline", "torch_fl/backends_metax_baseline.conf", "pure"),
        ("pure_metax_opt", "torch_fl/backends_metax_opt.conf", "pure"),
        ("pure_metax_v2", "torch_fl/backends_metax_v2.conf", "pure"),
    ],
    "hybrid": [
        ("hybrid_baseline", "torch_fl/backends_metax_flagos_py.conf", "hybrid"),
        ("hybrid_metax_opt", "torch_fl/backends_metax_flagos_py_opt.conf", "hybrid"),
        ("hybrid_metax_v2", "torch_fl/backends_metax_flagos_py_v2.conf", "hybrid"),
    ],
}


def build_env(stack: str) -> dict[str, str]:
    env = os.environ.copy()
    if stack == "hybrid":
        env.pop("FLAGOS_DISABLE_FLAGGEMS_PY", None)
        env.setdefault("GEMS_VENDOR", "metax")
        env.setdefault(
            "FLAGGEMS_SOURCE_DIR",
            "/home/hongzw/FlagGems/src/flag_gems",
        )
    else:
        env["FLAGOS_DISABLE_FLAGGEMS_PY"] = "1"
    return env


def run_variant(
    name: str,
    conf: str,
    stack: str,
    model: str,
    tokens: int,
    warmup: int,
    rounds: int,
    discard_first: int,
    out_dir: Path,
    cooldown: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_path = str(REPO / conf)
    json_path = out_dir / "top10_profile.json"
    log_path = out_dir / "top10_profile.log"

    env = build_env(stack)
    env["FLAGOS_BACKEND_CONFIG"] = conf_path

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "metax_top10_profile_benchmark.py"),
        "--model",
        model,
        "--tokens",
        str(tokens),
        "--warmup-rounds",
        str(warmup),
        "--rounds",
        str(rounds),
        "--discard-first",
        str(discard_first),
        "--json-out",
        str(json_path),
    ]

    print(f"\n=== {name} ===")
    print(f"  conf: {conf_path}")
    proc = subprocess.run(
        cmd, env=env, cwd=str(REPO), capture_output=True, text=True, timeout=2400
    )
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed, see {log_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["variant"] = name
    data["stack"] = stack
    if cooldown > 0:
        time.sleep(cooldown)
    return data


def write_report(
    path: Path, meta: dict, trials: list[dict[str, dict]]
) -> None:
    lines = [
        "# Top-10 算子低噪声 Profile 对比",
        "",
        f"生成时间: {meta['timestamp_utc']}",
        "",
        "## 协议",
        "",
        f"- Warmup **{meta['warmup_rounds']}** · Timed **{meta['timed_rounds']}** · "
        f"discard-first **{meta['discard_first']}**",
        f"- Trials **{meta['trials']}** · shuffle **{meta['shuffle']}** · "
        f"子进程隔离 · cooldown **{meta['cooldown_s']}s**",
        f"- 每轮独立 OpProfiler，Top10 取各轮 median",
        "",
    ]

    for trial_idx, trial_data in enumerate(trials, 1):
        lines.append(f"## Trial {trial_idx}")
        lines.append("")
        lines.append("### Top10 合计 (median ms/round)")
        lines.append("")
        lines.append("| variant | top10 sum | wall tok/s | top10 spread |")
        lines.append("|---------|-----------|------------|--------------|")
        for name, data in trial_data.items():
            t10 = data["top10_sum"]
            wall = data["wall"]
            lines.append(
                f"| {name} | **{t10['median_ms']:.1f}** ms | "
                f"{meta['tokens'] / wall['median_s']:.2f} | {t10['spread_pct']:.1f}% |"
            )
        lines.append("")

        lines.append("### 各算子 median 总耗时 (ms/round)")
        lines.append("")
        header = "| 算子 | " + " | ".join(trial_data.keys()) + " |"
        sep = "|------|" + "|".join(["------"] * len(trial_data)) + "|"
        lines.append(header)
        lines.append(sep)
        for op in TOP10_OPS:
            short = op.replace("aten.", "").replace(".default", "").replace(".Tensor", "")
            cells = []
            for name in trial_data:
                ms = trial_data[name]["ops"][op]["median_ms"]
                cells.append(f"{ms:.1f}")
            lines.append(f"| `{short}` | " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("### 各算子 spread (%)")
        lines.append("")
        lines.append(header.replace("算子", "算子 spread"))
        lines.append(sep)
        for op in TOP10_OPS:
            short = op.replace("aten.", "").replace(".default", "").replace(".Tensor", "")
            cells = [f"{trial_data[n]['ops'][op]['spread_pct']:.1f}" for n in trial_data]
            lines.append(f"| `{short}` | " + " | ".join(cells) + " |")
        lines.append("")

    if len(trials) > 1:
        lines.append("## 跨 Trial 汇总（Top10 sum median of medians）")
        lines.append("")
        lines.append("| variant | median ms | trial stdev |")
        lines.append("|---------|-----------|-------------|")
        all_names = list(trials[0].keys())
        for name in all_names:
            vals = [t[name]["top10_sum"]["median_ms"] for t in trials]
            med = statistics.median(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            lines.append(f"| {name} | **{med:.1f}** | {sd:.1f} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pure", "hybrid", "all"], default="pure")
    p.add_argument("--model", default="/home/hongzw/Qwen3-0.6B")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--warmup-rounds", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--discard-first", type=int, default=1)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument(
        "--out-dir",
        default=str(REPO / "profile_traces" / "metax_top10_lownoise"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, str, str]] = []
    if args.mode in ("pure", "all"):
        variants.extend(VARIANTS["pure"])
    if args.mode in ("hybrid", "all"):
        variants.extend(VARIANTS["hybrid"])

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model": args.model,
        "tokens": args.tokens,
        "warmup_rounds": args.warmup_rounds,
        "timed_rounds": args.rounds,
        "discard_first": args.discard_first,
        "trials": args.trials,
        "shuffle": args.shuffle,
        "cooldown_s": args.cooldown,
        "top10_ops": TOP10_OPS,
    }

    all_trials: list[dict[str, dict]] = []
    for trial in range(1, args.trials + 1):
        trial_dir = out_root / f"trial_{trial:02d}"
        order = list(variants)
        if args.shuffle:
            random.shuffle(order)
        print(f"\n######## Trial {trial}/{args.trials} ########")
        print("Order:", " -> ".join(n for n, _, _ in order))

        trial_data: dict[str, dict] = {}
        for name, conf, stack in order:
            data = run_variant(
                name, conf, stack, args.model, args.tokens,
                args.warmup_rounds, args.rounds, args.discard_first,
                trial_dir / name, args.cooldown,
            )
            trial_data[name] = data

        all_trials.append(trial_data)
        (trial_dir / "trial_summary.json").write_text(
            json.dumps(trial_data, indent=2), encoding="utf-8"
        )

    payload = {"meta": meta, "trials": all_trials}
    (out_root / "top10_lownoise_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_report(out_root / "TOP10_LOWNOISE_REPORT.md", meta, all_trials)
    print(f"\nReport: {out_root / 'TOP10_LOWNOISE_REPORT.md'}")


if __name__ == "__main__":
    main()
