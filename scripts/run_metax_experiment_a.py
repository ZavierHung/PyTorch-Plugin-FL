#!/usr/bin/env python3
"""
Low-noise Experiment A runner for MetaX torch_fl variants.

Design goals (vs run_metax_*_benchmark.sh):
  - Fresh subprocess per variant (no torch_fl / JIT bleed between configs)
  - Many timed rounds + median / stdev / IQR (default 5 warmup, 10 timed)
  - Optional shuffle of variant order + repeated trials
  - E2E-only by default (profile is optional and separated)
  - JSON aggregate for apples-to-apples comparison

Usage:
  # Hybrid suite (recommended defaults)
  python scripts/run_metax_experiment_a.py --mode hybrid

  # Hybrid + pure v2 control in one batch
  python scripts/run_metax_experiment_a.py --mode hybrid --include-pure-control

  # Pure metax only
  python scripts/run_metax_experiment_a.py --mode pure
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

VARIANTS: dict[str, list[tuple[str, str, str]]] = {
    "hybrid": [
        ("hybrid_baseline", "torch_fl/backends_metax_flagos_py.conf", "hybrid"),
        ("hybrid_metax_opt", "torch_fl/backends_metax_flagos_py_opt.conf", "hybrid"),
        ("hybrid_metax_v2", "torch_fl/backends_metax_flagos_py_v2.conf", "hybrid"),
    ],
    "pure": [
        ("pure_baseline", "torch_fl/backends_metax_baseline.conf", "pure"),
        ("pure_metax_opt", "torch_fl/backends_metax_opt.conf", "pure"),
        ("pure_metax_v2", "torch_fl/backends_metax_v2.conf", "pure"),
    ],
}

PURE_CONTROL = (
    "pure_metax_v2_control",
    "torch_fl/backends_metax_v2.conf",
    "pure",
)


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


def run_variant_e2e(
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
    json_path = out_dir / "e2e.json"
    log_path = out_dir / "e2e.log"

    env = build_env(stack)
    env["FLAGOS_BACKEND_CONFIG"] = conf_path

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "metax_e2e_benchmark.py"),
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
    print(f"  stack: {stack}")
    print(f"  conf:  {conf_path}")
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed (exit {proc.returncode}), see {log_path}")

    result = json.loads(json_path.read_text(encoding="utf-8"))
    result["variant"] = name
    result["stack"] = stack
    if cooldown > 0:
        time.sleep(cooldown)
    return result


def run_variant_profile(
    name: str,
    conf: str,
    stack: str,
    model: str,
    tokens: int,
    warmup: int,
    rounds: int,
    out_dir: Path,
    with_trace: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_path = str(REPO / conf)
    env = build_env(stack)
    env["FLAGOS_BACKEND_CONFIG"] = conf_path

    cmd = [
        sys.executable,
        str(REPO / "tests" / "perf" / "profile_qwen3_infer_cuda.py"),
        "--model",
        model,
        "--tokens",
        str(tokens),
        "--warmup-rounds",
        str(warmup),
        "--rounds",
        str(rounds),
    ]
    if with_trace:
        cmd.extend(["--trace-dir", str(out_dir)])

    print(f"\n=== profile: {name} (trace={with_trace}) ===")
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    (out_dir / "profile.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    print(proc.stdout, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"profile {name} failed")


def aggregate_trial(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        rows.append(
            {
                "variant": r["variant"],
                "stack": r["stack"],
                "conf": r["flagos_config"],
                "median_tps": r["median_tps"],
                "mean_tps": r["mean_tps"],
                "stdev_tps": r["stdev_tps"],
                "spread_pct": r["spread_pct"],
                "median_time_s": r["median_time_s"],
                "p25_time_s": r["p25_time_s"],
                "p75_time_s": r["p75_time_s"],
                "round_times_s": r["round_times_s"],
            }
        )
    return rows


def write_summary_md(path: Path, meta: dict, all_trials: list[list[dict]]) -> None:
    lines = [
        "# Experiment A (low-noise rerun)",
        "",
        f"Generated: {meta['timestamp_utc']}",
        "",
        "## Protocol",
        "",
        f"- Warmup: **{meta['warmup_rounds']}** · Timed: **{meta['timed_rounds']}**"
        f" · discard-first: **{meta['discard_first']}**",
        f"- Trials: **{meta['trials']}** · shuffle order: **{meta['shuffle']}**",
        f"- Subprocess per variant · cooldown: **{meta['cooldown_s']}s**",
        f"- Profile: **{meta['profile']}**",
        "",
        "## E2E median tok/s",
        "",
    ]

    # Table per trial
    for ti, trial in enumerate(all_trials, 1):
        lines.append(f"### Trial {ti}")
        lines.append("")
        lines.append("| variant | median tok/s | mean ± stdev | spread |")
        lines.append("|---------|--------------|--------------|--------|")
        for r in trial:
            lines.append(
                f"| {r['variant']} | **{r['median_tps']:.2f}** | "
                f"{r['mean_tps']:.2f} ± {r['stdev_tps']:.2f} | {r['spread_pct']:.1f}% |"
            )
        lines.append("")

    # Cross-trial median of medians if multiple trials
    if len(all_trials) > 1:
        by_variant: dict[str, list[float]] = {}
        for trial in all_trials:
            for r in trial:
                by_variant.setdefault(r["variant"], []).append(r["median_tps"])
        lines.append("### Cross-trial aggregate (median of medians)")
        lines.append("")
        lines.append("| variant | median tok/s | trial stdev |")
        lines.append("|---------|--------------|-------------|")
        for variant, vals in sorted(by_variant.items()):
            med = statistics.median(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            lines.append(f"| {variant} | **{med:.2f}** | {sd:.2f} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-noise MetaX Experiment A")
    p.add_argument("--mode", choices=["hybrid", "pure", "all"], default="hybrid")
    p.add_argument(
        "--include-pure-control",
        action="store_true",
        help="When mode=hybrid, also run pure_metax_v2 with identical protocol",
    )
    p.add_argument("--model", default="/home/hongzw/Qwen3-0.6B")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--warmup-rounds", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument(
        "--discard-first",
        type=int,
        default=1,
        help="Drop first N timed rounds per variant (after warmup)",
    )
    p.add_argument("--trials", type=int, default=1, help="Repeat full variant suite")
    p.add_argument("--shuffle", action="store_true", help="Shuffle variant order each trial")
    p.add_argument("--cooldown", type=float, default=3.0, help="Seconds between variants")
    p.add_argument(
        "--profile",
        action="store_true",
        help="Run profile pass after e2e (slower; uses OpProfiler hooks)",
    )
    p.add_argument(
        "--profile-trace",
        action="store_true",
        help="With --profile, also export Chrome trace (adds overhead)",
    )
    p.add_argument("--profile-rounds", type=int, default=3)
    p.add_argument("--profile-warmup", type=int, default=3)
    p.add_argument(
        "--out-dir",
        default=str(REPO / "profile_traces" / "metax_experiment_a_lownoise"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    variants: list[tuple[str, str, str]] = []
    if args.mode in ("hybrid", "all"):
        variants.extend(VARIANTS["hybrid"])
    if args.mode in ("pure", "all"):
        variants.extend(VARIANTS["pure"])
    if args.include_pure_control and PURE_CONTROL not in variants:
        variants.append(PURE_CONTROL)

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
        "profile": args.profile,
        "variants": [{"name": n, "conf": c, "stack": s} for n, c, s in variants],
    }

    all_trials: list[list[dict]] = []
    for trial in range(1, args.trials + 1):
        trial_dir = out_root / f"trial_{trial:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        order = list(variants)
        if args.shuffle:
            random.shuffle(order)

        print(f"\n######## Trial {trial}/{args.trials} ########")
        print("Order:", " -> ".join(n for n, _, _ in order))

        trial_results: list[dict] = []
        for name, conf, stack in order:
            vdir = trial_dir / name
            result = run_variant_e2e(
                name=name,
                conf=conf,
                stack=stack,
                model=args.model,
                tokens=args.tokens,
                warmup=args.warmup_rounds,
                rounds=args.rounds,
                discard_first=args.discard_first,
                out_dir=vdir,
                cooldown=args.cooldown,
            )
            trial_results.append(result)
            if args.profile:
                run_variant_profile(
                    name=name,
                    conf=conf,
                    stack=stack,
                    model=args.model,
                    tokens=args.tokens,
                    warmup=args.profile_warmup,
                    rounds=args.profile_rounds,
                    out_dir=vdir,
                    with_trace=args.profile_trace,
                )

        rows = aggregate_trial(trial_results)
        all_trials.append(rows)
        (trial_dir / "trial_summary.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

    payload = {"meta": meta, "trials": all_trials}
    summary_json = out_root / "experiment_a_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_md(out_root / "EXPERIMENT_A_LOWNOISE.md", meta, all_trials)

    print(f"\nSummary JSON: {summary_json}")
    print(f"Summary MD:   {out_root / 'EXPERIMENT_A_LOWNOISE.md'}")


if __name__ == "__main__":
    main()
