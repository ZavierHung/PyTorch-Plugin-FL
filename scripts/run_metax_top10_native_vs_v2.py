#!/usr/bin/env python3
"""
Low-noise Top10 comparison: native MACA (cuda:0) vs pure metax_v2 (torch_fl).

Each backend runs in a fresh subprocess with identical protocol:
  8 warmup + 10 timed rounds + discard-first 2

Usage:
  python scripts/run_metax_top10_native_vs_v2.py
  python scripts/run_metax_top10_native_vs_v2.py --warmup-rounds 10 --discard-first 3
"""

from __future__ import annotations

import argparse
import json
import os
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

VARIANTS = [
    ("native_maca", "native", None),
    ("pure_metax_v2", "torch_fl", "torch_fl/backends_metax_v2.conf"),
]


def build_env(stack: str, conf: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("FLAGOS_BACKEND_CONFIG", None)
    env.pop("FLAGOS_DISABLE_FLAGGEMS_PY", None)
    if stack == "torch_fl":
        env["FLAGOS_DISABLE_FLAGGEMS_PY"] = "1"
        if conf:
            env["FLAGOS_BACKEND_CONFIG"] = str(REPO / conf)
    return env


def run_variant(
    name: str,
    backend: str,
    conf: str | None,
    model: str,
    tokens: int,
    warmup: int,
    rounds: int,
    discard_first: int,
    out_dir: Path,
    cooldown: float,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "top10_profile.json"
    log_path = out_dir / "top10_profile.log"

    env = build_env(backend, conf)
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "metax_top10_profile_benchmark.py"),
        "--backend",
        backend,
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
    print(f"\n=== {name} ({backend}) ===")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed (exit {proc.returncode}); see {log_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    print(
        f"  top10_sum={data['top10_sum']['median_ms']:.1f}ms  "
        f"spread={data['top10_sum']['spread_pct']:.1f}%"
    )
    time.sleep(cooldown)
    return data


def write_summary(results: dict[str, dict], out_dir: Path, meta: dict) -> None:
    lines = [
        "# Top10: native MACA vs pure metax_v2",
        "",
        f"> {meta['timestamp_utc']}",
        "",
        "## Protocol",
        "",
        f"- warmup: {meta['warmup_rounds']} · timed: {meta['timed_rounds']} · "
        f"discard-first: {meta['discard_first']}",
        f"- model: `{meta['model']}` · tokens: {meta['tokens']}",
        "",
        "## Top10 sum (ms/round, median)",
        "",
        "| variant | top10 sum | spread |",
        "|---------|-----------|--------|",
    ]
    for name, data in results.items():
        t10 = data["top10_sum"]
        lines.append(
            f"| {name} | **{t10['median_ms']:.1f}** | {t10['spread_pct']:.1f}% |"
        )
    lines.extend(["", "## Per-op (ms/round, median)", ""])
    lines.append("| op | native MACA | pure metax_v2 | ratio (v2/native) |")
    lines.append("|----|-------------|---------------|-------------------|")
    native = results.get("native_maca", {}).get("ops", {})
    v2 = results.get("pure_metax_v2", {}).get("ops", {})
    for op in TOP10_OPS:
        short = op.replace("aten.", "").replace(".default", "").replace(".Tensor", "")
        n_ms = native.get(op, {}).get("median_ms", 0.0)
        v_ms = v2.get(op, {}).get("median_ms", 0.0)
        ratio = v_ms / n_ms if n_ms else 0.0
        lines.append(f"| `{short}` | {n_ms:.1f} | {v_ms:.1f} | {ratio:.2f}x |")

    (out_dir / "TOP10_NATIVE_VS_V2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/hongzw/Qwen3-0.6B")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--warmup-rounds", type=int, default=8)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--discard-first", type=int, default=2)
    p.add_argument("--cooldown", type=float, default=5.0)
    p.add_argument(
        "--out-dir",
        default=str(REPO / "profile_traces" / "metax_top10_native_vs_v2"),
    )
    args = p.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, backend, conf in VARIANTS:
        results[name] = run_variant(
            name,
            backend,
            conf,
            args.model,
            args.tokens,
            args.warmup_rounds,
            args.rounds,
            args.discard_first,
            out_root / name,
            args.cooldown,
        )

    meta = {
        "model": args.model,
        "tokens": args.tokens,
        "warmup_rounds": args.warmup_rounds,
        "timed_rounds": args.rounds,
        "discard_first": args.discard_first,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "top10_ops": TOP10_OPS,
    }
    payload = {"meta": meta, "results": results}
    (out_root / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_summary(results, out_root, meta)
    print(f"\nJSON: {out_root / 'summary.json'}")
    print(f"Report: {out_root / 'TOP10_NATIVE_VS_V2.md'}")


if __name__ == "__main__":
    main()
