#!/usr/bin/env python3
"""
Compare Qwen3 inference TPS: native MACA torch (cuda:0) vs optimized torch_fl (flagos:0).

Runs each backend in a **separate subprocess** to avoid torch_fl registration bleed.

Usage:
    python tests/perf/compare_maca_vs_torch_fl_qwen3.py \\
        --model /home/hongzw/Qwen3-0.6B --tokens 16 --warmup-rounds 1 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

WORKER = textwrap.dedent(
    r"""
    import json, os, sys, time
    backend = sys.argv[1]
    model = sys.argv[2]
    tokens = int(sys.argv[3])
    warmup = int(sys.argv[4])
    rounds = int(sys.argv[5])

    def patch_triton():
        try:
            import triton.language.extra.libdevice as ld
            import triton.language.math as tlm
            for n in ("pow","erf","exp","tanh","rsqrt","exp2"):
                if not hasattr(tlm,n) and hasattr(ld,n):
                    setattr(tlm,n,getattr(ld,n))
        except Exception:
            pass

    patch_triton()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if backend == "torch_fl":
        import torch_fl
        device = "flagos:0"
        def sync():
            torch_fl.flagos.synchronize()
        stack = {
            "backend": "torch_fl",
            "device": device,
            "flagos_config": os.environ.get("FLAGOS_BACKEND_CONFIG", ""),
            "registered_ops": len(torch_fl.get_registered_ops()),
        }
    else:
        device = "cuda:0"
        def sync():
            torch.cuda.synchronize()
        stack = {
            "backend": "native_maca_torch",
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        }

    stack["torch_version"] = torch.__version__

    tokenizer = AutoTokenizer.from_pretrained(model)
    model_obj = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, device_map="cpu"
    ).to(device)
    model_obj.eval()
    if hasattr(model_obj, "model"):
        model_obj.model.layers[0].self_attn.config._attn_implementation = "eager"

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Give me a short introduction to large language model."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    gen = dict(
        **inputs,
        max_new_tokens=tokens,
        min_new_tokens=tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    round_times = []
    for i in range(warmup):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model_obj.generate(**gen)
        sync()
        stack.setdefault("warmup_s", []).append(time.perf_counter() - t0)

    for _ in range(rounds):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model_obj.generate(**gen)
        sync()
        round_times.append(time.perf_counter() - t0)

    round_times.sort()
    med = round_times[len(round_times) // 2]
    stack.update({
        "input_tokens": input_len,
        "new_tokens": tokens,
        "warmup_rounds": warmup,
        "rounds": rounds,
        "round_times_s": round_times,
        "median_time_s": med,
        "median_tps": tokens / med,
        "output_tokens": out.shape[1] - input_len,
    })
    print(json.dumps(stack))
    """
)


def run_backend(
    backend: str,
    model: str,
    tokens: int,
    warmup: int,
    rounds: int,
    extra_env: dict[str, str] | None = None,
    torch_fl_conf: str | None = None,
    enable_flaggems_py: bool = False,
) -> dict:
    env = os.environ.copy()
    if enable_flaggems_py:
        env.pop("FLAGOS_DISABLE_FLAGGEMS_PY", None)
        env.setdefault("GEMS_VENDOR", "metax")
        env.setdefault(
            "FLAGGEMS_SOURCE_DIR",
            os.environ.get("FLAGGEMS_SOURCE_DIR", "/home/hongzw/FlagGems/src/flag_gems"),
        )
    else:
        env["FLAGOS_DISABLE_FLAGGEMS_PY"] = "1"
    if backend == "torch_fl":
        env["FLAGOS_BACKEND_CONFIG"] = torch_fl_conf or str(
            REPO / "torch_fl" / "backends_metax_v2.conf"
        )
    else:
        env.pop("FLAGOS_BACKEND_CONFIG", None)

    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable,
        "-c",
        WORKER,
        backend,
        model,
        str(tokens),
        str(warmup),
        str(rounds),
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(REPO),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{backend} failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def collect_env_meta() -> dict:
    import torch

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "maca_path": os.environ.get("MACA_PATH", "/opt/maca"),
        "metax_path": os.environ.get("METAX_PATH", ""),
        "container_image_expected": "maca-torch2.8-py312:mc3.7.0.7-ubuntu24.04-amd64",
        "stack_note": (
            "Native MACA = torch 2.8.0+metax3.7.0.7 on cuda:0 (MetaX C500 via cu-bridge); "
            "no separate torch_maca Python wheel in this image."
        ),
    }
    ver = Path("/opt/maca/Version.txt")
    if ver.is_file():
        meta["maca_version_txt"] = ver.read_text().strip()
    return meta


def write_report(path: Path, meta: dict, maca: dict, fl: dict) -> None:
    speedup = fl["median_tps"] / maca["median_tps"] if maca["median_tps"] else 0
    lines = [
        "# Qwen3 推理对比：原生 MACA torch vs 优化后 torch_fl",
        "",
        f"生成时间 (UTC): {meta['timestamp_utc']}",
        "",
        "## 环境（两组完全一致）",
        "",
        "| 项 | 值 |",
        "|----|-----|",
        f"| 容器镜像（预期） | `{meta['container_image_expected']}` |",
        f"| 主机名 | `{meta['hostname']}` |",
        f"| OS | {meta['platform']} |",
        f"| Python | {meta['python']} |",
        f"| PyTorch | `{meta['torch_version']}` |",
        f"| GPU | {meta['gpu_name']} |",
        f"| MACA 路径 | `{meta['maca_path']}` |",
        f"| MACA Version.txt | {meta.get('maca_version_txt', 'N/A')} |",
        "",
        meta["stack_note"],
        "",
        "## Benchmark 参数",
        "",
        f"- 模型: `{fl['model_path']}`",
        f"- 新 token 数: **{maca['new_tokens']}** (greedy, fixed)",
        f"- Warmup: {maca['warmup_rounds']} 轮",
        f"- 计时轮次: {maca['rounds']} 轮（取 median）",
        f"- Attention: eager",
        "",
        "## TPS 对比（主指标）",
        "",
        "| 后端 | 设备 | 配置 | Median 耗时 | **tok/s** |",
        "|------|------|------|-------------|-----------|",
        f"| 原生 MACA torch | `cuda:0` | PyTorch 自带 ATen/cu-bridge | {maca['median_time_s']:.3f}s | **{maca['median_tps']:.2f}** |",
        f"| torch_fl (优化 v2) | `flagos:0` | `backends_metax_v2.conf` | {fl['median_time_s']:.3f}s | **{fl['median_tps']:.2f}** |",
        "",
        f"**相对原生加速比**: {speedup:.2f}× （torch_fl / native）",
        "",
        "## 分项耗时",
        "",
        "### 原生 MACA",
        f"- Warmup: {[f'{x:.3f}s' for x in maca.get('warmup_s', [])]}",
        f"- Rounds: {[f'{x:.3f}s' for x in maca['round_times_s']]}",
        "",
        "### torch_fl v2",
        f"- FlagOS conf: `{fl.get('flagos_config', '')}`",
        f"- Registered ops: {fl.get('registered_ops', 'N/A')}",
        f"- Warmup: {[f'{x:.3f}s' for x in fl.get('warmup_s', [])]}",
        f"- Rounds: {[f'{x:.3f}s' for x in fl['round_times_s']]}",
        "",
        "## 结论",
        "",
    ]
    if speedup >= 1.05:
        lines.append(
            f"- 在相同 MACA 栈上，优化后 torch_fl 比原生 torch **快约 {(speedup-1)*100:.1f}%**（{fl['median_tps']:.2f} vs {maca['median_tps']:.2f} tok/s）。"
        )
    elif speedup <= 0.95:
        lines.append(
            f"- 本轮实测原生 torch 更快（{maca['median_tps']:.2f} vs {fl['median_tps']:.2f} tok/s），需结合算子覆盖与 conf 再分析。"
        )
    else:
        lines.append("- 两组 TPS 接近，差异在误差范围内。")

    lines.extend(
        [
            "",
            "## 与 torch_fl 内部对比（同栈、不同对照系）",
            "",
            "| 对照 | tok/s | 说明 |",
            "|------|-------|------|",
            "| torch_fl baseline (`metax`) | ~16.3 | 见 `METAX_TOP10_OPTIMIZATION.md` |",
            "| torch_fl v2 (`metax_v2`) | ~34.8 | 相对 baseline **2.1×** |",
            f"| **原生 MACA torch** | **{maca['median_tps']:.1f}** | 相对 torch_fl v2 **{maca['median_tps']/fl['median_tps']:.2f}×** |",
            "",
            "torch_fl Top10 优化解决的是 **PrivateUse1 路径上的 kernel 效率**；",
            "原生 `cuda:0` 仍走 MACA 高度优化的 ATen/cuBLAS，当前 **整体仍显著快于 torch_fl**。",
            "后续方向：减少 dispatch/boxing 开销，或让热点算子 delegate 回原生 MACA。",
            "",
            "- 原生路径走 `cuda:0`（MACA 驱动 + cu-bridge），不经 flagos dispatcher。",
            "- torch_fl 路径走 `flagos:0` + Top10 `metax_v2` 优化 kernel。",
            "",
            "## 复现",
            "",
            "```bash",
            "source /home/hongzw/setup_metax_env.sh",
            "export FLAGOS_DISABLE_FLAGGEMS_PY=1",
            "cd /home/hongzw/PyTorch-Plugin-FL",
            "python tests/perf/compare_maca_vs_torch_fl_qwen3.py \\",
            "  --model /home/hongzw/Qwen3-0.6B --tokens 16 --warmup-rounds 1 --rounds 3",
            "```",
            "",
            "原生单测:",
            "",
            "```bash",
            "pytest tests/integration/maca_native --confcutdir=tests/integration/maca_native -v -s \\",
            "  --model /home/hongzw/Qwen3-0.6B --max-new-tokens 16 --warmup-rounds 1",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/hongzw/Qwen3-0.6B")
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--warmup-rounds", type=int, default=1)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument(
        "--out-dir",
        default=str(REPO / "profile_traces" / "maca_vs_torch_fl"),
    )
    p.add_argument(
        "--torch-fl-conf",
        default=str(REPO / "torch_fl" / "backends_metax_v2.conf"),
        help="FLAGOS_BACKEND_CONFIG for torch_fl path",
    )
    p.add_argument(
        "--hybrid",
        action="store_true",
        help="Enable flagos_python (unset FLAGOS_DISABLE_FLAGGEMS_PY, GEMS_VENDOR=metax)",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = collect_env_meta()
    meta["model"] = args.model

    print("=== Native MACA torch (cuda:0) ===")
    maca = run_backend(
        "native_maca", args.model, args.tokens, args.warmup_rounds, args.rounds
    )
    maca["model_path"] = args.model
    print(f"  median: {maca['median_time_s']:.3f}s  {maca['median_tps']:.2f} tok/s")

    conf_label = Path(args.torch_fl_conf).name
    print(f"=== torch_fl ({conf_label}, flagos:0) ===")
    fl = run_backend(
        "torch_fl",
        args.model,
        args.tokens,
        args.warmup_rounds,
        args.rounds,
        torch_fl_conf=args.torch_fl_conf,
        enable_flaggems_py=args.hybrid,
    )
    fl["model_path"] = args.model
    print(f"  median: {fl['median_time_s']:.3f}s  {fl['median_tps']:.2f} tok/s")

    payload = {
        "env": meta,
        "native_maca": maca,
        "torch_fl": fl,
        "torch_fl_conf": args.torch_fl_conf,
        "hybrid_flaggems_py": args.hybrid,
    }
    json_path = out_dir / "compare_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = out_dir / "MACA_VS_TORCH_FL_REPORT.md"
    write_report(report_path, meta, maca, fl)

    print(f"\nJSON: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
