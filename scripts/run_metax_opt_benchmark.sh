#!/bin/bash
# Three-way MetaX Top-10 optimization benchmark: baseline / metax_opt / metax_v2
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate base
source /home/hongzw/setup_metax_env.sh

export FLAGOS_DISABLE_FLAGGEMS_PY="${FLAGOS_DISABLE_FLAGGEMS_PY:-1}"

REPO="/home/hongzw/PyTorch-Plugin-FL"
MODEL="${1:-/home/hongzw/Qwen3-0.6B}"
TOKENS="${2:-16}"
WARMUP="${3:-1}"
ROUNDS="${4:-1}"
OUT_ROOT="${5:-$REPO/profile_traces/metax_opt_cmp_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUT_ROOT"
mkdir -p "$(dirname "$OUT_ROOT/master.log" 2>/dev/null || echo "$OUT_ROOT")"

run_variant() {
  local name="$1"
  local conf="$2"
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"

  export FLAGOS_BACKEND_CONFIG="$conf"
  echo "=== $name ==="
  echo "  conf: $conf"
  echo "  out:  $dir"

  pkill -f "profile_qwen3_infer_cuda.py" 2>/dev/null || true
  pkill -f "e2e_qwen3_infer_cuda.py" 2>/dev/null || true
  sleep 2

  cd "$REPO"
  python "$REPO/scripts/metax_e2e_benchmark.py" \
    --model "$MODEL" \
    --tokens "$TOKENS" \
    --warmup-rounds "$WARMUP" \
    --rounds "$ROUNDS" \
    2>&1 | tee "$dir/e2e.log"

  python tests/perf/profile_qwen3_infer_cuda.py \
    --model "$MODEL" \
    --tokens "$TOKENS" \
    --rounds "$ROUNDS" \
    --warmup-rounds "$WARMUP" \
    --trace-dir "$dir" \
    2>&1 | tee "$dir/profile.log"

  echo "$name done" >> "$OUT_ROOT/run_manifest.txt"
}

: > "$OUT_ROOT/run_manifest.txt"
run_variant baseline "$REPO/torch_fl/backends_metax_baseline.conf"
run_variant metax_opt "$REPO/torch_fl/backends_metax_opt.conf"
run_variant metax_v2 "$REPO/torch_fl/backends_metax_v2.conf"

echo ""
echo "All variants finished. Results under: $OUT_ROOT"
