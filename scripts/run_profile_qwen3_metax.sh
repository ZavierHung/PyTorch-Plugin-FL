#!/bin/bash
# MetaX (muxi) Qwen3 op profiler + Chrome trace export for Perfetto / chrome://tracing
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate base
source /home/hongzw/setup_metax_env.sh

# torch_fl MetaX C++ kernels (mcblas bmm/mm, etc.) — avoids FlagGems Triton JIT on muxi
export FLAGOS_BACKEND_CONFIG="/home/hongzw/PyTorch-Plugin-FL/torch_fl/backends_metax_profile.conf"
export FLAGOS_DISABLE_FLAGGEMS_PY="${FLAGOS_DISABLE_FLAGGEMS_PY:-1}"

MODEL="${1:-/home/hongzw/Qwen3-0.6B}"
TOKENS="${2:-32}"
ROUNDS="${3:-1}"
WARMUP="${4:-1}"
TRACE_DIR="${5:-/home/hongzw/PyTorch-Plugin-FL/profile_traces/muxi_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$TRACE_DIR"

# Kill stale profile runs that hold GPU memory
pkill -f "profile_qwen3_infer_cuda.py" 2>/dev/null || true
sleep 2

echo "Trace output: $TRACE_DIR"

cd /home/hongzw/PyTorch-Plugin-FL
python tests/perf/profile_qwen3_infer_cuda.py \
  --model "$MODEL" \
  --tokens "$TOKENS" \
  --rounds "$ROUNDS" \
  --warmup-rounds "$WARMUP" \
  --trace-dir "$TRACE_DIR" \
  2>&1 | tee "$TRACE_DIR/run.log"

echo ""
echo "Done. Artifacts:"
echo "  Chrome trace: $TRACE_DIR/trace_round*.json  -> https://ui.perfetto.dev"
echo "  Op summary:   $TRACE_DIR/op_profile_summary.json"
echo "  Run log:      $TRACE_DIR/run.log"
