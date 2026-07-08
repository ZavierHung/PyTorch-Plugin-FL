// Copyright (c) 2026, BAAI. All rights reserved.
//
// MetaX v2 fused RMSNorm CUDA kernel.
//
//   out = weight * (x * rsqrt(mean(x^2, dim=-1) + eps))
//
// Fuses HF's RMSNorm chain (pow -> mean -> add -> rsqrt -> mul -> mul, 5~6
// separate launches) into a single kernel to cut launch overhead. I/O keeps
// the model dtype (fp16/bf16); the reduction runs in fp32 (at::opmath_type).
//
// WARNING: warpSize is 64 on MetaX C500, not the CUDA default 32. Every
// __shfl_down_sync mask/offset below assumes WARP=64; using 32 silently
// produces wrong reductions.

#pragma once

#include <cmath>
#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>
#include <c10/util/Exception.h>

#include "metax_elementwise_v2.cuh"

namespace at::native::flagos {

namespace {

// ============================================================================
// GPU kernel: per-row fused RMSNorm
// ============================================================================
//
// One block per row (last contiguous dim). Two-pass over the row: pass 1
// accumulates sum(x^2) in fp32 and derives scale = rsqrt(mean + eps); pass 2
// re-reads x/weight and writes out. The second global read hits L2 (hidden is
// small, 1024/4096), which is cheaper than holding the whole row in registers.

template <typename scalar_t, typename acc_t>
__global__ void RmsNormRowKernel(
    int64_t rows,                         // total rows = batch * seq_len
    int64_t hidden,                       // last-dim size
    acc_t eps,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    const scalar_t* __restrict__ weight)  // length == hidden

{
  // BLOCK=256 = 4 warps x 64 lanes. WARP=64 is the C500 wavefront width and
  // must match the hardware or the shuffle offsets below are wrong.
  constexpr int BLOCK = 256;
  constexpr int WARP = 64;
  constexpr int WARPS = BLOCK / WARP;  // = 4

  // grid may exceed rows; extra blocks return early.
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  const scalar_t* __restrict__ row_in = in + row * hidden;
  scalar_t* __restrict__ row_out = out + row * hidden;

  // Pass 1: sum(x^2) in fp32.
  acc_t thread_sum = acc_t(0);
  for (int64_t i = threadIdx.x; i < hidden; i += BLOCK) {
    const acc_t v = static_cast<acc_t>(__ldg(row_in + i));
    thread_sum += v * v;
  }

  // Warp-level shuffle reduction (WARP=64, mask = 64 ones).
  for (int offset = WARP / 2; offset > 0; offset >>= 1) {
    thread_sum += __shfl_down_sync(0xffffffffffffffffull, thread_sum, offset);
  }

  // Cross-warp reduction via shared memory (each warp's lane 0 writes).
  __shared__ acc_t warp_sums[WARPS];

  const int lane = threadIdx.x & (WARP - 1);
  const int warp_id = threadIdx.x / WARP;

  if (lane == 0) {
    warp_sums[warp_id] = thread_sum;
  }
  __syncthreads();

  // warp 0 finishes the reduction and computes scale = rsqrt(mean + eps).
  __shared__ acc_t scale_sh;

  if (warp_id == 0) {
    acc_t v = (threadIdx.x < WARPS) ? warp_sums[threadIdx.x] : acc_t(0);

    for (int offset = WARPS / 2; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xffffffffffffffffull, v, offset);
    }

    if (threadIdx.x == 0) {
      const acc_t mean = v / static_cast<acc_t>(hidden);
      scale_sh = acc_t(1) / ::sqrt(mean + eps);
    }
  }
  __syncthreads();

  const acc_t scale = scale_sh;

  // Pass 2: out = weight * x * scale.
  for (int64_t i = threadIdx.x; i < hidden; i += BLOCK) {
    const acc_t x = static_cast<acc_t>(__ldg(row_in + i));
    const acc_t w = static_cast<acc_t>(__ldg(weight + i));
    row_out[i] = static_cast<scalar_t>(w * x * scale);
  }
}

// ============================================================================
// Host launcher
// ============================================================================
// One block per row (gridDim = rows), clamped to kMaxGridV2 for large prefill;
// the in-kernel row >= rows check drops the surplus blocks.

template <typename scalar_t, typename acc_t>
void LaunchRmsNorm(
    int64_t rows,
    int64_t hidden,
    acc_t eps,
    scalar_t* out_ptr,
    const scalar_t* in_ptr,
    const scalar_t* weight_ptr,
    cudaStream_t stream) {
  constexpr int BLOCK = 256;

  if (rows == 0 || hidden == 0) {
    return;
  }

  // Clamp grid to the hardware limit (decode rows are tiny; prefill can be
  // large, e.g. batch=32, seq=2048 -> rows=65536).
  const int blocks =
      static_cast<int>(std::min<int64_t>(rows, metax::kMaxGridV2));

  RmsNormRowKernel<scalar_t, acc_t><<<blocks, BLOCK, 0, stream>>>(
      rows, hidden, eps, out_ptr, in_ptr, weight_ptr);

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX rmsnorm v2 launch failed: ",
      cudaGetErrorString(err));
}

} // namespace

// ============================================================================
// Public API: fused RMSNorm (out = weight * x * rsqrt(mean(x^2) + eps))
// ============================================================================
// Requires input dim >= 1, weight 1-D of length == input's last dim, and
// input/weight sharing dtype (fp16/bf16). Returns a new tensor like input.

inline at::Tensor RmsNormKernelV2(
    const at::Tensor& input,
    const at::Tensor& weight,
    double eps) {
  TORCH_CHECK(input.dim() >= 1, "rms_norm: input must have >= 1 dim");
  TORCH_CHECK(weight.dim() == 1, "rms_norm: weight must be 1-D");

  const int64_t hidden = input.size(input.dim() - 1);
  TORCH_CHECK(
      weight.size(0) == hidden,
      "rms_norm: weight size (", weight.size(0),
      ") must match last dim of input (", hidden, ")");
  TORCH_CHECK(
      input.scalar_type() == weight.scalar_type(),
      "rms_norm: input and weight must share dtype");

  // kernel assumes the last dim is contiguous.
  const at::Tensor in = input.is_contiguous() ? input : input.contiguous();
  const at::Tensor w =
      weight.is_contiguous() ? weight : weight.contiguous();

  at::Tensor output = at::empty_like(in);

  if (in.numel() == 0) {
    return output;
  }

  const int64_t rows = in.numel() / hidden;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      in.scalar_type(),
      "rms_norm_metax_v2",
      [&]() {
        // acc_t = fp32 for fp16/bf16, keeping the reduction in fp32.
        using acc_t = at::opmath_type<scalar_t>;
        LaunchRmsNorm<scalar_t, acc_t>(
            rows,
            hidden,
            static_cast<acc_t>(eps),
            output.data_ptr<scalar_t>(),
            in.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            metax::CurrentStream());
      });

  return output;
}

} // namespace at::native::flagos
