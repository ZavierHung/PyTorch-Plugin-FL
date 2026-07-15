// Copyright (c) 2026, BAAI. All rights reserved.
//
// MetaX v2 fused residual-add + RMSNorm CUDA kernel.
//
//   new_residual = residual + hidden                      (model dtype)
//   normed       = weight * new_residual * rsqrt(mean(new_residual^2) + eps)
//
// Fuses HF decoder layer's `residual + hidden` then post_attention_layernorm
// (6+ launches) into one.
//
// CORRECTNESS: new_residual is rounded back to the model dtype (fp16/bf16)
// before it feeds the reduction and is returned, matching HF's fp16 add
// bit-for-bit. Skipping the round lets residual drift accumulate and forks
// decode tokens. The reduction runs in fp32 (at::opmath_type).
//
// WARNING: warpSize is 64 on MetaX C500. The __shfl_down_sync masks below use
// 64 ones; using 32 silently produces wrong reductions.

#pragma once

#include <cmath>
#include <cstdint>
#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>
#include <c10/util/Exception.h>

#include "metax_elementwise_v2.cuh"

namespace at::native::flagos {

namespace {

// ============================================================================
// GPU kernel: per-row fused residual-add + RMSNorm
// ============================================================================
// Same one-block-per-row two-pass layout as RmsNormRowKernel. Pass 1 writes
// new_residual = round(residual + hidden) and accumulates its square; pass 2
// re-reads new_residual and writes the normalized output.

template <typename scalar_t, typename acc_t>
__global__ void AddRmsNormRowKernel(
    int64_t rows,                          // total rows = batch * seq_len
    int64_t hidden,                        // last-dim size
    acc_t eps,
    scalar_t* __restrict__ normed,         // out 1: normalized
    scalar_t* __restrict__ new_residual,   // out 2: residual + hidden (dtype)
    const scalar_t* __restrict__ res_in,
    const scalar_t* __restrict__ hid_in,
    const scalar_t* __restrict__ weight)   // length == hidden
{
  constexpr int BLOCK = 256;
  constexpr int WARP = 64;
  constexpr int WARPS = BLOCK / WARP;  // = 4

  __shared__ acc_t warp_sums[WARPS];
  __shared__ acc_t scale_sh;
  const int lane = threadIdx.x & (WARP - 1);
  const int warp_id = threadIdx.x / WARP;

  // Grid-stride over rows: gridDim.x is clamped to kMaxGridV2, so for
  // rows > kMaxGridV2 (large prefill) each block sweeps multiple rows instead
  // of leaving the tail uncomputed.
  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < rows;
       row += static_cast<int64_t>(gridDim.x)) {
    const scalar_t* __restrict__ row_res = res_in + row * hidden;
    const scalar_t* __restrict__ row_hid = hid_in + row * hidden;
    scalar_t* __restrict__ row_normed = normed + row * hidden;
    scalar_t* __restrict__ row_newres = new_residual + row * hidden;

    // Pass 1: new_residual = round(residual + hidden); accumulate sum(nr^2).
    acc_t thread_sum = acc_t(0);
    for (int64_t i = threadIdx.x; i < hidden; i += BLOCK) {
      // Add in fp32 then round to dtype, matching HF's fp16 add exactly.
      const acc_t sum_f =
          static_cast<acc_t>(__ldg(row_res + i)) +
          static_cast<acc_t>(__ldg(row_hid + i));
      const scalar_t nr = static_cast<scalar_t>(sum_f);
      row_newres[i] = nr;
      // Square the rounded value so it matches the pass-2 re-read.
      const acc_t v = static_cast<acc_t>(nr);
      thread_sum += v * v;
    }

    // Warp-level shuffle reduction (WARP=64, mask = 64 ones).
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
      thread_sum += __shfl_down_sync(0xffffffffffffffffull, thread_sum, offset);
    }

    // Cross-warp reduction via shared memory.
    if (lane == 0) {
      warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // warp 0 finishes and computes scale = rsqrt(mean + eps).
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

    // Pass 2: normed = weight * new_residual * scale.
    for (int64_t i = threadIdx.x; i < hidden; i += BLOCK) {
      const acc_t x = static_cast<acc_t>(row_newres[i]);
      const acc_t w = static_cast<acc_t>(__ldg(weight + i));
      row_normed[i] = static_cast<scalar_t>(w * x * scale);
    }

    // Barrier before the next row reuses warp_sums / scale_sh.
    __syncthreads();
  }
}

// ============================================================================
// Host launcher
// ============================================================================

template <typename scalar_t, typename acc_t>
void LaunchAddRmsNorm(
    int64_t rows,
    int64_t hidden,
    acc_t eps,
    scalar_t* normed_ptr,
    scalar_t* new_residual_ptr,
    const scalar_t* res_ptr,
    const scalar_t* hid_ptr,
    const scalar_t* weight_ptr,
    cudaStream_t stream) {
  constexpr int BLOCK = 256;

  if (rows == 0 || hidden == 0) {
    return;
  }

  const int blocks =
      static_cast<int>(std::min<int64_t>(rows, metax::kMaxGridV2));

  AddRmsNormRowKernel<scalar_t, acc_t><<<blocks, BLOCK, 0, stream>>>(
      rows, hidden, eps, normed_ptr, new_residual_ptr, res_ptr, hid_ptr,
      weight_ptr);

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX add_rms_norm v2 launch failed: ",
      cudaGetErrorString(err));
}

} // namespace

// ============================================================================
// Public API: fused residual-add + RMSNorm, returns (normed, new_residual)
// ============================================================================
// residual and hidden must share shape and dtype; weight is 1-D of length ==
// last dim. new_residual = residual + hidden (model dtype) feeds the next
// residual link.

inline std::tuple<at::Tensor, at::Tensor> AddRmsNormKernelV2(
    const at::Tensor& residual,
    const at::Tensor& hidden,
    const at::Tensor& weight,
    double eps) {
  TORCH_CHECK(hidden.dim() >= 1, "add_rms_norm: hidden must have >= 1 dim");
  TORCH_CHECK(
      residual.sizes().equals(hidden.sizes()),
      "add_rms_norm: residual and hidden must have the same shape (residual ",
      residual.sizes(), " vs hidden ", hidden.sizes(), ")");
  TORCH_CHECK(weight.dim() == 1, "add_rms_norm: weight must be 1-D");

  const int64_t hsize = hidden.size(hidden.dim() - 1);
  TORCH_CHECK(
      weight.size(0) == hsize,
      "add_rms_norm: weight size (", weight.size(0),
      ") must match last dim of hidden (", hsize, ")");
  TORCH_CHECK(
      residual.scalar_type() == hidden.scalar_type() &&
          hidden.scalar_type() == weight.scalar_type(),
      "add_rms_norm: residual, hidden and weight must share dtype");

  const at::Tensor res = residual.is_contiguous() ? residual : residual.contiguous();
  const at::Tensor hid = hidden.is_contiguous() ? hidden : hidden.contiguous();
  const at::Tensor w = weight.is_contiguous() ? weight : weight.contiguous();

  at::Tensor normed = at::empty_like(hid);
  at::Tensor new_residual = at::empty_like(hid);

  if (hid.numel() == 0) {
    return std::make_tuple(normed, new_residual);
  }

  const int64_t rows = hid.numel() / hsize;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      hid.scalar_type(),
      "add_rms_norm_metax_v2",
      [&]() {
        using acc_t = at::opmath_type<scalar_t>;
        LaunchAddRmsNorm<scalar_t, acc_t>(
            rows,
            hsize,
            static_cast<acc_t>(eps),
            normed.data_ptr<scalar_t>(),
            new_residual.data_ptr<scalar_t>(),
            res.data_ptr<scalar_t>(),
            hid.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            metax::CurrentStream());
      });

  return std::make_tuple(normed, new_residual);
}

} // namespace at::native::flagos
