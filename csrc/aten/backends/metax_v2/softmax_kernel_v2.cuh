// Copyright (c) 2026, BAAI. All rights reserved.
//
// MetaX v2 fused masked-softmax CUDA kernel.
//
// Per row of attn_weights (last dim k_len):
//   s[i]   = x[i] * scaling + mask[i]   (mask optional, broadcast via strides)
//   out[i] = exp(s[i] - max_i s) / sum_i exp(s - max)  -> cast to input dtype
//
// Fuses HF eager_attention_forward's scale + mask-add + softmax(fp32) + cast
// (6~10 launches) into one. I/O keeps the model dtype; max/exp/sum run in fp32
// (at::opmath_type), matching softmax(dtype=float32).to(input_dtype).
//
// Three passes (softmax needs two reductions, max then sum); the row is
// re-read from L2 each pass rather than held in registers, since k_len can be
// large (long-context decode).
//
// WARNING: warpSize is 64 on MetaX C500. The __shfl_down_sync masks below use
// 64 ones; using 32 silently produces wrong reductions.

#pragma once

#include <cmath>
#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include "metax_elementwise_v2.cuh"

namespace at::native::flagos {

namespace {

// ============================================================================
// GPU kernel: per-row fused masked softmax
// ============================================================================
// One block per row (last dim k_len). row -> (b, h, q) with
//   q = row % q_len;  h = (row / q_len) % heads;  b = row / (heads*q_len).
// mask element index = b*ms0 + h*ms1 + q*ms2 + k*ms3; a broadcast dim has
// stride 0 (PyTorch broadcasting). mask == nullptr means no mask.

template <typename scalar_t, typename acc_t, typename mask_t>
__global__ void MaskedSoftmaxKernel(
    int64_t rows,                 // total rows = batch * heads * q_len
    int64_t k_len,                // last dim (reduction dim)
    int64_t q_len,                // for row -> (b,h,q) decode
    int64_t heads,
    acc_t scaling,                // 1 / sqrt(head_dim)
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    const mask_t* __restrict__ mask,  // may be nullptr
    int64_t ms0, int64_t ms1, int64_t ms2, int64_t ms3)  // mask 4-D strides (elements)
{
  constexpr int BLOCK = 256;
  constexpr int WARP = 64;
  constexpr int WARPS = BLOCK / WARP;

  __shared__ acc_t warp_sums[WARPS];
  __shared__ acc_t max_sh;
  __shared__ acc_t sum_sh;
  const int lane = threadIdx.x & (WARP - 1);
  const int warp_id = threadIdx.x / WARP;

  // Grid-stride over rows: gridDim.x is clamped to kMaxGridV2, so for
  // rows > kMaxGridV2 (large batch/prefill) each block sweeps multiple rows
  // instead of leaving the tail uninitialized.
  for (int64_t row = static_cast<int64_t>(blockIdx.x); row < rows;
       row += static_cast<int64_t>(gridDim.x)) {
    // row -> (b, h, q)
    const int64_t q = row % q_len;
    const int64_t h = (row / q_len) % heads;
    const int64_t b = row / (q_len * heads);
    const int64_t mask_row_base =
        b * ms0 + h * ms1 + q * ms2;  // + k * ms3 added in the loop

    const scalar_t* __restrict__ row_in = in + row * k_len;
    scalar_t* __restrict__ row_out = out + row * k_len;

    // Pass 1: row max in fp32, after applying scaling + mask.
    acc_t thread_max = -INFINITY;
    for (int64_t i = threadIdx.x; i < k_len; i += BLOCK) {
      acc_t v = static_cast<acc_t>(__ldg(row_in + i)) * scaling;
      if (mask != nullptr) {
        v += static_cast<acc_t>(__ldg(mask + mask_row_base + i * ms3));
      }
      if (v > thread_max) {
        thread_max = v;
      }
    }

    // warp-level shuffle reduction (max)
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
      acc_t other = __shfl_down_sync(0xffffffffffffffffull, thread_max, offset);
      if (other > thread_max) {
        thread_max = other;
      }
    }

    if (lane == 0) {
      warp_sums[warp_id] = thread_max;
    }
    __syncthreads();

    if (warp_id == 0) {
      acc_t v = (threadIdx.x < WARPS) ? warp_sums[threadIdx.x] : -INFINITY;
      for (int offset = WARPS / 2; offset > 0; offset >>= 1) {
        acc_t other = __shfl_down_sync(0xffffffffffffffffull, v, offset);
        if (other > v) {
          v = other;
        }
      }
      if (threadIdx.x == 0) {
        max_sh = v;
      }
    }
    __syncthreads();
    const acc_t row_max = max_sh;

    // Pass 2: sum(exp(s - max)) in fp32.
    acc_t thread_sum = acc_t(0);
    for (int64_t i = threadIdx.x; i < k_len; i += BLOCK) {
      acc_t v = static_cast<acc_t>(__ldg(row_in + i)) * scaling;
      if (mask != nullptr) {
        v += static_cast<acc_t>(__ldg(mask + mask_row_base + i * ms3));
      }
      thread_sum += ::exp(v - row_max);
    }

    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
      thread_sum += __shfl_down_sync(0xffffffffffffffffull, thread_sum, offset);
    }

    if (lane == 0) {
      warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
      acc_t v = (threadIdx.x < WARPS) ? warp_sums[threadIdx.x] : acc_t(0);
      for (int offset = WARPS / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffffffffffull, v, offset);
      }
      if (threadIdx.x == 0) {
        sum_sh = v;
      }
    }
    __syncthreads();
    const acc_t row_sum = sum_sh;

    // Pass 3: write exp(s - max) / sum, cast to scalar_t.
    const acc_t inv_sum = (row_sum > acc_t(0)) ? (acc_t(1) / row_sum) : acc_t(0);
    for (int64_t i = threadIdx.x; i < k_len; i += BLOCK) {
      acc_t v = static_cast<acc_t>(__ldg(row_in + i)) * scaling;
      if (mask != nullptr) {
        v += static_cast<acc_t>(__ldg(mask + mask_row_base + i * ms3));
      }
      row_out[i] = static_cast<scalar_t>(::exp(v - row_max) * inv_sum);
    }

    // Barrier before the next row reuses warp_sums / max_sh / sum_sh.
    __syncthreads();
  }
}

// ============================================================================
// Host launcher (single dtype/mask combination)
// ============================================================================

template <typename scalar_t, typename acc_t, typename mask_t>
void LaunchMaskedSoftmax(
    int64_t rows,
    int64_t k_len,
    int64_t q_len,
    int64_t heads,
    acc_t scaling,
    scalar_t* out_ptr,
    const scalar_t* in_ptr,
    const mask_t* mask_ptr,
    int64_t ms0,
    int64_t ms1,
    int64_t ms2,
    int64_t ms3,
    cudaStream_t stream) {
  constexpr int BLOCK = 256;
  if (rows == 0 || k_len == 0) {
    return;
  }
  const int blocks =
      static_cast<int>(std::min<int64_t>(rows, metax::kMaxGridV2));
  MaskedSoftmaxKernel<scalar_t, acc_t, mask_t>
      <<<blocks, BLOCK, 0, stream>>>(
          rows, k_len, q_len, heads, scaling, out_ptr, in_ptr, mask_ptr,
          ms0, ms1, ms2, ms3);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX masked_softmax v2 launch failed: ",
      cudaGetErrorString(err));
}

// ============================================================================
// Public API: fused masked softmax (PyTorch Tensor interface)
// ============================================================================

inline at::Tensor MaskedSoftmaxKernelV2(
    const at::Tensor& attn_weights,
    const c10::optional<at::Tensor>& mask_opt,
    double scaling) {
  TORCH_CHECK(attn_weights.dim() >= 2, "masked_softmax: input must have >= 2 dims");
  const int64_t k_len = attn_weights.size(attn_weights.dim() - 1);
  // View input as [rows, k_len]; row -> (b,h,q) decode needs q_len and heads.
  // For 4-D [batch, heads, q_len, k_len]: q_len = size(-2), heads = size(-3).
  // Lower rank degrades to heads=1 and q_len = size(-2) (or 1 if <3-D).
  int64_t q_len = 1;
  int64_t heads = 1;
  if (attn_weights.dim() >= 3) {
    q_len = attn_weights.size(attn_weights.dim() - 2);
  }
  if (attn_weights.dim() >= 4) {
    heads = attn_weights.size(attn_weights.dim() - 3);
  }
  const int64_t rows = attn_weights.numel() / k_len;

  const at::Tensor in =
      attn_weights.is_contiguous() ? attn_weights : attn_weights.contiguous();
  at::Tensor out = at::empty_like(in);
  if (in.numel() == 0) {
    return out;
  }

  const bool has_mask = mask_opt.has_value() && mask_opt->defined();
  at::Tensor mask;
  if (has_mask) {
    mask = *mask_opt;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      in.scalar_type(),
      "masked_softmax_metax_v2",
      [&]() {
        using acc_t = at::opmath_type<scalar_t>;

        // mask_t: prefer input dtype (no cast); else fp32; else cast to fp32
        // (one copy launch, rare). mask strides are in elements, broadcast
        // dims have stride 0.
        int64_t ms0 = 0, ms1 = 0, ms2 = 0, ms3 = 0;
        const void* mask_ptr_void = nullptr;

        if (has_mask) {
          // Take the mask's last 4 dims' strides; missing high dims are size-1
          // (stride 0).
          auto mask_strides = mask.strides();
          auto mask_sizes = mask.sizes();
          const int64_t nd = mask_sizes.size();
          // ms3 = last dim stride, ms2 = 2nd-last, ms1 = 3rd-last, ms0 = 4th-last
          int64_t m_stride[4] = {0, 0, 0, 0};
          int64_t m_size[4] = {1, 1, 1, 1};
          for (int i = 0; i < 4; ++i) {
            int src = static_cast<int>(nd) - 1 - i;
            if (src >= 0) {
              m_size[3 - i] = mask_sizes[src];
              m_stride[3 - i] = mask_strides[src];
            }
          }
          // broadcast: size==1 -> stride 0
          if (m_size[0] == 1) ms0 = 0; else ms0 = m_stride[0];
          if (m_size[1] == 1) ms1 = 0; else ms1 = m_stride[1];
          if (m_size[2] == 1) ms2 = 0; else ms2 = m_stride[2];
          if (m_size[3] == 1) ms3 = 0; else ms3 = m_stride[3];
          mask_ptr_void = mask.data_ptr();
        }

        if (!has_mask) {
          LaunchMaskedSoftmax<scalar_t, acc_t, scalar_t>(
              rows, k_len, q_len, heads,
              static_cast<acc_t>(scaling),
              out.data_ptr<scalar_t>(),
              in.data_ptr<scalar_t>(),
              static_cast<const scalar_t*>(nullptr),
              0, 0, 0, 0,
              metax::CurrentStream());
        } else if (mask.scalar_type() ==
                   c10::CppTypeToScalarType<scalar_t>::value) {
          LaunchMaskedSoftmax<scalar_t, acc_t, scalar_t>(
              rows, k_len, q_len, heads,
              static_cast<acc_t>(scaling),
              out.data_ptr<scalar_t>(),
              in.data_ptr<scalar_t>(),
              static_cast<const scalar_t*>(mask_ptr_void),
              ms0, ms1, ms2, ms3,
              metax::CurrentStream());
        } else if (mask.scalar_type() == at::ScalarType::Float) {
          LaunchMaskedSoftmax<scalar_t, acc_t, float>(
              rows, k_len, q_len, heads,
              static_cast<acc_t>(scaling),
              out.data_ptr<scalar_t>(),
              in.data_ptr<scalar_t>(),
              static_cast<const float*>(mask_ptr_void),
              ms0, ms1, ms2, ms3,
              metax::CurrentStream());
        } else {
          // rare dtype: cast to fp32 then take the float path (one copy launch).
          at::Tensor mask_f = mask.to(at::ScalarType::Float);
          LaunchMaskedSoftmax<scalar_t, acc_t, float>(
              rows, k_len, q_len, heads,
              static_cast<acc_t>(scaling),
              out.data_ptr<scalar_t>(),
              in.data_ptr<scalar_t>(),
              mask_f.data_ptr<float>(),
              ms0, ms1, ms2, ms3,
              metax::CurrentStream());
        }
      });

  return out;
}

} // namespace

} // namespace at::native::flagos
