// Copyright (c) 2026, BAAI. All rights reserved.
//
// MetaX v2 fused RoPE CUDA kernel.
//
// Per head row (length d, half = d/2), i in [0, half):
//   out[i]      = q[i]*cos[i]      - q[i+half]*sin[i]
//   out[i+half] = q[i+half]*cos[i+half] + q[i]*sin[i+half]
//
// Fuses HF apply_rotary_pos_emb (unsqueeze/rotate_half/mul/add, 12~14
// launches) into 1~2. I/O keeps the model dtype; cos/sin are promoted to fp32
// (opmath_type) and all muls/adds run in fp32, matching HF's fp32 semantics.
// Pure elementwise (no reduction), so warpSize is irrelevant.

#pragma once

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
// GPU kernel: fused RoPE for one tensor (q or k)
// ============================================================================
// grid-stride over pairs: each thread handles one (i, i+half) pair.
// n_pairs = half * rows, rows = numel / d, half = d/2 (d must be even).
//
// cos/sin are host-broadcast to q's shape (unsqueeze then expand+contiguous to
// [batch,heads,seq,d]), so cos/sin share q's linear index in the kernel.

template <typename scalar_t, typename acc_t, typename cos_t>
__global__ void RoPEKernel(
    int64_t n_pairs,           // = half * rows
    int64_t half,              // d / 2
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const cos_t* __restrict__ cos,
    const cos_t* __restrict__ sin) {
  const int64_t stride =
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n_pairs;
       idx += stride) {
    // idx -> (row, j): j is the pair index within the row, in [0, half)
    const int64_t j = idx % half;
    const int64_t row = idx / half;
    const int64_t base = row * (half * 2);
    const int64_t i_lo = base + j;          // front half
    const int64_t i_hi = base + half + j;   // back half

    // read and promote to fp32
    const acc_t x_lo = static_cast<acc_t>(__ldg(x + i_lo));
    const acc_t x_hi = static_cast<acc_t>(__ldg(x + i_hi));
    const acc_t c_lo = static_cast<acc_t>(__ldg(cos + i_lo));
    const acc_t c_hi = static_cast<acc_t>(__ldg(cos + i_hi));
    const acc_t s_lo = static_cast<acc_t>(__ldg(sin + i_lo));
    const acc_t s_hi = static_cast<acc_t>(__ldg(sin + i_hi));

    // RoPE (rotate_half equivalent)
    out[i_lo] = static_cast<scalar_t>(x_lo * c_lo - x_hi * s_lo);
    out[i_hi] = static_cast<scalar_t>(x_hi * c_hi + x_lo * s_hi);
  }
}

// ============================================================================
// Host launcher (single tensor)
// ============================================================================

template <typename scalar_t, typename acc_t, typename cos_t>
void LaunchRoPE(
    int64_t rows,
    int64_t half,
    scalar_t* out_ptr,
    const scalar_t* x_ptr,
    const cos_t* cos_ptr,
    const cos_t* sin_ptr,
    cudaStream_t stream) {
  if (rows == 0 || half == 0) {
    return;
  }
  const int64_t n_pairs = half * rows;
  const int blocks = metax::ComputeGrid1d(n_pairs);
  if (blocks == 0) {
    return;
  }
  RoPEKernel<scalar_t, acc_t, cos_t>
      <<<blocks, metax::kBlockSizeV2, 0, stream>>>(
          n_pairs, half, out_ptr, x_ptr, cos_ptr, sin_ptr);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX rope v2 launch failed: ",
      cudaGetErrorString(err));
}

// ============================================================================
// Host helper: apply RoPE to one tensor
// ============================================================================
// Broadcasts cos/sin to x's shape, dispatches by dtype, launches the kernel.

template <typename scalar_t, typename acc_t>
at::Tensor ApplyRoPEOne(
    const at::Tensor& x,
    const at::Tensor& cos_in,
    const at::Tensor& sin_in) {
  using cos_t = acc_t;  // treat cos/sin as fp32 (acc_t = float for fp16/bf16)

  const int64_t d = x.size(x.dim() - 1);
  TORCH_CHECK(d % 2 == 0, "rope: last dim must be even, got ", d);
  const int64_t half = d / 2;
  const int64_t rows = x.numel() / d;

  // cos/sin must be broadcast to x's full shape and contiguous for linear
  // indexing.
  at::Tensor cos_b = BroadcastContiguousIfNeeded(cos_in, x.sizes());
  at::Tensor sin_b = BroadcastContiguousIfNeeded(sin_in, x.sizes());

  // cos/sin dtype may differ from x (commonly cos=fp32, x=fp16). Cast to acc_t
  // if needed (no-op if already acc_t).
  constexpr at::ScalarType kAccScalar = c10::CppTypeToScalarType<cos_t>::value;
  if (cos_b.scalar_type() != kAccScalar) {
    cos_b = cos_b.to(kAccScalar);
  }
  if (sin_b.scalar_type() != kAccScalar) {
    sin_b = sin_b.to(kAccScalar);
  }

  at::Tensor out = at::empty_like(x);
  if (out.numel() == 0) {
    return out;
  }

  LaunchRoPE<scalar_t, acc_t, cos_t>(
      rows,
      half,
      out.data_ptr<scalar_t>(),
      x.data_ptr<scalar_t>(),
      cos_b.data_ptr<cos_t>(),
      sin_b.data_ptr<cos_t>(),
      metax::CurrentStream());

  return out;
}

} // namespace

// ============================================================================
// Public API: fused RoPE, returns (q_embed, k_embed)
// ============================================================================

inline std::tuple<at::Tensor, at::Tensor> RoPEKernelV2(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t unsqueeze_dim) {
  TORCH_CHECK(q.dim() >= 2, "rope: q must have >= 2 dims");
  TORCH_CHECK(k.dim() == q.dim(), "rope: k must have same ndim as q");
  TORCH_CHECK(
      q.scalar_type() == k.scalar_type(),
      "rope: q and k must share dtype");

  // cos/sin: HF shape is typically [batch, seq, d]; unsqueeze(unsqueeze_dim)
  // gives [batch, 1, seq, d]. Broadcast to q/k's full [batch,heads,seq,d] in
  // the kernel path.
  at::Tensor cos_u = cos.unsqueeze(unsqueeze_dim);
  at::Tensor sin_u = sin.unsqueeze(unsqueeze_dim);

  // kernel assumes the last dim is contiguous (q/k usually already are).
  at::Tensor q_c = q.is_contiguous() ? q : q.contiguous();
  at::Tensor k_c = k.is_contiguous() ? k : k.contiguous();

  at::Tensor q_embed, k_embed;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q_c.scalar_type(),
      "rope_metax_v2",
      [&]() {
        using acc_t = at::opmath_type<scalar_t>;
        q_embed = ApplyRoPEOne<scalar_t, acc_t>(q_c, cos_u, sin_u);
        k_embed = ApplyRoPEOne<scalar_t, acc_t>(k_c, cos_u, sin_u);
      });

  return std::make_tuple(q_embed, k_embed);
}

} // namespace at::native::flagos
