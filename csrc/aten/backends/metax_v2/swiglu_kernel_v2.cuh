// Copyright (c) 2026, BAAI. All rights reserved.
//
// MetaX v2 fused SwiGLU CUDA kernel.
//
//   out = silu(gate) * up,  silu(x) = x / (1 + exp(-x))
//
// Fuses HF MLP's silu(gate) then mul(up) (2 launches) into one. silu runs in
// fp32 (at::opmath_type); the product is rounded back to the model dtype
// (fp16/bf16), matching HF elementwise. Pure elementwise, no reduction, so
// warpSize is irrelevant; a grid-stride loop covers all elements.

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

// gate/up/out are contiguous and same-shape (host-guaranteed), so they share
// one linear index.
template <typename scalar_t, typename acc_t>
__global__ void SwiGLUContigKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ gate,  // gate_proj output
    const scalar_t* __restrict__ up)    // up_proj output
{
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; idx < n; idx += stride) {
    const acc_t g = static_cast<acc_t>(__ldg(gate + idx));
    const acc_t u = static_cast<acc_t>(__ldg(up + idx));
    const acc_t silu = g / (acc_t(1) + ::exp(-g));  // fp32, then round on store
    out[idx] = static_cast<scalar_t>(silu * u);
  }
}

} // namespace

// ============================================================================
// Public API: fused SwiGLU (out = silu(gate) * up)
// ============================================================================
// gate and up must share shape and dtype (fp16/bf16/fp32). Returns a new
// tensor like gate.

inline at::Tensor SwiGLUKernelV2(
    const at::Tensor& gate,
    const at::Tensor& up) {
  TORCH_CHECK(
      gate.sizes().equals(up.sizes()),
      "swiglu: gate and up must have the same shape (gate ",
      gate.sizes(), " vs up ", up.sizes(), ")");
  TORCH_CHECK(
      gate.scalar_type() == up.scalar_type(),
      "swiglu: gate and up must share dtype");

  const at::Tensor g = gate.is_contiguous() ? gate : gate.contiguous();
  const at::Tensor u = up.is_contiguous() ? up : up.contiguous();

  at::Tensor output = at::empty_like(g);

  const int64_t n = g.numel();
  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      g.scalar_type(),
      "swiglu_metax_v2",
      [&]() {
        using acc_t = at::opmath_type<scalar_t>;
        metax::Launch1dGridStrideV2(
            n,
            SwiGLUContigKernel<scalar_t, acc_t>,
            output.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(),
            u.data_ptr<scalar_t>());
      });

  return output;
}

} // namespace at::native::flagos
