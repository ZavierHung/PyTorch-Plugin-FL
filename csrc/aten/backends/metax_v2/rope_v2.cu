// Copyright (c) 2026, BAAI. All rights reserved.
//
// mxcc-compiled definition of the fused RoPE host entry declared in
// csrc/aten/rope.h. Thin wrapper over the kernel in rope_kernel_v2.cuh.

#include "../../rope.h"

#include "rope_kernel_v2.cuh"

namespace at::native::flagos {

std::tuple<at::Tensor, at::Tensor> RoPEV2(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t unsqueeze_dim) {
  return RoPEKernelV2(q, k, cos, sin, unsqueeze_dim);
}

} // namespace at::native::flagos
