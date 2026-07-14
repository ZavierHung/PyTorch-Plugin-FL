// Copyright (c) 2026, BAAI. All rights reserved.
//
// mxcc-compiled definition of the fused residual-add + RMSNorm host entry
// declared in csrc/aten/add_rms_norm.h. Thin wrapper over the kernel in
// add_rms_norm_kernel_v2.cuh.

#include "../../add_rms_norm.h"

#include "add_rms_norm_kernel_v2.cuh"

namespace at::native::flagos {

std::tuple<at::Tensor, at::Tensor> AddRmsNormV2(
    const at::Tensor& residual,
    const at::Tensor& hidden,
    const at::Tensor& weight,
    double eps) {
  return AddRmsNormKernelV2(residual, hidden, weight, eps);
}

} // namespace at::native::flagos
