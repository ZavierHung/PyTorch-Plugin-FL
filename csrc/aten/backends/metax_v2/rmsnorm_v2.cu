// Copyright (c) 2026, BAAI. All rights reserved.
//
// mxcc-compiled definition of the fused RMSNorm host entry declared in
// csrc/aten/rmsnorm.h. Thin wrapper over the kernel in rmsnorm_kernel_v2.cuh.

#include "../../rmsnorm.h"

#include "rmsnorm_kernel_v2.cuh"

namespace at::native::flagos {

at::Tensor RmsNormV2(const at::Tensor& input, const at::Tensor& weight,
                     double eps) {
  return RmsNormKernelV2(input, weight, eps);
}

} // namespace at::native::flagos
