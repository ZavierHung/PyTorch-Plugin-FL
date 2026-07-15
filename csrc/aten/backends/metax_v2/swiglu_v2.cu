// Copyright (c) 2026, BAAI. All rights reserved.
//
// mxcc-compiled definition of the fused SwiGLU host entry declared in
// csrc/aten/swiglu.h. Thin wrapper over the kernel in swiglu_kernel_v2.cuh.

#include "../../swiglu.h"

#include "swiglu_kernel_v2.cuh"

namespace at::native::flagos {

at::Tensor SwiGLUV2(const at::Tensor& gate, const at::Tensor& up) {
  return SwiGLUKernelV2(gate, up);
}

} // namespace at::native::flagos
