// Copyright (c) 2026, BAAI. All rights reserved.
//
// mxcc-compiled definition of the fused masked-softmax host entry declared in
// csrc/aten/softmax.h. Thin wrapper over the kernel in softmax_kernel_v2.cuh.

#include "../../masked_softmax.h"

#include "softmax_kernel_v2.cuh"

namespace at::native::flagos {

at::Tensor MaskedSoftmaxV2(
    const at::Tensor& attn_weights,
    const c10::optional<at::Tensor>& mask,
    double scaling) {
  return MaskedSoftmaxKernelV2(attn_weights, mask, scaling);
}

} // namespace at::native::flagos
