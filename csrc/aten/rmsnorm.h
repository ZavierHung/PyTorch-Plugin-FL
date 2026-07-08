// Copyright (c) 2026, BAAI. All rights reserved.
//
// Host entry for the fused MetaX v2 RMSNorm custom op (flagos::rms_norm).
// Declared as a plain function (no CUDA types) so register.cc, compiled by the
// host C++ compiler, can bind the custom op to it without including the .cuh.
// The definition lives in backends/metax_v2/rmsnorm_v2.cu (compiled by mxcc).

#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::flagos {

// out = weight * x * rsqrt(mean(x^2, -1) + eps). Reduction in fp32; input and
// output stay in the model dtype (fp16/bf16/fp32). input and weight must share
// dtype and weight.size(0) must equal input's last dim.
at::Tensor RmsNormV2(const at::Tensor& input, const at::Tensor& weight,
                     double eps);

} // namespace at::native::flagos
