// Copyright (c) 2026, BAAI. All rights reserved.
//
// Host entry for the fused MetaX v2 residual-add + RMSNorm custom op
// (flagos::add_rms_norm). Declared as a plain function (no CUDA types) so
// register.cc, compiled by the host C++ compiler, can bind the custom op to it
// without including the .cuh. The definition lives in
// backends/metax_v2/add_rms_norm_v2.cu (compiled by mxcc).

#pragma once

#include <tuple>

#include <ATen/core/Tensor.h>

namespace at::native::flagos {

// Fused residual-add + RMSNorm. Replaces the HuggingFace decoder-layer chain
//   new_residual = residual + hidden      (fp16 elementwise add)
//   normed = rms_norm(new_residual)       (pow/mean/rsqrt/mul chain)
// with a single kernel. The add is rounded to the model dtype BEFORE feeding
// the RMSNorm reduction (matching HF's fp16 residual accumulation exactly), and
// the RMSNorm reduction runs in fp32.
//
// residual and hidden must share shape and dtype; weight must be 1-D with size
// equal to the last dim. Returns (normed, new_residual): the normalized output
// (fed to the next sub-layer) and the updated residual (fp16, carried forward).
std::tuple<at::Tensor, at::Tensor> AddRmsNormV2(
    const at::Tensor& residual,
    const at::Tensor& hidden,
    const at::Tensor& weight,
    double eps);

} // namespace at::native::flagos
