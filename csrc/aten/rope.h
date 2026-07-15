// Copyright (c) 2026, BAAI. All rights reserved.
//
// Host entry for the fused MetaX v2 RoPE custom op (flagos::rope).
// Declared as a plain function (no CUDA types) so register.cc, compiled by the
// host C++ compiler, can bind the custom op to it without including the .cuh.
// The definition lives in backends/metax_v2/rope_v2.cu (compiled by mxcc).

#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::flagos {

// Fused rotary position embedding. Equivalent to HuggingFace's
// apply_rotary_pos_emb (unsqueeze cos/sin + rotate_half + mul/add) but as a
// single kernel launch per (q, k) pair. Computes in fp32 internally; q/k stay
// in the model dtype (fp16/bf16/fp32). cos/sin may be fp16 or fp32 — they are
// cast to q's opmath_type (fp32) inside the kernel.
//
// Semantics per head row of length d (half = d/2):
//   out[i]       = q[i]*cos[i]       - q[i+half]*sin[i]        (i in [0, half))
//   out[i+half]  = q[i+half]*cos[i+half] + q[i]*sin[i+half]
// Same for k. cos/sin are expected pre-broadcast to q's last dim; the wrapper
// unsqueezes them along unsqueeze_dim so they broadcast over the head dim.
std::tuple<at::Tensor, at::Tensor> RoPEV2(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t unsqueeze_dim);

} // namespace at::native::flagos
