// Copyright (c) 2026, BAAI. All rights reserved.
//
// Host entry for the fused MetaX v2 SwiGLU custom op (flagos::swiglu).
// Declared as a plain function (no CUDA types) so register.cc, compiled by the
// host C++ compiler, can bind the custom op to it without including the .cuh.
// The definition lives in backends/metax_v2/swiglu_v2.cu (compiled by mxcc).

#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::flagos {

// Fused SwiGLU gated activation: out = silu(gate) * up, where
// silu(x) = x / (1 + exp(-x)). Replaces HuggingFace MLP's
// ``self.act_fn(self.gate_proj(x)) * self.up_proj(x)`` (a separate silu launch
// followed by a separate elementwise-mul launch) with a single kernel. silu is
// computed in fp32 internally and the product is cast back to the model dtype —
// numerically equivalent to the HF fp16 path. gate and up must share shape and
// dtype (fp16/bf16/fp32). Returns a new tensor of the same shape and dtype.
at::Tensor SwiGLUV2(const at::Tensor& gate, const at::Tensor& up);

} // namespace at::native::flagos
