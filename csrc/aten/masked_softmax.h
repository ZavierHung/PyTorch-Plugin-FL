// Copyright (c) 2026, BAAI. All rights reserved.
//
// Host entry for the fused MetaX v2 masked-softmax custom op
// (flagos::masked_softmax). Declared as a plain function (no CUDA types) so
// register.cc, compiled by the host C++ compiler, can bind the custom op to it
// without including the .cuh. The definition lives in
// backends/metax_v2/softmax_v2.cu (compiled by mxcc).
//
// Note: distinct from csrc/aten/softmax.h (the aten::_softmax dispatcher);
// this header is for the fused flagos::masked_softmax custom op.

#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>

namespace at::native::flagos {

// Fused scale + mask + numerically-stable softmax + dtype cast, covering both
// the "softmax chain" (max/sub/exp/sum/div) and the attention's surrounding
// scale/mask/cast chain in HuggingFace eager_attention_forward:
//
//   attn_weights = (matmul(q, k^T) * scaling) [+ attention_mask]
//   attn_weights = nn.functional.softmax(attn_weights, dim=-1,
//                                        dtype=torch.float32).to(query.dtype)
//
// Single kernel: for each row over the last dim,
//   s = x * scaling + mask   (mask optional, broadcast via strides)
//   m = max(s)               (fp32 reduction, warpSize=64)
//   e = exp(s - m)
//   z = sum(e)               (fp32 reduction)
//   out = (e / z)  cast to input dtype
//
// Input `attn_weights` must be contiguous in the last dim with shape
// [batch, heads, q_len, k_len] (or any 2+-D tensor); reduction is over the last
// dim. `mask` may be None or broadcastable to `attn_weights.sizes()`.
// `scaling` is the 1/sqrt(head_dim) factor. Returns a new tensor of the same
// shape and dtype as `attn_weights`.
at::Tensor MaskedSoftmaxV2(
    const at::Tensor& attn_weights,
    const c10::optional<at::Tensor>& mask,
    double scaling);

} // namespace at::native::flagos
