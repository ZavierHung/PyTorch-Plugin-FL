// Copyright (c) 2026, BAAI. All rights reserved.
// MetaX cat v2: fused-2D-grid kernel for non-fast-path cat, single bulk
// memcpy for fast path (post_dim==1, all contiguous).

#include "../../cat.h"

#include <include/flagos.h>

#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace at::native::flagos {

// Defined in cat_v2_fused.cu.
void LaunchCatFusedKernel(
    char* out_base,
    const char* in_base,
    int64_t elem_size,
    int64_t post_dim,
    int64_t chunk_bytes,
    int64_t out_dim_offset,
    int64_t out_row_stride,
    int64_t in_row_stride,
    int64_t pre_dim,
    cudaStream_t stream);

namespace {

at::Tensor CatKernelMetaxV2(const at::ITensorListRef& tensors, int64_t dim) {
  // Avoid std::vector<at::Tensor> materialization: tensors is already a
  // reference list; collect ptrs/sizes lazily.
  TORCH_CHECK(!tensors.empty(), "MetaX cat v2: expected a non-empty TensorList");

  bool all_contiguous = true;
  const at::Tensor* first_nonempty = nullptr;
  for (const auto& t : tensors) {
    all_contiguous = all_contiguous && t.is_contiguous();
    if (!first_nonempty && !(t.dim() == 1 && t.numel() == 0)) {
      first_nonempty = &t;
    }
  }
  if (!first_nonempty) {
    // All inputs are 1D-empty: clone the first (matches PyTorch semantics).
    at::Tensor first_empty;
    for (const auto& t : tensors) { first_empty = t; break; }
    return first_empty.clone();
  }
  const auto& first = *first_nonempty;
  bool all_same_dtype_device = true;
  for (const auto& t : tensors) {
    if (t.dim() == 1 && t.numel() == 0) {
      continue;
    }
    all_same_dtype_device = all_same_dtype_device &&
        (t.scalar_type() == first.scalar_type()) &&
        (t.device() == first.device());
  }
  // We only need same-dtype/device for our fused kernel.
  TORCH_CHECK(all_same_dtype_device, "MetaX cat v2: dtype/device mismatch");

  // If any input is non-contiguous, contiguous-ify it (one-time cost).
  std::vector<at::Tensor> contig_owned;
  std::vector<const at::Tensor*> views;
  contig_owned.reserve(tensors.size());
  views.reserve(tensors.size());
  if (!all_contiguous) {
    for (const auto& t : tensors) {
      if (!t.is_contiguous()) {
        contig_owned.push_back(t.contiguous());
        views.push_back(&contig_owned.back());
      } else {
        views.push_back(&t);
      }
    }
    all_contiguous = true;
  } else {
    for (const auto& t : tensors) {
      views.push_back(&t);
    }
  }

  dim = at::maybe_wrap_dim(dim, first.dim());
  TORCH_CHECK(dim < first.dim(), "MetaX cat v2: dimension out of range");

  int64_t out_dim_size = 0;
  for (const auto* tp : views) {
    const auto& tensor = *tp;
    if (tensor.dim() == 1 && tensor.numel() == 0) continue;
    TORCH_CHECK(tensor.dim() == first.dim(), "MetaX cat v2: dim mismatch");
    for (int64_t d = 0; d < first.dim(); ++d) {
      if (d == dim) continue;
      TORCH_CHECK(tensor.size(d) == first.size(d),
                  "MetaX cat v2: sizes must match except in cat dim");
    }
    out_dim_size += tensor.size(dim);
  }

  auto out_sizes = first.sizes().vec();
  out_sizes[dim] = out_dim_size;
  at::Tensor out = at::empty(out_sizes, first.options());
  if (out.numel() == 0) return out;

  const size_t elem_size = static_cast<size_t>(first.element_size());
  const int64_t ndim = first.dim();
  int64_t post_dim = 1;
  for (int64_t d = dim + 1; d < ndim; ++d) post_dim *= first.size(d);
  int64_t pre_dim = 1;
  for (int64_t d = 0; d < dim; ++d) pre_dim *= first.size(d);

  // Fast path: contiguous + post_dim==1 → single concatenated memcpy per
  // input tensor (already optimal — single DMA per input).
  if (all_contiguous && post_dim == 1) {
    char* dst = static_cast<char*>(out.data_ptr());
    size_t written = 0;
    for (const auto* tp : views) {
      const auto& tensor = *tp;
      if (tensor.dim() == 1 && tensor.numel() == 0) continue;
      const size_t nbytes = static_cast<size_t>(tensor.numel()) * elem_size;
      if (nbytes > 0) {
        Memcpy(dst + written, tensor.data_ptr(), nbytes, MemcpyDeviceToDevice);
        written += nbytes;
      }
    }
    return out;
  }

  // General path: fused 2D-grid kernel. Each block handles 1 input tensor
  // for 1 pre row + 1 post block. Avoids N×pre_dim Memcpy driver calls.
  int64_t out_dim_acc = 0;
  for (const auto* tp : views) {
    const auto& tensor = *tp;
    if (tensor.dim() == 1 && tensor.numel() == 0) {
      continue;
    }
    const int64_t chunk = tensor.size(dim);
    if (chunk == 0 || post_dim == 0) {
      out_dim_acc += chunk;
      continue;
    }
    const int64_t chunk_bytes = chunk * post_dim * elem_size;
    if (pre_dim > 0 && chunk_bytes > 0) {
      const char* in_base = static_cast<const char*>(tensor.data_ptr());
      char* out_base = static_cast<char*>(out.data_ptr());
      const int64_t out_row_stride = out_dim_size * post_dim * elem_size;
      const int64_t in_row_stride = chunk * post_dim * elem_size;
      LaunchCatFusedKernel(
          out_base, in_base,
          elem_size, post_dim, chunk_bytes,
          out_dim_acc, out_row_stride, in_row_stride,
          pre_dim, c10::cuda::getCurrentCUDAStream());
    }
    out_dim_acc += chunk;
  }
  return out;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(CatFn, cat_dispatcher, Backend::kMetaxV2, CatKernelMetaxV2)

} // namespace at::native::flagos