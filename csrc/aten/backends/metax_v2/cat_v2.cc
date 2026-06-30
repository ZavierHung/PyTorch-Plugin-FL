// Copyright (c) 2026, BAAI. All rights reserved.
// MetaX cat v2: CUDA structured-cat style — validate all_contiguous then bulk copy.

#include "../../cat.h"

#include <include/flagos.h>

#include <vector>

namespace at::native::flagos {

namespace {

bool AllSameDtypeDevice(const std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    return false;
  }
  const auto& ref = tensors.front();
  for (const auto& t : tensors) {
    if (t.scalar_type() != ref.scalar_type() || t.device() != ref.device()) {
      return false;
    }
  }
  return true;
}

at::Tensor CatKernelMetaxV2(const at::ITensorListRef& tensors, int64_t dim) {
  std::vector<at::Tensor> materialized;
  materialized.reserve(tensors.size());
  bool all_contiguous = true;
  for (const auto& tensor : tensors) {
    all_contiguous = all_contiguous && tensor.is_contiguous();
    materialized.push_back(tensor);
  }

  TORCH_CHECK(!materialized.empty(), "MetaX cat v2: expected a non-empty TensorList");

  int64_t ref_idx = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(materialized.size()); ++i) {
    if (!(materialized[i].dim() == 1 && materialized[i].numel() == 0)) {
      ref_idx = i;
      break;
    }
  }
  if (ref_idx < 0) {
    return materialized[0].clone();
  }

  if (!all_contiguous || !AllSameDtypeDevice(materialized)) {
    for (auto& t : materialized) {
      if (!t.is_contiguous()) {
        t = t.contiguous();
      }
    }
    all_contiguous = true;
  }

  const auto& first = materialized[ref_idx];
  dim = at::maybe_wrap_dim(dim, first.dim());

  int64_t out_dim_size = 0;
  for (const auto& tensor : materialized) {
    if (tensor.dim() == 1 && tensor.numel() == 0) {
      continue;
    }
    for (int64_t d = 0; d < first.dim(); ++d) {
      if (d == dim) {
        continue;
      }
      TORCH_CHECK(tensor.size(d) == first.size(d), "size mismatch");
    }
    out_dim_size += tensor.size(dim);
  }

  auto out_sizes = first.sizes().vec();
  out_sizes[dim] = out_dim_size;
  at::Tensor out = at::empty(out_sizes, first.options());
  if (out.numel() == 0) {
    return out;
  }

  const size_t elem_size = static_cast<size_t>(first.element_size());
  const int64_t ndim = first.dim();
  int64_t post_dim = 1;
  for (int64_t d = dim + 1; d < ndim; ++d) {
    post_dim *= first.size(d);
  }
  int64_t pre_dim = 1;
  for (int64_t d = 0; d < dim; ++d) {
    pre_dim *= first.size(d);
  }

  if (all_contiguous && post_dim == 1) {
    char* dst = static_cast<char*>(out.data_ptr());
    size_t written = 0;
    for (const auto& tensor : materialized) {
      if (tensor.dim() == 1 && tensor.numel() == 0) {
        continue;
      }
      const size_t nbytes = static_cast<size_t>(tensor.numel()) * elem_size;
      if (nbytes > 0) {
        Memcpy(dst + written, tensor.data_ptr(), nbytes, MemcpyDeviceToDevice);
        written += nbytes;
      }
    }
    return out;
  }

  const int64_t chunk_stride = post_dim;
  for (int64_t p = 0; p < pre_dim; ++p) {
    int64_t offset = 0;
    for (const auto& tensor : materialized) {
      if (tensor.dim() == 1 && tensor.numel() == 0) {
        continue;
      }
      const int64_t chunk = tensor.size(dim);
      const size_t nbytes =
          static_cast<size_t>(chunk * post_dim) * elem_size;
      if (nbytes == 0) {
        offset += chunk;
        continue;
      }
      const char* src = static_cast<const char*>(tensor.data_ptr()) +
          static_cast<size_t>(p * tensor.size(dim) * post_dim) * elem_size;
      char* dst = static_cast<char*>(out.data_ptr()) +
          static_cast<size_t>((p * out_dim_size + offset) * post_dim) * elem_size;
      Memcpy(dst, src, nbytes, MemcpyDeviceToDevice);
      offset += chunk;
    }
  }
  return out;
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(CatFn, cat_dispatcher, Backend::kMetaxV2, CatKernelMetaxV2)

} // namespace at::native::flagos
