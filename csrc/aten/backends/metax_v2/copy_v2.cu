// MetaX v2 strided copy kernel (copy/contiguous path).
#include "copy_path_v2.h"

#include <include/flagos.h>

#include "metax_elementwise_v2.cuh"

namespace at::native::flagos {

namespace {

__global__ void StridedCopyBytesKernel(
    int64_t n,
    char* __restrict__ dst,
    const char* __restrict__ src,
    int64_t dst_stride_bytes,
    int64_t src_stride_bytes) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dst[idx * dst_stride_bytes] = src[idx * src_stride_bytes];
}

} // namespace

void MetaxStridedCopyV2(const at::Tensor& src, at::Tensor& dst) {
  TORCH_CHECK(src.sizes().equals(dst.sizes()), "MetaX copy v2: shape mismatch");
  TORCH_CHECK(
      src.scalar_type() == dst.scalar_type(),
      "MetaX copy v2: dtype mismatch");
  const int64_t n = src.numel();
  if (n == 0) {
    return;
  }
  const size_t elem_size = static_cast<size_t>(src.element_size());

  if (src.is_contiguous() && dst.is_contiguous()) {
    Memcpy(
        dst.data_ptr(),
        src.data_ptr(),
        n * elem_size,
        MemcpyDeviceToDevice);
    return;
  }

  if (src.dim() == 1 && dst.dim() == 1 && !src.is_contiguous()) {
    const int64_t dst_stride_bytes =
        dst.stride(0) * static_cast<int64_t>(elem_size);
    const int64_t src_stride_bytes =
        src.stride(0) * static_cast<int64_t>(elem_size);
    metax::Launch1dV2(
        n,
        StridedCopyBytesKernel,
        static_cast<char*>(dst.data_ptr()),
        static_cast<const char*>(src.data_ptr()),
        dst_stride_bytes,
        src_stride_bytes);
    return;
  }

  const at::Tensor src_c = src.is_contiguous() ? src : src.contiguous();
  Memcpy(
      dst.data_ptr(),
      src_c.data_ptr(),
      static_cast<size_t>(n) * elem_size,
      MemcpyDeviceToDevice);
}

} // namespace at::native::flagos
