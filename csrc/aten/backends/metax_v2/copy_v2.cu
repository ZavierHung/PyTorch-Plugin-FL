// MetaX v2 strided copy kernel (copy/contiguous path).
#include "copy_path_v2.h"

#include <include/flagos.h>

#include <ATen/cuda/detail/OffsetCalculator.cuh>

#include "metax_elementwise_v2.cuh"

namespace at::native::flagos {

namespace {

constexpr int kCopyBlockSize = 256;
constexpr size_t kVecCopyMinBytes = 1024;
constexpr size_t kVec4CopyMinBytes = 16384;

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

__global__ void CopyBytesVec4Kernel(
    int64_t n4,
    uint32_t* __restrict__ dst,
    const uint32_t* __restrict__ src) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < n4; i += stride) {
    dst[i] = src[i];
  }
}

__global__ void CopyBytesUint4Kernel(
    int64_t n4,
    uint4* __restrict__ dst,
    const uint4* __restrict__ src) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < n4; i += stride) {
    dst[i] = src[i];
  }
}

__global__ void CopyOffsetBytesKernel(
    int64_t n,
    char* __restrict__ dst,
    const char* __restrict__ src,
    OffsetCalculator<2> calc) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t linear_idx = tid; linear_idx < n; linear_idx += stride) {
    const auto offsets = calc.get(static_cast<uint32_t>(linear_idx));
    dst[offsets[0]] = src[offsets[1]];
  }
}

void LaunchCopyBytesVec4(const void* dst, const void* src, size_t nbytes) {
  const int64_t n4 = static_cast<int64_t>(nbytes / 4);
  const int blocks = metax::ComputeGrid1d(n4);
  CopyBytesVec4Kernel<<<blocks, kCopyBlockSize, 0, metax::CurrentStream()>>>(
      n4,
      static_cast<uint32_t*>(const_cast<void*>(dst)),
      static_cast<const uint32_t*>(src));
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX v2 vector copy launch failed: ",
      cudaGetErrorString(err));
}

void LaunchCopyBytesUint4(const void* dst, const void* src, size_t nbytes) {
  const int64_t n4 = static_cast<int64_t>(nbytes / 16);
  const int blocks = metax::ComputeGrid1d(n4);
  CopyBytesUint4Kernel<<<blocks, kCopyBlockSize, 0, metax::CurrentStream()>>>(
      n4,
      static_cast<uint4*>(const_cast<void*>(dst)),
      static_cast<const uint4*>(src));
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX v2 uint4 copy launch failed: ",
      cudaGetErrorString(err));
}

void CopyContiguousBytes(void* dst, const void* src, size_t nbytes) {
  if (nbytes == 0) {
    return;
  }
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
  if (nbytes >= kVec4CopyMinBytes &&
      (nbytes % 16) == 0 &&
      (dst_addr & 15u) == 0 &&
      (src_addr & 15u) == 0) {
    LaunchCopyBytesUint4(dst, src, nbytes);
    return;
  }
  if (nbytes >= kVecCopyMinBytes &&
      (nbytes % 4) == 0 &&
      (dst_addr & 3u) == 0 &&
      (src_addr & 3u) == 0) {
    LaunchCopyBytesVec4(dst, src, nbytes);
    return;
  }
  Memcpy(dst, src, nbytes, MemcpyDeviceToDevice);
}

OffsetCalculator<2> MakeCopyOffsetCalculator(
    const at::Tensor& src,
    const at::Tensor& dst) {
  std::array<const int64_t*, 2> strides;
  int64_t element_sizes[2];
  strides[0] = dst.strides().data();
  strides[1] = src.strides().data();
  element_sizes[0] = dst.element_size();
  element_sizes[1] = src.element_size();
  return OffsetCalculator<2>(
      src.dim(),
      src.sizes().data(),
      strides.data(),
      element_sizes);
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
    CopyContiguousBytes(dst.data_ptr(), src.data_ptr(), n * elem_size);
    return;
  }

  if (src.dim() == 1 && dst.dim() == 1) {
    const int64_t dst_stride_bytes =
        dst.stride(0) * static_cast<int64_t>(elem_size);
    const int64_t src_stride_bytes =
        src.stride(0) * static_cast<int64_t>(elem_size);
    const int blocks = metax::ComputeGrid1d(n);
    StridedCopyBytesKernel<<<blocks, kCopyBlockSize, 0, metax::CurrentStream()>>>(
        n,
        static_cast<char*>(dst.data_ptr()),
        static_cast<const char*>(src.data_ptr()),
        dst_stride_bytes,
        src_stride_bytes);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "MetaX v2 1d strided copy launch failed: ",
        cudaGetErrorString(err));
    return;
  }

  if (dst.is_contiguous()) {
    const auto calc = MakeCopyOffsetCalculator(src, dst);
    const int blocks = metax::ComputeGrid1d(n);
    CopyOffsetBytesKernel<<<blocks, kCopyBlockSize, 0, metax::CurrentStream()>>>(
        n,
        static_cast<char*>(dst.data_ptr()),
        static_cast<const char*>(src.data_ptr()),
        calc);
    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "MetaX v2 offset copy launch failed: ",
        cudaGetErrorString(err));
    return;
  }

  const at::Tensor src_c = src.is_contiguous() ? src : src.contiguous();
  CopyContiguousBytes(
      dst.data_ptr(),
      src_c.data_ptr(),
      static_cast<size_t>(n) * elem_size);
}

} // namespace at::native::flagos
