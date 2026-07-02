// CUDA-style capped-grid elementwise launch on MetaX (no CUDALoops dependency).
#pragma once

#include <algorithm>
#include <cstdint>

#include <ATen/core/Tensor.h>

#include "../metax/metax_elementwise.cuh"

namespace at::native::flagos::metax {

constexpr int kBlockSizeV2 = 256;
constexpr int kMaxGridV2 = 65535;

template <typename Kernel, typename... Args>
inline void Launch1dV2(int64_t n, Kernel kernel, Args... args) {
  if (n == 0) {
    return;
  }
  const int64_t blocks_raw =
      (n + static_cast<int64_t>(kBlockSizeV2) - 1) / kBlockSizeV2;
  const int blocks = static_cast<int>(std::min<int64_t>(blocks_raw, kMaxGridV2));
  kernel<<<blocks, kBlockSizeV2, 0, CurrentStream()>>>(n, args...);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX v2 kernel launch failed: ",
      cudaGetErrorString(err));
}

// Compute grid size for grid-stride loops (cap by kMaxGridV2).
inline int ComputeGrid1d(int64_t n_per_block) {
  if (n_per_block <= 0) return 0;
  const int64_t blocks_raw =
      (n_per_block + static_cast<int64_t>(kBlockSizeV2) - 1) / kBlockSizeV2;
  return static_cast<int>(std::min<int64_t>(blocks_raw, kMaxGridV2));
}

// Capped-grid launch where each thread grid-strides over [0, n).
template <typename Kernel, typename... Args>
inline void Launch1dGridStrideV2(int64_t n, Kernel kernel, Args... args) {
  if (n == 0) {
    return;
  }
  const int blocks = ComputeGrid1d(n);
  if (blocks == 0) {
    return;
  }
  kernel<<<blocks, kBlockSizeV2, 0, CurrentStream()>>>(n, args...);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX v2 grid-stride kernel launch failed: ",
      cudaGetErrorString(err));
}

} // namespace at::native::flagos::metax

namespace at::native::flagos {

// Skip expand().contiguous() when tensor already matches output shape+layout.
inline at::Tensor BroadcastContiguousIfNeeded(
    const at::Tensor& tensor,
    at::IntArrayRef shape) {
  if (tensor.sizes().equals(shape) && tensor.is_contiguous()) {
    return tensor;
  }
  return tensor.expand(shape).contiguous();
}

} // namespace at::native::flagos
