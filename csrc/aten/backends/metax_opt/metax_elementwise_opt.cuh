// Optimized elementwise launch: larger blocks for better occupancy on MetaX.
#pragma once

#include "../metax/metax_elementwise.cuh"

namespace at::native::flagos::metax {

constexpr int kBlockSizeOpt = 512;

template <typename Kernel, typename... Args>
inline void Launch1dOpt(int64_t n, Kernel kernel, Args... args) {
  if (n == 0) {
    return;
  }
  const int blocks = static_cast<int>(
      (n + static_cast<int64_t>(kBlockSizeOpt) - 1) / kBlockSizeOpt);
  kernel<<<blocks, kBlockSizeOpt, 0, CurrentStream()>>>(n, args...);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX opt kernel launch failed: ",
      cudaGetErrorString(err));
}

} // namespace at::native::flagos::metax
