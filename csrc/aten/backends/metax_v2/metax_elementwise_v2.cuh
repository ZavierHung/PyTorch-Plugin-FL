// CUDA-style capped-grid elementwise launch on MetaX (no CUDALoops dependency).
#pragma once

#include <algorithm>
#include <cstdint>

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

} // namespace at::native::flagos::metax
