// Copyright (c) 2026, BAAI. All rights reserved.
// Fused row-copy kernel for MetaX cat v2 general path.

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

#include "metax_elementwise_v2.cuh"

namespace at::native::flagos {

namespace {

// Copy one pre-row (chunk * post_dim * elem_size bytes) with grid-stride.
// Grid: (blocks_for_row, pre_dim). Each block copies an entire row cooperatively.
template <int BLOCK_THREADS>
__global__ void CatRowCopyKernel(
    char* __restrict__ out_base,
    const char* __restrict__ in_base,
    int64_t row_bytes,
    int64_t out_dim_offset,
    int64_t out_row_stride,
    int64_t in_row_stride,
    int64_t post_dim,
    int64_t elem_size) {
  const int64_t pre_idx = static_cast<int64_t>(blockIdx.y);
  const int64_t row_base_out =
      pre_idx * out_row_stride + out_dim_offset * post_dim * elem_size;
  const int64_t row_base_in = pre_idx * in_row_stride;
  char* dst_row = out_base + row_base_out;
  const char* src_row = in_base + row_base_in;

  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;

  // 16-byte vectorized copy when row is aligned.
  if (row_bytes >= 16 && (row_bytes & 15) == 0) {
    const int64_t vec_count = row_bytes / 16;
    const uint4* src_v = reinterpret_cast<const uint4*>(src_row);
    uint4* dst_v = reinterpret_cast<uint4*>(dst_row);
    for (int64_t v = tid; v < vec_count; v += stride) {
      dst_v[v] = src_v[v];
    }
    return;
  }

  // 4-byte vectorized copy for fp32 rows aligned to 4 bytes.
  if (row_bytes >= 4 && (row_bytes & 3) == 0 && elem_size == 4) {
    const int64_t word_count = row_bytes / 4;
    const uint32_t* src_w = reinterpret_cast<const uint32_t*>(src_row);
    uint32_t* dst_w = reinterpret_cast<uint32_t*>(dst_row);
    for (int64_t w = tid; w < word_count; w += stride) {
      dst_w[w] = src_w[w];
    }
    return;
  }

  for (int64_t b = tid; b < row_bytes; b += stride) {
    dst_row[b] = src_row[b];
  }
}

} // namespace

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
    cudaStream_t stream) {
  (void)chunk_bytes;
  constexpr int BLOCK_THREADS = 256;
  if (pre_dim <= 0) {
    return;
  }
  const int64_t row_bytes = chunk_bytes;
  if (row_bytes <= 0) {
    return;
  }
  const int blocks_per_row = metax::ComputeGrid1d(row_bytes);
  if (blocks_per_row <= 0) {
    return;
  }
  dim3 grid(blocks_per_row, static_cast<unsigned int>(pre_dim), 1);
  dim3 threads(BLOCK_THREADS, 1, 1);
  CatRowCopyKernel<BLOCK_THREADS><<<grid, threads, 0, stream>>>(
      out_base,
      in_base,
      row_bytes,
      out_dim_offset,
      out_row_stride,
      in_row_stride,
      post_dim,
      elem_size);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX cat v2 fused kernel launch failed: ",
      cudaGetErrorString(err));
}

} // namespace at::native::flagos
