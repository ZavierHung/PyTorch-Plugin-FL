// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <c10/util/Exception.h>

#include "metax_elementwise_v2.cuh"

namespace at::native::flagos {

namespace {

std::vector<int64_t> ParseReductionDims(
    at::OptionalIntArrayRef opt_dims,
    int64_t ndim) {
  if (!opt_dims.has_value() || opt_dims->empty()) {
    std::vector<int64_t> dims(ndim);
    std::iota(dims.begin(), dims.end(), 0);
    return dims;
  }
  std::vector<int64_t> dims;
  dims.reserve(opt_dims->size());
  for (const auto dim : *opt_dims) {
    dims.push_back(at::maybe_wrap_dim(dim, ndim));
  }
  return dims;
}

std::vector<int64_t> ShapeReduce(
    at::IntArrayRef shape,
    at::IntArrayRef dims,
    bool keepdim) {
  std::vector<int64_t> out(shape.begin(), shape.end());
  std::vector<int64_t> sorted_dims(dims.begin(), dims.end());
  std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int64_t>());
  for (int64_t dim : sorted_dims) {
    if (keepdim) {
      out[dim] = 1;
    } else {
      out.erase(out.begin() + dim);
    }
  }
  return out;
}

// (Old per-thread serial kernel removed; replaced by MeanAlongDimFlatKernel
// and MeanAlongDimBlockReduceKernel below.)

// Optimized mean kernels (hand-rolled warp shuffle, no cub dependency).

// Cache-hinted load for arithmetic types; plain load for complex.
template <typename T>
__device__ __forceinline__
std::enable_if_t<std::is_arithmetic<T>::value, T> LoadInput(const T* ptr) {
  return __ldg(ptr);
}

template <typename T>
__device__ __forceinline__
std::enable_if_t<!std::is_arithmetic<T>::value, T> LoadInput(const T* ptr) {
  return *ptr;
}

// Vectorized accumulate along reduce axis when inner==1 (coalesced row).
template <typename scalar_t, typename acc_t>
__device__ __forceinline__ acc_t AccumulateReduceRow(
    const scalar_t* __restrict__ row,
    int64_t reduce) {
  acc_t sum = acc_t(0);
  if constexpr (sizeof(scalar_t) == 2 && std::is_arithmetic<scalar_t>::value) {
    const int64_t pairs = reduce / 2;
    const uint32_t* row_u32 = reinterpret_cast<const uint32_t*>(row);
    for (int64_t p = 0; p < pairs; ++p) {
      const uint32_t packed = __ldg(row_u32 + p);
      const uint16_t lo = static_cast<uint16_t>(packed & 0xffffu);
      const uint16_t hi = static_cast<uint16_t>(packed >> 16);
      scalar_t v0, v1;
      std::memcpy(&v0, &lo, sizeof(scalar_t));
      std::memcpy(&v1, &hi, sizeof(scalar_t));
      sum += static_cast<acc_t>(v0);
      sum += static_cast<acc_t>(v1);
    }
    if ((reduce & 1) != 0) {
      sum += static_cast<acc_t>(LoadInput(row + reduce - 1));
    }
    return sum;
  }
  if constexpr (sizeof(scalar_t) == 4 && std::is_arithmetic<scalar_t>::value) {
    const int64_t quads = reduce / 4;
    const float4* row_f4 = reinterpret_cast<const float4*>(row);
    for (int64_t q = 0; q < quads; ++q) {
      const float4 v = __ldg(row_f4 + q);
      sum += static_cast<acc_t>(v.x);
      sum += static_cast<acc_t>(v.y);
      sum += static_cast<acc_t>(v.z);
      sum += static_cast<acc_t>(v.w);
    }
    for (int64_t r = quads * 4; r < reduce; ++r) {
      sum += static_cast<acc_t>(LoadInput(row + r));
    }
    return sum;
  }
  for (int64_t r = 0; r < reduce; ++r) {
    sum += static_cast<acc_t>(LoadInput(row + r));
  }
  return sum;
}

template <typename scalar_t, typename acc_t>
__device__ __forceinline__ acc_t AccumulateReduceStrided(
    const scalar_t* __restrict__ base,
    int64_t reduce,
    int64_t inner) {
  acc_t sum = acc_t(0);
  for (int64_t r = 0; r < reduce; ++r) {
    sum += static_cast<acc_t>(LoadInput(base + r * inner));
  }
  return sum;
}

// BlockReduce body — always defined (no template instantiation issue);
// only compiled for arithmetic types since `__shfl_down_sync` requires it.
// Two overloads of the body — only one is selected via enable_if:
// - arithmetic: real implementation
// - non-arithmetic: no-op (never reached because dispatch routes complex
//   types to FlatKernel, but defined here so that the template still
//   instantiates without compile errors)
template <typename scalar_t, typename acc_t, typename out_t>
__device__ __forceinline__
std::enable_if_t<std::is_arithmetic<acc_t>::value, void>
MeanAlongDimBlockReduceBody(
    int64_t outer,
    int64_t reduce,
    int64_t inner,
    out_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    acc_t inv_reduce) {
  // MetaX C500 wavefront size is 64. All warp-shuffle constants below are
  // derived from WARP=64, not the CUDA-default 32.
  constexpr int BLOCK = 256;
  constexpr int WARP = 64;
  constexpr int WARPS = BLOCK / WARP;
  // One block reduces one (outer, inner) slot — all BLOCK threads cooperate
  // on the same reduce axis. Do NOT pack multiple inner slots per block: that
  // would make __shfl_down_sync mix sums from different slots.
  const int64_t outer_idx = static_cast<int64_t>(blockIdx.x);
  const int64_t inner_idx = static_cast<int64_t>(blockIdx.y);
  const bool valid = (outer_idx < outer) && (inner_idx < inner);
  const int64_t base_in = outer_idx * reduce * inner + inner_idx;
  // Every thread participates in the shuffle reduce (out-of-bounds threads
  // contribute 0); only the write-back is guarded by `valid`.
  acc_t thread_sum = acc_t(0);
  if (valid) {
    if (inner == 1) {
      const scalar_t* row = in + base_in;
      for (int64_t r = threadIdx.x; r < reduce; r += BLOCK) {
        thread_sum += static_cast<acc_t>(LoadInput(row + r));
      }
    } else {
      for (int64_t r = threadIdx.x; r < reduce; r += BLOCK) {
        thread_sum += static_cast<acc_t>(LoadInput(in + base_in + r * inner));
      }
    }
  }
  for (int offset = WARP / 2; offset > 0; offset >>= 1) {
    thread_sum += __shfl_down_sync(0xffffffffffffffffull, thread_sum, offset);
  }
  __shared__ acc_t warp_sums[WARPS];
  const int lane = threadIdx.x & (WARP - 1);
  const int warp_id = threadIdx.x / WARP;
  if (lane == 0) warp_sums[warp_id] = thread_sum;
  __syncthreads();
  if (warp_id == 0) {
    acc_t v = (threadIdx.x < WARPS) ? warp_sums[threadIdx.x] : acc_t(0);
    for (int offset = WARPS / 2; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xffffffffffffffffull, v, offset);
    }
    if (threadIdx.x == 0 && valid) {
      out[outer_idx * inner + inner_idx] = static_cast<out_t>(v * inv_reduce);
    }
  }
}

template <typename scalar_t, typename acc_t, typename out_t>
__device__ __forceinline__
std::enable_if_t<!std::is_arithmetic<acc_t>::value, void>
MeanAlongDimBlockReduceBody(
    int64_t, int64_t, int64_t, out_t*, const scalar_t*, acc_t) {
  // unreachable for complex types — dispatch routes to FlatKernel
}

template <typename scalar_t, typename acc_t, typename out_t>
__global__ void MeanAlongDimBlockReduceKernelImpl(
    int64_t outer,
    int64_t reduce,
    int64_t inner,
    out_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    acc_t inv_reduce) {
  // SFINAE prevents non-arithmetic instantiations from reaching this code.
  MeanAlongDimBlockReduceBody<scalar_t, acc_t, out_t>(
      outer, reduce, inner, out, in, inv_reduce);
}

// (Old SplitKMean kernels removed; replaced by MeanAlongDimSplitKReduceKernelImpl
// and MeanAlongDimSplitKFinalKernel below.)

template <typename scalar_t, typename acc_t, typename out_t>
__global__ void MeanAlongDimFlatKernel(
    int64_t outer,
    int64_t reduce,
    int64_t inner,
    int64_t total_slots,
    out_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    acc_t inv_reduce) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t slot = tid; slot < total_slots; slot += stride) {
    const int64_t outer_idx = slot / inner;
    const int64_t inner_idx = slot - outer_idx * inner;
    const int64_t base_in = outer_idx * reduce * inner + inner_idx;
    acc_t sum = acc_t(0);
    if (inner == 1) {
      sum = AccumulateReduceRow<scalar_t, acc_t>(in + base_in, reduce);
    } else {
      sum = AccumulateReduceStrided<scalar_t, acc_t>(
          in + base_in, reduce, inner);
    }
    out[slot] = static_cast<out_t>(sum * inv_reduce);
  }
}

// Multi-block-per-slot kernel for the degenerate case (inner=1, small outer,
// large reduce). Each block reduces a chunk of reduce axis for one outer slot.
// Many blocks share the same outer to give the GPU enough parallelism.
// Arithmetic-only (uses __shfl_down_sync).
// SplitKReduceBody — two overloads (arithmetic vs non-arithmetic).
template <typename scalar_t, typename acc_t, typename out_t>
__device__ __forceinline__
std::enable_if_t<std::is_arithmetic<acc_t>::value, void>
MeanAlongDimSplitKReduceBody(
    int64_t outer,
    int64_t reduce,
    int64_t inner,
    int64_t reduce_per_block,
    int64_t splits,
    acc_t* __restrict__ partials,
    const scalar_t* __restrict__ in) {
  constexpr int BLOCK = 256;
  constexpr int WARP = 64;
  constexpr int WARPS = BLOCK / WARP;
  const int64_t outer_idx = static_cast<int64_t>(blockIdx.x);
  const int64_t inner_idx = static_cast<int64_t>(blockIdx.y);
  const int64_t split_idx = static_cast<int64_t>(blockIdx.z);
  const bool valid = (outer_idx < outer) && (inner_idx < inner);
  const int64_t base_in = outer_idx * reduce * inner + inner_idx;
  const int64_t r_start = split_idx * reduce_per_block;
  const int64_t r_end = std::min(r_start + reduce_per_block, reduce);
  acc_t thread_sum = acc_t(0);
  if (valid) {
    if (inner == 1) {
      const scalar_t* row = in + base_in;
      for (int64_t r = r_start + threadIdx.x; r < r_end; r += BLOCK) {
        thread_sum += static_cast<acc_t>(LoadInput(row + r));
      }
    } else {
      for (int64_t r = r_start + threadIdx.x; r < r_end; r += BLOCK) {
        thread_sum += static_cast<acc_t>(LoadInput(in + base_in + r * inner));
      }
    }
  }
  for (int offset = WARP / 2; offset > 0; offset >>= 1) {
    thread_sum += __shfl_down_sync(0xffffffffffffffffull, thread_sum, offset);
  }
  __shared__ acc_t warp_sums[WARPS];
  const int lane = threadIdx.x & (WARP - 1);
  const int warp_id = threadIdx.x / WARP;
  if (lane == 0) warp_sums[warp_id] = thread_sum;
  __syncthreads();
  if (warp_id == 0) {
    acc_t v = (threadIdx.x < WARPS) ? warp_sums[threadIdx.x] : acc_t(0);
    for (int offset = WARPS / 2; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xffffffffffffffffull, v, offset);
    }
    if (threadIdx.x == 0 && valid) {
      partials[(outer_idx * inner + inner_idx) * splits + split_idx] = v;
    }
  }
}

template <typename scalar_t, typename acc_t, typename out_t>
__device__ __forceinline__
std::enable_if_t<!std::is_arithmetic<acc_t>::value, void>
MeanAlongDimSplitKReduceBody(
    int64_t, int64_t, int64_t, int64_t, int64_t,
    acc_t*, const scalar_t*) {
  // unreachable for complex types
}

template <typename scalar_t, typename acc_t, typename out_t>
__global__ void MeanAlongDimSplitKReduceKernelImpl(
    int64_t outer,
    int64_t reduce,
    int64_t inner,
    int64_t reduce_per_block,  // ceil(reduce / splits)
    int64_t splits,
    acc_t* __restrict__ partials,  // [outer * inner * splits]
    const scalar_t* __restrict__ in) {
  MeanAlongDimSplitKReduceBody<scalar_t, acc_t, out_t>(
      outer, reduce, inner, reduce_per_block, splits, partials, in);
}

template <typename scalar_t, typename acc_t, typename out_t>
__global__ void MeanAlongDimSplitKFinalKernel(
    int64_t total_slots,
    int64_t splits,
    acc_t inv_reduce,
    out_t* __restrict__ out,
    const acc_t* __restrict__ partials) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t slot = tid; slot < total_slots; slot += stride) {
    acc_t sum = acc_t(0);
    const acc_t* slot_partials = partials + slot * splits;
    for (int64_t s = 0; s < splits; ++s) {
      sum += slot_partials[s];
    }
    out[slot] = static_cast<out_t>(sum * inv_reduce);
  }
}

// Simple growing workspace cache for split-K partials.
template <typename T>
static T* AcquireWorkspace(int64_t bytes) {
  static T* buf = nullptr;
  static int64_t cur_bytes = 0;
  if (bytes > cur_bytes) {
    if (buf) {
      cudaFree(buf);
      buf = nullptr;
      cur_bytes = 0;
    }
    cudaMalloc(&buf, bytes);
    cur_bytes = bytes;
  }
  return buf;
}

template <typename scalar_t, typename acc_t, typename out_t>
void LaunchMeanAlongDim(
    int64_t outer,
    int64_t reduce,
    int64_t inner,
    out_t* out_ptr,
    const scalar_t* in_ptr,
    acc_t inv_reduce,
    cudaStream_t stream) {
  constexpr int BLOCK = 256;
  const int64_t total_slots = outer * inner;

  constexpr bool arith = std::is_arithmetic<acc_t>::value;

  // Decision matrix:
  // - Degenerate (inner=1, small outer*inner, large reduce): use split-K
  //   to multiply parallelism across the reduce axis.
  // - Many slots: use flat 1 thread per slot.
  // - Otherwise: block-reduce per (outer, inner).
  // - Complex: only flat is safe.
  const bool degenerate = arith && (inner <= 1) &&
      (total_slots < 256) && (reduce >= 1024);

  if (degenerate) {
    // Aim for ~64 blocks per outer slot for decent SM occupancy.
    const int64_t splits = std::min<int64_t>(
        64, (reduce + BLOCK - 1) / BLOCK);
    const int64_t reduce_per_block = (reduce + splits - 1) / splits;
    const int64_t partial_bytes = total_slots * splits * sizeof(acc_t);
    acc_t* partials = AcquireWorkspace<acc_t>(partial_bytes);
    // One block per (outer, inner, split) — grid Y = inner (=1 here, since
    // the degenerate path requires inner<=1).
    const dim3 blocks(
        static_cast<unsigned int>(outer),
        static_cast<unsigned int>(inner),
        static_cast<unsigned int>(splits));
    const dim3 threads(BLOCK, 1, 1);
    MeanAlongDimSplitKReduceKernelImpl<scalar_t, acc_t, out_t>
        <<<blocks, threads, 0, stream>>>(
            outer, reduce, inner, reduce_per_block, splits, partials, in_ptr);
    const int blocks_final = metax::ComputeGrid1d(total_slots);
    if (blocks_final > 0) {
      MeanAlongDimSplitKFinalKernel<scalar_t, acc_t, out_t>
          <<<blocks_final, BLOCK, 0, stream>>>(
              total_slots, splits, inv_reduce, out_ptr, partials);
    }
  } else if constexpr (arith) {
    // BlockReduce launches one block per (outer, inner) slot. Guard against
    // inner exceeding the grid-Y limit (65535); fall through to FlatKernel.
    const bool use_block_reduce =
        (inner <= 1) ||
        (inner <= metax::kMaxGridV2 &&
         total_slots < static_cast<int64_t>(metax::kMaxGridV2 * BLOCK));
    if (use_block_reduce) {
      const dim3 blocks(
          static_cast<unsigned int>(outer),
          static_cast<unsigned int>(inner),
          1);
      const dim3 threads(BLOCK, 1, 1);
      MeanAlongDimBlockReduceKernelImpl<scalar_t, acc_t, out_t>
          <<<blocks, threads, 0, stream>>>(
              outer, reduce, inner, out_ptr, in_ptr, inv_reduce);
    } else {
      const int blocks = metax::ComputeGrid1d(total_slots);
      if (blocks > 0) {
        MeanAlongDimFlatKernel<scalar_t, acc_t, out_t>
            <<<blocks, BLOCK, 0, stream>>>(
                outer, reduce, inner, total_slots, out_ptr, in_ptr, inv_reduce);
      }
    }
  } else {
    // Complex types: only flat kernel is safe.
    const int blocks = metax::ComputeGrid1d(total_slots);
    if (blocks > 0) {
      MeanAlongDimFlatKernel<scalar_t, acc_t, out_t>
          <<<blocks, BLOCK, 0, stream>>>(
              outer, reduce, inner, total_slots, out_ptr, in_ptr, inv_reduce);
    }
  }
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX mean launch failed: ",
      cudaGetErrorString(err));
}

template <typename scalar_t, typename acc_t, typename out_t>
at::Tensor MeanAlongDim(
    const at::Tensor& input,
    int64_t dim,
    bool keepdim) {
  const int64_t ndim = input.dim();
  TORCH_CHECK(dim >= 0 && dim < ndim, "invalid reduction dim");
  const at::Tensor& in_ref = input.is_contiguous() ? input : input.contiguous();
  const at::Tensor& in = in_ref;

  int64_t outer = 1;
  for (int64_t i = 0; i < dim; ++i) {
    outer *= in.size(i);
  }
  const int64_t reduce = in.size(dim);
  int64_t inner = 1;
  for (int64_t i = dim + 1; i < ndim; ++i) {
    inner *= in.size(i);
  }

  std::vector<int64_t> out_sizes;
  out_sizes.reserve(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    if (i == dim) {
      if (keepdim) {
        out_sizes.push_back(1);
      }
    } else {
      out_sizes.push_back(in.size(i));
    }
  }
  at::Tensor output = at::empty(
      out_sizes,
      in.options().dtype(c10::CppTypeToScalarType<out_t>::value));

  if (output.numel() == 0) {
    return output;
  }

  const acc_t inv_reduce = acc_t(1) / static_cast<acc_t>(reduce);

  LaunchMeanAlongDim<scalar_t, acc_t, out_t>(
      outer,
      reduce,
      inner,
      output.data_ptr<out_t>(),
      in.data_ptr<scalar_t>(),
      inv_reduce,
      metax::CurrentStream());

  return output;
}

template <typename scalar_t, typename acc_t, typename out_t>
at::Tensor MeanDimsTyped(
    at::Tensor input,
    std::vector<int64_t> dims,
    bool keepdim) {
  std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
  for (int64_t dim : dims) {
    input = MeanAlongDim<scalar_t, acc_t, out_t>(input, dim, keepdim);
  }
  return input;
}

} // namespace

inline at::Tensor MeanDimKernelV2(
    const at::Tensor& self,
    at::OptionalIntArrayRef opt_dims,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {
  auto out_dtype = at::native::get_dtype_from_self(self, dtype, true);
  const bool promote_lowp_to_f32 =
      (self.scalar_type() == at::kHalf ||
       self.scalar_type() == at::kBFloat16) &&
      out_dtype == at::kFloat;

  const std::vector<int64_t> dims = ParseReductionDims(opt_dims, self.dim());

  if (self.numel() == 0) {
    const auto shape = ShapeReduce(self.sizes(), dims, keepdim);
    return at::empty(shape, self.options().dtype(out_dtype));
  }

  at::Tensor result;
  if (promote_lowp_to_f32) {
    if (self.scalar_type() == at::kHalf) {
      result = MeanDimsTyped<at::Half, float, float>(self, dims, keepdim);
    } else {
      result = MeanDimsTyped<at::BFloat16, float, float>(self, dims, keepdim);
    }
    return result;
  }

  if (at::isComplexType(self.scalar_type())) {
    AT_DISPATCH_COMPLEX_TYPES(self.scalar_type(), "mean_metax", [&]() {
      TORCH_CHECK(
          out_dtype == self.scalar_type(),
          "MetaX mean: complex input does not support dtype argument");
      result = MeanDimsTyped<scalar_t, scalar_t, scalar_t>(self, dims, keepdim);
    });
    return result;
  }

  if (self.scalar_type() == at::kHalf) {
    if (out_dtype == at::kHalf) {
      result = MeanDimsTyped<at::Half, float, at::Half>(self, dims, keepdim);
    } else {
      TORCH_CHECK(out_dtype == at::kFloat, "MetaX mean: unsupported output dtype");
      result = MeanDimsTyped<at::Half, float, float>(self, dims, keepdim);
    }
    return result;
  }

  if (self.scalar_type() == at::kBFloat16) {
    if (out_dtype == at::kBFloat16) {
      result = MeanDimsTyped<at::BFloat16, float, at::BFloat16>(
          self, dims, keepdim);
    } else {
      TORCH_CHECK(out_dtype == at::kFloat, "MetaX mean: unsupported output dtype");
      result = MeanDimsTyped<at::BFloat16, float, float>(self, dims, keepdim);
    }
    return result;
  }

  AT_DISPATCH_ALL_TYPES(self.scalar_type(), "mean_metax", [&]() {
    using acc_t = at::opmath_type<scalar_t>;
    if (out_dtype == self.scalar_type()) {
      result = MeanDimsTyped<scalar_t, acc_t, scalar_t>(self, dims, keepdim);
    } else {
      TORCH_CHECK(
          out_dtype == at::kFloat,
          "MetaX mean: unsupported output dtype");
      result = MeanDimsTyped<scalar_t, acc_t, float>(self, dims, keepdim);
    }
  });
  return result;
}

} // namespace at::native::flagos
