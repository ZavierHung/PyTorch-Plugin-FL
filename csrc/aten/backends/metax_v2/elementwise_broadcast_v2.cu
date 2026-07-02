// Copyright (c) 2026, BAAI. All rights reserved.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <array>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>

#include "elementwise_broadcast_v2.cuh"
#include "elementwise_vec_v2.cuh"
#include "metax_elementwise_v2.cuh"

namespace at::native::flagos::elementwise_v2 {

LastDimBroadcastInfo DetectLastDimBroadcast(
    const at::TensorIteratorBase& iter) {
  LastDimBroadcastInfo info;
  if (iter.ninputs() != 2 || iter.numel() == 0) {
    return info;
  }
  if (!iter.output().is_contiguous()) {
    return info;
  }
  const int ndim = iter.ndim();
  if (ndim == 0) {
    return info;
  }
  const int64_t last = iter.shape()[ndim - 1];
  const int64_t out_last_stride = iter.strides(0)[ndim - 1];

  for (const int in : {1, 2}) {
    const int other = (in == 1) ? 2 : 1;
    bool broadcast_ok = true;
    for (int d = 0; d < ndim - 1; ++d) {
      if (iter.strides(in)[d] != 0) {
        broadcast_ok = false;
        break;
      }
    }
    if (!broadcast_ok) {
      continue;
    }
    if (iter.strides(in)[ndim - 1] != out_last_stride) {
      continue;
    }
    bool other_ok = true;
    for (int d = 0; d < ndim; ++d) {
      if (iter.strides(other)[d] != iter.strides(0)[d]) {
        other_ok = false;
        break;
      }
    }
    if (!other_ok) {
      continue;
    }
    info.broadcast_operand = in;
    info.last_dim_size = last;
    info.valid = true;
    return info;
  }
  return info;
}

ContigPlusBroadcastInfo DetectContigPlusBroadcast(
    const at::TensorIteratorBase& iter) {
  ContigPlusBroadcastInfo info;
  if (iter.ninputs() != 2 || iter.numel() == 0) {
    return info;
  }
  if (!iter.output().is_contiguous()) {
    return info;
  }
  for (const int in : {1, 2}) {
    const int other = (in == 1) ? 2 : 1;
    bool contig_ok = true;
    for (int d = 0; d < iter.ndim(); ++d) {
      if (iter.strides(in)[d] != iter.strides(0)[d]) {
        contig_ok = false;
        break;
      }
    }
    if (!contig_ok) {
      continue;
    }
    bool has_bcast = false;
    for (int d = 0; d < iter.ndim(); ++d) {
      if (iter.strides(other)[d] == 0 && iter.shape()[d] > 1) {
        has_bcast = true;
        break;
      }
    }
    if (!has_bcast) {
      continue;
    }
    info.contig_operand = in;
    info.broadcast_operand = other;
    info.valid = true;
    return info;
  }
  return info;
}

namespace {

constexpr int kBroadcastBlockSize = 512;

template <typename scalar_t, typename opmath_t>
__global__ void MulLastDimBroadcastKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ contig,
    const scalar_t* __restrict__ bcast,
    int64_t last_dim) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    if constexpr (std::is_same_v<scalar_t, bool>) {
      out[i] = contig[i] && bcast[i % last_dim];
    } else {
      out[i] = static_cast<scalar_t>(
          static_cast<opmath_t>(contig[i]) *
          static_cast<opmath_t>(bcast[i % last_dim]));
    }
  }
}

__global__ void MulHalf2LastDimBroadcastKernel(
    int64_t n2,
    __half* __restrict__ out,
    const __half* __restrict__ contig,
    const __half* __restrict__ bcast,
    int64_t last_dim) {
  const int64_t last_dim2 = last_dim / 2;
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const __half2* contig_h2 = reinterpret_cast<const __half2*>(contig);
  const __half2* bcast_h2 = reinterpret_cast<const __half2*>(bcast);
  __half2* out_h2 = reinterpret_cast<__half2*>(out);
  for (int64_t i = tid; i < n2; i += stride) {
    out_h2[i] = __hmul2(contig_h2[i], bcast_h2[i % last_dim2]);
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void AddLastDimBroadcastKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ contig,
    const scalar_t* __restrict__ bcast,
    int64_t last_dim,
    opmath_t alpha) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    out[i] = static_cast<scalar_t>(
        static_cast<opmath_t>(contig[i]) +
        alpha * static_cast<opmath_t>(bcast[i % last_dim]));
  }
}

__global__ void AddHalf2LastDimBroadcastKernel(
    int64_t n2,
    __half* __restrict__ out,
    const __half* __restrict__ contig,
    const __half* __restrict__ bcast,
    int64_t last_dim,
    float alpha) {
  const __half2 alpha_h2 = __floats2half2_rn(alpha, alpha);
  const int64_t last_dim2 = last_dim / 2;
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const __half2* contig_h2 = reinterpret_cast<const __half2*>(contig);
  const __half2* bcast_h2 = reinterpret_cast<const __half2*>(bcast);
  __half2* out_h2 = reinterpret_cast<__half2*>(out);
  for (int64_t i = tid; i < n2; i += stride) {
    out_h2[i] = __hfma2(bcast_h2[i % last_dim2], alpha_h2, contig_h2[i]);
  }
}

__global__ void AddHalf2LastDimBroadcastAlpha1Kernel(
    int64_t n2,
    __half* __restrict__ out,
    const __half* __restrict__ contig,
    const __half* __restrict__ bcast,
    int64_t last_dim) {
  const int64_t last_dim2 = last_dim / 2;
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const __half2* contig_h2 = reinterpret_cast<const __half2*>(contig);
  const __half2* bcast_h2 = reinterpret_cast<const __half2*>(bcast);
  __half2* out_h2 = reinterpret_cast<__half2*>(out);
  for (int64_t i = tid; i < n2; i += stride) {
    out_h2[i] = __hadd2(contig_h2[i], bcast_h2[i % last_dim2]);
  }
}

template <typename Kernel, typename... Args>
void LaunchBroadcastKernel(int64_t n, Kernel kernel, Args... args) {
  if (n == 0) {
    return;
  }
  const int blocks = metax::ComputeGrid1d(n);
  kernel<<<blocks, kBroadcastBlockSize, 0, metax::CurrentStream()>>>(
      n, args...);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX v2 broadcast kernel launch failed: ",
      cudaGetErrorString(err));
}

template <typename scalar_t, typename opmath_t>
bool LaunchMulBroadcast(
    at::TensorIteratorBase& iter,
    const LastDimBroadcastInfo& info) {
  const int64_t n = iter.numel();
  auto* out = static_cast<scalar_t*>(iter.data_ptr(0));
  const int contig_op = (info.broadcast_operand == 1) ? 2 : 1;
  const int bcast_op = info.broadcast_operand;
  const auto* contig =
      static_cast<const scalar_t*>(iter.data_ptr(contig_op));
  const auto* bcast =
      static_cast<const scalar_t*>(iter.data_ptr(bcast_op));

  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if ((info.last_dim_size % 2) == 0 && (n % 2) == 0 && n >= 256 &&
        PtrAligned4(out) && PtrAligned4(contig) && PtrAligned4(bcast)) {
      LaunchBroadcastKernel(
          n / 2,
          MulHalf2LastDimBroadcastKernel,
          reinterpret_cast<__half*>(out),
          reinterpret_cast<const __half*>(contig),
          reinterpret_cast<const __half*>(bcast),
          info.last_dim_size);
      return true;
    }
  }

  LaunchBroadcastKernel(
      n,
      MulLastDimBroadcastKernel<scalar_t, opmath_t>,
      out,
      contig,
      bcast,
      info.last_dim_size);
  return true;
}

template <typename scalar_t, typename opmath_t>
bool LaunchAddBroadcast(
    at::TensorIteratorBase& iter,
    const LastDimBroadcastInfo& info,
    opmath_t alpha) {
  const int64_t n = iter.numel();
  auto* out = static_cast<scalar_t*>(iter.data_ptr(0));
  const int contig_op = (info.broadcast_operand == 1) ? 2 : 1;
  const int bcast_op = info.broadcast_operand;
  const auto* contig =
      static_cast<const scalar_t*>(iter.data_ptr(contig_op));
  const auto* bcast =
      static_cast<const scalar_t*>(iter.data_ptr(bcast_op));

  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if ((info.last_dim_size % 2) == 0 && (n % 2) == 0 && n >= 256 &&
        PtrAligned4(out) && PtrAligned4(contig) && PtrAligned4(bcast)) {
      auto* out_h = reinterpret_cast<__half*>(out);
      const auto* contig_h = reinterpret_cast<const __half*>(contig);
      const auto* bcast_h = reinterpret_cast<const __half*>(bcast);
      if (alpha == opmath_t(1)) {
        LaunchBroadcastKernel(
            n / 2,
            AddHalf2LastDimBroadcastAlpha1Kernel,
            out_h,
            contig_h,
            bcast_h,
            info.last_dim_size);
      } else {
        LaunchBroadcastKernel(
            n / 2,
            AddHalf2LastDimBroadcastKernel,
            out_h,
            contig_h,
            bcast_h,
            info.last_dim_size,
            static_cast<float>(alpha));
      }
      return true;
    }
  }

  LaunchBroadcastKernel(
      n,
      AddLastDimBroadcastKernel<scalar_t, opmath_t>,
      out,
      contig,
      bcast,
      info.last_dim_size,
      alpha);
  return true;
}

OffsetCalculator<1> MakeBroadcastOperandCalculator(
    const at::TensorIteratorBase& iter,
    int operand) {
  std::array<const int64_t*, 1> strides;
  int64_t element_sizes[1];
  strides[0] = iter.strides(operand).data();
  element_sizes[0] = iter.element_size(operand);
  return OffsetCalculator<1>(
      iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <typename scalar_t, typename opmath_t>
__global__ void MulContigBroadcastKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ contig,
    const scalar_t* __restrict__ bcast,
    OffsetCalculator<1> calc) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    const auto boff = calc.get(static_cast<uint32_t>(i));
    if constexpr (std::is_same_v<scalar_t, bool>) {
      out[i] = contig[i] && bcast[boff[0]];
    } else {
      out[i] = static_cast<scalar_t>(
          static_cast<opmath_t>(contig[i]) *
          static_cast<opmath_t>(bcast[boff[0]]));
    }
  }
}

__global__ void MulContigBroadcastHalf2Kernel(
    int64_t n2,
    __half* __restrict__ out,
    const __half* __restrict__ contig,
    const __half* __restrict__ bcast,
    OffsetCalculator<1> calc) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const __half2* contig_h2 = reinterpret_cast<const __half2*>(contig);
  const __half* bcast_h = bcast;
  __half2* out_h2 = reinterpret_cast<__half2*>(out);
  for (int64_t i = tid; i < n2; i += stride) {
    const auto boff = calc.get(static_cast<uint32_t>(i * 2));
    const __half2 c = contig_h2[i];
    const __half bv0 = bcast_h[boff[0]];
    const auto boff1 = calc.get(static_cast<uint32_t>(i * 2 + 1));
    const __half bv1 = bcast_h[boff1[0]];
    out_h2[i] = __hmul2(c, __halves2half2(bv0, bv1));
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void AddContigBroadcastKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ contig,
    const scalar_t* __restrict__ bcast,
    opmath_t alpha,
    OffsetCalculator<1> calc) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    const auto boff = calc.get(static_cast<uint32_t>(i));
    out[i] = static_cast<scalar_t>(
        static_cast<opmath_t>(contig[i]) +
        alpha * static_cast<opmath_t>(bcast[boff[0]]));
  }
}

template <typename scalar_t, typename opmath_t>
bool LaunchMulContigBroadcast(
    at::TensorIteratorBase& iter,
    const ContigPlusBroadcastInfo& info) {
  const int64_t n = iter.numel();
  auto* out = static_cast<scalar_t*>(iter.data_ptr(0));
  const auto* contig =
      static_cast<const scalar_t*>(iter.data_ptr(info.contig_operand));
  const auto* bcast =
      static_cast<const scalar_t*>(iter.data_ptr(info.broadcast_operand));
  const auto calc = MakeBroadcastOperandCalculator(
      iter, info.broadcast_operand);

  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if ((n % 2) == 0 && n >= 256 && PtrAligned4(out) && PtrAligned4(contig) &&
        PtrAligned4(bcast)) {
      LaunchBroadcastKernel(
          n / 2,
          MulContigBroadcastHalf2Kernel,
          reinterpret_cast<__half*>(out),
          reinterpret_cast<const __half*>(contig),
          reinterpret_cast<const __half*>(bcast),
          calc);
      return true;
    }
  }

  LaunchBroadcastKernel(
      n,
      MulContigBroadcastKernel<scalar_t, opmath_t>,
      out,
      contig,
      bcast,
      calc);
  return true;
}

template <typename scalar_t, typename opmath_t>
bool LaunchAddContigBroadcast(
    at::TensorIteratorBase& iter,
    const ContigPlusBroadcastInfo& info,
    opmath_t alpha) {
  const int64_t n = iter.numel();
  auto* out = static_cast<scalar_t*>(iter.data_ptr(0));
  const auto* contig =
      static_cast<const scalar_t*>(iter.data_ptr(info.contig_operand));
  const auto* bcast =
      static_cast<const scalar_t*>(iter.data_ptr(info.broadcast_operand));
  const auto calc = MakeBroadcastOperandCalculator(
      iter, info.broadcast_operand);

  LaunchBroadcastKernel(
      n,
      AddContigBroadcastKernel<scalar_t, opmath_t>,
      out,
      contig,
      bcast,
      alpha,
      calc);
  return true;
}

} // namespace

bool TryLaunchMulContigBroadcast(at::TensorIteratorBase& iter) {
  const auto info = DetectContigPlusBroadcast(iter);
  if (!info.valid) {
    return false;
  }
  bool launched = false;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "mul_metax_contig_broadcast",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        launched = LaunchMulContigBroadcast<scalar_t, opmath_t>(iter, info);
      });
  return launched;
}

bool TryLaunchAddContigBroadcast(
    at::TensorIteratorBase& iter,
    const at::Scalar& alpha) {
  const auto info = DetectContigPlusBroadcast(iter);
  if (!info.valid) {
    return false;
  }
  bool launched = false;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "add_metax_contig_broadcast",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        launched = LaunchAddContigBroadcast<scalar_t, opmath_t>(
            iter, info, alpha.to<opmath_t>());
      });
  return launched;
}

bool TryLaunchMulBroadcastFast(at::TensorIteratorBase& iter) {
  const auto info = DetectLastDimBroadcast(iter);
  if (!info.valid) {
    return false;
  }
  bool launched = false;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "mul_metax_broadcast",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        launched = LaunchMulBroadcast<scalar_t, opmath_t>(iter, info);
      });
  return launched;
}

bool TryLaunchAddBroadcastFast(
    at::TensorIteratorBase& iter,
    const at::Scalar& alpha) {
  const auto info = DetectLastDimBroadcast(iter);
  if (!info.valid) {
    return false;
  }
  bool launched = false;
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "add_metax_broadcast",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        launched =
            LaunchAddBroadcast<scalar_t, opmath_t>(iter, info, alpha.to<opmath_t>());
      });
  return launched;
}

} // namespace at::native::flagos::elementwise_v2
