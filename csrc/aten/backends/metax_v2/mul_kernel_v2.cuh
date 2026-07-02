// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>
#include <type_traits>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>

#include "elementwise_offset_v2.cuh"
#include "elementwise_vec_v2.cuh"
#include "metax_elementwise_v2.cuh"

namespace at::native::flagos {

namespace {

constexpr int64_t kGridStrideThresholdV2 =
    static_cast<int64_t>(metax::kMaxGridV2) * metax::kBlockSizeV2;

using elementwise_v2::CanVectorizeBinary;
using elementwise_v2::CanVectorizeHalfBinary;
using elementwise_v2::LaunchMulBfloat162;
using elementwise_v2::LaunchMulFloat4;
using elementwise_v2::LaunchMulHalf2;

template <typename scalar_t, typename opmath_t>
__global__ void MulContigKernelSmall(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  if constexpr (std::is_same_v<scalar_t, bool>) {
    out[idx] = self[idx] && other[idx];
  } else {
    out[idx] = static_cast<scalar_t>(
        static_cast<opmath_t>(self[idx]) * static_cast<opmath_t>(other[idx]));
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void MulContigKernelLarge(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = tid; idx < n; idx += stride) {
    if constexpr (std::is_same_v<scalar_t, bool>) {
      out[idx] = self[idx] && other[idx];
    } else {
      out[idx] = static_cast<scalar_t>(
          static_cast<opmath_t>(self[idx]) * static_cast<opmath_t>(other[idx]));
    }
  }
}

template <typename scalar_t, typename opmath_t>
void LaunchMulContig(
    int64_t n,
    scalar_t* out,
    const scalar_t* self,
    const scalar_t* other) {
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if (CanVectorizeHalfBinary(n, out, self, other, 2) ||
        CanVectorizeBinary(n, out, self, other, 2)) {
      const int64_t n2 = n / 2;
      LaunchMulHalf2(
          n2,
          reinterpret_cast<__half*>(out),
          reinterpret_cast<const __half*>(self),
          reinterpret_cast<const __half*>(other));
      return;
    }
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    if (CanVectorizeHalfBinary(n, out, self, other, 2) ||
        CanVectorizeBinary(n, out, self, other, 2)) {
      const int64_t n2 = n / 2;
      LaunchMulBfloat162(
          n2,
          reinterpret_cast<__nv_bfloat16*>(out),
          reinterpret_cast<const __nv_bfloat16*>(self),
          reinterpret_cast<const __nv_bfloat16*>(other));
      return;
    }
  } else if constexpr (std::is_same_v<scalar_t, float>) {
    if (CanVectorizeBinary(n, out, self, other, 4)) {
      const int64_t n4 = n / 4;
      LaunchMulFloat4(n4, out, self, other);
      return;
    }
  }

  if (n > kGridStrideThresholdV2) {
    metax::Launch1dGridStrideV2(
        n, MulContigKernelLarge<scalar_t, opmath_t>, out, self, other);
  } else {
    metax::Launch1dV2(
        n, MulContigKernelSmall<scalar_t, opmath_t>, out, self, other);
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void MulSelfScalarKernelSmall(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    opmath_t other_val) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  if constexpr (std::is_same_v<scalar_t, bool>) {
    out[idx] = self[idx] && static_cast<bool>(other_val);
  } else {
    out[idx] = static_cast<scalar_t>(
        static_cast<opmath_t>(self[idx]) * other_val);
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void MulSelfScalarKernelLarge(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    opmath_t other_val) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = tid; idx < n; idx += stride) {
    if constexpr (std::is_same_v<scalar_t, bool>) {
      out[idx] = self[idx] && static_cast<bool>(other_val);
    } else {
      out[idx] = static_cast<scalar_t>(
          static_cast<opmath_t>(self[idx]) * other_val);
    }
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void MulOtherScalarKernelSmall(
    int64_t n,
    scalar_t* __restrict__ out,
    opmath_t self_val,
    const scalar_t* __restrict__ other) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  if constexpr (std::is_same_v<scalar_t, bool>) {
    out[idx] = static_cast<bool>(self_val) && other[idx];
  } else {
    out[idx] = static_cast<scalar_t>(
        self_val * static_cast<opmath_t>(other[idx]));
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void MulOtherScalarKernelLarge(
    int64_t n,
    scalar_t* __restrict__ out,
    opmath_t self_val,
    const scalar_t* __restrict__ other) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = tid; idx < n; idx += stride) {
    if constexpr (std::is_same_v<scalar_t, bool>) {
      out[idx] = static_cast<bool>(self_val) && other[idx];
    } else {
      out[idx] = static_cast<scalar_t>(
          self_val * static_cast<opmath_t>(other[idx]));
    }
  }
}

template <typename scalar_t, typename opmath_t>
void LaunchMulIter(at::TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.is_contiguous());
  const int64_t n = iter.numel();
  scalar_t* out = static_cast<scalar_t*>(iter.data_ptr(0));
  const bool use_grid_stride = n > kGridStrideThresholdV2;

  if (iter.is_cpu_scalar(1)) {
    const opmath_t other_val = iter.scalar_value<opmath_t>(1);
    iter.remove_operand(1);
    const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
    if (use_grid_stride) {
      metax::Launch1dGridStrideV2(
          n,
          MulSelfScalarKernelLarge<scalar_t, opmath_t>,
          out,
          self,
          other_val);
    } else {
      metax::Launch1dV2(
          n,
          MulSelfScalarKernelSmall<scalar_t, opmath_t>,
          out,
          self,
          other_val);
    }
  } else if (iter.is_cpu_scalar(2)) {
    const opmath_t self_val = iter.scalar_value<opmath_t>(2);
    iter.remove_operand(2);
    const scalar_t* other = static_cast<const scalar_t*>(iter.data_ptr(1));
    if (use_grid_stride) {
      metax::Launch1dGridStrideV2(
          n,
          MulOtherScalarKernelLarge<scalar_t, opmath_t>,
          out,
          self_val,
          other);
    } else {
      metax::Launch1dV2(
          n,
          MulOtherScalarKernelSmall<scalar_t, opmath_t>,
          out,
          self_val,
          other);
    }
  } else {
    TORCH_INTERNAL_ASSERT(iter.ninputs() == 2);
    const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
    const scalar_t* other = static_cast<const scalar_t*>(iter.data_ptr(2));
    LaunchMulContig<scalar_t, opmath_t>(n, out, self, other);
  }
}

} // namespace

namespace {

template <typename scalar_t, typename opmath_t>
void DispatchMulIter(at::TensorIteratorBase& iter) {
  if (!iter.is_contiguous()) {
    if (iter.is_cpu_scalar(1) || iter.is_cpu_scalar(2)) {
      LaunchMulIter<scalar_t, opmath_t>(iter);
      return;
    }
    elementwise_v2::LaunchMulOffsetIter(iter);
    return;
  }
  LaunchMulIter<scalar_t, opmath_t>(iter);
}

} // namespace

inline at::Tensor MulTensorKernelV2(
    const at::Tensor& self,
    const at::Tensor& other) {
  at::Tensor output;
  auto iter = at::TensorIteratorConfig()
                  .add_output(output)
                  .add_input(self)
                  .add_input(other)
                  .allow_cpu_scalars(true)
                  .promote_inputs_to_common_dtype(true)
                  .cast_common_dtype_to_outputs(true)
                  .enforce_safe_casting_to_output(true)
                  .build();

  if (!iter.is_contiguous() && iter.numel() > 0) {
    if (self.device() != other.device()) {
      if (self.device().is_cpu() && self.numel() == 1) {
        return MulTensorKernelV2(
            other, self.to(other.device(), other.scalar_type()));
      }
      if (other.device().is_cpu() && other.numel() == 1) {
        return MulTensorKernelV2(
            self, other.to(self.device(), self.scalar_type()));
      }
    }
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          at::ScalarType::Bool,
          sub_iter.common_dtype(),
          "mul_metax",
          [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            DispatchMulIter<scalar_t, opmath_t>(sub_iter);
          });
    }
    return iter.output();
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "mul_metax",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        DispatchMulIter<scalar_t, opmath_t>(iter);
      });

  return iter.output();
}

} // namespace at::native::flagos
