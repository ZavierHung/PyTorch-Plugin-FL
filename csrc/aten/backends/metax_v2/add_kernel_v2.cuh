// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>
#include <type_traits>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>
#include <c10/util/BFloat16.h>

#include "elementwise_broadcast_v2.cuh"
#include "elementwise_offset_v2.cuh"
#include "elementwise_vec_v2.cuh"
#include "metax_elementwise_v2.cuh"

namespace at::native::flagos {

namespace {

constexpr int64_t kGridStrideThresholdV2 =
    static_cast<int64_t>(metax::kMaxGridV2) * metax::kBlockSizeV2;

using elementwise_v2::CanVectorizeBinary;
using elementwise_v2::CanVectorizeHalfBinary;
using elementwise_v2::LaunchAddBfloat162;
using elementwise_v2::LaunchAddBfloat162Alpha1;
using elementwise_v2::LaunchAddFloat4;
using elementwise_v2::LaunchAddFloat4Alpha1;
using elementwise_v2::LaunchAddHalf2;
using elementwise_v2::LaunchAddHalf2Alpha1;

template <typename scalar_t, typename opmath_t>
__global__ void AddContigKernelSmall(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other,
    opmath_t alpha) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = static_cast<scalar_t>(
      static_cast<opmath_t>(self[idx]) +
      alpha * static_cast<opmath_t>(other[idx]));
}

template <typename scalar_t, typename opmath_t>
__global__ void AddContigKernelLarge(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other,
    opmath_t alpha) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = tid; idx < n; idx += stride) {
    out[idx] = static_cast<scalar_t>(
        static_cast<opmath_t>(self[idx]) +
        alpha * static_cast<opmath_t>(other[idx]));
  }
}

template <typename scalar_t, typename opmath_t>
void LaunchAddContig(
    int64_t n,
    scalar_t* out,
    const scalar_t* self,
    const scalar_t* other,
    opmath_t alpha) {
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if (CanVectorizeHalfBinary(n, out, self, other, 2) ||
        CanVectorizeBinary(n, out, self, other, 2)) {
      const int64_t n2 = n / 2;
      auto* out_h = reinterpret_cast<__half*>(out);
      const auto* self_h = reinterpret_cast<const __half*>(self);
      const auto* other_h = reinterpret_cast<const __half*>(other);
      if (alpha == opmath_t(1)) {
        LaunchAddHalf2Alpha1(n2, out_h, self_h, other_h);
      } else {
        LaunchAddHalf2(
            n2, out_h, self_h, other_h, static_cast<float>(alpha));
      }
      return;
    }
  } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
    if (CanVectorizeHalfBinary(n, out, self, other, 2) ||
        CanVectorizeBinary(n, out, self, other, 2)) {
      const int64_t n2 = n / 2;
      auto* out_b = reinterpret_cast<__nv_bfloat16*>(out);
      const auto* self_b = reinterpret_cast<const __nv_bfloat16*>(self);
      const auto* other_b = reinterpret_cast<const __nv_bfloat16*>(other);
      if (alpha == opmath_t(1)) {
        LaunchAddBfloat162Alpha1(n2, out_b, self_b, other_b);
      } else {
        LaunchAddBfloat162(
            n2, out_b, self_b, other_b, static_cast<float>(alpha));
      }
      return;
    }
  } else if constexpr (std::is_same_v<scalar_t, float>) {
    if (CanVectorizeBinary(n, out, self, other, 4)) {
      const int64_t n4 = n / 4;
      if (alpha == opmath_t(1)) {
        LaunchAddFloat4Alpha1(n4, out, self, other);
      } else {
        LaunchAddFloat4(
            n4, out, self, other, static_cast<float>(alpha));
      }
      return;
    }
  }

  if (n > kGridStrideThresholdV2) {
    metax::Launch1dGridStrideV2(
        n, AddContigKernelLarge<scalar_t, opmath_t>, out, self, other, alpha);
  } else {
    metax::Launch1dV2(
        n, AddContigKernelSmall<scalar_t, opmath_t>, out, self, other, alpha);
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void AddSelfScalarKernelSmall(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    opmath_t other_val,
    opmath_t alpha) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = static_cast<scalar_t>(
      static_cast<opmath_t>(self[idx]) + alpha * other_val);
}

template <typename scalar_t, typename opmath_t>
__global__ void AddSelfScalarKernelLarge(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    opmath_t other_val,
    opmath_t alpha) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = tid; idx < n; idx += stride) {
    out[idx] = static_cast<scalar_t>(
        static_cast<opmath_t>(self[idx]) + alpha * other_val);
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void AddOtherScalarKernelSmall(
    int64_t n,
    scalar_t* __restrict__ out,
    opmath_t self_val,
    const scalar_t* __restrict__ other,
    opmath_t alpha) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = static_cast<scalar_t>(
      self_val + alpha * static_cast<opmath_t>(other[idx]));
}

template <typename scalar_t, typename opmath_t>
__global__ void AddOtherScalarKernelLarge(
    int64_t n,
    scalar_t* __restrict__ out,
    opmath_t self_val,
    const scalar_t* __restrict__ other,
    opmath_t alpha) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = tid; idx < n; idx += stride) {
    out[idx] = static_cast<scalar_t>(
        self_val + alpha * static_cast<opmath_t>(other[idx]));
  }
}

template <typename scalar_t, typename opmath_t>
void LaunchAddIter(at::TensorIteratorBase& iter, opmath_t alpha) {
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
          AddSelfScalarKernelLarge<scalar_t, opmath_t>,
          out,
          self,
          other_val,
          alpha);
    } else {
      metax::Launch1dV2(
          n,
          AddSelfScalarKernelSmall<scalar_t, opmath_t>,
          out,
          self,
          other_val,
          alpha);
    }
  } else if (iter.is_cpu_scalar(2)) {
    const opmath_t self_val = iter.scalar_value<opmath_t>(2);
    iter.remove_operand(2);
    const scalar_t* other = static_cast<const scalar_t*>(iter.data_ptr(1));
    if (use_grid_stride) {
      metax::Launch1dGridStrideV2(
          n,
          AddOtherScalarKernelLarge<scalar_t, opmath_t>,
          out,
          self_val,
          other,
          alpha);
    } else {
      metax::Launch1dV2(
          n,
          AddOtherScalarKernelSmall<scalar_t, opmath_t>,
          out,
          self_val,
          other,
          alpha);
    }
  } else {
    TORCH_INTERNAL_ASSERT(iter.ninputs() == 2);
    const scalar_t* self = static_cast<const scalar_t*>(iter.data_ptr(1));
    const scalar_t* other = static_cast<const scalar_t*>(iter.data_ptr(2));
    LaunchAddContig<scalar_t, opmath_t>(n, out, self, other, alpha);
  }
}

} // namespace

namespace {

template <typename scalar_t, typename opmath_t>
void DispatchAddIter(at::TensorIteratorBase& iter, const at::Scalar& alpha) {
  if (!iter.is_contiguous()) {
    if (iter.is_cpu_scalar(1) || iter.is_cpu_scalar(2)) {
      LaunchAddIter<scalar_t, opmath_t>(iter, alpha.to<opmath_t>());
      return;
    }
    elementwise_v2::LaunchAddOffsetIter(iter, alpha);
    return;
  }
  LaunchAddIter<scalar_t, opmath_t>(iter, alpha.to<opmath_t>());
}

} // namespace

inline at::Tensor AddTensorKernelV2(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {
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
        return AddTensorKernelV2(
            other,
            self.to(other.device(), other.scalar_type()),
            alpha);
      }
      if (other.device().is_cpu() && other.numel() == 1) {
        return AddTensorKernelV2(
            self,
            other.to(self.device(), self.scalar_type()),
            alpha);
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
          "add_metax",
          [&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            DispatchAddIter<scalar_t, opmath_t>(sub_iter, alpha);
          });
    }
    return iter.output();
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "add_metax",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        DispatchAddIter<scalar_t, opmath_t>(iter, alpha);
      });

  return iter.output();
}

} // namespace at::native::flagos
