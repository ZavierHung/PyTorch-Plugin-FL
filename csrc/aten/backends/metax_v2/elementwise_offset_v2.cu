// Copyright (c) 2026, BAAI. All rights reserved.

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>
#include <c10/util/Exception.h>

#include "elementwise_broadcast_v2.cuh"
#include "elementwise_offset_v2.cuh"
#include "metax_elementwise_v2.cuh"

namespace at::native::flagos::elementwise_v2 {

namespace {

constexpr int kOffsetBlockSize = 256;

template <int NTENSORS>
OffsetCalculator<NTENSORS> MakeOffsetCalculator(
    const at::TensorIteratorBase& iter) {
  std::array<const int64_t*, NTENSORS> strides;
  int64_t element_sizes[NTENSORS];
  for (int i = 0; i < NTENSORS; ++i) {
    strides[i] = iter.strides(i).data();
    element_sizes[i] = iter.element_size(i);
  }
  return OffsetCalculator<NTENSORS>(
      iter.ndim(), iter.shape().data(), strides.data(), element_sizes);
}

template <typename scalar_t, typename opmath_t>
__global__ void MulOffsetKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other,
    OffsetCalculator<3> calc,
    bool out_contiguous) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t linear_idx = tid; linear_idx < n; linear_idx += stride) {
    const auto offsets = calc.get(static_cast<uint32_t>(linear_idx));
    const int64_t out_idx = out_contiguous ? linear_idx : offsets[0];
    if constexpr (std::is_same_v<scalar_t, bool>) {
      out[out_idx] = self[offsets[1]] && other[offsets[2]];
    } else {
      out[out_idx] = static_cast<scalar_t>(
          static_cast<opmath_t>(self[offsets[1]]) *
          static_cast<opmath_t>(other[offsets[2]]));
    }
  }
}

template <typename scalar_t, typename opmath_t>
__global__ void AddOffsetKernel(
    int64_t n,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ other,
    opmath_t alpha,
    OffsetCalculator<3> calc,
    bool out_contiguous) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t linear_idx = tid; linear_idx < n; linear_idx += stride) {
    const auto offsets = calc.get(static_cast<uint32_t>(linear_idx));
    const int64_t out_idx = out_contiguous ? linear_idx : offsets[0];
    out[out_idx] = static_cast<scalar_t>(
        static_cast<opmath_t>(self[offsets[1]]) +
        alpha * static_cast<opmath_t>(other[offsets[2]]));
  }
}

template <typename scalar_t, typename opmath_t>
void LaunchMulOffset(
    at::TensorIteratorBase& iter,
    scalar_t* out,
    const scalar_t* self,
    const scalar_t* other,
    int64_t n) {
  const auto calc = MakeOffsetCalculator<3>(iter);
  const bool out_contiguous = iter.output().is_contiguous();
  const int blocks = metax::ComputeGrid1d(n);
  MulOffsetKernel<scalar_t, opmath_t>
      <<<blocks, kOffsetBlockSize, 0, metax::CurrentStream()>>>(
          n, out, self, other, calc, out_contiguous);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX v2 mul offset kernel launch failed: ",
      cudaGetErrorString(err));
}

template <typename scalar_t, typename opmath_t>
void LaunchAddOffset(
    at::TensorIteratorBase& iter,
    scalar_t* out,
    const scalar_t* self,
    const scalar_t* other,
    opmath_t alpha,
    int64_t n) {
  const auto calc = MakeOffsetCalculator<3>(iter);
  const bool out_contiguous = iter.output().is_contiguous();
  const int blocks = metax::ComputeGrid1d(n);
  AddOffsetKernel<scalar_t, opmath_t>
      <<<blocks, kOffsetBlockSize, 0, metax::CurrentStream()>>>(
          n, out, self, other, alpha, calc, out_contiguous);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX v2 add offset kernel launch failed: ",
      cudaGetErrorString(err));
}

} // namespace

void LaunchMulOffsetIter(at::TensorIteratorBase& iter) {
  if (TryLaunchMulBroadcastFast(iter)) {
    return;
  }
  if (TryLaunchMulContigBroadcast(iter)) {
    return;
  }
  TORCH_INTERNAL_ASSERT(iter.ninputs() == 2);
  const int64_t n = iter.numel();
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "mul_metax_offset",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto* out = static_cast<scalar_t*>(iter.data_ptr(0));
        const auto* self = static_cast<const scalar_t*>(iter.data_ptr(1));
        const auto* other = static_cast<const scalar_t*>(iter.data_ptr(2));
        LaunchMulOffset<scalar_t, opmath_t>(iter, out, self, other, n);
      });
}

void LaunchAddOffsetIter(
    at::TensorIteratorBase& iter,
    const at::Scalar& alpha) {
  if (TryLaunchAddBroadcastFast(iter, alpha)) {
    return;
  }
  if (TryLaunchAddContigBroadcast(iter, alpha)) {
    return;
  }
  TORCH_INTERNAL_ASSERT(iter.ninputs() == 2);
  const int64_t n = iter.numel();
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.common_dtype(),
      "add_metax_offset",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto* out = static_cast<scalar_t*>(iter.data_ptr(0));
        const auto* self = static_cast<const scalar_t*>(iter.data_ptr(1));
        const auto* other = static_cast<const scalar_t*>(iter.data_ptr(2));
        LaunchAddOffset<scalar_t, opmath_t>(
            iter, out, self, other, alpha.to<opmath_t>(), n);
      });
}

} // namespace at::native::flagos::elementwise_v2
