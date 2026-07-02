// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <c10/util/Exception.h>
#include <c10/util/BFloat16.h>

#include "metax_elementwise_v2.cuh"

#if __has_include(<mcblas/mcblas.h>)
#include <mcblas/mcblas.h>
#define FLAGOS_METAX_HAS_MCBLAS 1
#else
#define FLAGOS_METAX_HAS_MCBLAS 0
#endif

namespace at::native::flagos {

namespace {

#if FLAGOS_METAX_HAS_MCBLAS
inline mcblasHandle_t GetMcblasHandle() {
  thread_local mcblasHandle_t handle = nullptr;
  if (handle == nullptr) {
    const mcblasStatus_t status = mcblasCreate(&handle);
    TORCH_CHECK(
        status == MCBLAS_STATUS_SUCCESS,
        "MetaX mm: mcblasCreate failed: ",
        mcblasGetStatusString(status));
  }
  return handle;
}

template <typename scalar_t>
struct GemmExConfig;

template <>
struct GemmExConfig<float> {
  using alpha_t = float;
  static constexpr macaDataType type = MACA_R_32F;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_32F;
};

template <>
struct GemmExConfig<double> {
  using alpha_t = double;
  static constexpr macaDataType type = MACA_R_64F;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_64F;
};

template <>
struct GemmExConfig<at::Half> {
  using alpha_t = float;
  static constexpr macaDataType type = MACA_R_16F;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_32F_FAST_16F;
};

template <>
struct GemmExConfig<at::BFloat16> {
  using alpha_t = float;
  static constexpr macaDataType type = MACA_R_16BF;
  static constexpr mcblasComputeType_t compute_type = MCBLAS_COMPUTE_32F_FAST_16BF;
};

struct PreparedMmMatrix {
  at::Tensor tensor;
  bool owned = false;
};

// PyTorch cublasCommonArgs-style layout resolution (2-D).
PreparedMmMatrix PrepareMatrixForMcblas(
    const at::Tensor& tensor,
    bool& transpose_tensor,
    bool transpose_result) {
  PreparedMmMatrix prepared;
  if (tensor.is_non_overlapping_and_dense()) {
    transpose_tensor = tensor.is_contiguous();
    prepared.tensor = tensor;
    return prepared;
  }

  const int64_t s0 = tensor.stride(0);
  const int64_t s1 = tensor.stride(1);
  const int64_t r = tensor.size(0);
  const int64_t c = tensor.size(1);
  if (s0 == 1 && s1 >= std::max<int64_t>(1, r)) {
    transpose_tensor = false;
    prepared.tensor = tensor;
  } else if (s1 == 1 && s0 >= std::max<int64_t>(1, c)) {
    transpose_tensor = true;
    prepared.tensor = tensor;
  } else {
    transpose_tensor = true;
    prepared.tensor = tensor.contiguous();
    prepared.owned = true;
  }
  return prepared;
}

bool PrepareResultTransposeFlag(const at::Tensor& result) {
  if (result.is_non_overlapping_and_dense()) {
    return result.is_contiguous();
  }
  const int64_t s0 = result.stride(0);
  const int64_t s1 = result.stride(1);
  const int64_t r = result.size(0);
  const int64_t c = result.size(1);
  if (s0 == 1 && s1 >= std::max<int64_t>(1, r)) {
    return false;
  }
  if (s1 == 1 && s0 >= std::max<int64_t>(1, c)) {
    return true;
  }
  return true;
}

template <typename scalar_t>
struct McblasMmLaunchArgs {
  const scalar_t* a_ptr = nullptr;
  const scalar_t* b_ptr = nullptr;
  scalar_t* c_ptr = nullptr;
  mcblasOperation_t op_a = MCBLAS_OP_N;
  mcblasOperation_t op_b = MCBLAS_OP_N;
  int m = 0;
  int n = 0;
  int k = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  at::Tensor out_work;
  PreparedMmMatrix mat_a_prepared;
  PreparedMmMatrix mat_b_prepared;
};

template <typename scalar_t>
McblasMmLaunchArgs<scalar_t> BuildMcblasMmArgs(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  McblasMmLaunchArgs<scalar_t> args;
  const bool transpose_result = PrepareResultTransposeFlag(out);

  bool transpose_a = false;
  bool transpose_b = false;
  args.mat_a_prepared = PrepareMatrixForMcblas(
      transpose_result ? mat2 : self, transpose_a, transpose_result);
  args.mat_b_prepared = PrepareMatrixForMcblas(
      transpose_result ? self : mat2, transpose_b, transpose_result);

  if (transpose_result) {
    transpose_a = !transpose_a;
    transpose_b = !transpose_b;
  }

  const at::Tensor& mata = args.mat_a_prepared.tensor;
  const at::Tensor& matb = args.mat_b_prepared.tensor;

  const int64_t m64 = mata.size(transpose_result ? 1 : 0);
  const int64_t k64 = mata.size(transpose_result ? 0 : 1);
  const int64_t n64 = matb.size(transpose_result ? 0 : 1);
  TORCH_CHECK(
      m64 <= std::numeric_limits<int>::max() &&
          n64 <= std::numeric_limits<int>::max() &&
          k64 <= std::numeric_limits<int>::max(),
      "MetaX mm: shape too large for mcblasGemmEx int32 dimensions");
  TORCH_CHECK(
      self.size(1) == mat2.size(0),
      "MetaX mm: shape mismatch");

  args.m = static_cast<int>(m64);
  args.k = static_cast<int>(k64);
  args.n = static_cast<int>(n64);
  args.lda = static_cast<int>(
      mata.stride((transpose_a == transpose_result) ? 1 : 0));
  args.ldb = static_cast<int>(
      matb.stride((transpose_b == transpose_result) ? 1 : 0));

  args.op_a = transpose_a ? MCBLAS_OP_T : MCBLAS_OP_N;
  args.op_b = transpose_b ? MCBLAS_OP_T : MCBLAS_OP_N;

  if (out.is_contiguous()) {
    args.out_work = out;
    args.ldc = static_cast<int>(out.stride(transpose_result ? 0 : 1));
    args.c_ptr = out.template data_ptr<scalar_t>();
  } else {
    args.out_work = at::empty(
        out.sizes(),
        out.options().memory_format(at::MemoryFormat::Contiguous));
    args.ldc = static_cast<int>(args.out_work.stride(0));
    args.c_ptr = args.out_work.template data_ptr<scalar_t>();
  }

  args.a_ptr = mata.template data_ptr<scalar_t>();
  args.b_ptr = matb.template data_ptr<scalar_t>();
  return args;
}

template <typename scalar_t>
void LaunchMmMcblasV2(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  auto args = BuildMcblasMmArgs<scalar_t>(self, mat2, out);

  const auto handle = GetMcblasHandle();
  const mcblasStatus_t set_stream_status =
      mcblasSetStream(handle, reinterpret_cast<mcStream_t>(metax::CurrentStream()));
  TORCH_CHECK(
      set_stream_status == MCBLAS_STATUS_SUCCESS,
      "MetaX mm: mcblasSetStream failed: ",
      mcblasGetStatusString(set_stream_status));

  using cfg = GemmExConfig<scalar_t>;
  const typename cfg::alpha_t alpha = static_cast<typename cfg::alpha_t>(1);
  const typename cfg::alpha_t beta = static_cast<typename cfg::alpha_t>(0);

  mcblasStatus_t status = MCBLAS_STATUS_SUCCESS;
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    at::Half alpha_h = at::Half(1.0f);
    at::Half beta_h = at::Half(0.0f);
    status = mcblasHgemm(
        handle,
        args.op_a,
        args.op_b,
        args.m,
        args.n,
        args.k,
        reinterpret_cast<const mcblas_half*>(&alpha_h),
        reinterpret_cast<const mcblas_half*>(args.a_ptr),
        args.lda,
        reinterpret_cast<const mcblas_half*>(args.b_ptr),
        args.ldb,
        reinterpret_cast<const mcblas_half*>(&beta_h),
        reinterpret_cast<mcblas_half*>(args.c_ptr),
        args.ldc);
  } else {
    status = mcblasGemmEx(
        handle,
        args.op_a,
        args.op_b,
        args.m,
        args.n,
        args.k,
        &alpha,
        args.a_ptr,
        cfg::type,
        args.lda,
        args.b_ptr,
        cfg::type,
        args.ldb,
        &beta,
        args.c_ptr,
        cfg::type,
        args.ldc,
        cfg::compute_type,
        MCBLAS_GEMM_DEFAULT);
  }

  TORCH_CHECK(
      status == MCBLAS_STATUS_SUCCESS,
      "MetaX mm: mcblas call failed: ",
      mcblasGetStatusString(status));

  if (!out.is_contiguous()) {
    out.copy_(args.out_work);
  }
}
#endif

} // namespace

inline void MmKernelMetaxV2(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2, "MetaX mm: inputs must be 2-D");
  TORCH_CHECK(
      self.scalar_type() == mat2.scalar_type(),
      "MetaX mm: expected self and mat2 to have the same dtype");
  at::native::resize_output(out, {self.size(0), mat2.size(1)});
  TORCH_CHECK(
      out.scalar_type() == self.scalar_type(),
      "MetaX mm: expected out to have the same dtype as inputs");

  if (out.numel() == 0) {
    return;
  }

#if FLAGOS_METAX_HAS_MCBLAS
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "mm_metax",
      [&]() { LaunchMmMcblasV2<scalar_t>(self, mat2, out); });
#else
  TORCH_CHECK(false, "MetaX mm requires mcblas headers and library");
#endif
}

} // namespace at::native::flagos
