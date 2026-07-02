// Copyright (c) 2026, BAAI. All rights reserved.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/util/Exception.h>

#include "elementwise_vec_v2.cuh"
#include "metax_elementwise_v2.cuh"

namespace at::native::flagos::elementwise_v2 {

namespace {

constexpr int kVecBlockSize = 512;

template <typename Kernel, typename... Args>
void LaunchVec1dImpl(int64_t n, Kernel kernel, Args... args) {
  if (n == 0) {
    return;
  }
  const int64_t blocks_raw =
      (n + static_cast<int64_t>(kVecBlockSize) - 1) / kVecBlockSize;
  const int blocks = static_cast<int>(
      std::min<int64_t>(blocks_raw, metax::kMaxGridV2));
  if (blocks == 0) {
    return;
  }
  kernel<<<blocks, kVecBlockSize, 0, metax::CurrentStream()>>>(n, args...);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX v2 vector kernel launch failed: ",
      cudaGetErrorString(err));
}

__global__ void AddHalf2Kernel(
    int64_t n2,
    __half* __restrict__ out,
    const __half* __restrict__ self,
    const __half* __restrict__ other,
    float alpha) {
  const __half2 alpha_h2 = __floats2half2_rn(alpha, alpha);
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const __half2* self_h2 = reinterpret_cast<const __half2*>(self);
  const __half2* other_h2 = reinterpret_cast<const __half2*>(other);
  __half2* out_h2 = reinterpret_cast<__half2*>(out);
  for (int64_t i = tid; i < n2; i += stride) {
    out_h2[i] = __hfma2(other_h2[i], alpha_h2, self_h2[i]);
  }
}

__global__ void AddHalf2Alpha1Kernel(
    int64_t n2,
    __half* __restrict__ out,
    const __half* __restrict__ self,
    const __half* __restrict__ other) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const __half2* self_h2 = reinterpret_cast<const __half2*>(self);
  const __half2* other_h2 = reinterpret_cast<const __half2*>(other);
  __half2* out_h2 = reinterpret_cast<__half2*>(out);
  for (int64_t i = tid; i < n2; i += stride) {
    out_h2[i] = __hadd2(self_h2[i], other_h2[i]);
  }
}

__global__ void MulHalf2Kernel(
    int64_t n2,
    __half* __restrict__ out,
    const __half* __restrict__ self,
    const __half* __restrict__ other) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const __half2* self_h2 = reinterpret_cast<const __half2*>(self);
  const __half2* other_h2 = reinterpret_cast<const __half2*>(other);
  __half2* out_h2 = reinterpret_cast<__half2*>(out);
  for (int64_t i = tid; i < n2; i += stride) {
    out_h2[i] = __hmul2(self_h2[i], other_h2[i]);
  }
}

__global__ void AddBfloat162Kernel(
    int64_t n2,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ self,
    const __nv_bfloat16* __restrict__ other,
    float alpha) {
  const __nv_bfloat162 alpha_b2 =
      __floats2bfloat162_rn(alpha, alpha);
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const __nv_bfloat162* self_b2 =
      reinterpret_cast<const __nv_bfloat162*>(self);
  const __nv_bfloat162* other_b2 =
      reinterpret_cast<const __nv_bfloat162*>(other);
  __nv_bfloat162* out_b2 = reinterpret_cast<__nv_bfloat162*>(out);
  for (int64_t i = tid; i < n2; i += stride) {
    out_b2[i] = __hfma2(other_b2[i], alpha_b2, self_b2[i]);
  }
}

__global__ void AddBfloat162Alpha1Kernel(
    int64_t n2,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ self,
    const __nv_bfloat16* __restrict__ other) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const __nv_bfloat162* self_b2 =
      reinterpret_cast<const __nv_bfloat162*>(self);
  const __nv_bfloat162* other_b2 =
      reinterpret_cast<const __nv_bfloat162*>(other);
  __nv_bfloat162* out_b2 = reinterpret_cast<__nv_bfloat162*>(out);
  for (int64_t i = tid; i < n2; i += stride) {
    out_b2[i] = __hadd2(self_b2[i], other_b2[i]);
  }
}

__global__ void MulBfloat162Kernel(
    int64_t n2,
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ self,
    const __nv_bfloat16* __restrict__ other) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const __nv_bfloat162* self_b2 =
      reinterpret_cast<const __nv_bfloat162*>(self);
  const __nv_bfloat162* other_b2 =
      reinterpret_cast<const __nv_bfloat162*>(other);
  __nv_bfloat162* out_b2 = reinterpret_cast<__nv_bfloat162*>(out);
  for (int64_t i = tid; i < n2; i += stride) {
    out_b2[i] = __hmul2(self_b2[i], other_b2[i]);
  }
}

__global__ void AddFloat4Kernel(
    int64_t n4,
    float* __restrict__ out,
    const float* __restrict__ self,
    const float* __restrict__ other,
    float alpha) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const float4* self_v = reinterpret_cast<const float4*>(self);
  const float4* other_v = reinterpret_cast<const float4*>(other);
  float4* out_v = reinterpret_cast<float4*>(out);
  for (int64_t i = tid; i < n4; i += stride) {
    const float4 a = self_v[i];
    const float4 b = other_v[i];
    float4 c;
    c.x = a.x + alpha * b.x;
    c.y = a.y + alpha * b.y;
    c.z = a.z + alpha * b.z;
    c.w = a.w + alpha * b.w;
    out_v[i] = c;
  }
}

__global__ void AddFloat4Alpha1Kernel(
    int64_t n4,
    float* __restrict__ out,
    const float* __restrict__ self,
    const float* __restrict__ other) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const float4* self_v = reinterpret_cast<const float4*>(self);
  const float4* other_v = reinterpret_cast<const float4*>(other);
  float4* out_v = reinterpret_cast<float4*>(out);
  for (int64_t i = tid; i < n4; i += stride) {
    const float4 a = self_v[i];
    const float4 b = other_v[i];
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    out_v[i] = c;
  }
}

__global__ void MulFloat4Kernel(
    int64_t n4,
    float* __restrict__ out,
    const float* __restrict__ self,
    const float* __restrict__ other) {
  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * blockDim.x;
  const float4* self_v = reinterpret_cast<const float4*>(self);
  const float4* other_v = reinterpret_cast<const float4*>(other);
  float4* out_v = reinterpret_cast<float4*>(out);
  for (int64_t i = tid; i < n4; i += stride) {
    const float4 a = self_v[i];
    const float4 b = other_v[i];
    float4 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;
    c.w = a.w * b.w;
    out_v[i] = c;
  }
}

} // namespace

void LaunchAddHalf2(
    int64_t n2,
    __half* out,
    const __half* self,
    const __half* other,
    float alpha) {
  LaunchVec1dImpl(n2, AddHalf2Kernel, out, self, other, alpha);
}

void LaunchAddHalf2Alpha1(
    int64_t n2,
    __half* out,
    const __half* self,
    const __half* other) {
  LaunchVec1dImpl(n2, AddHalf2Alpha1Kernel, out, self, other);
}

void LaunchMulHalf2(
    int64_t n2,
    __half* out,
    const __half* self,
    const __half* other) {
  LaunchVec1dImpl(n2, MulHalf2Kernel, out, self, other);
}

void LaunchAddBfloat162(
    int64_t n2,
    __nv_bfloat16* out,
    const __nv_bfloat16* self,
    const __nv_bfloat16* other,
    float alpha) {
  LaunchVec1dImpl(n2, AddBfloat162Kernel, out, self, other, alpha);
}

void LaunchAddBfloat162Alpha1(
    int64_t n2,
    __nv_bfloat16* out,
    const __nv_bfloat16* self,
    const __nv_bfloat16* other) {
  LaunchVec1dImpl(n2, AddBfloat162Alpha1Kernel, out, self, other);
}

void LaunchMulBfloat162(
    int64_t n2,
    __nv_bfloat16* out,
    const __nv_bfloat16* self,
    const __nv_bfloat16* other) {
  LaunchVec1dImpl(n2, MulBfloat162Kernel, out, self, other);
}

void LaunchAddFloat4(
    int64_t n4,
    float* out,
    const float* self,
    const float* other,
    float alpha) {
  LaunchVec1dImpl(n4, AddFloat4Kernel, out, self, other, alpha);
}

void LaunchAddFloat4Alpha1(
    int64_t n4,
    float* out,
    const float* self,
    const float* other) {
  LaunchVec1dImpl(n4, AddFloat4Alpha1Kernel, out, self, other);
}

void LaunchMulFloat4(
    int64_t n4,
    float* out,
    const float* self,
    const float* other) {
  LaunchVec1dImpl(n4, MulFloat4Kernel, out, self, other);
}

} // namespace at::native::flagos::elementwise_v2
