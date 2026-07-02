// Vectorized elementwise helpers for metax_v2 (fp16 __half2, bf16, fp32 float4).
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace at::native::flagos::elementwise_v2 {

constexpr int64_t kVecMinElements = 8192;
constexpr int64_t kVecMinElementsHalf = 512;

inline bool PtrAligned4(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 3u) == 0;
}

inline bool CanVectorizeBinary(
    int64_t n,
    const void* out,
    const void* a,
    const void* b,
    int width) {
  if (n < kVecMinElements) {
    return false;
  }
  if ((n % width) != 0) {
    return false;
  }
  return PtrAligned4(out) && PtrAligned4(a) && PtrAligned4(b);
}

inline bool CanVectorizeHalfBinary(
    int64_t n,
    const void* out,
    const void* a,
    const void* b,
    int width) {
  if (n < kVecMinElementsHalf) {
    return false;
  }
  if ((n % width) != 0) {
    return false;
  }
  return PtrAligned4(out) && PtrAligned4(a) && PtrAligned4(b);
}

void LaunchAddHalf2(
    int64_t n2,
    __half* out,
    const __half* self,
    const __half* other,
    float alpha);

void LaunchAddHalf2Alpha1(
    int64_t n2,
    __half* out,
    const __half* self,
    const __half* other);

void LaunchMulHalf2(
    int64_t n2,
    __half* out,
    const __half* self,
    const __half* other);

void LaunchAddBfloat162(
    int64_t n2,
    __nv_bfloat16* out,
    const __nv_bfloat16* self,
    const __nv_bfloat16* other,
    float alpha);

void LaunchAddBfloat162Alpha1(
    int64_t n2,
    __nv_bfloat16* out,
    const __nv_bfloat16* self,
    const __nv_bfloat16* other);

void LaunchMulBfloat162(
    int64_t n2,
    __nv_bfloat16* out,
    const __nv_bfloat16* self,
    const __nv_bfloat16* other);

void LaunchAddFloat4(
    int64_t n4,
    float* out,
    const float* self,
    const float* other,
    float alpha);

void LaunchAddFloat4Alpha1(
    int64_t n4,
    float* out,
    const float* self,
    const float* other);

void LaunchMulFloat4(
    int64_t n4,
    float* out,
    const float* self,
    const float* other);

} // namespace at::native::flagos::elementwise_v2
