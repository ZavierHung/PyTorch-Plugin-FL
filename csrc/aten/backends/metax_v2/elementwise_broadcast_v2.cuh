#pragma once

#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>

namespace at::native::flagos::elementwise_v2 {

struct LastDimBroadcastInfo {
  int broadcast_operand = -1; // 1 or 2 (iter operand index)
  int64_t last_dim_size = 0;
  bool valid = false;
};

struct ContigPlusBroadcastInfo {
  int contig_operand = -1; // 1 or 2
  int broadcast_operand = -1;
  bool valid = false;
};

LastDimBroadcastInfo DetectLastDimBroadcast(const at::TensorIteratorBase& iter);
ContigPlusBroadcastInfo DetectContigPlusBroadcast(
    const at::TensorIteratorBase& iter);

bool TryLaunchMulBroadcastFast(at::TensorIteratorBase& iter);
bool TryLaunchAddBroadcastFast(
    at::TensorIteratorBase& iter,
    const at::Scalar& alpha);
bool TryLaunchMulContigBroadcast(at::TensorIteratorBase& iter);
bool TryLaunchAddContigBroadcast(
    at::TensorIteratorBase& iter,
    const at::Scalar& alpha);

} // namespace at::native::flagos::elementwise_v2
