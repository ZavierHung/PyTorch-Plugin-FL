#pragma once

#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>

namespace at::native::flagos::elementwise_v2 {

void LaunchMulOffsetIter(at::TensorIteratorBase& iter);
void LaunchAddOffsetIter(at::TensorIteratorBase& iter, const at::Scalar& alpha);

} // namespace at::native::flagos::elementwise_v2
