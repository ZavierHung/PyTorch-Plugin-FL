#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::flagos {

void MetaxStridedCopyV2(const at::Tensor& src, at::Tensor& dst);

} // namespace at::native::flagos
