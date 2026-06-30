// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mul.h"

#include "mul_kernel_v2.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    MulTensorFn, mul_tensor_dispatcher, Backend::kMetaxV2, MulTensorKernelV2)

} // namespace at::native::flagos
