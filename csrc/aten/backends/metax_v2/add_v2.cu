// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../add.h"

#include "add_kernel_v2.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    AddTensorFn, add_tensor_dispatcher, Backend::kMetaxV2, AddTensorKernelV2)

} // namespace at::native::flagos
