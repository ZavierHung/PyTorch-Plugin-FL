// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../add.h"

#include "add_kernel_opt.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    AddTensorFn, add_tensor_dispatcher, Backend::kMetaxOpt, AddTensorKernelOpt)

} // namespace at::native::flagos
