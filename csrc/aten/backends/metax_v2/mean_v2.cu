// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mean.h"

#include "mean_kernel_v2.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    MeanDimFn, mean_dim_dispatcher, Backend::kMetaxV2, MeanDimKernelV2)

} // namespace at::native::flagos
