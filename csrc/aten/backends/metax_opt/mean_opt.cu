// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mean.h"

#include "mean_kernel_opt.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    MeanDimFn, mean_dim_dispatcher, Backend::kMetaxOpt, MeanDimKernelOpt)

} // namespace at::native::flagos
