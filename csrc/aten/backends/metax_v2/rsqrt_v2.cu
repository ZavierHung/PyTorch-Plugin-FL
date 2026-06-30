// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../rsqrt.h"

#include "rsqrt_kernel_v2.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    RsqrtFn, rsqrt_dispatcher, Backend::kMetaxV2, RsqrtKernelMetaxV2)

} // namespace at::native::flagos
