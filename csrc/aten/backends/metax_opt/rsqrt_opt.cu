// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../rsqrt.h"

#include "rsqrt_kernel_opt.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    RsqrtFn, rsqrt_dispatcher, Backend::kMetaxOpt, RsqrtKernelMetax)

} // namespace at::native::flagos
