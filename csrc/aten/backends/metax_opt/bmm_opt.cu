// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bmm.h"

#include "bmm_kernel_opt.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(BmmFn, bmm_dispatcher, Backend::kMetaxOpt, BmmKernelMetaxOpt)

} // namespace at::native::flagos
