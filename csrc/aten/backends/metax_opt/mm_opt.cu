// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mm.h"

#include "mm_kernel_opt.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kMetaxOpt, MmKernelMetaxOpt)

} // namespace at::native::flagos
