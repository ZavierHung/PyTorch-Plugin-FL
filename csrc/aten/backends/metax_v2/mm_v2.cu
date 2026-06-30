// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../mm.h"

#include "mm_kernel_v2.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(MmFn, mm_dispatcher, Backend::kMetaxV2, MmKernelMetaxV2)

} // namespace at::native::flagos
