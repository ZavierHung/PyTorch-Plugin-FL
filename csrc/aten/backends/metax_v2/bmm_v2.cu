// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../bmm.h"

#include "bmm_kernel_v2.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(BmmFn, bmm_dispatcher, Backend::kMetaxV2, BmmKernelMetaxV2)

} // namespace at::native::flagos
