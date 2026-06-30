// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../pow.h"

#include "pow_kernel_opt.cuh"

namespace at::native::flagos {

REGISTER_IMPL_TO_DISPATCHER(
    PowTensorScalarFn,
    pow_tensor_scalar_dispatcher,
    Backend::kMetaxOpt,
    PowTensorScalarKernelMetax)

} // namespace at::native::flagos
