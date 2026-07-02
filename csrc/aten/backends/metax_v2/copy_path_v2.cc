// MetaX _to_copy v2: same-device fast path without DeviceBoxingGuard (aligned with opt).
#include "../../copy_dispatcher.h"
#include "../../contiguous_ops.h"
#include "../../copy_ops.h"

#include <include/flagos.h>

#include "copy_path_v2.h"

namespace at::native::flagos {

namespace {

at::Tensor CloneContiguousD2dV2(const at::Tensor& self) {
  auto result = at::empty_like(self);
  MetaxStridedCopyV2(self, result);
  return result;
}

at::Tensor ToCopyKernelMetaxV2(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format) {
  auto resolved_device = device.value_or(self.device());
  auto resolved_dtype = dtype.value_or(self.scalar_type());
  auto resolved_format = memory_format.value_or(c10::MemoryFormat::Preserve);

  if (resolved_device == self.device() && resolved_dtype == self.scalar_type()) {
    if (resolved_format == c10::MemoryFormat::Preserve) {
      if (self.is_contiguous()) {
        return CloneContiguousD2dV2(self);
      }
      resolved_format = c10::MemoryFormat::Contiguous;
    }
    if (self.is_contiguous(resolved_format)) {
      return CloneContiguousD2dV2(self);
    }
    auto result = at::empty_like(self, self.options().memory_format(resolved_format));
    MetaxStridedCopyV2(self, result);
    return result;
  }

  return at::native::flagos::_to_copy(
      self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(
    ToCopyFn, to_copy_dispatcher, Backend::kMetaxV2, ToCopyKernelMetaxV2)

} // namespace at::native::flagos
