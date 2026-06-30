// MetaX _to_copy opt: skip redundant contiguous() on hot paths.
#include "../../copy_dispatcher.h"
#include "../../contiguous_ops.h"
#include "../../copy_ops.h"

#include <include/flagos.h>

namespace at::native::flagos {

namespace {

at::Tensor CloneContiguousD2d(const at::Tensor& self) {
  auto result = at::empty_like(self);
  const size_t nbytes =
      static_cast<size_t>(self.numel()) * static_cast<size_t>(self.element_size());
  if (nbytes > 0) {
    Memcpy(result.data_ptr(), self.data_ptr(), nbytes, MemcpyDeviceToDevice);
  }
  return result;
}

at::Tensor ToCopyKernelMetaxOpt(
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
        return CloneContiguousD2d(self);
      }
      resolved_format = c10::MemoryFormat::Contiguous;
    }
    if (self.is_contiguous(resolved_format)) {
      return CloneContiguousD2d(self);
    }
    return contiguous(self, resolved_format);
  }

  return at::native::flagos::_to_copy(
      self, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

} // namespace

REGISTER_IMPL_TO_DISPATCHER(
    ToCopyFn, to_copy_dispatcher, Backend::kMetaxOpt, ToCopyKernelMetaxOpt)

} // namespace at::native::flagos
