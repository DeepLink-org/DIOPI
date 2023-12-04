#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu::native {

at::Tensor& NPUNativeFunctions::npu_format_cast_(at::Tensor& self, const at::Tensor& src) {
    auto device = self.device();
    auto x = self.cpu().copy_(src.cpu()).to(device);
    return self.copy_(x);
}

}  // namespace at_npu::native
