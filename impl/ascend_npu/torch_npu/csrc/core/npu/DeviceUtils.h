#pragma once
#include <c10/core/TensorOptions.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>
//#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
//#include "torch_npu/csrc/utils/LazyInit.h"


namespace torch_npu {
namespace utils {

inline bool is_npu(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  return tensor.device().type() != at::DeviceType::CPU;
}

inline bool is_npu(const at::TensorOptions& options) {
  return options.device().type() != at::DeviceType::CPU;
}

inline bool is_npu(const at::Device& device) {
  return device.type() != at::DeviceType::CPU;
}

inline void torch_check_npu(const at::Tensor& tensor) {
  TORCH_CHECK(is_npu(tensor),
              "Expected NPU tensor, please check whether the input tensor device is correct.");
}

inline void torch_check_npu(const at::TensorOptions& options) {
  TORCH_CHECK(is_npu(options),
              "Expected NPU tensor, please check whether the input tensor device is correct.");
}

inline void torch_check_npu(const at::Device& device) {
  TORCH_CHECK(is_npu(device),
              "Expected NPU tensor, please check whether the input tensor device is correct.");
}

inline c10::DeviceType get_npu_device_type() {
  return c10::DeviceType::PrivateUse1;
}

inline void maybe_initialize_npu(const at::TensorOptions& options) {

}

inline void maybe_initialize_npu(const at::Device& device) {

}

inline void maybe_initialize_npu(const c10::optional<at::Device>& device) {
  
}

}
}
