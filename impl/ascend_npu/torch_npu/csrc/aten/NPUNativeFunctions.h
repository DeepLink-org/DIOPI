#pragma once

#include <ATen/ATen.h>

namespace at_npu {

namespace native {

namespace NPUNativeFunctions {

int64_t get_storage_size(const at::Tensor& self);
at::Tensor& npu_format_cast_(at::Tensor& dst, const at::Tensor& src);

// conver self to dst'format, write the result into new result tensor
at::Tensor npu_format_cast(const at::Tensor& src, const at::Tensor& dst);

// conver self to acl_format, write the result into self
at::Tensor& npu_format_cast_(at::Tensor& src, int64_t acl_format);

int64_t get_npu_format(const at::Tensor& src);
at::Tensor _npu_format_cast(const at::Tensor& self, int64_t acl_format);

at::Tensor npu_format_cast(const at::Tensor& self, int64_t acl_format);

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking);

};  // namespace NPUNativeFunctions

};  // namespace native

};  // namespace at_npu
