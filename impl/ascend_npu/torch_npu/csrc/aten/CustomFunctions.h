#pragma once

#pragma once

#include <ATen/ATen.h>

namespace at_npu {

namespace native {

void npu_fast_reshape_(at::Tensor& tensor);

namespace custom_ops {

at::Tensor npu_dtype_cast(const at::Tensor& self, at::ScalarType dtype);

at::Tensor npu_format_cast(const at::Tensor& self, int acl_format);

at::Tensor& npu_format_cast_(at::Tensor& self, int acl_format);

int64_t get_npu_format(const at::Tensor& src);

at::Tensor _npu_format_cast(const at::Tensor& self, int64_t acl_format);

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking);


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_cell(
    const at::Tensor& input, const at::Tensor& w_ih, const at::Tensor& w_hh, const at::Tensor& h, const at::Tensor& c,
    const c10::optional<at::Tensor>& bias_opt);

std::tuple<at::Tensor, at::Tensor> _npu_ciou(const at::Tensor& self, const at::Tensor& gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag);
std::tuple<at::Tensor, at::Tensor> npu_ciou_backward(const at::Tensor& grad, const at::Tensor& bboxes, const at::Tensor& gtboxes,
                                                     const c10::optional<at::Tensor>& atan_sub_opt, bool trans, bool is_cross, int64_t mode);

at::Tensor npu_ciou(const at::Tensor& self, const at::Tensor& gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag);

}  // namespace custom_ops
}  // namespace native
}  // namespace at_npu
