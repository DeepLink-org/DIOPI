#pragma once

#include <ATen/ATen.h>

namespace acl_op {

at::Tensor add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor& add_out(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& result);

at::Tensor adaptive_avg_pool2d(const at::Tensor& self, at::IntArrayRef output_size);

at::Tensor& fill_(at::Tensor& self, const at::Tensor& other);
at::Tensor& fill_(at::Tensor& self, const at::Scalar& other);

at::Tensor& ones_out(at::IntArrayRef size, at::Tensor& result);

at::Tensor ones(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                c10::optional<bool> pin_memory_opt);
at::Tensor ones(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt,
                c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt);
at::Tensor& one_(at::Tensor& self);

at::Tensor& zero_out(at::IntArrayRef size, at::Tensor& result);

at::Tensor zero(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                c10::optional<bool> pin_memory_opt);
at::Tensor zero(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt,
                c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt);
at::Tensor& one_(at::Tensor& self);

at::Tensor& mul_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);
at::Tensor mul(const at::Tensor& self, const at::Tensor& other);
at::Tensor mul(const at::Tensor& self, const at::Scalar& other);
at::Tensor& mul_(at::Tensor& self, const at::Tensor& other);
at::Tensor& mul_(at::Tensor& self, const at::Scalar& other);

at::Tensor npu_transpose(const at::Tensor& self, at::IntArrayRef perm, bool require_contiguous);
at::Tensor& npu_transpose_out(const at::Tensor& self, at::IntArrayRef perm, bool require_contiguous, at::Tensor& result);
at::Tensor npu_broadcast(const at::Tensor& self, at::IntArrayRef size);

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> native_batch_norm_out(const at::Tensor& self, const c10::optional<at::Tensor>& weight_opt,
                                                                        const c10::optional<at::Tensor>& bias_opt,
                                                                        const c10::optional<at::Tensor>& running_mean_opt,
                                                                        const c10::optional<at::Tensor>& running_var_opt, bool train, double momentum,
                                                                        double eps, at::Tensor& result, at::Tensor& save_mean, at::Tensor& save_invstd);

at::Tensor& bitwise_xor_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result);
at::Tensor& bitwise_or_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result);
at::Tensor& bitwise_and_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result);
at::Tensor eq(const at::Tensor& self, const at::Tensor& other);
at::Tensor eq(const at::Tensor& self, const at::Scalar& other);
at::Tensor& eq_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);
at::Tensor& eq_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result);

at::Tensor& ne_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);
at::Tensor& ne_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result);
at::Tensor ne(const at::Tensor& self, const at::Tensor& other);
at::Tensor ne(const at::Tensor& self, const at::Scalar& other);
at::Tensor& ne_(at::Tensor& self, const at::Tensor& other);
at::Tensor& ne_(at::Tensor& self, const at::Scalar& other);

at::Tensor& ge_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);
at::Tensor& ge_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result);
at::Tensor ge(const at::Tensor& self, const at::Tensor& other);
at::Tensor ge(const at::Tensor& self, const at::Scalar& other);
at::Tensor& ge_(at::Tensor& self, const at::Tensor& other);
at::Tensor& ge_(at::Tensor& self, const at::Scalar& other);

at::Tensor& le_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);
at::Tensor& le_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result);
at::Tensor le(const at::Tensor& self, const at::Tensor& other);
at::Tensor le(const at::Tensor& self, const at::Scalar& other);
at::Tensor& le_(at::Tensor& self, const at::Tensor& other);
at::Tensor& le_(at::Tensor& self, const at::Scalar& other);

at::Tensor gt(const at::Tensor& self, const at::Tensor& other);
at::Tensor gt(const at::Tensor& self, const at::Scalar& other);
at::Tensor& gt_(at::Tensor& self, const at::Tensor& other);
at::Tensor& gt_(at::Tensor& self, const at::Scalar& other);
at::Tensor& gt_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);
at::Tensor& gt_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result);

at::Tensor lt(const at::Tensor& self, const at::Tensor& other);
at::Tensor lt(const at::Tensor& self, const at::Scalar& other);
at::Tensor& lt_(at::Tensor& self, const at::Tensor& other);
at::Tensor& lt_(at::Tensor& self, const at::Scalar& other);
at::Tensor& lt_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);
at::Tensor& lt_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result);

std::tuple<at::Tensor&, at::Tensor&> max_out(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& output, at::Tensor& indices);

std::tuple<at::Tensor&, at::Tensor&> max_out(const at::Tensor& self, at::Dimname dim, bool keepdim, at::Tensor& output, at::Tensor& indices);

std::tuple<at::Tensor, at::Tensor> max(const at::Tensor& self, int64_t dim, bool keepdim);

std::tuple<at::Tensor, at::Tensor> max(const at::Tensor& self, at::Dimname dim, bool keepdim);
at::Tensor& max_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);
at::Tensor& maximum_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);
at::Tensor maximum(const at::Tensor& self, const at::Tensor& other);
at::Tensor amax(const at::Tensor& self, at::IntArrayRef dims, bool keepdim);

at::Tensor& amax_out(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, at::Tensor& result);

at::Tensor npu_dtype_cast(const at::Tensor& self, at::ScalarType dtype);


at::Tensor& sum_out(const at::Tensor& self, at::DimnameList dim, bool keepdim, c10::optional<c10::ScalarType> dtype, at::Tensor& result);
at::Tensor& sum_out(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, c10::optional<c10::ScalarType> dtype, at::Tensor& result);
at::Tensor& sum_out(at::Tensor&, const at::Tensor&, at::OptionalIntArrayRef, bool, c10::optional<at::ScalarType>);
at::Tensor& sum_out(const at::Tensor&, at::IntArrayRef dim, bool, c10::ScalarType, at::Tensor& out);
at::Tensor sum(const at::Tensor& self, at::DimnameList dim, bool keepdim, c10::optional<c10::ScalarType> dtype);
at::Tensor sum(const at::Tensor& self, c10::optional<c10::ScalarType> dtype);

at::Tensor reflection_pad2d(const at::Tensor& self, at::IntArrayRef padding);
at::Tensor& reflection_pad2d_out(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& result);
at::Tensor replication_pad2d_backward(const at::Tensor& grad_output, const at::Tensor& input, at::IntArrayRef padding);
at::Tensor& replication_pad2d_backward_out(const at::Tensor& grad_output, const at::Tensor& input, at::IntArrayRef padding, at::Tensor& grad_input);


at::Tensor& reflection_pad2d_backward_out(const at::Tensor& grad_output, const at::Tensor& input, at::IntArrayRef padding, at::Tensor& grad_input);

at::Tensor reflection_pad2d_backward(const at::Tensor& grad_output, const at::Tensor& input, at::IntArrayRef padding);

at::Tensor& replication_pad2d_out(const at::Tensor& self, at::IntArrayRef padding, at::Tensor& result);

at::Tensor replication_pad2d(const at::Tensor& self, at::IntArrayRef padding);

at::Tensor& amin_out(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, at::Tensor& result);
at::Tensor& amax_out(const at::Tensor& self, at::IntArrayRef dims, bool keepdim, at::Tensor& result);

std::tuple<at::Tensor, at::Tensor> min(const at::Tensor& self, int64_t dim, bool keepdim);
std::tuple<at::Tensor, at::Tensor> max(const at::Tensor& self, int64_t dim, bool keepdim);
at::Tensor min(const at::Tensor& self);
at::Tensor max(const at::Tensor& self);

at::Tensor& index_select_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, at::Tensor& result);

at::Tensor index_select(const at::Tensor& self, int64_t dim, const at::Tensor& index);

at::Tensor& index_select_out(const at::Tensor& self, at::Dimname dim, const at::Tensor& index, at::Tensor& result);

at::Tensor index_select(const at::Tensor& self, at::Dimname dim, const at::Tensor& index);

at::Tensor& trunc_(at::Tensor& self);
at::Tensor trunc(const at::Tensor& self);

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose3d_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
                                                                             at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride,
                                                                             at::IntArrayRef dilation, int64_t groups, std::array<bool, 3> output_mask);

at::Tensor npu_conv_transpose2d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias_opt, at::IntArrayRef padding,
                                at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups);

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv2d_backward(const at::Tensor &input, const at::Tensor &grad,
                                                                   const at::Tensor &weight, at::IntArrayRef stride,
                                                                   at::IntArrayRef padding, at::IntArrayRef dilation,
                                                                   int64_t groups, std::array<bool, 3> grad_input_mask);

at::Tensor floor_divide(const at::Tensor& self, const at::Scalar& other);
at::Tensor floor_divide(const at::Tensor& self, const at::Tensor& other);
at::Tensor& floor_divide_(at::Tensor& self, const at::Scalar& other);
at::Tensor& floor_divide_(at::Tensor& self, const at::Tensor& other);
at::Tensor& floor_divide_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);

at::Tensor& npu_indexing_out(const at::Tensor& self, c10::IntArrayRef begin, c10::IntArrayRef end, c10::IntArrayRef strides, int64_t begin_mask,
                             int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor& result);
at::Tensor npu_indexing(const at::Tensor& self, c10::IntArrayRef begin, c10::IntArrayRef end, c10::IntArrayRef strides, int64_t begin_mask, int64_t end_mask,
                        int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask);

at::Tensor& npu_reshape_out(const at::Tensor& src, at::IntArrayRef shape, bool can_refresh, at::Tensor& result);
at::Tensor npu_reshape(const at::Tensor& self, at::IntArrayRef shape, bool can_refresh);

at::Tensor& remainder_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& result);
at::Tensor& remainder_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& result);
at::Tensor remainder(const at::Tensor& self, const at::Tensor& other);
at::Tensor remainder(const at::Tensor& self, const at::Scalar& other);
at::Tensor& remainder_(at::Tensor& self, const at::Tensor& other);
at::Tensor& remainder_(at::Tensor& self, const at::Scalar& other);
at::Tensor remainder(const at::Scalar& self, const at::Tensor& other);

}  // namespace acl_op

namespace at_npu {

namespace native {

void npu_fast_reshape_(at::Tensor& tensor);

namespace custom_ops {

at::Tensor npu_dtype_cast(const at::Tensor& self, at::ScalarType dtype);

at::Tensor npu_format_cast(const at::Tensor& self, int acl_format);

at::Tensor& npu_format_cast_(at::Tensor& self, int acl_format);

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
