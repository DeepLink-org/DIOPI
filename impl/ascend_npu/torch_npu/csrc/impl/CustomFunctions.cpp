#include <ATen/core/dispatch/Dispatcher.h>

#include "torch_npu/csrc/aten/CustomFunctions.h"


namespace at_npu {
namespace native {
namespace custom_ops {

int64_t npu_change_data_ptr(const at::Tensor & dst, const at::Tensor & src, int64_t index) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_change_data_ptr", "").typed<int64_t (const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(dst, src, index);
}
int64_t get_npu_format(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::get_npu_format", "").typed<int64_t (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_format_cast(const at::Tensor & self, const at::Tensor & dst) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_format_cast", "Tensor").typed<at::Tensor (const at::Tensor &, const at::Tensor &)>();
    return op.call(self, dst);
}
at::Tensor & npu_format_cast_(at::Tensor & self, int64_t acl_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_format_cast_", "acl_format").typed<at::Tensor & (at::Tensor &, int64_t)>();
    return op.call(self, acl_format);
}
at::Tensor & npu_format_cast_(at::Tensor & self, const at::Tensor & src) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_format_cast_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &)>();
    return op.call(self, src);
}
at::Tensor empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, int64_t acl_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::empty_with_format", "").typed<at::Tensor (at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>, int64_t)>();
    return op.call(size, dtype, layout, device, pin_memory, acl_format);
}
at::Tensor unsafe_empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, int64_t acl_format, bool keep_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::unsafe_empty_with_format", "").typed<at::Tensor (at::IntArrayRef, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>, int64_t, bool)>();
    return op.call(size, dtype, layout, device, pin_memory, acl_format, keep_format);
}
at::Tensor empty_with_format(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, int64_t acl_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::empty_with_format", "names").typed<at::Tensor (at::IntArrayRef, c10::optional<at::DimnameList>, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>, int64_t)>();
    return op.call(size, names, dtype, layout, device, pin_memory, acl_format);
}
at::Tensor & copy_memory_(at::Tensor & self, const at::Tensor & src, bool non_blocking) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::copy_memory_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &, bool)>();
    return op.call(self, src, non_blocking);
}
at::Tensor format_contiguous(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::format_contiguous", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
bool check_match(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::check_match", "").typed<bool (const at::Tensor &)>();
    return op.call(self);
}
void check_memory_overlaps(at::TensorList inputs, at::TensorList outputs) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::check_memory_overlaps", "").typed<void (at::TensorList, at::TensorList)>();
    return op.call(inputs, outputs);
}
int64_t get_storage_size(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::get_storage_size", "").typed<int64_t (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_format_cast(const at::Tensor & self, int64_t acl_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_format_cast", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(self, acl_format);
}
at::Tensor _npu_format_cast(const at::Tensor & self, int64_t acl_format) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_npu_format_cast", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(self, acl_format);
}
at::Tensor & npu_view_copy(at::Tensor & self, const at::Tensor & other, bool non_blocking) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_view_copy", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &, bool)>();
    return op.call(self, other, non_blocking);
}
at::Tensor npu_transpose(const at::Tensor & self, at::IntArrayRef perm, bool require_contiguous) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_transpose", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, bool)>();
    return op.call(self, perm, require_contiguous);
}
at::Tensor & npu_transpose_out(const at::Tensor & self, at::IntArrayRef perm, bool require_contiguous, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_transpose", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, bool, at::Tensor &)>();
    return op.call(self, perm, require_contiguous, out);
}
at::Tensor npu_broadcast(const at::Tensor & self, at::IntArrayRef size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_broadcast", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef)>();
    return op.call(self, size);
}
at::Tensor & npu_broadcast_out(const at::Tensor & self, at::IntArrayRef size, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_broadcast", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, at::Tensor &)>();
    return op.call(self, size, out);
}
at::Tensor & npu_dtype_cast_(at::Tensor & self, const at::Tensor & src) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_dtype_cast_", "").typed<at::Tensor & (at::Tensor &, const at::Tensor &)>();
    return op.call(self, src);
}
at::Tensor npu_alloc_float_status(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_alloc_float_status", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_get_float_status(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_get_float_status", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_clear_float_status(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_clear_float_status", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor & one_(at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::one_", "").typed<at::Tensor & (at::Tensor &)>();
    return op.call(self);
}
at::Tensor fast_gelu(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::fast_gelu", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor npu_fast_gelu_backward(const at::Tensor & grad, const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_fast_gelu_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &)>();
    return op.call(grad, self);
}
bool _amp_foreach_non_finite_check(at::TensorList scaled_grads) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_amp_foreach_non_finite_check", "").typed<bool (at::TensorList)>();
    return op.call(scaled_grads);
}
at::Tensor npu_sign_bits_pack(const at::Tensor & self, int64_t size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_sign_bits_pack", "").typed<at::Tensor (const at::Tensor &, int64_t)>();
    return op.call(self, size);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_bert_apply_adam(const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const at::Scalar & max_grad_norm, const at::Scalar & global_grad_norm, const at::Scalar & weight_decay, const c10::optional<at::Scalar> & step_size, int64_t adam_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_bert_apply_adam", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const c10::optional<at::Scalar> &, int64_t)>();
    return op.call(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_bert_apply_adam_out(const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const at::Scalar & max_grad_norm, const at::Scalar & global_grad_norm, const at::Scalar & weight_decay, const c10::optional<at::Scalar> & step_size, int64_t adam_mode, at::Tensor & var, at::Tensor & m, at::Tensor & v) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_bert_apply_adam", "out").typed<::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const c10::optional<at::Scalar> &, int64_t, at::Tensor &, at::Tensor &, at::Tensor &)>();
    return op.call(lr, beta1, beta2, epsilon, grad, max_grad_norm, global_grad_norm, weight_decay, step_size, adam_mode, var, m, v);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv_transpose2d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_conv_transpose2d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv_transpose3d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_conv_transpose3d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_convolution_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_convolution_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}
at::Tensor npu_conv_transpose2d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_conv_transpose2d", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(input, weight, bias, padding, output_padding, stride, dilation, groups);
}
at::Tensor npu_conv2d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_conv2d", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor & npu_conv2d_out(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_conv2d", "out").typed<at::Tensor & (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, at::Tensor &)>();
    return op.call(input, weight, bias, stride, padding, dilation, groups, out);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv2d_backward(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_conv2d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad_output, weight, stride, padding, dilation, groups, output_mask);
}
at::Tensor npu_conv3d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_conv3d", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor & npu_conv3d_out(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_conv3d", "out").typed<at::Tensor & (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, at::Tensor &)>();
    return op.call(input, weight, bias, stride, padding, dilation, groups, out);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_conv3d_backward(const at::Tensor & input, const at::Tensor & grad, const at::Tensor & weight, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_conv3d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, ::std::array<bool,3>)>();
    return op.call(input, grad, weight, stride, padding, dilation, groups, output_mask);
}
at::Tensor npu_stride_add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & offset1, const at::Scalar & offset2, const at::Scalar & c1_len) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_stride_add", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Scalar &, const at::Scalar &)>();
    return op.call(self, other, offset1, offset2, c1_len);
}
at::Tensor npu_slice(const at::Tensor & self, at::IntArrayRef offsets, at::IntArrayRef size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_slice", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef)>();
    return op.call(self, offsets, size);
}
at::Tensor & npu_slice_out(const at::Tensor & self, at::IntArrayRef offsets, at::IntArrayRef size, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_slice", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::Tensor &)>();
    return op.call(self, offsets, size, out);
}
at::Tensor npu_indexing(const at::Tensor & self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_indexing", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, int64_t, int64_t, int64_t, int64_t)>();
    return op.call(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask);
}
at::Tensor & npu_indexing_out(const at::Tensor & self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask, int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_indexing", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, int64_t, int64_t, int64_t, int64_t, at::Tensor &)>();
    return op.call(self, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, out);
}
at::Tensor npu_softmax_cross_entropy_with_logits_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & labels) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_softmax_cross_entropy_with_logits_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grad, self, labels);
}
at::Tensor npu_stride_copy(const at::Tensor & self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar & storage_offset) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_stride_copy", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, const at::Scalar &)>();
    return op.call(self, shape, stride, storage_offset);
}
at::Tensor & npu_stride_copy_out(const at::Tensor & self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar & storage_offset, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_stride_copy", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, const at::Scalar &, at::Tensor &)>();
    return op.call(self, shape, stride, storage_offset, out);
}
at::Tensor npu_roi_align(const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sample_num, int64_t roi_end_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_roi_align", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double, int64_t, int64_t, int64_t, int64_t)>();
    return op.call(self, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode);
}
at::Tensor npu_roi_alignbk(const at::Tensor & self, const at::Tensor & rois, at::IntArrayRef xdiff_shape, int64_t pooled_width, int64_t pooled_height, double spatial_scale, int64_t sample_num, c10::optional<int64_t> roi_end_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_roi_alignbk", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, at::IntArrayRef, int64_t, int64_t, double, int64_t, c10::optional<int64_t>)>();
    return op.call(self, rois, xdiff_shape, pooled_width, pooled_height, spatial_scale, sample_num, roi_end_mode);
}
at::Tensor & npu_sort_v2_out(const at::Tensor & self, int64_t dim, bool descending, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_sort_v2", "out").typed<at::Tensor & (const at::Tensor &, int64_t, bool, at::Tensor &)>();
    return op.call(self, dim, descending, out);
}
at::Tensor npu_sort_v2(const at::Tensor & self, int64_t dim, bool descending) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_sort_v2", "").typed<at::Tensor (const at::Tensor &, int64_t, bool)>();
    return op.call(self, dim, descending);
}
at::Tensor npu_one_hot(const at::Tensor & self, int64_t num_classes, int64_t depth, const at::Scalar & on_value, const at::Scalar & off_value) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_one_hot", "").typed<at::Tensor (const at::Tensor &, int64_t, int64_t, const at::Scalar &, const at::Scalar &)>();
    return op.call(self, num_classes, depth, on_value, off_value);
}
::std::tuple<at::Tensor,at::Tensor> npu_linear_backward(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & weight) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_linear_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grad, input, weight);
}
at::Tensor npu_anchor_response_flags(const at::Tensor & self, at::IntArrayRef featmap_size, at::IntArrayRef stride, int64_t num_base_anchors) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_anchor_response_flags", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(self, featmap_size, stride, num_base_anchors);
}
at::Tensor npu_dropout_backward(const at::Tensor & grad_output, const at::Tensor & mask, double p) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_dropout_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double)>();
    return op.call(grad_output, mask, p);
}
::std::tuple<at::Tensor,at::Tensor> npu_nms_rotated(const at::Tensor & self, const at::Tensor & scores, double iou_threshold, double scores_threshold, int64_t max_output_size, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_nms_rotated", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, double, double, int64_t, int64_t)>();
    return op.call(self, scores, iou_threshold, scores_threshold, max_output_size, mode);
}
at::Tensor npu_masked_fill_range(const at::Tensor & self, const at::Tensor & start, const at::Tensor & end, const at::Tensor & value, int64_t axis) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_masked_fill_range", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(self, start, end, value, axis);
}
at::Tensor npu_sub_sample(const at::Tensor & self, int64_t per_images, double positive_fraction) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_sub_sample", "").typed<at::Tensor (const at::Tensor &, int64_t, double)>();
    return op.call(self, per_images, positive_fraction);
}
at::Tensor npu_yolo_boxes_encode(const at::Tensor & self, const at::Tensor & gt_bboxes, const at::Tensor & stride, bool performance_mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_yolo_boxes_encode", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, bool)>();
    return op.call(self, gt_bboxes, stride, performance_mode);
}
at::Tensor npu_scatter(const at::Tensor & self, const at::Tensor & indices, const at::Tensor & updates, int64_t dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_scatter", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(self, indices, updates, dim);
}
at::Tensor npu_layer_norm_eval(const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_layer_norm_eval", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, double)>();
    return op.call(input, normalized_shape, weight, bias, eps);
}
at::Tensor npu_rotated_box_encode(const at::Tensor & self, const at::Tensor & gt_bboxes, const at::Tensor & weight) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_rotated_box_encode", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(self, gt_bboxes, weight);
}
at::Tensor npu_rotated_box_decode(const at::Tensor & self, const at::Tensor & deltas, const at::Tensor & weight) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_rotated_box_decode", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(self, deltas, weight);
}
at::Tensor npu_rotated_overlaps(const at::Tensor & self, const at::Tensor & query_boxes, bool trans) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_rotated_overlaps", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, bool)>();
    return op.call(self, query_boxes, trans);
}
at::Tensor npu_silu_backward(const at::Tensor & grad_output, const at::Tensor & x0, const at::Tensor & x1) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_silu_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grad_output, x0, x1);
}
at::Tensor npu_rotated_iou(const at::Tensor & self, const at::Tensor & query_boxes, bool trans, int64_t mode, bool is_cross, double v_threshold, double e_threshold) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_rotated_iou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, bool, int64_t, bool, double, double)>();
    return op.call(self, query_boxes, trans, mode, is_cross, v_threshold, e_threshold);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nms_with_mask(const at::Tensor & input, const at::Scalar & iou_threshold) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_nms_with_mask", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Scalar &)>();
    return op.call(input, iou_threshold);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_gru_backward(const c10::optional<at::Tensor> & grady, const c10::optional<at::Tensor> & gradh, const at::Tensor & input, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, const at::Tensor & hx, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & output_updata, const at::Tensor & output_reset, const at::Tensor & output_new, const at::Tensor & hidden_new) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_gru_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grady, gradh, input, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, hx, y_output, h_output, output_updata, output_reset, output_new, hidden_new);
}
at::Tensor npu_mish_backward(const at::Tensor & grad, const at::Tensor & input) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_mish_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &)>();
    return op.call(grad, input);
}
at::Tensor npu_reshape(const at::Tensor & self, at::IntArrayRef shape, bool can_refresh) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_reshape", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, bool)>();
    return op.call(self, shape, can_refresh);
}
at::Tensor & npu_reshape_out(const at::Tensor & self, at::IntArrayRef shape, bool can_refresh, at::Tensor & out) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_reshape", "out").typed<at::Tensor & (const at::Tensor &, at::IntArrayRef, bool, at::Tensor &)>();
    return op.call(self, shape, can_refresh, out);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_batch_nms(const at::Tensor & self, const at::Tensor & scores, double score_threshold, double iou_threshold, int64_t max_size_per_class, int64_t max_total_size, bool change_coordinate_frame, bool transpose_box) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_batch_nms", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, double, double, int64_t, int64_t, bool, bool)>();
    return op.call(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame, transpose_box);
}
at::Tensor npu_bounding_box_encode(const at::Tensor & anchor_box, const at::Tensor & ground_truth_box, double means0, double means1, double means2, double means3, double stds0, double stds1, double stds2, double stds3) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_bounding_box_encode", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double, double, double, double, double, double, double, double)>();
    return op.call(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3);
}
at::Tensor npu_bounding_box_decode(const at::Tensor & rois, const at::Tensor & deltas, double means0, double means1, double means2, double means3, double stds0, double stds1, double stds2, double stds3, at::IntArrayRef max_shape, double wh_ratio_clip) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_bounding_box_decode", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double, double, double, double, double, double, double, double, at::IntArrayRef, double)>();
    return op.call(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_apply_adam(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, c10::optional<bool> use_locking, c10::optional<bool> use_nesterov) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_apply_adam", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, c10::optional<bool>, c10::optional<bool>)>();
    return op.call(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_apply_adam_out(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, c10::optional<bool> use_locking, c10::optional<bool> use_nesterov, at::Tensor & var, at::Tensor & m, at::Tensor & v) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_apply_adam", "out").typed<::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, c10::optional<bool>, c10::optional<bool>, at::Tensor &, at::Tensor &, at::Tensor &)>();
    return op.call(beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, use_locking, use_nesterov, var, m, v);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_apply_adam_w(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & weight_decay, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const c10::optional<at::Tensor> & max_grad_norm, c10::optional<bool> amsgrad, c10::optional<bool> maximize) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_apply_adam_w", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::optional<bool>, c10::optional<bool>)>();
    return op.call(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> npu_apply_adam_w_out(const at::Scalar & beta1_power, const at::Scalar & beta2_power, const at::Scalar & lr, const at::Scalar & weight_decay, const at::Scalar & beta1, const at::Scalar & beta2, const at::Scalar & epsilon, const at::Tensor & grad, const c10::optional<at::Tensor> & max_grad_norm, c10::optional<bool> amsgrad, c10::optional<bool> maximize, at::Tensor & var, at::Tensor & m, at::Tensor & v) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_apply_adam_w", "out").typed<::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> (const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Scalar &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::optional<bool>, c10::optional<bool>, at::Tensor &, at::Tensor &, at::Tensor &)>();
    return op.call(beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm, amsgrad, maximize, var, m, v);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_deformable_conv2dbk(const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & offset_out, const at::Tensor & weight, const at::Tensor & offset, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_deformable_conv2dbk", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, int64_t, bool)>();
    return op.call(input, grad_output, offset_out, weight, offset, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
}
::std::tuple<at::Tensor,at::Tensor> npu_giou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_giou_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, bool, int64_t)>();
    return op.call(grad, bboxes, gtboxes, trans, is_cross, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_diou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_diou_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, bool, int64_t)>();
    return op.call(grad, bboxes, gtboxes, trans, is_cross, mode);
}
at::Tensor npu_iou(const at::Tensor & bboxes, const at::Tensor & gtboxes, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_iou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(bboxes, gtboxes, mode);
}
::std::tuple<at::Tensor,at::Tensor> npu_nms_v4(const at::Tensor & self, const at::Tensor & scores, const at::Scalar & max_output_size, const at::Tensor & iou_threshold, const at::Tensor & scores_threshold, bool pad_to_max_output_size) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_nms_v4", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Scalar &, const at::Tensor &, const at::Tensor &, bool)>();
    return op.call(self, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size);
}
at::Tensor npu_pad(const at::Tensor & input, at::IntArrayRef paddings) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_pad", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef)>();
    return op.call(input, paddings);
}
::std::tuple<at::Tensor,at::Tensor> npu_random_choice_with_mask(const at::Tensor & x, int64_t count, int64_t seed, int64_t seed2) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_random_choice_with_mask", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, int64_t, int64_t, int64_t)>();
    return op.call(x, count, seed, seed2);
}
at::Tensor npu_normalize_batch(const at::Tensor & self, const at::Tensor & seq_len, int64_t normalize_type) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_normalize_batch", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(self, seq_len, normalize_type);
}
at::Tensor npu_ptiou(const at::Tensor & bboxes, const at::Tensor & gtboxes, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_ptiou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, int64_t)>();
    return op.call(bboxes, gtboxes, mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_backward(const c10::optional<at::Tensor> & grady, const c10::optional<at::Tensor> & gradh, const c10::optional<at::Tensor> & gradc, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & hx, const at::Tensor & cx, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & c_output, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_lstm_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grady, gradh, gradc, input, weight, bias, hx, cx, y_output, h_output, c_output, i, j, f, o, tanhc);
}
at::Tensor _dropout_with_byte_mask_backward(const at::Tensor & grad_output, const at::Tensor & mask, double p) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_dropout_with_byte_mask_backward", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double)>();
    return op.call(grad_output, mask, p);
}
at::Tensor dropout_with_byte_mask(const at::Tensor & self, double p, bool train) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::dropout_with_byte_mask", "").typed<at::Tensor (const at::Tensor &, double, bool)>();
    return op.call(self, p, train);
}
::std::tuple<at::Tensor,at::Tensor> npu_dropout_with_add_softmax_backward(const at::Tensor & grad, const at::Tensor & mask, const at::Tensor & softmax_out, const at::Scalar & alpha, double prob, int64_t dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_dropout_with_add_softmax_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, double, int64_t)>();
    return op.call(grad, mask, softmax_out, alpha, prob, dim);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention_backward(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & out_proj_weight, const c10::optional<at::Tensor> & query_bias, const c10::optional<at::Tensor> & key_bias, const c10::optional<at::Tensor> & value_bias, const c10::optional<at::Tensor> & out_proj_bias, const at::Tensor & query_res, const at::Tensor & key_res, const at::Tensor & value_res, const at::Tensor & attn_scores, const at::Tensor & attn_res, const at::Tensor & context, const at::Tensor & y_grad, const at::Tensor & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_multi_head_attention_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t, int64_t, double, bool)>();
    return op.call(query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, query_res, key_res, value_res, attn_scores, attn_res, context, y_grad, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}
at::Tensor npu_dropout_gen_mask(at::IntArrayRef size, double p, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_dropout_gen_mask", "").typed<at::Tensor (at::IntArrayRef, double, c10::optional<at::ScalarType>, c10::optional<at::Layout>, c10::optional<at::Device>, c10::optional<bool>)>();
    return op.call(size, p, dtype, layout, device, pin_memory);
}
::std::tuple<at::Tensor,at::Tensor> npu_ciou_backward(const at::Tensor & grad, const at::Tensor & bboxes, const at::Tensor & gtboxes, const c10::optional<at::Tensor> & atan_sub, bool trans, bool is_cross, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_ciou_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, bool, bool, int64_t)>();
    return op.call(grad, bboxes, gtboxes, atan_sub, trans, is_cross, mode);
}
at::Tensor npu_sign_bits_unpack(const at::Tensor & input, int64_t size, at::ScalarType dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_sign_bits_unpack", "").typed<at::Tensor (const at::Tensor &, int64_t, at::ScalarType)>();
    return op.call(input, size, dtype);
}
at::Tensor decode_jpeg(const at::Tensor & self, at::IntArrayRef image_shape, int64_t channels, bool try_recover_truncated) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::decode_jpeg", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, int64_t, bool)>();
    return op.call(self, image_shape, channels, try_recover_truncated);
}
at::Tensor crop_and_resize(const at::Tensor & self, c10::optional<at::ArrayRef<double>> boxes, at::IntArrayRef box_index, at::IntArrayRef crop_size, double extrapolation_value, c10::string_view method) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::crop_and_resize", "").typed<at::Tensor (const at::Tensor &, c10::optional<at::ArrayRef<double>>, at::IntArrayRef, at::IntArrayRef, double, c10::string_view)>();
    return op.call(self, boxes, box_index, crop_size, extrapolation_value, method);
}
at::Tensor reverse(const at::Tensor & self, at::IntArrayRef axis) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::reverse", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef)>();
    return op.call(self, axis);
}
at::Tensor image_normalize(const at::Tensor & self, c10::optional<at::ArrayRef<double>> mean, c10::optional<at::ArrayRef<double>> variance, int64_t dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::image_normalize", "").typed<at::Tensor (const at::Tensor &, c10::optional<at::ArrayRef<double>>, c10::optional<at::ArrayRef<double>>, int64_t)>();
    return op.call(self, mean, variance, dtype);
}
at::Tensor & image_normalize_(at::Tensor & self, c10::optional<at::ArrayRef<double>> mean, c10::optional<at::ArrayRef<double>> variance, int64_t dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::image_normalize_", "").typed<at::Tensor & (at::Tensor &, c10::optional<at::ArrayRef<double>>, c10::optional<at::ArrayRef<double>>, int64_t)>();
    return op.call(self, mean, variance, dtype);
}
at::Tensor img_to_tensor(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::img_to_tensor", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
::std::tuple<at::Tensor,at::Tensor> _conv_depthwise2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,2> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_conv_depthwise2d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, ::std::array<bool,2>)>();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_dilated2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::slow_conv_dilated2d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, ::std::array<bool,3>)>();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> slow_conv_transpose2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, ::std::array<bool,3> output_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::slow_conv_transpose2d_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, ::std::array<bool,3>)>();
    return op.call(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, output_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_cell_backward(const c10::optional<at::Tensor> & grady, const c10::optional<at::Tensor> & gradh, const c10::optional<at::Tensor> & gradc, const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const at::Tensor & y_output, const at::Tensor & h_output, const at::Tensor & c_output, const at::Tensor & i, const at::Tensor & j, const at::Tensor & f, const at::Tensor & o, const at::Tensor & tanhc) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_lstm_cell_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grady, gradh, gradc, input, w_ih, w_hh, h, c, y_output, h_output, c_output, i, j, f, o, tanhc);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_reduce(const at::Tensor & input, double eps) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::batch_norm_reduce", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, double)>();
    return op.call(input, eps);
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_update(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::batch_norm_gather_stats_update", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, double, double, const at::Tensor &)>();
    return op.call(input, mean, invstd, running_mean, running_var, momentum, eps, counts);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_backward(const at::Tensor & grad_output, const at::Tensor & softmax_output, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool value_transpose, bool dx_transpose) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_fused_attention_score_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, double, bool, bool, bool, bool)>();
    return op.call(grad_output, softmax_output, query_layer, key_layer, value_layer, mask, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_fwd(const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_fused_attention_score_fwd", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, double, bool, bool, bool, bool, bool, bool)>();
    return op.call(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_attention_score_grad(const at::Tensor & grad_output, const at::Tensor & softmax_output, const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool value_transpose, bool dx_transpose) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_fused_attention_score_grad", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, double, bool, bool, bool, bool)>();
    return op.call(grad_output, softmax_output, query_layer, key_layer, value_layer, mask, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
}
::std::vector<at::Tensor> npu_fused_attention_qkv_grad(const at::Tensor & grad_output_query, const at::Tensor & grad_output_key, const at::Tensor & grad_output_value, const at::Tensor & query_kernel, const at::Tensor & key_kernel, const at::Tensor & value_kernel, const at::Tensor & hidden_states, const at::Tensor & grad_output_ln) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_fused_attention_qkv_grad", "").typed<::std::vector<at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grad_output_query, grad_output_key, grad_output_value, query_kernel, key_kernel, value_kernel, hidden_states, grad_output_ln);
}
::std::vector<at::Tensor> npu_fused_attention_layernorm_qkv_fwd(const at::Tensor & x, const at::Tensor & kernel_query, const at::Tensor & kernel_key, const at::Tensor & kernel_value, const at::Tensor & gamma, const at::Tensor & beta, const c10::optional<at::Tensor> & bias_query, const c10::optional<at::Tensor> & bias_key, const c10::optional<at::Tensor> & bias_value, int64_t seq_len, int64_t num_heads, double eps) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_fused_attention_layernorm_qkv_fwd", "").typed<::std::vector<at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, int64_t, int64_t, double)>();
    return op.call(x, kernel_query, kernel_key, kernel_value, gamma, beta, bias_query, bias_key, bias_value, seq_len, num_heads, eps);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_layernorm_grad(const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_layernorm_grad", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, at::IntArrayRef, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &)>();
    return op.call(grad_out, input, normalized_shape, mean, rstd, weight, bias);
}
::std::tuple<at::Tensor,at::Tensor> npu_ifmr(const at::Tensor & data, const at::Tensor & data_min, const at::Tensor & data_max, const at::Tensor & cumsum, double min_percentile, double max_percentile, double search_start, double search_end, double search_step, bool with_offset) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_ifmr", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, double, double, double, double, double, bool)>();
    return op.call(data, data_min, data_max, cumsum, min_percentile, max_percentile, search_start, search_end, search_step, with_offset);
}
at::Tensor npu_grid_assign_positive(const at::Tensor & self, const at::Tensor & overlaps, const at::Tensor & box_responsible_flags, const at::Tensor & max_overlaps, const at::Tensor & argmax_overlaps, const at::Tensor & gt_max_overlaps, const at::Tensor & gt_argmax_overlaps, int64_t num_gts, double pos_iou_thr, double min_pos_iou, bool gt_max_assign_all) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_grid_assign_positive", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, double, double, bool)>();
    return op.call(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all);
}
at::Tensor npu_rotary_mul(const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_rotary_mul", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(self, r1, r2);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_rotary_mul_backward(const at::Tensor & grad, const at::Tensor & self, const at::Tensor & r1, const at::Tensor & r2) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_rotary_mul_backward", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &)>();
    return op.call(grad, self, r1, r2);
}
at::Tensor npu_convolution(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_convolution", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(input, weight, bias, stride, padding, dilation, groups);
}
at::Tensor npu_convolution_transpose(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_convolution_transpose", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t)>();
    return op.call(input, weight, bias, padding, output_padding, stride, dilation, groups);
}
at::Tensor npu_confusion_transpose(const at::Tensor & self, at::IntArrayRef perm, at::IntArrayRef shape, bool transpose_first) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_confusion_transpose", "").typed<at::Tensor (const at::Tensor &, at::IntArrayRef, at::IntArrayRef, bool)>();
    return op.call(self, perm, shape, transpose_first);
}
at::Tensor npu_ps_roi_pooling(const at::Tensor & self, const at::Tensor & rois, double spatial_scale, int64_t group_size, int64_t output_dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_ps_roi_pooling", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, double, int64_t, int64_t)>();
    return op.call(self, rois, spatial_scale, group_size, output_dim);
}
at::Tensor npu_linear(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & bias) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_linear", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &)>();
    return op.call(input, weight, bias);
}
::std::tuple<at::Tensor,at::Tensor> _npu_dropout(const at::Tensor & self, double p) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_npu_dropout", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, double)>();
    return op.call(self, p);
}
at::Tensor npu_softmax_cross_entropy_with_logits(const at::Tensor & self, const at::Tensor & labels) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_softmax_cross_entropy_with_logits", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &)>();
    return op.call(self, labels);
}
::std::tuple<at::Tensor,at::Tensor> npu_max(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_max", "dim").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, int64_t, bool)>();
    return op.call(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_max(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_max", "names_dim").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::Dimname, bool)>();
    return op.call(self, dim, keepdim);
}
at::Tensor npu_bmmV2(const at::Tensor & self, const at::Tensor & mat2, at::IntArrayRef output_sizes) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_bmmV2", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, at::IntArrayRef)>();
    return op.call(self, mat2, output_sizes);
}
at::Tensor npu_dtype_cast(const at::Tensor & self, at::ScalarType dtype) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_dtype_cast", "").typed<at::Tensor (const at::Tensor &, at::ScalarType)>();
    return op.call(self, dtype);
}
at::Tensor npu_silu(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_silu", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
at::Tensor & npu_silu_(at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_silu_", "").typed<at::Tensor & (at::Tensor &)>();
    return op.call(self);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_gru(const at::Tensor & input, const at::Tensor & hx, const at::Tensor & weight_input, const at::Tensor & weight_hidden, const at::Tensor & bias_input, const at::Tensor & bias_hidden, const at::Tensor & seq_length, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_gru", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, int64_t, double, bool, bool, bool)>();
    return op.call(input, hx, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, has_biases, num_layers, dropout, train, bidirectional, batch_first);
}
at::Tensor npu_mish(const at::Tensor & self) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_mish", "").typed<at::Tensor (const at::Tensor &)>();
    return op.call(self);
}
::std::tuple<at::Tensor,at::Tensor> npu_min(const at::Tensor & self, int64_t dim, bool keepdim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_min", "dim").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, int64_t, bool)>();
    return op.call(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_min(const at::Tensor & self, at::Dimname dim, bool keepdim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_min", "names_dim").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, at::Dimname, bool)>();
    return op.call(self, dim, keepdim);
}
::std::tuple<at::Tensor,at::Tensor> npu_deformable_conv2d(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & offset, const c10::optional<at::Tensor> & bias, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups, bool modulated) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_deformable_conv2d", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, at::IntArrayRef, int64_t, int64_t, bool)>();
    return op.call(input, weight, offset, bias, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
}
at::Tensor npu_giou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_giou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, bool, bool, int64_t)>();
    return op.call(self, gtboxes, trans, is_cross, mode);
}
at::Tensor npu_diou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_diou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, bool, bool, int64_t)>();
    return op.call(self, gtboxes, trans, is_cross, mode);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_lstm", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, int64_t, double, bool, bool, bool, bool, bool)>();
    return op.call(input, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_data(const at::Tensor & input, const at::Tensor & batch_sizes, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & seq_mask, const at::Tensor & h, const at::Tensor & c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_lstm_data", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, bool, int64_t, double, bool, bool, bool, bool, bool)>();
    return op.call(input, batch_sizes, weight, bias, seq_mask, h, c, has_biases, num_layers, dropout, train, bidirectional, batch_first, flag_seq, direction);
}
::std::tuple<at::Tensor,at::Tensor> _dropout_with_byte_mask(const at::Tensor & self, double p) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_dropout_with_byte_mask", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, double)>();
    return op.call(self, p);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_dropout_with_add_softmax(const at::Tensor & self, const at::Tensor & x1, const at::Scalar & alpha, double prob, int64_t dim) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_dropout_with_add_softmax", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Scalar &, double, int64_t)>();
    return op.call(self, x1, alpha, prob, dim);
}
at::Tensor npu_scaled_masked_softmax(const at::Tensor & x, const at::Tensor & mask, const at::Scalar & scale, bool fixed_triu_mask) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_scaled_masked_softmax", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Scalar &, bool)>();
    return op.call(x, mask, scale, fixed_triu_mask);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_multi_head_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & query_weight, const at::Tensor & key_weight, const at::Tensor & value_weight, const at::Tensor & attn_mask, const at::Tensor & out_proj_weight, const c10::optional<at::Tensor> & query_bias, const c10::optional<at::Tensor> & key_bias, const c10::optional<at::Tensor> & value_bias, const c10::optional<at::Tensor> & out_proj_bias, const c10::optional<at::Tensor> & dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_multi_head_attention", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, int64_t, int64_t, int64_t, int64_t, double, bool)>();
    return op.call(query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> npu_fusion_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t head_num, c10::string_view input_layout, const c10::optional<at::Tensor> & pse, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, double scale, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, bool gen_mask_parallel, bool sync) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_fusion_attention", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t> (const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, c10::string_view, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, double, double, int64_t, int64_t, int64_t, bool, bool)>();
    return op.call(query, key, value, head_num, input_layout, pse, padding_mask, atten_mask, scale, keep_prob, pre_tockens, next_tockens, inner_precise, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_fusion_attention_grad(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & dy, int64_t head_num, c10::string_view input_layout, const c10::optional<at::Tensor> & pse, const c10::optional<at::Tensor> & padding_mask, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & softmax_max, const c10::optional<at::Tensor> & softmax_sum, const c10::optional<at::Tensor> & softmax_in, const c10::optional<at::Tensor> & attention_in, double scale_value, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, int64_t seed, int64_t offset, int64_t numels, bool gen_mask_parallel, bool sync) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_fusion_attention_grad", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, c10::string_view, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, const c10::optional<at::Tensor> &, double, double, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, bool)>();
    return op.call(query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in, scale_value, keep_prob, pre_tockens, next_tockens, inner_precise, seed, offset, numels, gen_mask_parallel, sync);
}
::std::tuple<at::Tensor,at::Tensor> npu_dropout_do_mask(const at::Tensor & self, const at::Tensor & mask, double p) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_dropout_do_mask", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, double)>();
    return op.call(self, mask, p);
}
at::Tensor npu_ciou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_ciou", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, bool, bool, int64_t, bool)>();
    return op.call(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_lstm_cell(const at::Tensor & input, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & h, const at::Tensor & c, const c10::optional<at::Tensor> & bias) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_lstm_cell", "").typed<::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &)>();
    return op.call(input, w_ih, w_hh, h, c, bias);
}
at::Tensor npu_fused_attention_score(const at::Tensor & query_layer, const at::Tensor & key_layer, const at::Tensor & value_layer, const at::Tensor & attention_mask, const at::Scalar & scale, double keep_prob, bool query_transpose, bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::npu_fused_attention_score", "").typed<at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Scalar &, double, bool, bool, bool, bool, bool, bool)>();
    return op.call(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b, value_transpose, dx_transpose);
}
::std::tuple<at::Tensor,at::Tensor> _npu_ciou(const at::Tensor & self, const at::Tensor & gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::_npu_ciou", "").typed<::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &, const at::Tensor &, bool, bool, int64_t, bool)>();
    return op.call(self, gtboxes, trans, is_cross, mode, atan_sub_flag);
}

}  // namespace custom_ops
}  // namespace native
}  // namespace at_npu
