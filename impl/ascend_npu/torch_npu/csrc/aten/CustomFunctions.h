#pragma once
#include <ATen/ATen.h>

#include <iostream>
#include <tuple>
#include <vector>

#include "torch_npu/csrc/framework/DIOPIAdapter.h"

namespace at_npu {
namespace native {
namespace custom_ops {

int64_t npu_change_data_ptr(const at::Tensor& dst, const at::Tensor& src, int64_t index);
int64_t get_npu_format(const at::Tensor& self);
at::Tensor npu_format_cast(const at::Tensor& self, const at::Tensor& dst);
at::Tensor& npu_format_cast_(at::Tensor& self, int64_t acl_format);
at::Tensor& npu_format_cast_(at::Tensor& self, const at::Tensor& src);
at::Tensor empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype = {}, c10::optional<at::Layout> layout = {},
                             c10::optional<at::Device> device = {}, c10::optional<bool> pin_memory = {}, int64_t acl_format = 2);
at::Tensor unsafe_empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype = {}, c10::optional<at::Layout> layout = {},
                                    c10::optional<at::Device> device = {}, c10::optional<bool> pin_memory = {}, int64_t acl_format = 2,
                                    bool keep_format = false);
at::Tensor empty_with_format(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype = {},
                             c10::optional<at::Layout> layout = {}, c10::optional<at::Device> device = {}, c10::optional<bool> pin_memory = {},
                             int64_t acl_format = 2);
at::Tensor& copy_memory_(at::Tensor& self, const at::Tensor& src, bool non_blocking = false);
int64_t get_storage_size(const at::Tensor& self);
at::Tensor npu_format_cast(const at::Tensor& self, int64_t acl_format);
at::Tensor _npu_format_cast(const at::Tensor& self, int64_t acl_format);
at::Tensor& npu_view_copy(at::Tensor& self, const at::Tensor& other, bool non_blocking);
at::Tensor npu_transpose(const at::Tensor& self, at::IntArrayRef perm, bool require_contiguous = true);
at::Tensor& npu_transpose_out(const at::Tensor& self, at::IntArrayRef perm, bool require_contiguous, at::Tensor& out);
at::Tensor npu_broadcast(const at::Tensor& self, at::IntArrayRef size);
at::Tensor& npu_broadcast_out(const at::Tensor& self, at::IntArrayRef size, at::Tensor& out);
at::Tensor& npu_dtype_cast_(at::Tensor& self, const at::Tensor& src);
at::Tensor npu_alloc_float_status(const at::Tensor& self);
at::Tensor npu_get_float_status(const at::Tensor& self);
at::Tensor npu_clear_float_status(const at::Tensor& self);
at::Tensor& one_(at::Tensor& self);
at::Tensor npu_fast_gelu(const at::Tensor& self);
at::Tensor npu_fast_gelu_backward(const at::Tensor& grad, const at::Tensor& self);
bool _amp_foreach_non_finite_check(at::TensorList scaled_grads);
at::Tensor npu_sign_bits_pack(const at::Tensor& self, int64_t size);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_bert_apply_adam(const at::Scalar& lr, const at::Scalar& beta1, const at::Scalar& beta2,
                                                                     const at::Scalar& epsilon, const at::Tensor& grad, const at::Scalar& max_grad_norm,
                                                                     const at::Scalar& global_grad_norm, const at::Scalar& weight_decay,
                                                                     const c10::optional<at::Scalar>& step_size = c10::nullopt, int64_t adam_mode = 0);
::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_bert_apply_adam_out(const at::Scalar& lr, const at::Scalar& beta1, const at::Scalar& beta2,
                                                                            const at::Scalar& epsilon, const at::Tensor& grad, const at::Scalar& max_grad_norm,
                                                                            const at::Scalar& global_grad_norm, const at::Scalar& weight_decay,
                                                                            const c10::optional<at::Scalar>& step_size, int64_t adam_mode, at::Tensor& var,
                                                                            at::Tensor& m, at::Tensor& v);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose2d_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
                                                                               at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride,
                                                                               at::IntArrayRef dilation, int64_t groups, ::std::array<bool, 3> output_mask);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose3d_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
                                                                               at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride,
                                                                               at::IntArrayRef dilation, int64_t groups, ::std::array<bool, 3> output_mask);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
                                                                          at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
                                                                          int64_t groups, ::std::array<bool, 3> output_mask);
at::Tensor npu_conv_transpose2d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef padding,
                                at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups);
at::Tensor npu_conv2d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding,
                      at::IntArrayRef dilation, int64_t groups);
at::Tensor& npu_conv2d_out(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
                           at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor& out);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv2d_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
                                                                     at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups,
                                                                     ::std::array<bool, 3> output_mask);
at::Tensor npu_conv3d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding,
                      at::IntArrayRef dilation, int64_t groups);
at::Tensor& npu_conv3d_out(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
                           at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor& out);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv3d_backward(const at::Tensor& input, const at::Tensor& grad, const at::Tensor& weight,
                                                                     at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups,
                                                                     ::std::array<bool, 3> output_mask);
at::Tensor npu_stride_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& offset1, const at::Scalar& offset2, const at::Scalar& c1_len);
at::Tensor npu_slice(const at::Tensor& self, at::IntArrayRef offsets, at::IntArrayRef size);
__attribute__((__visibility__("default"))) at::Tensor& npu_slice_out(const at::Tensor& self, at::IntArrayRef offsets, at::IntArrayRef size, at::Tensor& out);
at::Tensor npu_indexing(const at::Tensor& self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask = 0,
                        int64_t end_mask = 0, int64_t ellipsis_mask = 0, int64_t new_axis_mask = 0, int64_t shrink_axis_mask = 0);
at::Tensor& npu_indexing_out(const at::Tensor& self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask,
                             int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor& out);
at::Tensor npu_softmax_cross_entropy_with_logits_backward(const at::Tensor& grad, const at::Tensor& self, const at::Tensor& labels);
at::Tensor npu_stride_copy(const at::Tensor& self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar& storage_offset);
at::Tensor& npu_stride_copy_out(const at::Tensor& self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar& storage_offset, at::Tensor& out);
at::Tensor npu_roi_align(const at::Tensor& self, const at::Tensor& rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sample_num,
                         int64_t roi_end_mode);
at::Tensor npu_roi_alignbk(const at::Tensor& self, const at::Tensor& rois, at::IntArrayRef xdiff_shape, int64_t pooled_width, int64_t pooled_height,
                           double spatial_scale, int64_t sample_num, c10::optional<int64_t> roi_end_mode = c10::nullopt);
at::Tensor& npu_sort_v2_out(const at::Tensor& self, int64_t dim, bool descending, at::Tensor& out);
at::Tensor npu_sort_v2(const at::Tensor& self, int64_t dim = -1, bool descending = false);
at::Tensor npu_one_hot(const at::Tensor& self, int64_t num_classes = -1, int64_t depth = 1, const at::Scalar& on_value = 1, const at::Scalar& off_value = 0);
::std::tuple<at::Tensor, at::Tensor> npu_linear_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight);
at::Tensor npu_anchor_response_flags(const at::Tensor& self, at::IntArrayRef featmap_size, at::IntArrayRef stride, int64_t num_base_anchors);
at::Tensor npu_dropout_backward(const at::Tensor& grad_output, const at::Tensor& mask, double p);
::std::tuple<at::Tensor, at::Tensor> npu_nms_rotated(const at::Tensor& self, const at::Tensor& scores, double iou_threshold, double scores_threshold = 0,
                                                     int64_t max_output_size = -1, int64_t mode = 0);
at::Tensor npu_masked_fill_range(const at::Tensor& self, const at::Tensor& start, const at::Tensor& end, const at::Tensor& value, int64_t axis = -1);
at::Tensor npu_sub_sample(const at::Tensor& self, int64_t per_images, double positive_fraction);
at::Tensor npu_yolo_boxes_encode(const at::Tensor& self, const at::Tensor& gt_bboxes, const at::Tensor& stride, bool performance_mode = false);
at::Tensor npu_scatter(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t dim);
at::Tensor npu_layer_norm_eval(const at::Tensor& input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor>& weight = {},
                               const c10::optional<at::Tensor>& bias = {}, double eps = 1e-05);
at::Tensor npu_rotated_box_encode(const at::Tensor& self, const at::Tensor& gt_bboxes, const at::Tensor& weight);
at::Tensor npu_rotated_box_decode(const at::Tensor& self, const at::Tensor& deltas, const at::Tensor& weight);
at::Tensor npu_rotated_overlaps(const at::Tensor& self, const at::Tensor& query_boxes, bool trans = false);
at::Tensor npu_silu_backward(const at::Tensor& grad_output, const at::Tensor& x0, const at::Tensor& x1);
at::Tensor npu_rotated_iou(const at::Tensor& self, const at::Tensor& query_boxes, bool trans = false, int64_t mode = 0, bool is_cross = true,
                           double v_threshold = 0.0, double e_threshold = 0.0);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_nms_with_mask(const at::Tensor& input, const at::Scalar& iou_threshold);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gru_backward(
    const c10::optional<at::Tensor>& grady, const c10::optional<at::Tensor>& gradh, const at::Tensor& input, const at::Tensor& weight_input,
    const at::Tensor& weight_hidden, const at::Tensor& bias_input, const at::Tensor& bias_hidden, const at::Tensor& seq_length, const at::Tensor& hx,
    const at::Tensor& y_output, const at::Tensor& h_output, const at::Tensor& output_updata, const at::Tensor& output_reset, const at::Tensor& output_new,
    const at::Tensor& hidden_new);
at::Tensor npu_mish_backward(const at::Tensor& grad, const at::Tensor& input);
at::Tensor npu_reshape(const at::Tensor& self, at::IntArrayRef shape, bool can_refresh = false);
at::Tensor& npu_reshape_out(const at::Tensor& self, at::IntArrayRef shape, bool can_refresh, at::Tensor& out);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_batch_nms(const at::Tensor& self, const at::Tensor& scores, double score_threshold,
                                                                           double iou_threshold, int64_t max_size_per_class, int64_t max_total_size,
                                                                           bool change_coordinate_frame = false, bool transpose_box = false);
at::Tensor npu_bounding_box_encode(const at::Tensor& anchor_box, const at::Tensor& ground_truth_box, double means0, double means1, double means2, double means3,
                                   double stds0, double stds1, double stds2, double stds3);
at::Tensor npu_bounding_box_decode(const at::Tensor& rois, const at::Tensor& deltas, double means0, double means1, double means2, double means3, double stds0,
                                   double stds1, double stds2, double stds3, at::IntArrayRef max_shape, double wh_ratio_clip);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_apply_adam(const at::Scalar& beta1_power, const at::Scalar& beta2_power, const at::Scalar& lr,
                                                                const at::Scalar& beta1, const at::Scalar& beta2, const at::Scalar& epsilon,
                                                                const at::Tensor& grad, c10::optional<bool> use_locking, c10::optional<bool> use_nesterov);
::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_apply_adam_out(const at::Scalar& beta1_power, const at::Scalar& beta2_power, const at::Scalar& lr,
                                                                       const at::Scalar& beta1, const at::Scalar& beta2, const at::Scalar& epsilon,
                                                                       const at::Tensor& grad, c10::optional<bool> use_locking,
                                                                       c10::optional<bool> use_nesterov, at::Tensor& var, at::Tensor& m, at::Tensor& v);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_apply_adam_w(const at::Scalar& beta1_power, const at::Scalar& beta2_power, const at::Scalar& lr,
                                                                  const at::Scalar& weight_decay, const at::Scalar& beta1, const at::Scalar& beta2,
                                                                  const at::Scalar& epsilon, const at::Tensor& grad,
                                                                  const c10::optional<at::Tensor>& max_grad_norm, c10::optional<bool> amsgrad,
                                                                  c10::optional<bool> maximize);
::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_apply_adam_w_out(const at::Scalar& beta1_power, const at::Scalar& beta2_power, const at::Scalar& lr,
                                                                         const at::Scalar& weight_decay, const at::Scalar& beta1, const at::Scalar& beta2,
                                                                         const at::Scalar& epsilon, const at::Tensor& grad,
                                                                         const c10::optional<at::Tensor>& max_grad_norm, c10::optional<bool> amsgrad,
                                                                         c10::optional<bool> maximize, at::Tensor& var, at::Tensor& m, at::Tensor& v);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_deformable_conv2dbk(const at::Tensor& input, const at::Tensor& grad_output,
                                                                                     const at::Tensor& offset_out, const at::Tensor& weight,
                                                                                     const at::Tensor& offset, at::IntArrayRef kernel_size,
                                                                                     at::IntArrayRef stride, at::IntArrayRef padding,
                                                                                     at::IntArrayRef dilation = {1, 1, 1, 1}, int64_t groups = 1,
                                                                                     int64_t deformable_groups = 1, bool modulated = true);
::std::tuple<at::Tensor, at::Tensor> npu_giou_backward(const at::Tensor& grad, const at::Tensor& bboxes, const at::Tensor& gtboxes, bool trans = false,
                                                       bool is_cross = false, int64_t mode = 0);
::std::tuple<at::Tensor, at::Tensor> npu_diou_backward(const at::Tensor& grad, const at::Tensor& bboxes, const at::Tensor& gtboxes, bool trans = false,
                                                       bool is_cross = false, int64_t mode = 0);
at::Tensor npu_iou(const at::Tensor& bboxes, const at::Tensor& gtboxes, int64_t mode = 0);
::std::tuple<at::Tensor, at::Tensor> npu_nms_v4(const at::Tensor& self, const at::Tensor& scores, const at::Scalar& max_output_size,
                                                const at::Tensor& iou_threshold, const at::Tensor& scores_threshold, bool pad_to_max_output_size = false);
at::Tensor npu_pad(const at::Tensor& input, at::IntArrayRef paddings);
::std::tuple<at::Tensor, at::Tensor> npu_random_choice_with_mask(const at::Tensor& x, int64_t count = 256, int64_t seed = 0, int64_t seed2 = 0);
at::Tensor npu_normalize_batch(const at::Tensor& self, const at::Tensor& seq_len, int64_t normalize_type = 0);
at::Tensor npu_ptiou(const at::Tensor& bboxes, const at::Tensor& gtboxes, int64_t mode = 0);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_backward(
    const c10::optional<at::Tensor>& grady, const c10::optional<at::Tensor>& gradh, const c10::optional<at::Tensor>& gradc, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& hx, const at::Tensor& cx, const at::Tensor& y_output, const at::Tensor& h_output,
    const at::Tensor& c_output, const at::Tensor& i, const at::Tensor& j, const at::Tensor& f, const at::Tensor& o, const at::Tensor& tanhc);
at::Tensor _dropout_with_byte_mask_backward(const at::Tensor& grad_output, const at::Tensor& mask, double p);
at::Tensor dropout_with_byte_mask(const at::Tensor& self, double p, bool train);
::std::tuple<at::Tensor, at::Tensor> npu_dropout_with_add_softmax_backward(const at::Tensor& grad, const at::Tensor& mask, const at::Tensor& softmax_out,
                                                                           const at::Scalar& alpha, double prob, int64_t dim);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_multi_head_attention_backward(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& query_weight,
                                  const at::Tensor& key_weight, const at::Tensor& value_weight, const at::Tensor& out_proj_weight,
                                  const c10::optional<at::Tensor>& query_bias, const c10::optional<at::Tensor>& key_bias,
                                  const c10::optional<at::Tensor>& value_bias, const c10::optional<at::Tensor>& out_proj_bias, const at::Tensor& query_res,
                                  const at::Tensor& key_res, const at::Tensor& value_res, const at::Tensor& attn_scores, const at::Tensor& attn_res,
                                  const at::Tensor& context, const at::Tensor& y_grad, const at::Tensor& dropout_mask, int64_t attn_head_num,
                                  int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float);
::std::tuple<at::Tensor, at::Tensor> npu_rms_norm(const at::Tensor& self, const at::Tensor& gamma, double epsilon = 1e-06);
::std::tuple<at::Tensor, at::Tensor> npu_rms_norm_backward(const at::Tensor& dy, const at::Tensor& self, const at::Tensor& gamma, const at::Tensor& rstd);
at::Tensor npu_dropout_gen_mask(at::IntArrayRef size, double p, c10::optional<at::ScalarType> dtype = {}, c10::optional<at::Layout> layout = {},
                                c10::optional<at::Device> device = {}, c10::optional<bool> pin_memory = {});
::std::tuple<at::Tensor, at::Tensor> npu_ciou_backward(const at::Tensor& grad, const at::Tensor& bboxes, const at::Tensor& gtboxes,
                                                       const c10::optional<at::Tensor>& atan_sub, bool trans = false, bool is_cross = true, int64_t mode = 0);
at::Tensor npu_sign_bits_unpack(const at::Tensor& input, int64_t size, at::ScalarType dtype);
at::Tensor decode_jpeg(const at::Tensor& self, at::IntArrayRef image_shape, int64_t channels = 3, bool try_recover_truncated = false);
at::Tensor crop_and_resize(const at::Tensor& self, c10::optional<at::ArrayRef<double>> boxes, at::IntArrayRef box_index, at::IntArrayRef crop_size,
                           double extrapolation_value = 0, c10::string_view method = "bilinear");
at::Tensor reverse(const at::Tensor& self, at::IntArrayRef axis);
at::Tensor image_normalize(const at::Tensor& self, c10::optional<at::ArrayRef<double>> mean, c10::optional<at::ArrayRef<double>> variance, int64_t dtype = 0);
at::Tensor& image_normalize_(at::Tensor& self, c10::optional<at::ArrayRef<double>> mean, c10::optional<at::ArrayRef<double>> variance, int64_t dtype = 0);
at::Tensor img_to_tensor(const at::Tensor& self);
::std::tuple<at::Tensor, at::Tensor> _conv_depthwise2d_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& weight,
                                                                at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                                                                at::IntArrayRef dilation, ::std::array<bool, 2> output_mask);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> slow_conv_dilated2d_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& weight,
                                                                              at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                                                                              at::IntArrayRef dilation, ::std::array<bool, 3> output_mask);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> slow_conv_transpose2d_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& weight,
                                                                                at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                                                                                at::IntArrayRef output_padding, at::IntArrayRef dilation,
                                                                                ::std::array<bool, 3> output_mask);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_cell_backward(
    const c10::optional<at::Tensor>& grady, const c10::optional<at::Tensor>& gradh, const c10::optional<at::Tensor>& gradc, const at::Tensor& input,
    const at::Tensor& w_ih, const at::Tensor& w_hh, const at::Tensor& h, const at::Tensor& c, const at::Tensor& y_output, const at::Tensor& h_output,
    const at::Tensor& c_output, const at::Tensor& i, const at::Tensor& j, const at::Tensor& f, const at::Tensor& o, const at::Tensor& tanhc);
::std::tuple<at::Tensor, at::Tensor> batch_norm_reduce(const at::Tensor& input, double eps);
::std::tuple<at::Tensor, at::Tensor> batch_norm_gather_stats_update(const at::Tensor& input, const at::Tensor& mean, const at::Tensor& invstd,
                                                                    const c10::optional<at::Tensor>& running_mean, const c10::optional<at::Tensor>& running_var,
                                                                    double momentum, double eps, const at::Tensor& counts);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_backward(const at::Tensor& grad_output, const at::Tensor& softmax_output,
                                                                                    const at::Tensor& query_layer, const at::Tensor& key_layer,
                                                                                    const at::Tensor& value_layer, const at::Tensor& mask,
                                                                                    const at::Scalar& scale, double keep_prob, bool query_transpose = false,
                                                                                    bool key_transpose = false, bool value_transpose = false,
                                                                                    bool dx_transpose = false);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_fwd(const at::Tensor& query_layer, const at::Tensor& key_layer,
                                                                               const at::Tensor& value_layer, const at::Tensor& attention_mask,
                                                                               const at::Scalar& scale, double keep_prob, bool query_transpose = false,
                                                                               bool key_transpose = false, bool bmm_score_transpose_a = false,
                                                                               bool bmm_score_transpose_b = false, bool value_transpose = false,
                                                                               bool dx_transpose = false);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_grad(const at::Tensor& grad_output, const at::Tensor& softmax_output,
                                                                                const at::Tensor& query_layer, const at::Tensor& key_layer,
                                                                                const at::Tensor& value_layer, const at::Tensor& mask, const at::Scalar& scale,
                                                                                double keep_prob, bool query_transpose = false, bool key_transpose = false,
                                                                                bool value_transpose = false, bool dx_transpose = false);
::std::vector<at::Tensor> npu_fused_attention_qkv_grad(const at::Tensor& grad_output_query, const at::Tensor& grad_output_key,
                                                       const at::Tensor& grad_output_value, const at::Tensor& query_kernel, const at::Tensor& key_kernel,
                                                       const at::Tensor& value_kernel, const at::Tensor& hidden_states, const at::Tensor& grad_output_ln);
::std::vector<at::Tensor> npu_fused_attention_layernorm_qkv_fwd(const at::Tensor& x, const at::Tensor& kernel_query, const at::Tensor& kernel_key,
                                                                const at::Tensor& kernel_value, const at::Tensor& gamma, const at::Tensor& beta,
                                                                const c10::optional<at::Tensor>& bias_query = {},
                                                                const c10::optional<at::Tensor>& bias_key = {},
                                                                const c10::optional<at::Tensor>& bias_value = {}, int64_t seq_len = 128, int64_t num_heads = 12,
                                                                double eps = 1e-05);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_layernorm_grad(const at::Tensor& grad_out, const at::Tensor& input, at::IntArrayRef normalized_shape,
                                                                    const at::Tensor& mean, const at::Tensor& rstd, const c10::optional<at::Tensor>& weight,
                                                                    const c10::optional<at::Tensor>& bias);
::std::tuple<at::Tensor, at::Tensor> npu_ifmr(const at::Tensor& data, const at::Tensor& data_min, const at::Tensor& data_max, const at::Tensor& cumsum,
                                              double min_percentile, double max_percentile, double search_start, double search_end, double search_step,
                                              bool with_offset);
at::Tensor npu_grid_assign_positive(const at::Tensor& self, const at::Tensor& overlaps, const at::Tensor& box_responsible_flags, const at::Tensor& max_overlaps,
                                    const at::Tensor& argmax_overlaps, const at::Tensor& gt_max_overlaps, const at::Tensor& gt_argmax_overlaps, int64_t num_gts,
                                    double pos_iou_thr, double min_pos_iou, bool gt_max_assign_all);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rotary_mul_backward(const at::Tensor& grad, const at::Tensor& self, const at::Tensor& r1,
                                                                         const at::Tensor& r2);
at::Tensor npu_convolution(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
                           at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups);
at::Tensor npu_convolution_transpose(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef padding,
                                     at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups);
at::Tensor fast_gelu(const at::Tensor& self);
at::Tensor npu_confusion_transpose(const at::Tensor& self, at::IntArrayRef perm, at::IntArrayRef shape, bool transpose_first);
at::Tensor npu_ps_roi_pooling(const at::Tensor& self, const at::Tensor& rois, double spatial_scale, int64_t group_size, int64_t output_dim);
at::Tensor npu_linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias = {});
::std::tuple<at::Tensor, at::Tensor> _npu_dropout(const at::Tensor& self, double p);
at::Tensor npu_softmax_cross_entropy_with_logits(const at::Tensor& self, const at::Tensor& labels);
::std::tuple<at::Tensor, at::Tensor> npu_max(const at::Tensor& self, int64_t dim, bool keepdim = false);
::std::tuple<at::Tensor, at::Tensor> npu_max(const at::Tensor& self, at::Dimname dim, bool keepdim = false);
at::Tensor npu_bmmV2(const at::Tensor& self, const at::Tensor& mat2, at::IntArrayRef output_sizes);
at::Tensor npu_dtype_cast(const at::Tensor& self, at::ScalarType dtype);
at::Tensor npu_silu(const at::Tensor& self);
at::Tensor& npu_silu_(at::Tensor& self);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gru(const at::Tensor& input, const at::Tensor& hx,
                                                                                             const at::Tensor& weight_input, const at::Tensor& weight_hidden,
                                                                                             const at::Tensor& bias_input, const at::Tensor& bias_hidden,
                                                                                             const at::Tensor& seq_length, bool has_biases, int64_t num_layers,
                                                                                             double dropout, bool train, bool bidirectional, bool batch_first);
at::Tensor npu_mish(const at::Tensor& self);
::std::tuple<at::Tensor, at::Tensor> npu_min(const at::Tensor& self, int64_t dim, bool keepdim = false);
::std::tuple<at::Tensor, at::Tensor> npu_min(const at::Tensor& self, at::Dimname dim, bool keepdim = false);
::std::tuple<at::Tensor, at::Tensor> npu_deformable_conv2d(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& offset,
                                                           const c10::optional<at::Tensor>& bias, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                           at::IntArrayRef padding, at::IntArrayRef dilation = {1, 1, 1, 1}, int64_t groups = 1,
                                                           int64_t deformable_groups = 1, bool modulated = true);
at::Tensor npu_giou(const at::Tensor& self, const at::Tensor& gtboxes, bool trans = false, bool is_cross = false, int64_t mode = 0);
at::Tensor npu_diou(const at::Tensor& self, const at::Tensor& gtboxes, bool trans = false, bool is_cross = false, int64_t mode = 0);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& seq_mask, const at::Tensor& h, const at::Tensor& c,
    bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_data(
    const at::Tensor& input, const at::Tensor& batch_sizes, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& seq_mask, const at::Tensor& h,
    const at::Tensor& c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction);
::std::tuple<at::Tensor, at::Tensor> _dropout_with_byte_mask(const at::Tensor& self, double p);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dropout_with_add_softmax(const at::Tensor& self, const at::Tensor& x1, const at::Scalar& alpha,
                                                                              double prob, int64_t dim);
at::Tensor npu_scaled_masked_softmax(const at::Tensor& x, const at::Tensor& mask, const at::Scalar& scale = 1, bool fixed_triu_mask = false);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_multi_head_attention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& query_weight, const at::Tensor& key_weight,
    const at::Tensor& value_weight, const at::Tensor& attn_mask, const at::Tensor& out_proj_weight, const c10::optional<at::Tensor>& query_bias,
    const c10::optional<at::Tensor>& key_bias, const c10::optional<at::Tensor>& value_bias, const c10::optional<at::Tensor>& out_proj_bias,
    const c10::optional<at::Tensor>& dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob,
    bool softmax_use_float);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t> npu_fusion_attention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, int64_t head_num, c10::string_view input_layout,
    const c10::optional<at::Tensor>& pse = {}, const c10::optional<at::Tensor>& padding_mask = {}, const c10::optional<at::Tensor>& atten_mask = {},
    double scale = 1., double keep_prob = 1., int64_t pre_tockens = 2147483647, int64_t next_tockens = 2147483647, int64_t inner_precise = 0,
    at::OptionalIntArrayRef prefix = c10::nullopt, int64_t sparse_mode = 0, bool gen_mask_parallel = true, bool sync = false);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_fusion_attention_grad(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& dy, int64_t head_num, c10::string_view input_layout,
    const c10::optional<at::Tensor>& pse = {}, const c10::optional<at::Tensor>& padding_mask = {}, const c10::optional<at::Tensor>& atten_mask = {},
    const c10::optional<at::Tensor>& softmax_max = {}, const c10::optional<at::Tensor>& softmax_sum = {}, const c10::optional<at::Tensor>& softmax_in = {},
    const c10::optional<at::Tensor>& attention_in = {}, double scale_value = 1., double keep_prob = 1., int64_t pre_tockens = 2147483647,
    int64_t next_tockens = 2147483647, int64_t inner_precise = 0, int64_t seed = 0, int64_t offset = 0, int64_t numels = 0,
    at::OptionalIntArrayRef prefix = c10::nullopt, int64_t sparse_mode = 0, bool gen_mask_parallel = true, bool sync = false);
::std::tuple<at::Tensor, at::Tensor> npu_dropout_do_mask(const at::Tensor& self, const at::Tensor& mask, double p);
at::Tensor npu_ciou(const at::Tensor& self, const at::Tensor& gtboxes, bool trans = false, bool is_cross = true, int64_t mode = 0, bool atan_sub_flag = false);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_cell(
    const at::Tensor& input, const at::Tensor& w_ih, const at::Tensor& w_hh, const at::Tensor& h, const at::Tensor& c,
    const c10::optional<at::Tensor>& b_ih = {}, const c10::optional<at::Tensor>& b_hh = {});
at::Tensor npu_fused_attention_score(const at::Tensor& query_layer, const at::Tensor& key_layer, const at::Tensor& value_layer,
                                     const at::Tensor& attention_mask, const at::Scalar& scale, double keep_prob, bool query_transpose = false,
                                     bool key_transpose = false, bool bmm_score_transpose_a = false, bool bmm_score_transpose_b = false,
                                     bool value_transpose = false, bool dx_transpose = false);
at::Tensor npu_rotary_mul(const at::Tensor& self, const at::Tensor& r1, const at::Tensor& r2);
::std::tuple<at::Tensor, at::Tensor> _npu_ciou(const at::Tensor& self, const at::Tensor& gtboxes, bool trans = false, bool is_cross = true, int64_t mode = 0,
                                               bool atan_sub_flag = false);
at::Tensor npu_prompt_flash_attention(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                                      const c10::optional<at::Tensor>& padding_mask = {}, const c10::optional<at::Tensor>& atten_mask = {},
                                      at::OptionalIntArrayRef actual_seq_lengths = c10::nullopt, int64_t num_heads = 1, double scale_value = 1.0,
                                      int64_t pre_tokens = 2147473647, int64_t next_tokens = 0, c10::string_view input_layout = "BSH",
                                      int64_t num_key_value_heads = 0);
at::Tensor npu_incre_flash_attention(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                                     const c10::optional<at::Tensor>& padding_mask = {}, const c10::optional<at::Tensor>& atten_mask = {},
                                     at::OptionalIntArrayRef actual_seq_lengths = c10::nullopt, int64_t num_heads = 1, double scale_value = 1.0,
                                     c10::string_view input_layout = "BSH", int64_t num_key_value_heads = 0);
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_transpose_backward(const at::Tensor& input, const at::Tensor& grad, const at::Tensor& weight,
                                                                                    at::IntArrayRef padding, at::IntArrayRef output_padding,
                                                                                    at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups,
                                                                                    ::std::array<bool, 3> grad_input_mask);
at::Tensor npu_scaled_masked_softmax_backward(const at::Tensor& y_grad, const at::Tensor& y, const at::Tensor& mask, const at::Scalar& scale,
                                              bool fixed_triu_mask);
at::Tensor npu_dtype_cast_backward(const at::Tensor& grad, at::ScalarType dtype);
at::Tensor npu_binary_cross_entropy_with_logits_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
                                                         const c10::optional<at::Tensor>& weight_opt, const c10::optional<at::Tensor>& pos_weight_opt,
                                                         int64_t reduction);
at::Tensor npu_confusion_transpose_backward(const at::Tensor& grad, at::IntArrayRef perm, c10::SymIntArrayRef shape, bool transpose_first);
at::Tensor npu_max_backward(const at::Tensor& grad, int64_t dim, const at::Tensor& indices, c10::SymIntArrayRef sizes, bool keepdim);
at::Tensor npu_min_backward(const at::Tensor& grad, int64_t dim, const at::Tensor& indices, c10::SymIntArrayRef sizes, bool keepdim);
at::Tensor npu_ps_roi_pooling_backward(const at::Tensor& output_grad, const at::Tensor& rois, double spatial_scale, int64_t group_size, int64_t output_dim,
                                       c10::SymIntArrayRef input_size);
at::Tensor npu_bmm_v2_mat1_backward(const at::Tensor& grad, const at::Tensor& mat1, const at::Tensor& mat2, c10::SymIntArrayRef size);
at::Tensor npu_bmm_v2_mat2_backward(const at::Tensor& grad, const at::Tensor& mat1, const at::Tensor& mat2, c10::SymIntArrayRef size);
at::Tensor selu_backward(const at::Tensor& grad_output, const at::Tensor& result);
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_data_backward(
    const c10::optional<at::Tensor>& grady_opt, const c10::optional<at::Tensor>& gradh_opt, const c10::optional<at::Tensor>& gradc_opt, const at::Tensor& input,
    const at::Tensor& batch_sizes, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& init_h, const at::Tensor& init_c, const at::Tensor& y,
    const at::Tensor& h, const at::Tensor& c, const at::Tensor& i, const at::Tensor& j, const at::Tensor& f, const at::Tensor& o, const at::Tensor& tanhc,
    bool flag_direction);
at::Tensor l1_loss_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, int64_t reduction);
at::Tensor kl_div_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target, int64_t reduction = at::Reduction::Mean,
                           bool log_target = false);
at::Tensor npu_mm_reduce_scatter_base(const at::Tensor& self, const at::Tensor& x2, c10::string_view hcom, int64_t world_size,
                                      c10::string_view reduce_op = "sum", const c10::optional<at::Tensor>& bias = {}, int64_t comm_turn = 0);
::std::tuple<at::Tensor, at::Tensor> npu_all_gather_base_mm(const at::Tensor& self, const at::Tensor& x2, c10::string_view hcom, int64_t world_size,
                                                            const c10::optional<at::Tensor>& bias = {}, int64_t gather_index = 0, bool gather_output = true,
                                                            int64_t comm_turn = 0);
at::Tensor scatter_update(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t axis);
at::Tensor& scatter_update_(at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t axis);

}  // namespace custom_ops
}  // namespace native
}  // namespace at_npu
