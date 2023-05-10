/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include <iostream>

#include "../mlu_helper.hpp"
#include "../diopi_helper.hpp"
#include "../cnnl_helper.hpp"

namespace impl {

namespace camb {

void KernelRoiAlign(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                    cnrtQueue_t queue, const cnrtDataType_t d_type,
                    const void *input, const void *rois, const int channels,
                    const bool aligned, const int pooled_height,
                    const int pooled_width, const int input_height,
                    const int input_width, const int sampling_ratio,
                    const float spatial_scale, const int num_rois,
                    void *output);

void KernelRoiAlignBackward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                            cnrtQueue_t queue, const cnrtDataType_t dtype,
                            const void *grads, const void *boxes,
                            void *grads_image, const int boxes_num,
                            const int hi, const int wi, const int c,
                            const int no, const int ho, const int wo,
                            const float spatial_scale, const int sampling_ratio,
                            const bool aligned);

}  // namespace camb

}  // namespace impl

extern "C" DIOPI_API diopiError_t diopiRoiAlignMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output_, diopiTensorHandle_t argmax_y_, diopiTensorHandle_t argmax_x_, 
                               diopiConstTensorHandle_t input_, diopiConstTensorHandle_t rois_,
                               int64_t aligned_height, int64_t aligned_width, int64_t sampling_ratio, int64_t pool_mode,
                               float spatial_scale, bool aligned) {
  auto input = impl::camb::DiopiTensor(input_);
  auto rois = impl::camb::DiopiTensor(rois_);
  auto output = impl::camb::DiopiTensor(output_);
  auto argmax_y = impl::camb::DiopiTensor(argmax_y_);
  auto argmax_x = impl::camb::DiopiTensor(argmax_x_);

  auto memory_format = impl::camb::MemoryFormat::ChannelsLast;
  auto input_tensor = input.contiguous(ctx, memory_format);
  cnnlHandle_t handle = impl::camb::cnnlHandlePool.get(ctx);
  cnnl_transpose(ctx, handle, input, input_tensor, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  if (output.numel() == 0) {
    auto dtype = input.dtype();
    output = impl::camb::requiresTensor(ctx, {num_rois, channels, aligned_height, aligned_width}, dtype);
    diopiScalar_t scalar = {dtype, 0.0};
    if (impl::camb::DiopiDataType().isInteger(dtype)) scalar = {dtype, 0};
    diopiFill(ctx, diopiTensorHandle_t(output), &scalar);
    return diopiSuccess;
  }

  auto output_temp = impl::camb::requiresTensor(ctx, {num_rois, channels, aligned_height, aligned_width}, input.dtype());
  auto output_tmp = output_temp.contiguous(ctx, memory_format);

  // get compute queue
  auto queue = impl::camb::getStream(ctx);

  cnrtJobType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  k_dim.x = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim.y = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
  k_dim.z = 1;
  cnrtDataType_t data_type = impl::camb::dtype2CnrtDtype(input.dtype());

  impl::camb::KernelRoiAlign(k_dim, k_type, queue, data_type, input_tensor.data(), rois.data(), channels,
                 aligned, aligned_height, aligned_width, height, width,
                 sampling_ratio, spatial_scale, num_rois, output_tmp.data());
  cnnl_transpose(ctx, handle, output_tmp, output, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);
  return diopiSuccess;
}

static int nearestPower2(int x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

extern "C" diopiError_t diopiRoiAlignBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input__, diopiConstTensorHandle_t grad_output_, diopiConstTensorHandle_t rois_, 
                             diopiConstTensorHandle_t argmax_y_, diopiConstTensorHandle_t argmax_x_, 
                             int64_t aligned_height, int64_t aligned_width, int64_t sampling_ratio,
                             int64_t pool_mode, float spatial_scale,  bool aligned) {

  auto grad = impl::camb::DiopiTensor(grad_output_);
  auto rois = impl::camb::DiopiTensor(rois_);
  auto argmax_y = impl::camb::DiopiTensor(argmax_y_);
  auto argmax_x = impl::camb::DiopiTensor(argmax_x_);
  auto grad_input = impl::camb::DiopiTensor(grad_input__);

  int batch_size = grad_input.size(0);
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  auto memory_format = impl::camb::MemoryFormat::ChannelsLast;
  auto grad_ = grad.contiguous(ctx, memory_format);
  cnnlHandle_t handle = impl::camb::cnnlHandlePool.get(ctx);
  cnnl_transpose(ctx, handle, grad, grad_, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

  auto dtype = grad.dtype();
  auto grad_input_tmp = impl::camb::requiresTensor(ctx, {batch_size, channels, height, width}, dtype);
  diopiScalar_t scalar = {dtype, 0.0};
  if (impl::camb::DiopiDataType().isInteger(dtype)) {
     scalar = {dtype, 0};
  }
  diopiFill(ctx, diopiTensorHandle_t(grad_input_tmp), &scalar);

  auto grad_input_ = grad_input_tmp.contiguous(ctx, memory_format);
  cnnl_transpose(ctx, handle, grad_input_tmp, grad_input_, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

  int boxes_num = rois.size(0);
  int hi = grad.size(2);
  int wi = grad.size(3);
  int c = grad.size(1);

  int no = grad_input.size(0);
  int ho = grad_input.size(2);
  int wo = grad_input.size(3);

  // get compute queue
  auto queue = impl::camb::getStream(ctx);

  cnrtJobType_t k_type = CNRT_FUNC_TYPE_UNION1;
  int need_core = nearestPower2(boxes_num);
  int union_number = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
  uint32_t dim_x = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
  uint32_t dim_y = (need_core - 1) / dim_x + 1;
  dim_y = (dim_y > union_number) ? union_number : dim_y;
  cnrtDim3_t k_dim = {dim_x, dim_y, 1};
  cnrtDataType_t k_dtype = impl::camb::dtype2CnrtDtype(grad.dtype());

  impl::camb::KernelRoiAlignBackward(k_dim, k_type, queue, k_dtype, grad_.data(), rois.data(),
                         grad_input_.data(), boxes_num, hi, wi, c, no, ho, wo,
                         spatial_scale, sampling_ratio, aligned);

  cnnl_transpose(ctx, handle, grad_input_, grad_input, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);
  return diopiSuccess;
}
