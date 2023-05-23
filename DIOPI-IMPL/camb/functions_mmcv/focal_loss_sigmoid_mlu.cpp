/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include <string>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../diopi_helper.hpp"
#include "../mlu_helper.hpp"

namespace impl {

namespace camb {

void KernelFocalLossSigmoidForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                   cnrtQueue_t queue,
                                   const cnrtDataType_t d_type,
                                   const void *input, const void *target,
                                   const void *weight, const int32_t N,
                                   const int32_t C, const float alpha,
                                   const float gamma, void *output);

void KernelFocalLossSigmoidBackward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                    cnrtQueue_t queue,
                                    const cnrtDataType_t d_type,
                                    const void *input, const void *target,
                                    const void *weight, const float gamma,
                                    const float alpha, const int32_t dim_n,
                                    const int32_t deal_n, const int32_t dim_c,
                                    void *output);

}  // namespace camb

}  // namespace impl

// Policy Function for Forward
static void policyFuncForward(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type,
                              impl::camb::DiopiTensor input, impl::camb::DiopiTensor target,
                              impl::camb::DiopiTensor weight) {
  auto N = input.size(0);
  auto C = input.size(1);

  const size_t nram_size = impl::camb::getDeviceAttr(cnrtAttrNramSizePerMcore);
  const size_t c_align_size = PAD_UP((C * input.elemsize()), NFU_ALIGN_SIZE);
  const int split_target_num = 2;
  const int split_pipeline_num = 6;
  const int has_weight = weight.data() != nullptr;
  const int target_data_width = (target.dtype() == diopi_dtype_int64 || target.dtype() == diopi_dtype_uint64)
                                    ? target.elemsize() / 2
                                    : target.elemsize();
  const int threshold_c =
      PAD_DOWN((nram_size - split_target_num * sizeof(int)) /
                   (split_pipeline_num + has_weight),
               NFU_ALIGN_SIZE) /
      input.elemsize();

  int n_seg = 1;
  if (C <= threshold_c) {
    int c_size = C * input.elemsize();
    int reservered_align_size =
        (split_target_num + split_pipeline_num) * NFU_ALIGN_SIZE;
    int wegiht_size = 0;
    if (has_weight) {
      c_size = c_align_size;
      reservered_align_size = split_target_num * NFU_ALIGN_SIZE;
      wegiht_size = c_align_size;
    }
    // n_seg * c_size * split_pipeline_num + n_seg * target.elemsize() *
    // split_target_num
    //     + weight_size + reservered_align_size <= nram_size
    n_seg = (nram_size - wegiht_size - reservered_align_size) /
            (split_pipeline_num * c_size + split_target_num * sizeof(int32_t));
  }
  auto seg_num = n_seg == 0 ? N : (N + n_seg - 1) / n_seg;
  auto core_dim = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
  auto cluster_num = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
  auto core_num = core_dim * cluster_num;

  k_dim->x = *k_type;
  k_dim->y =
      seg_num > core_num ? cluster_num : (seg_num + core_dim - 1) / core_dim;
  k_dim->z = 1;
}

// Policy Function for Backward
static void policyFuncBackward(cnrtDim3_t *k_dim, cnrtFunctionType_t *k_type) {
  // set Union1 Job
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
  k_dim->z = 1;
}

extern "C" DIOPI_API diopiError_t diopiSigmoidFocalLossMmcv(diopiContextHandle_t ctx,
                                                 diopiTensorHandle_t output_,
                                                 diopiConstTensorHandle_t input_,
                                                 diopiConstTensorHandle_t target_,
                                                 diopiConstTensorHandle_t weight_,
                                                 float gamma,
                                                 float alpha) {
  // // params check
  // TORCH_CHECK(gamma >= 0, "gamma should be greater than or equal to 0. ",
  //             "But now gamma is ", gamma, ".");

  // // check dtype
  // TORCH_CHECK(
  //     input.dtype() == at::kFloat || input.dtype() == at::kHalf,
  //     "Data type of input should be Float or Half. But now input type is ",
  //     input.dtype(), ".");

  // TORCH_CHECK(
  //     (target.dtype() == at::kInt || target.dtype() == at::kLong),
  //     "target type should be Int or Long. ", "But now target type is ",
  //     target.dtype(), ".");

  // if (weight.data() != nullptr) {
  //   TORCH_CHECK(weight.dtype() == input.dtype(),
  //               "Data types of input and weight should be the same. But now "
  //               "input type is ",
  //               input.dtype(), ", weight type is ", weight.dtype(),
  //               ".");
  // } else {
  //   CNLOG(INFO) << "weight is a empty tensor.";
  // }
  auto output = impl::camb::DiopiTensor(output_);
  auto input = impl::camb::DiopiTensor(input_);
  auto target = impl::camb::DiopiTensor(target_);
  auto weight = impl::camb::DiopiTensor(weight_);

  // return if zero-element
  if (input.numel() == 0 || target.numel() == 0 || output.numel() == 0) {
    return diopiSuccess;
  }

  // calculate task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  policyFuncForward(&k_dim, &k_type, input, target, weight);
  auto core_dim = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);

  // get compute queue
  auto queue = impl::camb::getStream(ctx);

  // get dtype of input
  cnrtDataType_t d_type = impl::camb::dtype2CnrtDtype(input.dtype());

  // CNLOG(INFO) << "Launch Kernel KernelFocalLossSigmoidForward<<<Union"
  //             << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
  //             << k_dim.z << ">>>";
  // launch kernel
  impl::camb::KernelFocalLossSigmoidForward(k_dim, k_type, queue, d_type, input.data(),
                                target.data(), weight.data(), input.size(0),
                                input.size(1), alpha, gamma, output.data());
  return diopiSuccess;
}

void getDealNAndThresholdC(const int compute_data_bytes,
                           const int target_data_bytes, const int total_c,
                           int *deal_n_ptr, int *threshold_c_ptr,
                           const bool has_weight, const bool is_half) {
  /* NRAM partition:
   *
   * |-----------------ping pong--------------------|
   * |input | pt | alpha_t | temp | output | target | flt_min | gamma | weight|
   *
   * split_pipeline_num is 5: including input, pt, alpha_t, temp, output.
   */
  const int nram_split_num = 5;
  const int nram_split_pingpong = 2;
  const int max_nram_size = impl::camb::getDeviceAttr(cnrtAttrNramSizePerMcore);
  int32_t compute_align_size = NFU_ALIGN_SIZE;
  if (is_half) {
    compute_align_size += NFU_ALIGN_SIZE;
  }
  const int32_t compute_align_num = compute_align_size / compute_data_bytes;
  // reservered_align_size: including input(ping pong), pt(ping pong),
  //                        alpha_t(ping pong), temp(ping pong),
  //                        output(ping pong), target(ping pong),
  //                        flt_min and gamma.
  const int reservered_align_size =
      ((nram_split_num + 1) * nram_split_pingpong + 2) * compute_align_size;
  int nram_pingpong_size = max_nram_size - reservered_align_size;

  int compute_c = total_c;
  int threshold_c = 0;
  if (has_weight) {
    // reserved space for weight to align
    nram_pingpong_size -= NFU_ALIGN_SIZE;

    // threshold_c * nram_split_pingpong * compute_data_bytes * nram_split_num +
    //     nram_split_pingpong * target_data_bytes +
    //     threshold_c * compute_data_bytes <= nram_pingpong_size
    threshold_c =
        (nram_pingpong_size - nram_split_pingpong * target_data_bytes) /
        (compute_data_bytes * (nram_split_num * nram_split_pingpong + 1));
    threshold_c = PAD_DOWN(threshold_c, compute_align_num);
    int weight_space = PAD_UP(total_c * compute_data_bytes, NFU_ALIGN_SIZE);

    // reserved space for weight
    nram_pingpong_size -= weight_space;
    compute_c = PAD_UP(total_c, compute_align_num);
  } else {
    // threshold_c * nram_split_pingpong * compute_data_bytes * nram_split_num +
    //     nram_split_pingpong * target_data_bytes <= nram_pingpong_size
    threshold_c =
        (nram_pingpong_size / nram_split_pingpong - target_data_bytes) /
        (nram_split_num * compute_data_bytes);
  }
  // deal_n * compute_c * nram_split_pingpong * compute_data_bytes *
  //     nram_split_num + deal_n * nram_split_pingpong * target_data_bytes <=
  //     nram_pingpong_size
  *deal_n_ptr =
      nram_pingpong_size /
      ((nram_split_num * compute_c * compute_data_bytes + target_data_bytes) *
       nram_split_pingpong);
  *threshold_c_ptr = threshold_c;
}

extern "C" DIOPI_API diopiError_t diopiSigmoidFocalLossBackwardMmcv(diopiContextHandle_t ctx,
                                                         diopiTensorHandle_t grad_input_,
                                                         diopiConstTensorHandle_t input_,
                                                         diopiConstTensorHandle_t target_,
                                                         diopiConstTensorHandle_t weight_,
                                                         float gamma,
                                                         float alpha) {
  auto output = impl::camb::DiopiTensor(grad_input_);
  auto input = impl::camb::DiopiTensor(input_);
  auto target = impl::camb::DiopiTensor(target_);
  auto weight = impl::camb::DiopiTensor(weight_);

  // // params check
  // TORCH_CHECK(gamma >= 0, "gamma should be greater than or equal to 0. ",
  //             "But now gamma is ", gamma, ".");
  // // check dtype
  // TORCH_CHECK(
  //     input.dtype() == at::kFloat || input.dtype() == at::kHalf,
  //     "Data type of input should be Float or Half. But now input type is ",
  //     input.dtype(), ".");

  // TORCH_CHECK(
  //     (target.dtype() == at::kInt || target.dtype() == at::kLong),
  //     "target type should be Int or Long. ", "But now target type is ",
  //     target.dtype(), ".");

  bool has_weight = false;
  if (weight.data() != nullptr) {
    // TORCH_CHECK(weight.dtype() == input.dtype(),
    //             "Data types of input and weight should be the same. But now "
    //             "input type is ",
    //             input.dtype(), ", weight type is ", weight.dtype(),
    //             ".");
    has_weight = true;
  }
  // else {
  //   CNLOG(INFO) << "weight is a empty tensor.";
  // }

  auto dim_c = input.size(1);
  const int compute_data_bytes = sizeof(float);
  // target supports only INT on MLU device while it keeps LONG on host side,
  // so target.elemsize() / 2
  const int target_data_bytes = (target.dtype() == diopi_dtype_int64 || target.dtype() == diopi_dtype_uint64)
                                    ? (target.elemsize() / 2)
                                    : target.elemsize();
  int deal_n = 0;
  int threshold_c = 0;
  bool is_half = false;
  if (input.dtype() == diopi_dtype_float16) {
    is_half = true;
  }
  // calculate deal_n and threshold_c
  getDealNAndThresholdC(compute_data_bytes, target_data_bytes, dim_c, &deal_n,
                        &threshold_c, has_weight, is_half);

  // // check C
  // TORCH_CHECK(threshold_c >= dim_c,
  //             "input.size(1) should be in the range of [0, ", threshold_c,
  //             "]. ", "But now input.size(1) is ", dim_c, ".");

  if (input.numel() == 0 || target.numel() == 0 || output.numel() == 0) {
    // return if zero-element
    return diopiSuccess;
  }

  // set task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFuncBackward(&k_dim, &k_type);

  // get compute queue
  auto queue = impl::camb::getStream(ctx);

  // get dtype of input
  cnrtDataType_t d_type = impl::camb::dtype2CnrtDtype(input.dtype());
  auto core_dim = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
  auto dim_n = input.size(0);

  // CNLOG(INFO) << "Launch Kernel KernelFocalLossSigmoidBackward<<<Union"
  //             << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
  //             << k_dim.z << ">>>";

  // launch kernel
  impl::camb::KernelFocalLossSigmoidBackward(k_dim, k_type, queue, d_type, input.data(),
                                 target.data(), weight.data(), gamma, alpha, dim_n,
                                 deal_n, dim_c, output.data());
  return diopiSuccess;
}
