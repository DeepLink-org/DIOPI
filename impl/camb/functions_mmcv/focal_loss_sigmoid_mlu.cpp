/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../diopi_helper.hpp"
#include "../mlu_helper.hpp"

namespace impl {

namespace camb {

void kernelFocalLossSigmoidForward(cnrtDim3_t kDim, cnrtFunctionType_t kType, cnrtQueue_t queue, const cnrtDataType_t dType, const void *input,
                                   const void *target, const void *weight, const int32_t n, const int32_t c, const float alpha, const float gamma,
                                   void *output);

void kernelFocalLossSigmoidBackward(cnrtDim3_t kDim, cnrtFunctionType_t kType, cnrtQueue_t queue, const cnrtDataType_t dType, const void *input,
                                    const void *target, const void *weight, const float gamma, const float alpha, const int32_t dimN, const int32_t dealN,
                                    const int32_t dimC, void *output);

}  // namespace camb

}  // namespace impl

// Policy Function for Forward
static void policyFuncForward(cnrtDim3_t *kDim, cnrtFunctionType_t *kType, impl::camb::DiopiTensor input, impl::camb::DiopiTensor target,
                              impl::camb::DiopiTensor weight) {
    auto n = input.size(0);
    auto c = input.size(1);

    const size_t nramSize = impl::camb::getDeviceAttr(cnrtAttrNramSizePerMcore);
    const size_t cAlignSize = PAD_UP((c * input.elemsize()), NFU_ALIGN_SIZE);
    const int splitTargetNum = 2;
    const int splitPipelineNum = 6;
    const int hasWeight = weight.data() != nullptr;
    const int targetDataWidth = (target.dtype() == diopi_dtype_int64 || target.dtype() == diopi_dtype_uint64) ? target.elemsize() / 2 : target.elemsize();
    const int thresholdC = PAD_DOWN((nramSize - splitTargetNum * sizeof(int)) / (splitPipelineNum + hasWeight), NFU_ALIGN_SIZE) / input.elemsize();

    int nSeg = 1;
    if (c <= thresholdC) {
        int cSize = c * input.elemsize();
        int reserveredAlignSize = (splitTargetNum + splitPipelineNum) * NFU_ALIGN_SIZE;
        int wegihtSize = 0;
        if (hasWeight) {
            cSize = cAlignSize;
            reserveredAlignSize = splitTargetNum * NFU_ALIGN_SIZE;
            wegihtSize = cAlignSize;
        }
        // n_seg * c_size * split_pipeline_num + n_seg * target.elemsize() *
        // split_target_num
        //     + weight_size + reservered_align_size <= nram_size
        nSeg = (nramSize - wegihtSize - reserveredAlignSize) / (splitPipelineNum * cSize + splitTargetNum * static_cast<int>(sizeof(int32_t)));
    }
    auto segNum = nSeg == 0 ? n : (n + nSeg - 1) / nSeg;
    auto coreDim = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    auto clusterNum = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
    auto coreNum = coreDim * clusterNum;

    kDim->x = *kType;
    kDim->y = segNum > coreNum ? clusterNum : (segNum + coreDim - 1) / coreDim;
    kDim->z = 1;
}

// Policy Function for Backward
static void policyFuncBackward(cnrtDim3_t *kDim, cnrtFunctionType_t *kType) {
    // set Union1 Job
    *kType = CNRT_FUNC_TYPE_UNION1;
    kDim->x = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    kDim->y = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
    kDim->z = 1;
}

extern "C" DIOPI_API diopiError_t diopiSigmoidFocalLossMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiConstTensorHandle_t input,
                                                            diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, float gamma, float alpha) {
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
    auto outputTr = impl::camb::DiopiTensor(output);
    auto inputTr = impl::camb::DiopiTensor(input);
    auto targetTr = impl::camb::DiopiTensor(target);
    auto weightTr = impl::camb::DiopiTensor(weight);

    // return if zero-element
    if (inputTr.numel() == 0 || targetTr.numel() == 0 || outputTr.numel() == 0) {
        return diopiSuccess;
    }

    // calculate task dimension
    cnrtDim3_t kDim;
    cnrtFunctionType_t kType = CNRT_FUNC_TYPE_UNION1;
    policyFuncForward(&kDim, &kType, inputTr, targetTr, weightTr);
    auto coreDim = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);

    // get compute queue
    auto queue = impl::camb::getStream(ctx);

    // get dtype of input
    cnrtDataType_t dType = impl::camb::dtype2CnrtDtype(inputTr.dtype());

    // CNLOG(INFO) << "Launch Kernel KernelFocalLossSigmoidForward<<<Union"
    //             << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
    //             << k_dim.z << ">>>";
    // launch kernel
    impl::camb::kernelFocalLossSigmoidForward(
        kDim, kType, queue, dType, inputTr.data(), targetTr.data(), weightTr.data(), inputTr.size(0), inputTr.size(1), alpha, gamma, outputTr.data());
    return diopiSuccess;
}

void getDealNAndThresholdC(const int computeDataBytes, const int targetDataBytes, const int totalC, int *dealNPtr, int *thresholdCPtr, const bool hasWeight,
                           const bool isHalf) {
    /* NRAM partition:
     *
     * |-----------------ping pong--------------------|
     * |input | pt | alpha_t | temp | output | target | flt_min | gamma | weight|
     *
     * split_pipeline_num is 5: including input, pt, alpha_t, temp, output.
     */
    const int nramSplitNum = 5;
    const int nramSplitPingpong = 2;
    const int maxNramSize = impl::camb::getDeviceAttr(cnrtAttrNramSizePerMcore);
    int32_t computeAlignSize = NFU_ALIGN_SIZE;
    if (isHalf) {
        computeAlignSize += NFU_ALIGN_SIZE;
    }
    const int32_t computeAlignNum = computeAlignSize / computeDataBytes;
    // reservered_align_size: including input(ping pong), pt(ping pong),
    //                        alpha_t(ping pong), temp(ping pong),
    //                        output(ping pong), target(ping pong),
    //                        flt_min and gamma.
    const int reserveredAlignSize = ((nramSplitNum + 1) * nramSplitPingpong + 2) * computeAlignSize;
    int nramPingpongSize = maxNramSize - reserveredAlignSize;

    int computeC = totalC;
    int thresholdC = 0;
    if (hasWeight) {
        // reserved space for weight to align
        nramPingpongSize -= NFU_ALIGN_SIZE;

        // threshold_c * nram_split_pingpong * compute_data_bytes * nram_split_num +
        //     nram_split_pingpong * target_data_bytes +
        //     threshold_c * compute_data_bytes <= nram_pingpong_size
        thresholdC = (nramPingpongSize - nramSplitPingpong * targetDataBytes) / (computeDataBytes * (nramSplitNum * nramSplitPingpong + 1));
        thresholdC = PAD_DOWN(thresholdC, computeAlignNum);
        int weightSpace = PAD_UP(totalC * computeDataBytes, NFU_ALIGN_SIZE);

        // reserved space for weight
        nramPingpongSize -= weightSpace;
        computeC = PAD_UP(totalC, computeAlignNum);
    } else {
        // threshold_c * nram_split_pingpong * compute_data_bytes * nram_split_num +
        //     nram_split_pingpong * target_data_bytes <= nram_pingpong_size
        thresholdC = (nramPingpongSize / nramSplitPingpong - targetDataBytes) / (nramSplitNum * computeDataBytes);
    }
    // deal_n * compute_c * nram_split_pingpong * compute_data_bytes *
    //     nram_split_num + deal_n * nram_split_pingpong * target_data_bytes <=
    //     nram_pingpong_size
    *dealNPtr = nramPingpongSize / ((nramSplitNum * computeC * computeDataBytes + targetDataBytes) * nramSplitPingpong);
    *thresholdCPtr = thresholdC;
}

extern "C" DIOPI_API diopiError_t diopiSigmoidFocalLossBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t input,
                                                                    diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, float gamma,
                                                                    float alpha) {
    auto outputTr = impl::camb::DiopiTensor(gradInput);
    auto inputTr = impl::camb::DiopiTensor(input);
    auto targetTr = impl::camb::DiopiTensor(target);
    auto weightTr = impl::camb::DiopiTensor(weight);

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

    bool hasWeight = false;
    if (weightTr.data() != nullptr) {
        // TORCH_CHECK(weight.dtype() == input.dtype(),
        //             "Data types of input and weight should be the same. But now "
        //             "input type is ",
        //             input.dtype(), ", weight type is ", weight.dtype(),
        //             ".");
        hasWeight = true;
    }
    // else {
    //   CNLOG(INFO) << "weight is a empty tensor.";
    // }

    auto dimC = inputTr.size(1);
    const int computeDataBytes = sizeof(float);
    // target supports only INT on MLU device while it keeps LONG on host side,
    // so target.elemsize() / 2
    const int targetDataBytes =
        (targetTr.dtype() == diopi_dtype_int64 || targetTr.dtype() == diopi_dtype_uint64) ? (targetTr.elemsize() / 2) : targetTr.elemsize();
    int dealN = 0;
    int thresholdC = 0;
    bool isHalf = false;
    if (inputTr.dtype() == diopi_dtype_float16) {
        isHalf = true;
    }
    // calculate deal_n and threshold_c
    getDealNAndThresholdC(computeDataBytes, targetDataBytes, dimC, &dealN, &thresholdC, hasWeight, isHalf);

    // // check C
    // TORCH_CHECK(threshold_c >= dim_c,
    //             "input.size(1) should be in the range of [0, ", threshold_c,
    //             "]. ", "But now input.size(1) is ", dim_c, ".");

    if (inputTr.numel() == 0 || targetTr.numel() == 0 || outputTr.numel() == 0) {
        // return if zero-element
        return diopiSuccess;
    }

    // set task dimension
    cnrtDim3_t kDim;
    cnrtFunctionType_t kType;
    policyFuncBackward(&kDim, &kType);

    // get compute queue
    auto queue = impl::camb::getStream(ctx);

    // get dtype of input
    cnrtDataType_t dType = impl::camb::dtype2CnrtDtype(inputTr.dtype());
    auto coreDim = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    auto dimN = inputTr.size(0);

    // CNLOG(INFO) << "Launch Kernel KernelFocalLossSigmoidBackward<<<Union"
    //             << k_type / core_dim << ", " << k_dim.x << ", " << k_dim.y << ", "
    //             << k_dim.z << ">>>";

    // launch kernel
    impl::camb::kernelFocalLossSigmoidBackward(
        kDim, kType, queue, dType, inputTr.data(), targetTr.data(), weightTr.data(), gamma, alpha, dimN, dealN, dimC, outputTr.data());
    return diopiSuccess;
}
