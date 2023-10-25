/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <diopi/functions.h>
#include <math.h>
#include <torch/nn.h>
#include <torch/optim.h>

#include <cstring>

#ifdef USE_HIP
#include <miopen/version.h>
#endif

#define FLT_MIN __FLT_MIN__

#include "context.h"
#include "helper.hpp"
#include "vision_kernel.h"

extern "C" {

static const char* name = "CudaDevice";
static char version[1024] = {0};

const char* diopiGetVendorName() { return name; }

const char* diopiGetImplVersion() {
    if (strlen(version) == 0) {
#ifdef USE_HIP
        sprintf(version,
                "HIP Version: %d; MIOPEN Version: %d.%d.%d; DIOPI Version: %d.%d.%d",
                HIP_VERSION,
                MIOPEN_VERSION_MAJOR,
                MIOPEN_VERSION_MINOR,
                MIOPEN_VERSION_PATCH,
                DIOPI_VER_MAJOR,
                DIOPI_VER_MINOR,
                DIOPI_VER_PATCH);

#else
        sprintf(version,
                "Cuda Version: %d; Cudnn Version: %d; DIOPI Version: %d.%d.%d",
                CUDART_VERSION,
                CUDNN_VERSION,
                DIOPI_VER_MAJOR,
                DIOPI_VER_MINOR,
                DIOPI_VER_PATCH);
#endif
    }
    return version;
}

const char* diopiGetLastErrorString() { return cuda_get_last_error_string(); }

diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::relu_out, atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::relu_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atSlope = impl::aten::buildAtScalar(negative_slope);
    impl::aten::invokeATenFuncInp(ctx, at::leaky_relu_out, atOut, atInput, atSlope);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atSlope = impl::aten::buildAtScalar(negative_slope);
    impl::aten::invokeATenFuncInp(ctx, at::leaky_relu_, atInput, atSlope);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                            diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    bool atCeilMode = ceil_mode;
    impl::aten::invokeATenFuncRet(ctx, at::max_pool2d, out, atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    bool atCeilMode = ceil_mode;
    at::max_pool2d_with_indices_out(atOut, atIndices, atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

/**
 * @brief
 * @param rounding_mode supported in pytorch>=1.8
 */
diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      diopiRoundMode_t rounding_mode) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    auto roundingMode = impl::aten::getRoundingMode(rounding_mode);
    at::div_out(atOut, atInput, atOther, roundingMode);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

/**
 * @brief
 * @param rounding_mode supported in pytorch>=1.8
 */
diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    auto roundingMode = impl::aten::getRoundingMode(rounding_mode);
    atInput.div_(atOther, roundingMode);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

/**
 * @brief
 * @param rounding_mode supported in pytorch>=1.8.0
 */
diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            diopiRoundMode_t rounding_mode) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildAtScalar(other);
    auto roundingMode = impl::aten::getRoundingMode(rounding_mode);
    auto atOut = impl::aten::buildATen(out);
    at::div_out(atOut, atInput, atOther, roundingMode);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

/**
 * @brief
 * @param rounding_mode supported in pytorch>=1.8.0
 */
diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildAtScalar(other);
    auto roundingMode = impl::aten::getRoundingMode(rounding_mode);
    atInput.div_(atOther, roundingMode);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    auto atDilation = impl::aten::buildAtIntArray(dilation);
    impl::aten::invokeATenFuncInp(
        ctx, at::convolution_out, atOut, atInput, atWeight, atBias, atStride, atPadding, atDilation, false, at::IntArrayRef(0), groups);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

/**
 * @brief
 *
 * @param ignore_index supported in torch >= 1.10.0
 * @param label_smoothing supported in torch >= 1.10.0
 */
diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                   diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    auto atWeight = impl::aten::buildATen(weight);
#if TORCH_MM_VERSION >= TORCH_1_10_MM_VERSION
    auto atOut = at::cross_entropy_loss(atInput, atTarget, atWeight, reduction, ignore_index, label_smoothing);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
#elif TORCH_MM_VERSION == TORCH_1_9_MM_VERSION
    NOT_SUPPORTED("param label_smoothing in torch 1.9")
    auto atOut = at::cross_entropy_loss(atInput, atTarget, atWeight, reduction, ignore_index);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
#else
    ATEN_NOT_IMPLEMENT();
#endif
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMat2 = impl::aten::buildATen(mat2);
    auto atOut = impl::aten::buildATen(out);
    at::bmm_out(atOut, atInput, atMat2);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1,
                          diopiConstTensorHandle_t batch2, double beta, double alpha) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atBatch1 = impl::aten::buildATen(batch1);
    auto atBatch2 = impl::aten::buildATen(batch2);
    at::baddbmm_out(atOut, atInput, atBatch1, atBatch2, beta, alpha);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta,
                             double alpha) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atBatch1 = impl::aten::buildATen(batch1);
    auto atBatch2 = impl::aten::buildATen(batch2);
    atInput.baddbmm_(atBatch1, atBatch2, beta, alpha);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                          diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    auto atTensor1 = impl::aten::buildATen(tensor1);
    auto atTensor2 = impl::aten::buildATen(tensor2);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncInp(ctx, at::addcmul_out, atOut, atInput, atTensor1, atTensor2, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                             const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atTensor1 = impl::aten::buildATen(tensor1);
    auto atTensor2 = impl::aten::buildATen(tensor2);
    auto atValue = impl::aten::buildAtScalar(value);
    atInput.addcmul_(atTensor1, atTensor2, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildATen(other);
    // Note(huqingqing): pytorch optimize the bmm case by folding the batch into the first dimension.
    // It changes the shape of output and causes warnning when using matmul_out.
    impl::aten::invokeATenFuncInp(ctx, at::matmul_out, atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                          diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atTensor1 = impl::aten::buildATen(tensor1);
    auto atTensor2 = impl::aten::buildATen(tensor2);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = impl::aten::buildATen(out);
    at::addcdiv_out(atOut, atInput, atTensor1, atTensor2, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                             const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atTensor1 = impl::aten::buildATen(tensor1);
    auto atTensor2 = impl::aten::buildATen(tensor2);
    auto atValue = impl::aten::buildAtScalar(value);
    atInput.addcdiv_(atTensor1, atTensor2, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

// CAFFE2_API Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                        diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMax1 = impl::aten::buildATen(mat1);
    auto atMax2 = impl::aten::buildATen(mat2);
    auto atBeta = impl::aten::buildAtScalar(beta);
    auto atAlpha = impl::aten::buildAtScalar(alpha);
    auto atOut = impl::aten::buildATen(out);
    at::addmm_out(atOut, atInput, atMax1, atMax2, atBeta, atAlpha);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

// NOTE(fengsibo): add int, short, bool test case
diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atDim = impl::aten::buildAtIntArray(dim);
    bool keepdim = false;
    if (atInput.dim() == atOut.dim()) {
        keepdim = true;
    }
    at::mean_out(atOut, atInput, atDim, keepdim);  // TODO(fengsibo): use default type instead
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

// NOTE(fengsibo): add int, short, bool test case
diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atDim = impl::aten::buildAtIntArray(dim);
    // TODO(fengsibo): use default type instead
    bool keepdim = false;
    if (atInput.dim() == atOut.dim()) {
        keepdim = true;
    }
    at::sum_out(atOut, atInput, atDim, keepdim);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atDim = impl::aten::buildAtIntArray(dim);
    bool keepdim = false;
    if (atInput.dim() == atOut.dim()) {
        keepdim = true;
    }
    at::std_out(atOut, atInput, atDim, unbiased, keepdim);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices, diopiConstTensorHandle_t input, int64_t dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(min);
    auto atIndices = impl::aten::buildATen(min_indices);
    bool keepdim = false;
    if (atInput.dim() == atOut.dim()) {
        keepdim = true;
    }
    at::min_out(atOut, atIndices, atInput, dim, keepdim);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet<at::Tensor (*)(at::Tensor const&)>(ctx, at::min, min, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices, diopiConstTensorHandle_t input, int64_t dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(max);
    auto atIndices = impl::aten::buildATen(max_indices);
    bool keepdim = false;
    if (atInput.dim() == atOut.dim()) {
        keepdim = true;
    }
    at::max_out(atOut, atIndices, atInput, dim, keepdim);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet<at::Tensor (*)(at::Tensor const&)>(ctx, at::max, max, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    bool keepdim = false;
    if (atInput.dim() == atOut.dim()) {
        keepdim = true;
    }
    if (dim == nullptr) {
        at::any_out(atOut, atInput);
    } else {
        at::any_out(atOut, atInput, *dim, keepdim);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    bool keepdim = false;
    if (atInput.dim() == atOut.dim()) {
        keepdim = true;
    }
    if (dim == nullptr) {
        at::all_out(atOut, atInput);
    } else {
        at::all_out(atOut, atInput, *dim, keepdim);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::softmax(atInput, dim);  // TODO(fengsibo): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::log_softmax(atInput, dim);  // TODO(fengsibo): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atOut = impl::aten::buildATen(out);
    at::index_select_out(atOut, atInput, dim, atIndex);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    at::Tensor atOut = at::select(atInput, dim, index).contiguous();
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                diopiConstTensorHandle_t source) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atSource = impl::aten::buildATen(source);
    impl::aten::invokeATenFuncRet(ctx, at::masked_scatter, out, atInput, atMask, atSource);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets, diopiConstTensorHandle_t scores, double iouThreshold) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(out);
    auto atDets = impl::aten::buildATen(dets);
    auto atScores = impl::aten::buildATen(scores);
    auto atOut = vision::ops::nms_kernel(atDets, atScores, iouThreshold);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(out);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::nonzero(atInput);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                         diopiConstTensorHandle_t bias) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);
    impl::aten::invokeATenFuncInp(ctx, at::linear_out, atOut, atInput, atWeight, atBias);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois,
                           double spatialScale, int64_t pooledHeight, int64_t pooledWidth, int64_t samplingRatio, bool aligned) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atRois = impl::aten::buildATen(rois);
    auto atOut = vision::ops::roi_align_forward_kernel(atInput, atRois, spatialScale, pooledHeight, pooledWidth, samplingRatio, aligned);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf, double learningrate, double momentum,
                      double dampening, double weightDecay, bool nesterov) {
    impl::aten::setCurCtx(ctx);
    auto atW = impl::aten::buildATen(w);
    auto atDw = impl::aten::buildATen(dw);
    auto atBuf = impl::aten::buildATen(buf);

    atW.requires_grad_(true);
    atW.mutable_grad() = atDw;

    // Implementation in pytorch v1.10.2 sgd.cpp.
    auto& p = atW;
    auto d_p = p.grad().data();
    if (weightDecay != 0) {
        d_p = d_p.add(p.data(), weightDecay);
    }
    if (momentum != 0) {
        if (buf != nullptr) {
            atBuf.mul_(momentum).add_(d_p, 1 - dampening);
            impl::aten::updateATen2Tensor(ctx, atBuf, buf);
        } else {
            atBuf = d_p;
        }
        if (nesterov) {
            d_p = d_p.add(atBuf, momentum);
        } else {
            d_p = atBuf;
        }
    }
    p.data().add_(d_p, -1 * learningrate);

    impl::aten::updateATen2Tensor(ctx, atW, w);
    impl::aten::updateATen2Tensor(ctx, atDw, dw);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

/**
 * @brief
 * @param errorIfNonfinite supported in pytorch ?
 * @return diopiError_t
 */
diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t* grads, int64_t num_grads, double maxNorm, double normType,
                               bool errorIfNonfinite) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK(grads != nullptr && out != nullptr, "Not supported: out or parameters is nullptr");
    auto atGrads = impl::aten::buildATenList(grads, num_grads);
    at::Tensor total_norm_tensor;
    if (normType == std::numeric_limits<double>::infinity()) {
        std::vector<at::Tensor> norms;
        norms.reserve(atGrads.size());
        for (const auto& grad : atGrads) {
            norms.emplace_back(grad.abs().max());
        }
        total_norm_tensor = (norms.size() == 1) ? norms[0] : torch::max(torch::stack(norms));
    } else if (normType == 0) {
        total_norm_tensor = torch::full({}, static_cast<double>(atGrads.size()));
    } else {
        std::vector<at::Tensor> norms;
        norms.reserve(atGrads.size());
        for (const auto& grad : atGrads) {
            norms.emplace_back(grad.norm(normType));
        }
        total_norm_tensor = (norms.size() == 1) ? norms[0] : torch::stack(norms).norm(normType);
    }

    c10::optional<double> total_norm = c10::nullopt;
    if (errorIfNonfinite) {
        total_norm = total_norm_tensor.item().toDouble();
        DIOPI_CHECK(std::isfinite(*total_norm), "The total norm for gradients from `parameters` is non-finite");
    }

    auto clip_coef = maxNorm / (total_norm_tensor + 1e-6);
    auto clip_coef_clamped = torch::clamp(clip_coef, c10::nullopt /* min */, 1.0 /* max */);
    for (auto& grad : atGrads) {
        grad.mul_(clip_coef_clamped);
    }

    if (!total_norm.has_value()) {
        total_norm = total_norm_tensor.item().toDouble();
    }
    *out = *total_norm;
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiConstTensorHandle_t indices, double max_norm, double norm_type) {
    impl::aten::setCurCtx(ctx);
    auto atSelf = impl::aten::buildATen(inout);
    auto atIndices = impl::aten::buildATen(indices);
    at::embedding_renorm_(atSelf, atIndices, max_norm, norm_type);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                            int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    impl::aten::setCurCtx(ctx);
    auto atWeight = impl::aten::buildATen(weight);
    auto atIndices = impl::aten::buildATen(indices);
    impl::aten::invokeATenFuncRet(ctx, at::embedding, out, atWeight, atIndices, paddingIdx, scaleGradByFreq, sparse);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::tril_out(atOut, atInput, diagonal);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t insNum, int64_t dim) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(tensors);
    auto tensorList = impl::aten::buildATenList(tensors, insNum);
    auto atOut = impl::aten::buildATen(out);
    at::cat_out(atOut, tensorList, dim);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t outsNum, diopiConstTensorHandle_t input,
                                 const diopiSize_t splitSizes, int64_t dim) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(outs);
    auto atInput = impl::aten::buildATen(input);
    auto atSizes = impl::aten::buildAtIntArray(splitSizes);
    auto atOuts = at::split_with_sizes(atInput, atSizes, dim);
    for (int i = 0; i < outsNum; ++i) {
        impl::aten::updateATen2Tensor(ctx, atOuts[i].contiguous(), outs[i]);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(tensors);
    auto tensorList = impl::aten::buildATenList(tensors, numTensors);

    auto atOut = impl::aten::buildATen(out);
    at::stack_out(atOut, tensorList, dim);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim,
                       bool descending, const bool* stable) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atValues = impl::aten::buildATen(values);
    auto atIndices = impl::aten::buildATen(indices);
#if TORCH_MM_VERSION <= TORCH_1_8_MM_VERSION
    at::sort_out(atValues, atIndices, atInput, dim, descending);
#else
    c10::optional<bool> atStable = stable ? c10::optional<bool>(*stable) : c10::optional<bool>(false);
    at::sort_out(atValues, atIndices, atInput, atStable, dim, descending);
#endif
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t k,
                       int64_t dim, bool largest, bool sorted) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atValues = impl::aten::buildATen(values);
    auto atIndices = impl::aten::buildATen(indices);
    at::topk_out(atValues, atIndices, atInput, k, dim, largest, sorted);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet<at::Tensor (*)(at::Tensor const&, int64_t, int64_t)>(ctx, at::transpose, out, atInput, dim0, dim1);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numClasses) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::one_hot, out, atInput, numClasses);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                        diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    auto atCondition = impl::aten::buildATen(condition);
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildATen(other);
    // impl::aten::invokeATenFuncRet<at::Tensor (*)(at::Tensor const&, at::Tensor const&, at::Tensor const&)>(ctx, at::where, out, atCondition, atInput,
    // atOther);
    at::where_out(atOut, atCondition, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sin_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::sin_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::cos_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::cos_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAbs(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::abs_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::abs_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sqrt_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::sqrt_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRsqrt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::rsqrt_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRsqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::rsqrt_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::floor_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::floor_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNeg(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::neg_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::neg_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sign_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::tanh_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::tanh_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAtan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::atan_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAtanInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::atan_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSigmoid(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sigmoid_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::sigmoid_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSiluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::silu_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSilu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::silu_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}
diopiError_t diopiSiluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::silu_backward_out, atGradInput, atGradOutput, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::exp_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::exp_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::log_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::log_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLog2(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::log2_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::log2_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLog10(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::log10_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::log10_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiErf(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::erf_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::erf_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atExponent = impl::aten::buildATen(exponent);
    at::Scalar atInput = impl::aten::buildAtScalar(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::pow_out(atOut, atInput, atExponent);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atExponent = impl::aten::buildAtScalar(exponent);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::pow_out(atOut, atInput, atExponent);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atExponent = impl::aten::buildAtScalar(exponent);
    atInput.pow_(atExponent);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atExponent = impl::aten::buildATen(exponent);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::pow_out(atOut, atInput, atExponent);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atExponent = impl::aten::buildATen(exponent);
    atInput.pow_(atExponent);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::add_out(atOut, atInput, atOther, atAlpha);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    atInput.add_(atOther, atAlpha);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::add_out(atOut, atInput, atOther, atAlpha);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    atInput.add_(atOther, atAlpha);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other,
                      const diopiScalar_t* alpha) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sub_out(atOut, atInput, atOther, atAlpha);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSubInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    atInput.sub_(atOther, atAlpha);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other,
                            const diopiScalar_t* alpha) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sub_out(atOut, atInput, atOther, atAlpha);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSubInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    atInput.sub_(atOther, atAlpha);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::mul_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.mul_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::mul_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    atInput.mul_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::ge_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.ge_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::ge_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    atInput.ge_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::gt_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.gt_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::gt_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    atInput.gt_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::le_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.le_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::le_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    atInput.le_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::lt_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.lt_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::lt_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLtInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    atInput.lt_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::eq_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiEqInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.eq_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::eq_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiEqInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    atInput.eq_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::ne_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNeInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.ne_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::ne_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNeInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    atInput.ne_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::bitwise_and_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBitwiseAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.bitwise_and_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::bitwise_and_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBitwiseAndInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    atInput.bitwise_and_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::bitwise_or_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.bitwise_or_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::bitwise_or_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    atInput.bitwise_or_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLogicalAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::logical_and_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLogicalAndInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.logical_and_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLogicalOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::logical_or_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLogicalOrInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    atInput.logical_or_(atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    c10::optional<at::Scalar> atMin = c10::optional<at::Scalar>();
    if (min != nullptr) {
        atMin = impl::aten::buildAtScalar(min);
    }
    c10::optional<at::Scalar> atMax = c10::optional<at::Scalar>();
    if (max != nullptr) {
        atMax = impl::aten::buildAtScalar(max);
    }
    at::clamp_(atInput, atMin, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min,
                              const diopiScalar_t* max) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    c10::optional<at::Scalar> atMin = c10::optional<at::Scalar>();
    if (min != nullptr) {
        atMin = impl::aten::buildAtScalar(min);
    }
    c10::optional<at::Scalar> atMax = c10::optional<at::Scalar>();
    if (max != nullptr) {
        atMax = impl::aten::buildAtScalar(max);
    }
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_out(atOut, atInput, atMin, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::clamp_max_(atInput, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_max_out(atOut, atInput, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

#if TORCH_MM_VERSION > TORCH_1_9_MM_VERSION
diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    c10::optional<at::Tensor> atMin = c10::optional<at::Tensor>();
    if (min != nullptr) {
        atMin = impl::aten::buildATen(min);
    }
    c10::optional<at::Tensor> atMax = c10::optional<at::Tensor>();
    if (max != nullptr) {
        atMax = impl::aten::buildATen(max);
    }
    at::clamp_(atInput, atMin, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    c10::optional<at::Tensor> atMin = c10::optional<at::Tensor>();
    if (min != nullptr) {
        atMin = impl::aten::buildATen(min);
    }
    c10::optional<at::Tensor> atMax = c10::optional<at::Tensor>();
    if (max != nullptr) {
        atMax = impl::aten::buildATen(max);
    }
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_out(atOut, atInput, atMin, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMax = impl::aten::buildATen(max);
    at::clamp_max_(atInput, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMax = impl::aten::buildATen(max);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_max_out(atOut, atInput, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMin = impl::aten::buildATen(min);
    at::clamp_(atInput, atMin);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMin = impl::aten::buildATen(min);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_out(atOut, atInput, atMin);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}
#endif

diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::clamp_(atInput, atMin);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_out(atOut, atInput, atMin);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atValue = impl::aten::buildAtScalar(value);
    at::fill_(atInput, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::adaptive_avg_pool2d_out(atOut, atInput, atOutSize);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    auto atOuts = at::adaptive_max_pool2d(atInput, atOutSize);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                               diopiSize_t output_size) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    at::adaptive_max_pool2d_out(atOut, atIndices, atInput, atOutSize);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atGradOutput = impl::aten::buildATen(grad_output);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    at::Tensor atGradInput = impl::aten::buildATen(grad_input);
    at::adaptive_max_pool2d_backward_out(atGradInput, atGradOutput, atInput, atIndices);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                            diopiSize_t padding, bool ceil_mode, bool count_include_pad, const int64_t* divisor_override) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    c10::optional<int64_t> atDivisorOverride = divisor_override ? c10::optional<int64_t>(*divisor_override) : c10::nullopt;
    at::Tensor atOut = impl::aten::buildATen(out);
    at::avg_pool2d_out(atOut, atInput, atKernelSize, atStride, atPadding, ceil_mode, count_include_pad, atDivisorOverride);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train,
                          diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    if (train) {
        at::Generator gen = impl::aten::buildGenerator(ctx, generator);
        at::Tensor atOut = impl::aten::buildATen(out);
        at::Tensor atMask = impl::aten::buildATen(mask);
        if (atInput.numel() == atMask.numel()) {
            at::_fused_dropout_out(atOut, atMask, atInput, 1 - p, gen);
        } else {
            atMask.bernoulli_(1 - p, gen);
            at::mul_out(atOut, atInput, atMask);
            atOut.div_(1 - p);
        }
        impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    } else {
        impl::aten::updateATen2Tensor(ctx, atInput, out);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train,
                             diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    if (train) {
        at::Generator gen = impl::aten::buildGenerator(ctx, generator);
        at::Tensor atInput = impl::aten::buildATen(input);
        at::Tensor atMask = impl::aten::buildATen(mask);
        if (atInput.numel() == atMask.numel()) {
            at::_fused_dropout_out(atInput, atMask, atInput, 1 - p, gen);
        } else {
            atMask.bernoulli_(1 - p, gen);
            atInput.mul_(atMask).div_(1 - p);
        }
        impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiReduction_t reduction) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    // Note(huqingqing): at::mse_loss_out reduce in the 0 dimension, which is different from at::mse_loss.
    // at::mse_loss reduce over all the dimensions.
    if (reduction == 0) {
        at::Tensor atOut = impl::aten::buildATen(out);
        at::mse_loss_out(atOut, atInput, atTarget, reduction);
    } else {
        impl::aten::invokeATenFuncRet(ctx, at::mse_loss, out, atInput, atTarget, reduction);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs, diopiConstTensorHandle_t targets,
                                   float alpha, float gamma, diopiReduction_t reduction) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(inputs);
    at::Tensor atTarget = impl::aten::buildATen(targets);
    at::Tensor atP = at::sigmoid(atInput);
    at::Tensor atTerm1 = at::pow(1 - atP, gamma) * at::log(atP);
    at::Tensor atTerm2 = at::pow(atP, gamma) * at::log(1 - atP);
    at::Tensor atRes = -atTarget * atTerm1 * alpha - (1 - atTarget) * atTerm2 * (1 - alpha);
    if (reduction == 0) {
        impl::aten::updateATen2Tensor(ctx, atRes, out);
    } else if (reduction == 1) {
        at::Tensor atOut = at::mean(atRes);
        impl::aten::updateATen2Tensor(ctx, atOut, out);
    } else if (reduction == 2) {
        at::Tensor atOut = at::sum(atRes);
        impl::aten::updateATen2Tensor(ctx, atOut, out);
    } else {
        NOT_SUPPORTED("sigmoid reduction type");
        return diopiErrorOccurred;
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean,
                            diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atWeight = impl::aten::buildATen(weight);
    at::Tensor atBias = impl::aten::buildATen(bias);
    at::Tensor atRunningMean = impl::aten::buildATen(running_mean);
    at::Tensor atRunningVar = impl::aten::buildATen(running_var);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atSaveMean = impl::aten::buildATen(save_mean);
    at::Tensor atSaveInvstd = impl::aten::buildATen(save_invstd);
    at::native_batch_norm_out(atOut, atSaveMean, atSaveInvstd, atInput, atWeight, atBias, atRunningMean, atRunningVar, training, momentum, eps);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end,
                        int64_t step) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = at::slice(atInput, dim, start, end, step).contiguous();
    impl::aten::updateATen2Tensor(ctx, atOut, null_out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t* indices, int64_t nums) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK(out != nullptr && indices != nullptr, "Not supported: out or indices is nullptr");
    at::Tensor atInput = impl::aten::buildATen(input);
    c10::List<c10::optional<at::Tensor>> vecIdx;
    vecIdx.reserve(nums);
    for (size_t i = 0; i < nums; ++i) {
        if (indices[i] == nullptr) {
            vecIdx.emplace_back(c10::nullopt);
        } else {
            at::Tensor atIndex = impl::aten::buildATen(indices[i]);
            vecIdx.emplace_back(atIndex);
        }
    }
    at::Tensor atOut = at::index(atInput, vecIdx).contiguous();
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    c10::optional<at::Tensor> atWeight = weight ? c10::optional<at::Tensor>(impl::aten::buildATen(weight)) : c10::nullopt;
    c10::optional<at::Tensor> atPosWeight = pos_weight ? c10::optional<at::Tensor>(impl::aten::buildATen(pos_weight)) : c10::nullopt;

    impl::aten::invokeATenFuncInp(ctx, at::binary_cross_entropy_with_logits_out, atOut, atInput, atTarget, atWeight, atPosWeight, reduction);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min_val,
                           const diopiScalar_t* max_val) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMin = impl::aten::buildAtScalar(min_val);
    auto atMax = impl::aten::buildAtScalar(max_val);
    auto atOut = impl::aten::buildATen(out);
    at::hardtanh_out(atOut, atInput, atMin, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMin = impl::aten::buildAtScalar(min_val);
    auto atMax = impl::aten::buildAtScalar(max_val);
    impl::aten::invokeATenFuncInp(ctx, at::hardtanh_, atInput, atMin, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiHardswish(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::hardswish_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiHardswishInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::hardswish_, atInput);
    return diopiSuccess;
}

diopiError_t diopiHardswishBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::hardswish_backward_out, atGradInput, atGradOutput, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* threshold,
                            const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atThreshold = impl::aten::buildAtScalar(threshold);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = impl::aten::buildATen(out);
    at::threshold_out(atOut, atInput, atThreshold, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold, const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atThreshold = impl::aten::buildAtScalar(threshold);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncInp(ctx, at::threshold_, atInput, atThreshold, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const char* approximate) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    c10::string_view atApproximate(approximate, strlen(approximate));
    impl::aten::invokeATenFuncInp(ctx, at::gelu_out, atOut, atInput, atApproximate);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atTarget = impl::aten::buildATen(target);
    auto atWeight = impl::aten::buildATen(weight);
    auto dim = atInput.dim();
    DIOPI_CHECK(dim > 1, "dim <=1 is not supported");
    if (dim != 2 && dim != 4) {
        auto n = atInput.size(0);
        auto c = atInput.size(1);
        int64_t inputLastSize = 1;
        int64_t targetLastSize = 1;
        for (int i = 2; i < atInput.dim(); ++i) {
            inputLastSize *= atInput.size(i);
        }
        for (int i = 1; i < atTarget.dim(); ++i) {
            targetLastSize *= atTarget.size(i);
        }
        std::vector<int64_t> inputShape = {n, c, 1, inputLastSize};
        std::vector<int64_t> targetShape = {n, 1, targetLastSize};
        atInput = atInput.reshape(inputShape);
        atTarget = atTarget.reshape(targetShape);
    }

    if (dim == 2) {
        at::nll_loss_out(atOut, atInput, atTarget, atWeight, reduction, ignore_index);
    } else {
        at::nll_loss2d_out(atOut, atInput, atTarget, atWeight, reduction, ignore_index);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes,
                                int64_t dim, int64_t start, int64_t end, int64_t step) {
    impl::aten::setCurCtx(ctx);
    at::IntArrayRef atInputSizes = impl::aten::buildAtIntArray(input_sizes);
    at::Tensor atGradOutput = impl::aten::buildATen(grad_output);
    at::Tensor atGradInput = impl::aten::buildATen(grad_input);
    impl::aten::invokeATenFuncInp(ctx, at::slice_backward_out, atGradInput, atGradOutput, atInputSizes, dim, start, end, step);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input,
                                diopiConstTensorHandle_t* indices, int64_t nums, diopiConstTensorHandle_t grad) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(indices);
    at::Tensor atZerosInput = impl::aten::buildATen(zeros_like_input);
    at::Tensor atGrad = impl::aten::buildATen(grad);
    at::Tensor atGradInput = impl::aten::buildATen(grad_input);
    c10::List<c10::optional<at::Tensor>> vecIdx;
    vecIdx.reserve(nums);
    for (size_t i = 0; i < nums; ++i) {
        if (indices[i] == nullptr) {
            vecIdx.emplace_back(c10::nullopt);
        } else {
            at::Tensor atIndex = impl::aten::buildATen(indices[i]);
            vecIdx.emplace_back(atIndex);
        }
    }
    impl::aten::invokeATenFuncInp(ctx, at::_index_put_impl_out, atGradInput, atZerosInput, vecIdx, atGrad, true, true);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSigmoidFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                           diopiConstTensorHandle_t target, diopiTensorHandle_t grad_input, float gamma, float alpha,
                                           diopiReduction_t reduction) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    at::Tensor atGrad = impl::aten::buildATen(grad_output);
    at::Tensor atGradOutput = at::empty_like(atInput);
    if (reduction == 1) {
        atGradOutput.copy_(atGrad.expand_as(atInput) / atInput.numel());
    } else if (reduction == 2) {
        atGradOutput.copy_(atGrad.expand_as(atInput));
    } else if (reduction == 0) {
        atGradOutput.copy_(atGrad);
    } else {
        NOT_SUPPORTED("sigmoid reduction type");
        return diopiErrorOccurred;
    }

    at::Tensor atP = at::sigmoid(atInput);
    // (1-p)**g * (1 - p - g*p*log(p))
    at::Tensor atTerm1 = at::pow(1 - atP, gamma) * (1 - atP - gamma * atP * at::log(at::clamp_min(atP, FLT_MIN)));
    // (p**g) * (g*(1-p)*log(1-p) - p)
    at::Tensor atTerm2 = at::pow(atP, gamma) * (gamma * (1 - atP) * at::log(at::clamp_min(1 - atP, FLT_MIN)) - atP);
    at::Tensor atRes = -atTarget * atTerm1 * alpha - (1 - atTarget) * atTerm2 * (1 - alpha);
    atGradOutput *= atRes;
    impl::aten::updateATen2Tensor(ctx, atGradOutput, grad_input);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t rois,
                                   double spatialScale, int64_t pooledHeight, int64_t pooledWidth, int64_t batchSize, int64_t channels, int64_t height,
                                   int64_t width, int64_t samplingRatio, bool aligned) {
    impl::aten::setCurCtx(ctx);
    auto atGrad = impl::aten::buildATen(grad);
    auto atRois = impl::aten::buildATen(rois);
    auto atOut = vision::ops::roi_align_backward_kernel(
        atGrad, atRois, spatialScale, pooledHeight, pooledWidth, batchSize, channels, height, width, samplingRatio, aligned);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                        diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                        diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                        int64_t groups) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGradWeight = impl::aten::buildATen(grad_weight);
    auto atGradBias = impl::aten::buildATen(grad_bias);
    c10::OptionalIntArrayRef bias_sizes_opt = bias_sizes ? c10::make_optional<c10::ArrayRef<int64_t>>(impl::aten::buildAtIntArray(*bias_sizes)) : c10::nullopt;

    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad_output);
    auto atWeight = impl::aten::buildATen(weight);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    auto atDilation = impl::aten::buildAtIntArray(dilation);
    std::vector<int64_t> outputPadding(padding.len, 0);
    at::IntArrayRef atOutputPadding(outputPadding.data(), outputPadding.size());
    diopi_tensor_list vecOut = {grad_input, grad_weight};
#ifdef USE_HIP
    auto grad_input_mask = std::array<bool, 3>{true, true, false};
    impl::aten::invokeATenFuncRet(
        ctx, at::miopen_convolution_backward, vecOut, atInput, atGrad, atWeight, atPadding, atStride, atDilation, groups, false, false, grad_input_mask);
    if (bias_sizes != nullptr && grad_bias != nullptr) {
        auto atBias = impl::aten::buildATen(grad_bias);
        at::Tensor atTmp = atGrad;
        int64_t size = atGrad.dim() - 1;
        while (atBias.dim() != size) {
            atTmp = at::sum(atTmp, -1, false);
            size -= 1;
        }
        atTmp = at::sum(atTmp, 0, false);
        impl::aten::updateATen2Tensor(ctx, atTmp, grad_bias);
    }
#else
    {
        if (grad_input || grad_weight) {
            auto grad_inputs = at::convolution_backward(
                atGrad, atInput, atWeight, c10::nullopt, atStride, atPadding, atDilation, false, atOutputPadding, groups, {true, true, false});
            if (grad_input) {
                impl::aten::updateATen2Tensor(ctx, std::get<0>(grad_inputs), grad_input);
            }
            if (grad_weight) {
                impl::aten::updateATen2Tensor(ctx, std::get<1>(grad_inputs), grad_weight);
            }
        }

        if (bias_sizes != nullptr && grad_bias != nullptr) {
            auto atBias = impl::aten::buildATen(grad_bias);
            at::Tensor atTmp = atGrad;
            int64_t size = atGrad.dim() - 1;
            while (atBias.dim() != size) {
                atTmp = at::sum(atTmp, -1, false);
                size -= 1;
            }
            atTmp = at::sum(atTmp, 0, false);
            impl::aten::updateATen2Tensor(ctx, atTmp, grad_bias);
        }
    }

#endif
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiConvTranspose2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                          diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding,
                                          diopiSize_t dilation, diopiSize_t output_padding, int64_t groups) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad_output);
    auto atWeight = impl::aten::buildATen(weight);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    auto atOutputPadding = impl::aten::buildAtIntArray(output_padding);
    auto atDilation = impl::aten::buildAtIntArray(dilation);
#ifdef USE_HIP
    impl::aten::invokeATenFuncRet(ctx,
                                  at::miopen_convolution_transpose_backward,
                                  vecOut,
                                  atInput,
                                  atGrad,
                                  atWeight,
                                  atPadding,
                                  atOutputPadding,
                                  atStride,
                                  atDilation,
                                  groups,
                                  false,
                                  false,
                                  grad_input_mask);
#else
    auto grad_inputs =
        at::convolution_backward(atGrad, atInput, atWeight, c10::nullopt, atStride, atPadding, atDilation, true, atOutputPadding, groups, {true, true, false});
    impl::aten::updateATen2Tensor(ctx, std::get<0>(grad_inputs), grad_input);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(grad_inputs), grad_weight);
#endif
    if (bias_sizes != nullptr && grad_bias != nullptr) {
        auto atBias = impl::aten::buildATen(grad_bias);
        at::Tensor atTmp = atGrad;
        int64_t size = atGrad.dim() - 1;
        while (atBias.dim() != size) {
            atTmp = at::sum(atTmp, -1, false);
            size -= 1;
        }
        atTmp = at::sum(atTmp, 0, false);
        impl::aten::updateATen2Tensor(ctx, atTmp, grad_bias);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices,
                                    int64_t numWeights, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    impl::aten::setCurCtx(ctx);
    auto atGrad = impl::aten::buildATen(grad);
    auto atIndices = impl::aten::buildATen(indices);
    impl::aten::invokeATenFuncRet(ctx, at::embedding_backward, out, atGrad, atIndices, numWeights, paddingIdx, scaleGradByFreq, sparse);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::_adaptive_avg_pool2d_backward_out, atGradInput, atGradOutput, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atSlope = impl::aten::buildAtScalar(negative_slope);
    impl::aten::invokeATenFuncInp(ctx, at::leaky_relu_backward_out, atGradInput, atGradOutput, atInput, atSlope, input_is_result);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                   diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atMin = impl::aten::buildAtScalar(min_val);
    auto atMax = impl::aten::buildAtScalar(max_val);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::hardtanh_backward_out(atGradInput, atGradOutput, atInput, atMin, atMax);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                               const char* approximate) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    c10::string_view atApproximate(approximate, strlen(approximate));
    impl::aten::invokeATenFuncRet(ctx, at::gelu_backward, grad_input, atGradOutput, atInput, atApproximate);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                    bool count_include_pad, const int64_t* divisor_override) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    c10::optional<int64_t> atDivisorOverride = divisor_override ? c10::optional<int64_t>(*divisor_override) : c10::nullopt;
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::avg_pool2d_backward_out(atGradInput, atGradOutput, atInput, atKernelSize, atStride, atPadding, ceil_mode, count_include_pad, atDivisorOverride);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::mse_loss_backward_out(atGradInput, atGradOutput, atInput, atTarget, reduction);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::tanh_backward_out(atGradInput, atGradOutput, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad, diopiSize_t input_sizes,
                                      int64_t dim, diopiConstTensorHandle_t index) {
    impl::aten::setCurCtx(ctx);
    auto atGrad = impl::aten::buildATen(grad);
    at::IntArrayRef atInputSize = impl::aten::buildAtIntArray(input_sizes);
    auto atIndex = impl::aten::buildATen(index);
    impl::aten::invokeATenFuncRet(ctx, at::index_select_backward, grad_input, atGrad, atInputSize, dim, atIndex);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes,
                                 int64_t dim, int64_t index) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    at::IntArrayRef atInputSize = impl::aten::buildAtIntArray(input_sizes);
    impl::aten::invokeATenFuncRet(ctx, at::select_backward, grad_input, atGradOutput, atInputSize, dim, index);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t output, int64_t dim) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atOutput = impl::aten::buildATen(output);
    // TODO(huqingqing): use default type instead
    impl::aten::invokeATenFuncInp(ctx, at::_softmax_backward_data_out, atGradInput, atGradOutput, atOutput, dim, atOutput.scalar_type());
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                     diopiConstTensorHandle_t output, int64_t dim) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atOutput = impl::aten::buildATen(output);
    // TODO(huqingqing): use default type instead
    impl::aten::invokeATenFuncInp(ctx, at::_log_softmax_backward_data_out, atGradInput, atGradOutput, atOutput, dim, atOutput.scalar_type());
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t output) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atOutput = impl::aten::buildATen(output);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::sigmoid_backward_out(atGradInput, atGradOutput, atOutput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atThreshold = impl::aten::buildAtScalar(threshold);
    impl::aten::invokeATenFuncInp(ctx, at::threshold_backward_out, atGradInput, atGradOutput, atInput, atThreshold);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                        diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                        diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);

    at::Tensor atGradInput;
    if (pos_weight) {
        auto atPosWeight = impl::aten::buildATen(pos_weight);
        // pos_weight need to be broadcasted, thus mul(target) is not inplace.
        auto atT = atPosWeight.mul(atTarget);
        atGradInput = atT.add(1).sub_(atTarget).mul_(atInput.sigmoid()).sub_(atT).mul_(atGradOutput);
    } else {
        atGradInput = (atInput.sigmoid() - atTarget).mul_(atGradOutput);
    }

    if (weight) {
        auto atWeight = impl::aten::buildATen(weight);
        atGradInput.mul_(atWeight);
    }

    if (reduction == diopiReduction_t::ReductionMean) {
        atGradInput = atGradInput / atInput.numel();
    }
    impl::aten::updateATen2Tensor(ctx, atGradInput, grad_input);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction,
                                  int64_t ignore_index) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    auto atGradInput = impl::aten::nllLossNdBackward(atInput, atGradOutput, atTarget, weight, reduction, ignore_index);
    impl::aten::updateATen2Tensor(ctx, atGradInput, grad_input);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                    bool ceil_mode, diopiConstTensorHandle_t indices) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    auto atIndices = impl::aten::buildATen(indices);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::max_pool2d_with_indices_backward_out(atGradInput, atGradOutput, atInput, atKernelSize, atStride, atPadding, atDilation, ceil_mode, atIndices);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                    diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean,
                                    diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    impl::aten::setCurCtx(ctx);

    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    c10::optional<at::Tensor> atRunningMean = running_mean ? c10::optional<at::Tensor>(impl::aten::buildATen(running_mean)) : c10::nullopt;
    c10::optional<at::Tensor> atRunningVar = running_var ? c10::optional<at::Tensor>(impl::aten::buildATen(running_var)) : c10::nullopt;
    c10::optional<at::Tensor> atSaveMean = save_mean ? c10::optional<at::Tensor>(impl::aten::buildATen(save_mean)) : c10::nullopt;
    c10::optional<at::Tensor> atSaveVar = save_invstd ? c10::optional<at::Tensor>(impl::aten::buildATen(save_invstd)) : c10::nullopt;

    if (grad_input && grad_weight && grad_bias) {
        auto grad_input_mask = std::array<bool, 3>{true, true, true};
        auto atGradInput = impl::aten::buildATen(grad_input);
        auto atGradWeight = impl::aten::buildATen(grad_weight);
        auto atGradBias = impl::aten::buildATen(grad_bias);
        at::native_batch_norm_backward_out(atGradInput,
                                           atGradWeight,
                                           atGradBias,
                                           atGradOutput,
                                           atInput,
                                           atWeight,
                                           atRunningMean,
                                           atRunningVar,
                                           atSaveMean,
                                           atSaveVar,
                                           training,
                                           eps,
                                           grad_input_mask);
    } else {
        auto grad_input_mask = std::array<bool, 3>{grad_input != nullptr, grad_weight != nullptr, grad_bias != nullptr};
        auto atOut =
            at::native_batch_norm_backward(atGradOutput, atInput, atWeight, atRunningMean, atRunningVar, atSaveMean, atSaveVar, training, eps, grad_input_mask);
        if (grad_input) {
            impl::aten::updateATen2Tensor(ctx, std::get<0>(atOut), grad_input);
        }
        if (grad_weight) {
            impl::aten::updateATen2Tensor(ctx, std::get<1>(atOut), grad_weight);
        }
        if (grad_bias) {
            impl::aten::updateATen2Tensor(ctx, std::get<2>(atOut), grad_bias);
        }
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, const diopiScalar_t* step) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atStart = impl::aten::buildAtScalar(start);
    auto atEnd = impl::aten::buildAtScalar(end);
    auto atStep = impl::aten::buildAtScalar(step);
    at::arange_out(atOut, atStart, atEnd, atStep);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::randperm_out(atOut, n, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atInOut = impl::aten::buildATen(inout);
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::native::uniform_(atInOut, from, to, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atInOut = impl::aten::buildATen(inout);
    c10::optional<int64_t> atTo = to ? c10::optional<int64_t>(*to) : c10::nullopt;
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::native::random_(atInOut, from, atTo, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atInOut = impl::aten::buildATen(inout);
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::bernoulli_out(atInOut, atInOut, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::bernoulli_out(atOut, atInput, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::native::bernoulli_(atOut, p, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atSize = atOut.sizes();
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::normal_out(atOut, mean, std, atSize, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atInOut = impl::aten::buildATen(inout);
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::native::normal_(atInOut, mean, std, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    return diopiSuccess;
}

diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std,
                                     diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atMean = impl::aten::buildATen(mean);
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::normal_out(atOut, atMean, std, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std,
                                     diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atStd = impl::aten::buildATen(std);
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::normal_out(atOut, mean, atStd, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std,
                               diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atMean = impl::aten::buildATen(mean);
    auto atStd = impl::aten::buildATen(std);
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::normal_out(atOut, atMean, atStd, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                             diopiConstTensorHandle_t value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildATen(value);
    auto atOut = at::masked_fill(atInput, atMask, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildATen(value);
    atInput.masked_fill_(atMask, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                   const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = at::masked_fill(atInput, atMask, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildAtScalar(value);
    atInput.masked_fill_(atMask, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMeshGrid(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, diopiConstTensorHandle_t* inputs, int64_t inputsNum) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(outs);
    DIOPI_CHECK_PTR(inputs);
    auto outsNum = inputsNum;
    auto atInputs = impl::aten::buildATenList(inputs, inputsNum);
    auto atOuts = impl::aten::buildATenList(outs, outsNum);
    atOuts = at::meshgrid(atInputs);
    for (int i = 0; i < outsNum; ++i) {
        impl::aten::updateATen2Tensor(ctx, atOuts[i].contiguous(), outs[i]);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                        diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay,
                        int64_t step, bool amsgrad) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad);
    auto atExpAvg = impl::aten::buildATen(exp_avg);
    auto atExpAvgSq = impl::aten::buildATen(exp_avg_sq);
    auto atMaxExpAvgSq = impl::aten::buildATen(max_exp_avg_sq);

    atInput = atInput.mul(1 - lr * weight_decay);
    auto& param = atInput;
    auto grad_d = atGrad.data();
    auto bias_correction1 = 1 - pow(beta1, step);
    auto bias_correction2 = 1 - pow(beta2, step);
    atExpAvg.mul_(beta1).add_(grad_d, 1 - beta1);
    atExpAvgSq.mul_(beta2).addcmul_(grad_d, grad_d, 1 - beta2);

    at::Tensor denom;
    if (amsgrad) {
        at::maximum_out(atMaxExpAvgSq, atMaxExpAvgSq, atExpAvgSq);
        denom = (atMaxExpAvgSq.sqrt() / sqrt(bias_correction2)).add_(eps);
    } else {
        denom = (atExpAvgSq.sqrt() / sqrt(bias_correction2)).add_(eps);
    }
    auto stepSize = lr / bias_correction1;
    param = param.addcdiv(atExpAvg, denom, -1 * stepSize);

    impl::aten::updateATen2Tensor(ctx, atInput, input);
    impl::aten::updateATen2Tensor(ctx, atGrad, grad);
    impl::aten::updateATen2Tensor(ctx, atExpAvg, exp_avg);
    impl::aten::updateATen2Tensor(ctx, atExpAvgSq, exp_avg_sq);
    impl::aten::updateATen2Tensor(ctx, atMaxExpAvgSq, max_exp_avg_sq);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t exp_avg,
                       diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, float lr, float beta1, float beta2, float eps, float weight_decay,
                       int64_t step, bool amsgrad) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad);
    auto atExpAvg = impl::aten::buildATen(exp_avg);
    auto atExpAvgSq = impl::aten::buildATen(exp_avg_sq);
    auto atMaxExpAvgSq = impl::aten::buildATen(max_exp_avg_sq);

    auto& param = atInput;
    auto grad_d = atGrad.data();
    auto bias_correction1 = 1 - pow(beta1, step);
    auto bias_correction2 = 1 - pow(beta2, step);

    if (weight_decay != 0) {
        grad_d = grad_d.add(param, weight_decay);
    }
    atExpAvg.mul_(beta1).add_(grad_d, 1 - beta1);
    atExpAvgSq.mul_(beta2).addcmul_(grad_d, grad_d.conj(), 1 - beta2);

    at::Tensor denom;
    if (amsgrad) {
        at::maximum_out(atMaxExpAvgSq, atMaxExpAvgSq, atExpAvgSq);
        denom = (atMaxExpAvgSq.sqrt() / sqrt(bias_correction2)).add_(eps);
    } else {
        denom = (atExpAvgSq.sqrt() / sqrt(bias_correction2)).add_(eps);
    }
    auto stepSize = lr / bias_correction1;
    param = param.addcdiv(atExpAvg, denom, -1 * stepSize);

    impl::aten::updateATen2Tensor(ctx, atInput, input);
    impl::aten::updateATen2Tensor(ctx, atGrad, grad);
    impl::aten::updateATen2Tensor(ctx, atExpAvg, exp_avg);
    impl::aten::updateATen2Tensor(ctx, atExpAvgSq, exp_avg_sq);
    impl::aten::updateATen2Tensor(ctx, atMaxExpAvgSq, max_exp_avg_sq);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
                           diopiTensorHandle_t acc_delta, float lr, float rho, float eps, float weight_decay) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad);
    auto atSquareAvg = impl::aten::buildATen(square_avg);
    auto atAccDelta = impl::aten::buildATen(acc_delta);

    auto& param = atInput;
    auto grad_d = atGrad.data();
    if (weight_decay != 0) {
        grad_d = grad_d.add(param, weight_decay);
    }
    atSquareAvg.mul_(rho).addcmul_(grad_d, grad_d, 1 - rho);
    auto std = atSquareAvg.add(eps).sqrt_();
    auto delta = atAccDelta.add(eps).sqrt_().div_(std).mul_(grad_d);
    param.add_(delta, -lr);
    atAccDelta.mul_(rho).addcmul_(delta, delta, 1 - rho);
    impl::aten::updateATen2Tensor(ctx, atInput, input);
    impl::aten::updateATen2Tensor(ctx, atGrad, grad);
    impl::aten::updateATen2Tensor(ctx, atSquareAvg, square_avg);
    impl::aten::updateATen2Tensor(ctx, atAccDelta, acc_delta);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRmsprop(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, diopiTensorHandle_t square_avg,
                          diopiTensorHandle_t grad_avg, diopiTensorHandle_t momentum_buf, float lr, float alpha, float eps, float weight_decay, float momentum,
                          bool centered) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad);
    auto atSquareAvg = impl::aten::buildATen(square_avg);
    auto atGradAvg = impl::aten::buildATen(grad_avg);
    auto atBuf = impl::aten::buildATen(momentum_buf);

    if (weight_decay != 0) {
        atGrad = atGrad.add(atInput, weight_decay);
    }
    atSquareAvg.mul_(alpha).addcmul_(atGrad, atGrad, 1 - alpha);
    at::Tensor atAvg;

    if (centered) {
        atGradAvg.mul_(alpha).add_(atGrad, 1 - alpha);
        atAvg = atSquareAvg.addcmul(atGradAvg, atGradAvg, -1).sqrt_().add_(eps);
    } else {
        atAvg = atSquareAvg.sqrt().add_(eps);
    }

    if (momentum > 0) {
        atBuf.mul_(momentum).addcdiv_(atGrad, atAvg);
        atInput.add_(atBuf, -lr);
    } else {
        atInput.addcdiv_(atGrad, atAvg, -lr);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                  diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t output_padding, int64_t groups,
                                  diopiSize_t dilation) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    auto atOutputPadding = impl::aten::buildAtIntArray(output_padding);
    auto atDilation = impl::aten::buildAtIntArray(dilation);
    impl::aten::invokeATenFuncRet(ctx, at::conv_transpose2d, out, atInput, atWeight, atBias, atStride, atPadding, atOutputPadding, groups, atDilation);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::cumsum_out(atOut, atInput, dim);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p,
                        const int64_t* compute_mode) {
    impl::aten::setCurCtx(ctx);
    auto atInput1 = impl::aten::buildATen(input1);
    auto atInput2 = impl::aten::buildATen(input2);
    c10::optional<int64_t> atComputMode = compute_mode ? c10::optional<int64_t>(*compute_mode) : c10::nullopt;
    impl::aten::invokeATenFuncRet(ctx, at::cdist, out, atInput1, atInput2, p, atComputMode);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input1,
                                diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput1 = impl::aten::buildATen(input1);
    auto atInput2 = impl::aten::buildATen(input2);
    auto atCdist = impl::aten::buildATen(cdist);
    impl::aten::invokeATenFuncInp(ctx, at::_cdist_backward_out, atGradInput, atGradOutput, atInput1, atInput2, p, atCdist);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::reciprocal_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    at::reciprocal_(atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::bitwise_not_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBitwiseNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    atInput.bitwise_not_();
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLogicalNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::logical_not_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLogicalNotInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    atInput.logical_not_();
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    c10::optional<int64_t> atDim = dim ? c10::optional<int64_t>(*dim) : c10::nullopt;
    impl::aten::invokeATenFuncInp(ctx, at::argmax_out, atOut, atInput, atDim, keepdim);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                               diopiReduction_t reduction, double beta) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    if (reduction == 0) {
        at::Tensor atOut = impl::aten::buildATen(out);
        at::smooth_l1_loss_out(atOut, atInput, atTarget, reduction, beta);
    } else {
        impl::aten::invokeATenFuncRet(ctx, at::smooth_l1_loss, out, atInput, atTarget, reduction, beta);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                       diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::smooth_l1_loss_backward_out(atGradInput, atGradOutput, atInput, atTarget, reduction, beta);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildATen(other);
    auto atOut = impl::aten::buildATen(out);
    at::maximum_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildATen(other);
    auto atOut = impl::aten::buildATen(out);
    at::minimum_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMat2 = impl::aten::buildATen(mat2);
    auto atOut = impl::aten::buildATen(out);
    at::mm_out(atOut, atInput, atMat2);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiConvolution3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    auto atDilation = impl::aten::buildAtIntArray(dilation);
    impl::aten::invokeATenFuncInp(
        ctx, at::convolution_out, atOut, atInput, atWeight, atBias, atStride, atPadding, atDilation, false, at::IntArrayRef(0), groups);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiConvolution3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                        diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                        diopiConstTensorHandle_t weight, diopiSize_t* bias_sizes, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                        int64_t groups) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad_output);
    auto atWeight = impl::aten::buildATen(weight);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    std::vector<int64_t> outputPadding(padding.len, 0);
    at::IntArrayRef atOutputPadding(outputPadding.data(), outputPadding.size());
    auto atDilation = impl::aten::buildAtIntArray(dilation);
    diopi_tensor_list vecOut = {grad_input, grad_weight};
#ifdef USE_HIP
    auto grad_input_mask = std::array<bool, 3>{true, true, false};
    impl::aten::invokeATenFuncRet(
        ctx, at::miopen_convolution_backward, vecOut, atInput, atGrad, atWeight, atPadding, atStride, atDilation, groups, false, false, grad_input_mask);
#else
    auto grad_inputs =
        at::convolution_backward(atGrad, atInput, atWeight, c10::nullopt, atStride, atPadding, atDilation, false, atOutputPadding, groups, {true, true, false});
    impl::aten::updateATen2Tensor(ctx, std::get<0>(grad_inputs), grad_input);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(grad_inputs), grad_weight);
#endif

    if (bias_sizes != nullptr && grad_bias != nullptr) {
        auto atBias = impl::aten::buildATen(grad_bias);
        at::Tensor atTmp = atGrad;
        int64_t size = atGrad.dim() - 1;
        while (atBias.dim() != size) {
            atTmp = at::sum(atTmp, -1, false);
            size -= 1;
        }
        if (atBias.size(0) != atTmp.size(0)) {
            atTmp = at::sum(atTmp, 0, false);
        }
        impl::aten::updateATen2Tensor(ctx, atTmp, grad_bias);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    diopiSize_t size;
    diopiGetTensorShape(out, &size);
    auto atSize = impl::aten::buildAtIntArray(size);
    auto atOut = at::native::expand(atInput, atSize).clone();
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    // must use contiguous rather than clone in this case
    auto atOut = at::native::unfold(atInput, dim, size, step).contiguous();
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes,
                                 int64_t dim, int64_t size, int64_t step) {
    impl::aten::setCurCtx(ctx);
    auto atGradInput = impl::aten::buildATen(grad_input);
    auto atGrad = impl::aten::buildATen(grad_output);
    auto atInputSize = impl::aten::buildAtIntArray(input_sizes);
    impl::aten::invokeATenFuncInp(ctx, at::unfold_backward_out, atGradInput, atGrad, atInputSize, dim, size, step);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(out);
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atOut = at::masked_select(atInput, atMask);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                       diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    impl::aten::invokeATenFuncRet(ctx, at::masked_select_backward, grad_input, atGradOutput, atInput, atMask);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                                  diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = at::index_fill(atInput, dim, atIndex, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index,
                            diopiConstTensorHandle_t value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildATen(value);
    auto atOut = at::index_fill(atInput, dim, atIndex, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index,
                                     const diopiScalar_t* value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildAtScalar(value);
    atInput.index_fill_(dim, atIndex, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index,
                               diopiConstTensorHandle_t value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildATen(value);
    atInput.index_fill_(dim, atIndex, atValue);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    impl::aten::setCurCtx(ctx);
    auto atStart = impl::aten::buildAtScalar(start);
    auto atEnd = impl::aten::buildAtScalar(end);
    c10::optional<int64_t> atStep(steps);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::linspace_out(atOut, atStart, atEnd, steps);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    at::IntArrayRef atShifts = impl::aten::buildAtIntArray(shifts);
    at::IntArrayRef atDims = impl::aten::buildAtIntArray(dims);
    auto atOut = at::roll(atInput, atShifts, atDims);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atP = impl::aten::buildAtScalar(p);
    at::IntArrayRef atDim = impl::aten::buildAtIntArray(dim);
    bool keepdim = false;
    if (atInput.dim() == atOut.dim()) {
        keepdim = true;
    }
    at::norm_out(atOut, atInput, atP, atDim, keepdim);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups, double eps) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atWeight = impl::aten::buildATen(weight);
    at::Tensor atBias = impl::aten::buildATen(bias);
    const int64_t N = atInput.size(0);
    const int64_t C = atInput.size(1);
    const auto input_shape = atInput.sizes();
    const int64_t HxW = c10::multiply_integers(input_shape.cbegin() + 2, input_shape.cend());
    diopi_tensor_list vecOut = {out, save_mean, save_invstd};
    impl::aten::invokeATenFuncRet(ctx, at::native_group_norm, vecOut, atInput, atWeight, atBias, N, C, HxW, num_groups, eps);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                    diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, int64_t num_groups) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atSaveMean = impl::aten::buildATen(mean);
    auto atSaveVar = impl::aten::buildATen(rstd);
    const int64_t N = atInput.size(0);
    const int64_t C = atInput.size(1);
    const auto input_shape = atInput.sizes();
    const int64_t HxW = c10::multiply_integers(input_shape.cbegin() + 2, input_shape.cend());
    auto grad_input_mask = std::array<bool, 3>{true, grad_weight != nullptr, grad_bias != nullptr};
    auto atOut = at::native_group_norm_backward(atGradOutput, atInput, atSaveMean, atSaveVar, atWeight, N, C, HxW, num_groups, grad_input_mask);

    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOut), grad_input);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(atOut), grad_weight);
    impl::aten::updateATen2Tensor(ctx, std::get<2>(atOut), grad_bias);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiConstTensorHandle_t weight, diopiReduction_t reduction) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    c10::optional<at::Tensor> atWeight = weight ? c10::optional<at::Tensor>(impl::aten::buildATen(weight)) : c10::nullopt;
    if (reduction == 0) {
        at::binary_cross_entropy_out(atOut, atInput, atTarget, atWeight, reduction);
    } else {
        impl::aten::invokeATenFuncRet(ctx, at::binary_cross_entropy, out, atInput, atTarget, atWeight, reduction);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                  diopiReduction_t reduction) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    c10::optional<at::Tensor> atWeight = weight ? c10::optional<at::Tensor>(impl::aten::buildATen(weight)) : c10::nullopt;
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::binary_cross_entropy_backward_out(atGradInput, atGradOutput, atInput, atTarget, atWeight, reduction);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalized_shape,
                            double eps) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atSaveMean = impl::aten::buildATen(save_mean);
    at::Tensor atSaveInvstd = impl::aten::buildATen(save_invstd);

    at::Tensor atInput = impl::aten::buildATen(input);
    c10::optional<at::Tensor> atWeight = weight ? c10::optional<at::Tensor>(impl::aten::buildATen(weight)) : c10::nullopt;
    c10::optional<at::Tensor> atBias = bias ? c10::optional<at::Tensor>(impl::aten::buildATen(bias)) : c10::nullopt;
    auto atNormalizedShape = impl::aten::buildAtIntArray(normalized_shape);
    // TODO(zhaoguochun): check dtype: when input is half, atSaveInvstd, atInput should be float?
    //  at::native_layer_norm_out(atOut, atSaveMean, atSaveInvstd, atInput, atNormalizedShape, atWeight, atBias, eps);
    diopi_tensor_list vecOut = {out, save_mean, save_invstd};
    impl::aten::invokeATenFuncRet(ctx, at::native_layer_norm, vecOut, atInput, atNormalizedShape, atWeight, atBias, eps);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                    diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape) {
    impl::aten::setCurCtx(ctx);
    diopiDtype_t mDtype, rDtype;
    if (rstd) {
        diopiGetTensorDtype(rstd, &rDtype);
    }
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atNormalizedShape = impl::aten::buildAtIntArray(normalized_shape);
    c10::optional<at::Tensor> atWeight;
    c10::optional<at::Tensor> atBias;
    auto grad_input_mask = std::array<bool, 3>{true, false, false};
    if (weight != nullptr) {
        atWeight = c10::optional<at::Tensor>(impl::aten::buildATen(weight));
        grad_input_mask.at(1) = true;
    }
    if (bias != nullptr) {
        atBias = c10::optional<at::Tensor>(impl::aten::buildATen(bias));
        grad_input_mask.at(2) = true;
    }

    auto atSaveMean = impl::aten::buildATen(mean);
    diopiGetTensorDtype(mean, &mDtype);
    if (diopiDtype_t::diopi_dtype_float16 == mDtype) {
        atSaveMean = at::native::to(atSaveMean, impl::aten::getATenType(diopiDtype_t::diopi_dtype_float32).toScalarType(), false, true, c10::nullopt);
    }
    auto atSaveVar = impl::aten::buildATen(rstd);
    diopiGetTensorDtype(rstd, &rDtype);
    if (diopiDtype_t::diopi_dtype_float16 == rDtype) {
        atSaveVar = at::native::to(atSaveVar, impl::aten::getATenType(diopiDtype_t::diopi_dtype_float32).toScalarType(), false, true, c10::nullopt);
    }

    if (grad_input && grad_weight && grad_bias) {
        auto atGradInput = impl::aten::buildATen(grad_input);
        auto atGradWeight = impl::aten::buildATen(grad_weight);
        auto atGradBias = impl::aten::buildATen(grad_bias);
        at::native_layer_norm_backward_out(
            atGradInput, atGradWeight, atGradBias, atGradOutput, atInput, atNormalizedShape, atSaveMean, atSaveVar, atWeight, atBias, grad_input_mask);
    } else {
        auto atOut = at::native_layer_norm_backward(atGradOutput, atInput, atNormalizedShape, atSaveMean, atSaveVar, atWeight, atBias, grad_input_mask);
        if (grad_input) {
            impl::aten::updateATen2Tensor(ctx, std::get<0>(atOut), grad_input);
        }
        if (grad_weight) {
            impl::aten::updateATen2Tensor(ctx, std::get<1>(atOut), grad_weight);
        }
        if (grad_bias) {
            impl::aten::updateATen2Tensor(ctx, std::get<2>(atOut), grad_bias);
        }
    }

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    auto atOut = impl::aten::buildATen(out);
    at::adaptive_avg_pool3d_out(atOut, atInput, atOutSize);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::adaptive_avg_pool3d_backward_out(atGradInput, atGradOutput, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    auto atOuts = at::adaptive_max_pool3d(atInput, atOutSize);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                               diopiSize_t output_size) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    auto atOut = impl::aten::buildATen(out);
    auto atIndices = impl::aten::buildATen(indices);
    at::adaptive_max_pool3d_out(atOut, atIndices, atInput, atOutSize);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atGradOutput = impl::aten::buildATen(grad_output);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    at::Tensor atGradInput = impl::aten::buildATen(grad_input);
    at::adaptive_max_pool3d_backward_out(atGradInput, atGradOutput, atInput, atIndices);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
                            diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    bool atCeilMode = ceil_mode;
    impl::aten::invokeATenFuncRet(ctx, at::max_pool3d, out, atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t indices, diopiConstTensorHandle_t input,
                                       diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    bool atCeilMode = ceil_mode;
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    at::max_pool3d_with_indices_out(atOut, atIndices, atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation,
                                    bool ceil_mode, diopiConstTensorHandle_t indices) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    auto atIndices = impl::aten::buildATen(indices);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::max_pool3d_with_indices_backward_out(atGradInput, atGradOutput, atInput, atKernelSize, atStride, atPadding, atDilation, ceil_mode, atIndices);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atDims = impl::aten::buildAtIntArray(dims);
    impl::aten::invokeATenFuncRet(ctx, at::permute, out, atInput, atDims);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t dest) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atDest = impl::aten::buildATen(dest);
    at::Tensor atSrc = impl::aten::buildATen(src);
    // Set non_blocking true to avoid stream sync thus improving performance.
    // The data is not ready when diopiCopyInp returns.
    // If you need to use it immediately, please call cudaStreamSynchronize first.
    at::native::copy_(atDest, atSrc, true);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atOut = impl::aten::buildATen(out);
    at::gather_out(atOut, atInput, dim, atIndex);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                 int64_t dim, diopiConstTensorHandle_t index) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    bool sparse_grad = false;
    auto atOut = at::gather_backward(atGradOutput, atInput, dim, atIndex, sparse_grad);
    impl::aten::updateATen2Tensor(ctx, atOut, grad_input);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildATen(other);
    auto atOut = impl::aten::buildATen(out);
    at::remainder_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildAtScalar(other);
    auto atOut = impl::aten::buildATen(out);
    at::remainder_out(atOut, atInput, atOther);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t other) {
    impl::aten::setCurCtx(ctx);
    auto atInputScalar = impl::aten::buildAtScalar(input);
    auto atOther = impl::aten::buildATen(other);
    auto atOut = at::remainder(atInputScalar, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha,
                          diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths,
                          diopiConstTensorHandle_t target_lengths, int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
    impl::aten::setCurCtx(ctx);
    auto atLogProbs = impl::aten::buildATen(log_probs);
    auto atTarget = impl::aten::buildATen(targets);
    auto atInputLength = impl::aten::buildATen(input_lengths).to(at::Device(at::kCPU), at::kLong).contiguous();
    auto atTargetLength = impl::aten::buildATen(target_lengths).to(at::Device(at::kCPU), at::kLong).contiguous();
    at::IntArrayRef il(atInputLength.data_ptr<int64_t>(), atInputLength.numel());
    at::IntArrayRef tl(atTargetLength.data_ptr<int64_t>(), atTargetLength.numel());
    auto atOut = at::native::ctc_loss_gpu(atLogProbs, atTarget, il, tl, blank, zero_infinity);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOut), neg_log_likelihood);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(atOut), log_alpha);
    auto atRes = std::get<0>(atOut);
    if (zero_infinity) {
        atRes = at::where(atRes == at::Scalar(std::numeric_limits<double>::infinity()), at::zeros({}, atRes.options()), atRes);
    }
    if (reduction == 1) {
        auto target_lengths_t = at::tensor(tl, atRes.options()).clamp_min(1);
        atRes = (atRes / target_lengths_t).mean();
    } else if (reduction == 2) {
        atRes = atRes.sum();
    }
    impl::aten::updateATen2Tensor(ctx, atRes, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths,
                                  diopiConstTensorHandle_t target_lengths, diopiConstTensorHandle_t neg_log_likelihood, diopiConstTensorHandle_t log_alpha,
                                  int64_t blank, diopiReduction_t reduction, bool zero_infinity) {
    impl::aten::setCurCtx(ctx);
    auto atLogProbs = impl::aten::buildATen(log_probs);
    auto atTarget = impl::aten::buildATen(targets);
    auto atInputLength = impl::aten::buildATen(input_lengths).to(at::Device(at::kCPU), at::kLong).contiguous();
    auto atTargetLength = impl::aten::buildATen(target_lengths).to(at::Device(at::kCPU), at::kLong).contiguous();
    at::IntArrayRef il(atInputLength.data_ptr<int64_t>(), atInputLength.numel());
    at::IntArrayRef tl(atTargetLength.data_ptr<int64_t>(), atTargetLength.numel());
    int64_t batch_size = atLogProbs.size(1);
    std::vector<int64_t> expand_shape = {batch_size};
    at::IntArrayRef shape(expand_shape.data(), expand_shape.size());
    auto atGrad = impl::aten::buildATen(grad_output);
    if (reduction == 1) {
        atGrad = at::native::expand(atGrad, shape).clone();
        auto target_lengths_t = at::tensor(tl, atGrad.options()).clamp_min(1);
        atGrad = atGrad / target_lengths_t;
        atGrad.mul_(1. / batch_size);
    } else if (reduction == 2) {
        atGrad = at::native::expand(atGrad, shape);
    }
    auto atNegLogLikehood = impl::aten::buildATen(neg_log_likelihood);
    auto atLogAlpha = impl::aten::buildATen(log_alpha);
    auto atOut = at::native::ctc_loss_backward_gpu(atGrad, atLogProbs, atTarget, il, tl, atNegLogLikehood, atLogAlpha, blank, zero_infinity);
    impl::aten::updateATen2Tensor(ctx, atOut, grad_input);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices,
                              int64_t indices_counts, bool accumulate) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(indices);
    auto atInput = impl::aten::buildATen(input);
    auto atValues = impl::aten::buildATen(values);
    torch::List<c10::optional<at::Tensor>> atIndicesList;
    assert(indices_counts >= 1);
    for (int i = 0; i < indices_counts; ++i) {
        auto atIndices = c10::optional<at::Tensor>(impl::aten::buildATen(indices[i]));
        atIndicesList.emplace_back(atIndices);
    }
    at::Tensor atOut = at::index_put(atInput, atIndicesList, atValues, accumulate);
    impl::aten::updateATen2Tensor(ctx, atOut, input);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values,
                                     diopiConstTensorHandle_t* indices, int64_t indices_counts, bool accumulate) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(indices);
    auto atInput = impl::aten::buildATen(input);
    auto atValues = impl::aten::buildATen(values);
    torch::List<c10::optional<at::Tensor>> atIndicesList;
    assert(indices_counts >= 1);
    for (int i = 0; i < indices_counts; ++i) {
        auto atIndices = c10::optional<at::Tensor>(impl::aten::buildATen(indices[i]));
        atIndicesList.emplace_back(atIndices);
    }
    at::Tensor atOut = at::index_put(atInput, atIndicesList, atValues, accumulate);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index,
                             const char* reduce) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atSrc = impl::aten::buildATen(src);
    auto atIndex = impl::aten::buildATen(index);
    at::Tensor atOut;
    if (0 == strcmp(reduce, "add") || 0 == strcmp(reduce, "multiply")) {
        c10::string_view atReduce(reduce, strlen(reduce));
        atOut = at::scatter(atInput, dim, atIndex, atSrc, atReduce);
    } else {
        atOut = at::scatter(atInput, dim, atIndex, atSrc);
    }
    impl::aten::updateATen2Tensor(ctx, atOut, input);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index,
                                   const char* reduce) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atIndex = impl::aten::buildATen(index);
    at::Tensor atOut;
    if (0 == strcmp(reduce, "add") || 0 == strcmp(reduce, "multiply")) {
        c10::string_view atReduce(reduce, strlen(reduce));
        atOut = at::scatter(atInput, dim, atIndex, atValue, atReduce);
    } else {
        atOut = at::scatter(atInput, dim, atIndex, atValue);
    }
    impl::aten::updateATen2Tensor(ctx, atOut, input);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src,
                          diopiConstTensorHandle_t index, const char* reduce) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atSrc = impl::aten::buildATen(src);
    auto atIndex = impl::aten::buildATen(index);
    at::Tensor atOut;
    if (0 == strcmp(reduce, "add") || 0 == strcmp(reduce, "multiply")) {
        c10::string_view atReduce(reduce, strlen(reduce));
        atOut = at::scatter(atInput, dim, atIndex, atSrc, atReduce);
    } else {
        atOut = at::scatter(atInput, dim, atIndex, atSrc);
    }
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                diopiConstTensorHandle_t index, const char* reduce) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atIndex = impl::aten::buildATen(index);
    at::Tensor atOut;
    if (0 == strcmp(reduce, "add") || 0 == strcmp(reduce, "multiply")) {
        c10::string_view atReduce(reduce, strlen(reduce));
        atOut = at::scatter(atInput, dim, atIndex, atValue, atReduce);
    } else {
        atOut = at::scatter(atInput, dim, atIndex, atValue);
    }
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::IntArrayRef atSize = impl::aten::buildAtIntArray(size);
    if (atInput.dim() == 3) {
        at::upsample_nearest1d_out(atOut, atInput, atSize);
    } else if (atInput.dim() == 4) {
        at::upsample_nearest2d_out(atOut, atInput, atSize);
    } else if (atInput.dim() == 5) {
        at::upsample_nearest3d_out(atOut, atInput, atSize);
    } else {
        NOT_SUPPORTED("input dim < 3 or >5");
        return diopiErrorOccurred;
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t out_size,
                                          diopiSize_t in_size) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atGradOut = impl::aten::buildATen(grad_output);
    at::Tensor atGradInput = impl::aten::buildATen(grad_input);
    at::IntArrayRef atOutSize = impl::aten::buildAtIntArray(out_size);
    at::IntArrayRef atInSize = impl::aten::buildAtIntArray(in_size);
    if (atGradInput.dim() == 3) {
        at::upsample_nearest1d_backward_out(atGradInput, atGradOut, atOutSize, atInSize);
    } else if (atGradInput.dim() == 4) {
        at::upsample_nearest2d_backward_out(atGradInput, atGradOut, atOutSize, atInSize);
    } else if (atGradInput.dim() == 5) {
        at::upsample_nearest3d_backward_out(atGradInput, atGradOut, atOutSize, atInSize);
    } else {
        NOT_SUPPORTED("grad_input dim < 3 or >5");
        return diopiErrorOccurred;
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool align_corners,
                                 const char* mode) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::IntArrayRef atSize = impl::aten::buildAtIntArray(size);
    if (3 == atInput.dim() && 0 == strcmp(mode, "linear")) {
        at::upsample_linear1d_out(atOut, atInput, atSize, align_corners);
    } else if (4 == atInput.dim()) {
        if (0 == strcmp(mode, "bilinear")) {
            at::upsample_bilinear2d_out(atOut, atInput, atSize, align_corners);
        } else if (0 == strcmp(mode, "bicubic")) {
            at::upsample_bicubic2d_out(atOut, atInput, atSize, align_corners);
        } else {
            NOT_SUPPORTED("interpolate mode type");
            return diopiErrorOccurred;
        }
    } else if (5 == atInput.dim() && 0 == strcmp(mode, "trilinear")) {
        at::upsample_trilinear3d_out(atOut, atInput, atSize, align_corners);
    } else {
        NOT_SUPPORTED("interpolate mode type");
        return diopiErrorOccurred;
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiSize_t out_size,
                                         diopiSize_t in_size, bool align_corners, const char* mode) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atGradOut = impl::aten::buildATen(grad_output);
    at::Tensor atGradInput = impl::aten::buildATen(grad_input);
    at::IntArrayRef atOutSize = impl::aten::buildAtIntArray(out_size);
    at::IntArrayRef atInSize = impl::aten::buildAtIntArray(in_size);
    if (3 == atGradInput.dim() && 0 == strcmp(mode, "linear")) {
        at::upsample_linear1d_backward_out(atGradInput, atGradOut, atOutSize, atInSize, align_corners);
    } else if (4 == atGradInput.dim()) {
        if (0 == strcmp(mode, "bilinear")) {
            at::upsample_bilinear2d_backward_out(atGradInput, atGradOut, atOutSize, atInSize, align_corners);
        } else if (0 == strcmp(mode, "bicubic")) {
            at::upsample_bicubic2d_backward_out(atGradInput, atGradOut, atOutSize, atInSize, align_corners);
        } else {
            NOT_SUPPORTED("interpolate mode type");
            return diopiErrorOccurred;
        }
    } else if (5 == atGradInput.dim() && 0 == strcmp(mode, "trilinear")) {
        at::upsample_trilinear3d_backward_out(atGradInput, atGradOut, atOutSize, atInSize, align_corners);
    } else {
        NOT_SUPPORTED("interpolate mode type");
        return diopiErrorOccurred;
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode,
                      const double* value) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atPad = impl::aten::buildAtIntArray(pad);
    torch::nn::functional::PadFuncOptions::mode_t pad_mode;
    double atValue = 0;
    if (strcmp(mode, "constant") == 0) {
        DIOPI_CHECK_PTR(value);
        atValue = *value;
        pad_mode = torch::kConstant;
    } else if (strcmp(mode, "reflect") == 0) {
        pad_mode = torch::kReflect;
    } else if (strcmp(mode, "replicate") == 0) {
        pad_mode = torch::kReplicate;
    } else if (strcmp(mode, "circular") == 0) {
        pad_mode = torch::kCircular;
    } else {
        NOT_SUPPORTED("padding mode");
    }
    auto atOut = torch::nn::functional::detail::pad(atInput, atPad, pad_mode, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted,
                         bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
    impl::aten::setCurCtx(ctx);
    DIOPI_CHECK_PTR(out);
    auto atInput = impl::aten::buildATen(input);
    bool return_inverse = indices ? true : false;
    std::tuple<at::Tensor, at::Tensor, at::Tensor> atOuts;

    if (!dim) {
        atOuts = at::_unique2(atInput, sorted, return_inverse, return_counts);
    } else {
        atOuts = at::unique_dim(atInput, *dim, sorted, return_inverse, return_counts);
    }
    impl::aten::buildDiopiTensor(ctx, std::get<0>(atOuts), out);
    if (return_inverse) {
        impl::aten::updateATen2Tensor(ctx, std::get<1>(atOuts), indices);
    }
    if (return_counts) {
        DIOPI_CHECK_PTR(counts);
        impl::aten::buildDiopiTensor(ctx, std::get<2>(atOuts), counts);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    if (dim == nullptr) {
        auto atTmp = at::prod(atInput);
        impl::aten::updateATen2Tensor(ctx, atTmp, out);
    } else {
        bool keepdim = false;
        if (atInput.dim() == atOut.dim()) {
            keepdim = true;
        }
        at::prod_out(atOut, atInput, *dim, keepdim);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                 diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);

    if (grad_input) {
        auto atGradInput = impl::aten::buildATen(grad_input);
        at::matmul_out(atGradInput, atGradOutput, atWeight);
    }

    int64_t dims = atInput.dim();
    if (grad_weight) {
        auto atGradWeightTemp = at::matmul(atInput.transpose(dims - 2, dims - 1), atGradOutput);
        atGradWeightTemp.transpose_(dims - 2, dims - 1);
        if (dims > 2) {
            std::vector<int64_t> sumDim;
            for (int i = 0; i < dims - 2; ++i) {
                sumDim.push_back(i);
            }
            at::IntArrayRef atSumDim(sumDim.data(), sumDim.size());
            auto atGradWeight = impl::aten::buildATen(grad_weight);
            at::sum_out(atGradWeight, atGradWeightTemp, atSumDim);
        } else {
            impl::aten::updateATen2Tensor(ctx, atGradWeightTemp, grad_weight);
        }
    }

    if (grad_bias) {
        std::vector<int64_t> sumDim;
        for (int i = 0; i < dims - 1; ++i) {
            sumDim.push_back(i);
        }
        at::IntArrayRef atSumDim(sumDim.data(), sumDim.size());
        auto atGradBias = impl::aten::buildATen(grad_bias);
        at::sum_out(atGradBias, atGradOutput, atSumDim);
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                           diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    impl::aten::setCurCtx(ctx);
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);

    at::Tensor atGradInput;
    // case 1
    if (atInput.sizes() == atTarget.sizes()) {
        atGradInput = impl::aten::crossEntropyLossProbTargetBackward(atInput, atGradOutput, atTarget, weight, reduction, label_smoothing);
        // case 2
    } else if (label_smoothing > 0.0) {
        atGradInput = impl::aten::crossEntropyLossLabelSmoothingBackward(atInput, atGradOutput, atTarget, weight, reduction, ignore_index, label_smoothing);
        // case 3
    } else {
        auto atLogSoftmaxOutput = at::log_softmax(atInput, 1, atInput.scalar_type());
        auto atGradInputNllLoss = impl::aten::nllLossNdBackward(atLogSoftmaxOutput, atGradOutput, atTarget, weight, reduction, ignore_index);
        atGradInput = at::_log_softmax_backward_data(atGradInputNllLoss, atLogSoftmaxOutput, 1, atInput.scalar_type());
    }
    impl::aten::updateATen2Tensor(ctx, atGradInput, grad_input);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiErfinv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::erfinv_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiErfinvInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::erfinv_out, atInput, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiIm2Col(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t dilation,
                         diopiSize_t padding, diopiSize_t stride) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);

    at::im2col_out(atOut, atInput, atKernelSize, atDilation, atPadding, atStride);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCol2Im(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t output_size, diopiSize_t kernel_size,
                         diopiSize_t dilation, diopiSize_t padding, diopiSize_t stride) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::IntArrayRef atOutSize = impl::aten::buildAtIntArray(output_size);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);

    at::col2im_out(atOut, atInput, atOutSize, atKernelSize, atDilation, atPadding, atStride);

    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atDims = impl::aten::buildAtIntArray(dims);
    impl::aten::invokeATenFuncInp(ctx, at::flip_out, atOut, atInput, atDims);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCholesky(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t info, diopiConstTensorHandle_t mat, bool upper,
                           bool checkerror) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atMat = impl::aten::buildATen(mat);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atInfo = impl::aten::buildATen(info);
    at::linalg_cholesky_ex_out(atOut, atInfo, atMat, upper, checkerror);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCholeskyBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_mat, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t L,
                                   bool upper) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atGradMat = impl::aten::buildATen(grad_mat);
    at::Tensor atL = impl::aten::buildATen(L);
    at::Tensor atGradOut = impl::aten::buildATen(grad_output);
    if (upper) {
        atL = atL.transpose(-1, -2).conj();
        atGradOut = atGradOut.transpose(-1, -2).conj();
    }
    auto L_inverse = std::get<0>(at::triangular_solve(at::eye(atL.size(-1), atL.options()), atL, /*upper=*/false));
    auto phi = at::matmul(atL.transpose(-1, -2).conj(), atGradOut);
    phi.tril_().diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).mul_(0.5);

    auto grad_input = at::matmul(at::matmul(L_inverse.transpose(-1, -2).conj(), phi), L_inverse);
    auto out = grad_input.add(grad_input.transpose(-1, -2).conj()).mul_(0.5);  // Symmetrizing the gradient
    impl::aten::updateATen2Tensor(ctx, out, grad_mat);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiTriangularSolve(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t cloned_mat, diopiConstTensorHandle_t b,
                                  diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atClonedMat = impl::aten::buildATen(cloned_mat);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atb = impl::aten::buildATen(b);
    at::Tensor atMat = impl::aten::buildATen(mat);
    at::triangular_solve_out(atOut, atClonedMat, atb, atMat, upper, transpose, unitriangular);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiTriangularSolveBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_b, diopiTensorHandle_t grad_mat,
                                                    diopiConstTensorHandle_t grad_x, diopiConstTensorHandle_t grad_cloned_mat, diopiConstTensorHandle_t x,
                                                    diopiConstTensorHandle_t b, diopiConstTensorHandle_t mat, bool upper, bool transpose, bool unitriangular) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atGradB = impl::aten::buildATen(grad_b);
    at::Tensor atGradM = impl::aten::buildATen(grad_mat);

    at::Tensor atGradx = impl::aten::buildATen(grad_x);
    at::Tensor atGradCloneMat = impl::aten::buildATen(grad_cloned_mat);

    at::Tensor atx = impl::aten::buildATen(x);
    at::Tensor atb = impl::aten::buildATen(b);
    at::Tensor atMat = impl::aten::buildATen(mat);

    at::Tensor atGradb, atGradMat;
    if (atGradx.defined() || atGradCloneMat.defined()) {
        if (atGradx.defined()) {
            atGradb = std::get<0>(atGradx.triangular_solve(atMat.conj(), upper, !transpose, unitriangular));
            if (grad_mat != nullptr) {
                atGradMat = transpose ? -atx.conj().matmul(atGradb.transpose(-1, -2)) : -atGradb.matmul(atx.transpose(-1, -2).conj());
                if (upper) {
                    atGradMat = atGradMat.triu(static_cast<int>(unitriangular));
                } else {
                    atGradMat = atGradMat.tril(-static_cast<int>(unitriangular));
                }
            }
        }
        if (!atGradMat.defined()) {
            atGradMat = at::zeros({1}, atMat.options()).expand_as(atMat);
        }
        if (!atGradb.defined()) {
            atGradb = at::zeros({1}, atb.options()).expand_as(atb);
        }
        if (grad_mat != nullptr && atGradCloneMat.defined()) {
            atGradMat = atGradMat.add(atGradCloneMat);
        }
        int64_t nums = atGradMat.numel() / atGradM.numel();
        std::vector<int64_t> newShape{nums, atGradMat.size(-2), -1};
        if (nums != 1) {
            at::IntArrayRef atShape(newShape.data(), newShape.size());
            at::sum_out(atGradM, atGradMat.reshape(atShape), 0, false);
        } else {
            impl::aten::updateATen2Tensor(ctx, atGradMat, grad_mat);
        }

        nums = atGradb.numel() / atGradB.numel();
        if (nums != 1) {
            newShape[0] = nums;
            at::IntArrayRef atShape(newShape.data(), newShape.size());
            at::sum_out(atGradB, atGradb.reshape(atShape), 0, false);
        } else {
            impl::aten::updateATen2Tensor(ctx, atGradb, grad_b);
        }
    }
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeats_size) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atRepeatsSize = impl::aten::buildAtIntArray(repeats_size);
    auto atOut = at::native::repeat(atInput, atRepeatsSize);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiMultinomial(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t num_samples, bool replacement,
                              diopiGeneratorHandle_t generator) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::Generator gen = impl::aten::buildGenerator(ctx, generator);
    at::multinomial_out(atOut, atInput, num_samples, replacement, gen);
    impl::aten::updateGeneratorHandleState(ctx, gen, generator);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiCastDtype(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    auto atOut = at::native::to(atInput, impl::aten::getATenType(dtype).toScalarType(), false, true, c10::nullopt);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiPolar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t abs, diopiConstTensorHandle_t angle) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atAbs = impl::aten::buildATen(abs);
    auto atAngle = impl::aten::buildATen(angle);
    at::polar_out(atOut, atAbs, atAngle);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCeilInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::ceil_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiCeil(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    at::ceil_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAsinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::asin_, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAsin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    at::asin_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLerpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                                       diopiConstTensorHandle_t weight) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    auto atEnd = impl::aten::buildATen(end);
    auto atWeight = impl::aten::buildATen(weight);
    at::lerp_out(atOut, atInput, atEnd, atWeight);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLerpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t end,
                                       const diopiScalar_t* weight) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    auto atEnd = impl::aten::buildATen(end);
    at::Scalar atWeight = impl::aten::buildAtScalar(weight);
    at::lerp_out(atOut, atInput, atEnd, atWeight);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiTriu(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t diagonal) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    at::triu_out(atOut, atInput, diagonal);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiTriuInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t diagonal) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    at::triu_out(atInput, atInput, diagonal);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSgn(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atOut = impl::aten::buildATen(out);
    auto atInput = impl::aten::buildATen(input);
    at::sgn_out(atOut, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiSgnInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::sgn_out, atInput, atInput);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiIsNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto inp_out = at::isnan(atInput);
    impl::aten::updateATen2Tensor(ctx, inp_out, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiLinalgQR(diopiContextHandle_t ctx, diopiConstTensorHandle_t A, const char* mode, diopiTensorHandle_t Q, diopiTensorHandle_t R) {
    impl::aten::setCurCtx(ctx);
    auto atA = impl::aten::buildATen(A);
    auto atQ = impl::aten::buildATen(Q);
    auto atR = impl::aten::buildATen(R);
    c10::string_view atMode(mode, strlen(mode));
    at::linalg_qr_out(atQ, atR, atA, mode);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiAmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t self, diopiSize_t dim, bool keepdim) {
    impl::aten::setCurCtx(ctx);
    at::IntArrayRef atDim = impl::aten::buildAtIntArray(dim);
    auto atOut = impl::aten::buildATen(out);
    auto atSelf = impl::aten::buildATen(self);
    at::amax_out(atOut, atSelf, atDim, keepdim);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiBatchNormStats(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input, double eps) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atOuts = at::batch_norm_stats(atInput, eps);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), mean);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(atOuts), invstd);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormGatherStatsWithCounts(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd,
                                                           diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean_all,
                                                           diopiConstTensorHandle_t invstd_all, diopiTensorHandle_t running_mean,
                                                           diopiTensorHandle_t running_var, float momentum, float eps, diopiConstTensorHandle_t counts) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMean_all = impl::aten::buildATen(mean_all);
    auto atInvstd_all = impl::aten::buildATen(invstd_all);
    auto atRunning_mean = impl::aten::buildATen(running_mean);
    auto atRunning_var = impl::aten::buildATen(running_var);
    auto atCounts = impl::aten::buildATen(counts);
    auto atOuts = at::batch_norm_gather_stats_with_counts(atInput, atMean_all, atInvstd_all, atRunning_mean, atRunning_var, momentum, eps, atCounts);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), mean);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(atOuts), invstd);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormBackwardReduce(diopiContextHandle_t ctx, diopiTensorHandle_t sum_dy, diopiTensorHandle_t sum_dy_xmu,
                                                    diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_out,
                                                    diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd,
                                                    diopiConstTensorHandle_t weight, bool input_g, bool weight_g, bool bias_g) {
    impl::aten::setCurCtx(ctx);
    auto atGrad_out = impl::aten::buildATen(grad_out);
    auto atInput = impl::aten::buildATen(input);
    auto atMean = impl::aten::buildATen(mean);
    auto atInvstd = impl::aten::buildATen(invstd);
    auto atWeight = impl::aten::buildATen(weight);
    auto atOuts = at::batch_norm_backward_reduce(atGrad_out, atInput, atMean, atInvstd, atWeight, input_g, weight_g, bias_g);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), sum_dy);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(atOuts), sum_dy_xmu);
    impl::aten::updateATen2Tensor(ctx, std::get<2>(atOuts), grad_weight);
    impl::aten::updateATen2Tensor(ctx, std::get<3>(atOuts), grad_bias);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormBackwardElemt(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_out,
                                                   diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd,
                                                   diopiConstTensorHandle_t weight, diopiConstTensorHandle_t sum_dy, diopiConstTensorHandle_t sum_dy_xmu,
                                                   diopiConstTensorHandle_t count) {
    impl::aten::setCurCtx(ctx);
    auto atGrad_out = impl::aten::buildATen(grad_out);
    auto atInput = impl::aten::buildATen(input);
    auto atMean = impl::aten::buildATen(mean);
    auto atInvstd = impl::aten::buildATen(invstd);
    auto atWeight = impl::aten::buildATen(weight);
    auto atSum_dy = impl::aten::buildATen(sum_dy);
    auto atSum_dy_xmu = impl::aten::buildATen(sum_dy_xmu);
    auto atCount = impl::aten::buildATen(count);
    auto atOuts = at::batch_norm_backward_elemt(atGrad_out, atInput, atMean, atInvstd, atWeight, atSum_dy, atSum_dy_xmu, atCount);
    impl::aten::updateATen2Tensor(ctx, atOuts, grad_input);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormElemt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, float eps) {
    impl::aten::setCurCtx(ctx);
    auto atInput = impl::aten::buildATen(input);
    auto atMean = impl::aten::buildATen(mean);
    auto atInvstd = impl::aten::buildATen(invstd);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);
    auto atOuts = at::batch_norm_elemt(atInput, atWeight, atBias, atMean, atInvstd, eps);
    impl::aten::updateATen2Tensor(ctx, atOuts, out);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

}  // extern "C"
