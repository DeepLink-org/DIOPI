/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#include <math.h>
#include <diopi/functions.h>
#include <../../aten/operators/cnnl/internal/binaryops_util.h>
#include <../../aten/generated/mlu_type.h>
#include <cnrt.h>
#include <cstring>

static thread_local diopiContextHandle_t context = nullptr;
#include "aten_helper.hpp"

#define FLT_MIN  __FLT_MIN__

using namespace torch_mlu::cnnl::ops;

extern "C" {

__attribute__((constructor)) static void libdiopi_init() {
    torch_mlu::RegisterAtenOperators();
}

static const char* name = "CambDevice";
static char version[1024] = {0};

const char* diopiGetVendorName() {
    return name;
}

const char* diopiGetImplVersion() {
    if (strlen(version) == 0) {
        sprintf(version, "Cnrt Version: %d; CNNL Version: %d; DIOPI Version: %d.%d.%d",
                CNRT_VERSION, CNNL_VERSION, DIOPI_VER_MAJOR, DIOPI_VER_MINOR, DIOPI_VER_PATCH);
    }
    return version;
}

const char* diopiGetLastErrorString() {
    return camb_get_last_error_string();
}

diopiError_t diopiRelu(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncRet(ctx, at::relu, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::relu_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        const diopiScalar_t* negative_slope) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atSlope = camb::aten::buildAtScalar(negative_slope);
    camb::aten::invokeATenFuncRet(ctx, at::leaky_relu, out, atInput, atSlope);
    return diopiSuccess;
}

diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx,
        diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atSlope = camb::aten::buildAtScalar(negative_slope);
    camb::aten::invokeATenFuncInp(ctx, at::leaky_relu_, atInput, atSlope);
    return diopiSuccess;
}

// TOCHECK
diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
        diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::IntArrayRef atKernelSize = camb::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = camb::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = camb::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = camb::aten::buildAtIntArray(dilation);
    bool atCeilMode = ceil_mode;
    if (3 == atInput.dim()) {
       atInput = atInput.unsqueeze(0);
    }
    // cnnl has not out version and supports only 4d Tensor
    if (4 != atInput.dim()) {
        NOT_SUPPORTED("dim < 3 or dim > 4");
        return diopiErrorOccurred;
    }
    if (dilation.len != 0 && (dilation.data[0] != 1 || dilation.data[1] != 1)) {
        NOT_SUPPORTED("dilation != 1");
        return diopiErrorOccurred;
    }
    auto atOuts = cnnl_max_pool2d_with_indices(atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
    at::Tensor atOut = (std::get<0>(atOuts)).contiguous();
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
        diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::IntArrayRef atKernelSize = camb::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = camb::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = camb::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = camb::aten::buildAtIntArray(dilation);
    bool atCeilMode = ceil_mode;

    if (3 == atInput.dim()) {
       atInput = atInput.unsqueeze(0);
    }
    // cnnl has not out version and supports only 4d Tensor
    if (4 != atInput.dim()) {
        NOT_SUPPORTED("dim < 3 or dim > 4");
        return diopiErrorOccurred;
    }
    if (dilation.len != 0 && (dilation.data[0] != 1 || dilation.data[1] != 1)) {
        NOT_SUPPORTED("dilation != 1");
        return diopiErrorOccurred;
    }
    auto atOuts = cnnl_max_pool2d_with_indices(atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);

    at::Tensor atIndices = (std::get<1>(atOuts)).contiguous();
    at::Tensor atOut = (std::get<0>(atOuts)).contiguous();
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    camb::aten::updateATen2Tensor(ctx, atIndices, indices);
    return diopiSuccess;
}

/**
 * @brief
 * @param rounding_mode not supported
 */
diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOther = camb::aten::buildATen(other);
    at::Tensor atOut = camb::aten::buildATen(out);
    cnnl_div_out_internal(atOut, atInput, atOther);
    return diopiSuccess;
}

/**
 * @brief
 * @param rounding_mode not supported
 */
diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOther = camb::aten::buildAtScalar(other);
    at::Tensor atOut = camb::aten::buildATen(out);
#if CNRT_VERSION >= 60002
    cnnl_div_out_internal(atOut, atInput, wrapped_scalar_tensor(atOther));
#else
    cnnl_div_out(atOut, atInput, at::scalar_tensor(atOther, at::device(at::kMLU).dtype(atInput.scalar_type())));
#endif
    return diopiSuccess;
}

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
        diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atWeight = camb::aten::buildATen(weight);
    auto atBias = camb::aten::buildATen(bias);
    auto atStride = camb::aten::buildAtIntArray(stride);
    auto atPadding = camb::aten::buildAtIntArray(padding);
    auto atDilation = camb::aten::buildAtIntArray(dilation);
    camb::aten::invokeATenFuncRet(ctx, cnnl_convolution_overrideable, out,
        atInput, atWeight, atBias, atStride, atPadding, atDilation, false, at::IntArrayRef(0), groups);
    return diopiSuccess;
}

/**
 * @brief
 * @param label_smoothing not supported
 * @param target prob target not supported
 */
diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
        diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    if (label_smoothing != 0.0) {
        NOT_SUPPORTED("param label_smoothing");
        return diopiErrorOccurred;
    }
    at::Tensor atIn = at::log_softmax(atInput, 1);
    return camb::aten::nll_loss_internal(ctx, out, atIn, target, weight, reduction, ignore_index);
    // auto atOut = camb::aten::buildATen(out);
    // printf("loss : %f\n", ((float *)(cnnl_sum(atOut).cpu().data_ptr()))[0]);
}

diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
        diopiReduction_t reduction, int64_t ignore_index, double label_smoothing) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atGradInput = camb::aten::buildATen(grad_input);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atTarget = camb::aten::buildATen(target);
    // case 1
    if (atInput.sizes() == atTarget.sizes()) {
        NOT_SUPPORTED("target size");
        return diopiErrorOccurred;
    // case 2
    } else if (label_smoothing > 0.0) {
        NOT_SUPPORTED("param label_smoothing");
        return diopiErrorOccurred;
    // case 3
    } else {
        auto atLogSoftmaxOutput = at::log_softmax(atInput, 1, atInput.scalar_type());
        camb::aten::nll_loss_bp_internal(ctx, grad_input, grad_output, atLogSoftmaxOutput, target,
                             weight, reduction, ignore_index);
        atGradInput = at::_log_softmax_backward_data(atGradInput, atLogSoftmaxOutput, 1, atLogSoftmaxOutput);
    }
    camb::aten::updateATen2Tensor(ctx, atGradInput, grad_input);
    return diopiSuccess;
}

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atMat2 = camb::aten::buildATen(mat2);
    // cnnl has not out version
    camb::aten::invokeATenFuncRet(ctx, at::bmm, out, atInput, atMat2);
    return diopiSuccess;
}

diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOut = camb::aten::buildATen(out);
    auto atTensor1 = camb::aten::buildATen(tensor1);
    auto atTensor2 = camb::aten::buildATen(tensor2);
    auto atValue = camb::aten::buildAtScalar(value);
    cnnl_addcmul_internal(atOut, atInput, atTensor1, atTensor2, atValue);
    return diopiSuccess;
}

diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOther = camb::aten::buildATen(other);
    camb::aten::invokeATenFuncRet(ctx, at::matmul, out, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atTensor1 = camb::aten::buildATen(tensor1);
    auto atTensor2 = camb::aten::buildATen(tensor2);
    auto atValue = camb::aten::buildAtScalar(value);
    camb::aten::invokeATenFuncRet(ctx, at::addcdiv, out, atInput, atTensor1, atTensor2, atValue);
    return diopiSuccess;
}

diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
        diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOut = camb::aten::buildATen(out);
    auto atMax1 = camb::aten::buildATen(mat1);
    auto atMax2 = camb::aten::buildATen(mat2);
    auto atBeta = camb::aten::buildAtScalar(beta);
    auto atAlpha = camb::aten::buildAtScalar(alpha);
    at::addmm_out(atOut, atInput, atMax1, atMax2, atBeta, atAlpha);
    return diopiSuccess;
}

// NOTE(fengsibo): add int, short, bool test case
diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atDim = camb::aten::buildAtIntArray(dim);
    if (dim.len != 0) {
        auto atOut = camb::aten::buildATen(out);
        at::mean_out(atOut, atInput, atDim, false, camb::aten::getAtScalarType(dtype));
    } else {
        auto atOutCpu = at::mean(atInput.cpu(), atDim, false, camb::aten::getAtScalarType(dtype));
        camb::aten::updateATen2Tensor(ctx, atOutCpu, out);
    }
    return diopiSuccess;
}

// NOTE(fengsibo): add int, short, bool test case
diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atDim = camb::aten::buildAtIntArray(dim);
    if (dim.len != 0) {
        auto atOut = camb::aten::buildATen(out);
        at::sum_out(atOut, atInput, atDim, false, camb::aten::getAtScalarType(dtype));
    } else {
        auto atOutCpu = at::sum(atInput.cpu(), atDim, false, camb::aten::getAtScalarType(dtype));
        camb::aten::updateATen2Tensor(ctx, atOutCpu, out);
    }
    return diopiSuccess;
}

// TODO(huqingqing): only support 1 dimension
diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atDim = camb::aten::buildAtIntArray(dim);
    auto self_contiguous = cnnl_contiguous(atInput, c10::MemoryFormat::Contiguous);
    at::Tensor& atOut = self_contiguous;
    for (int i = 0; i < dim.len; i++) {
        atOut = cnnl_std_internal(atOut, dim.data[i], unbiased, true);
    }
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices,
        diopiConstTensorHandle_t input, int64_t dim) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atIndices = camb::aten::buildATen(min_indices);
    auto atOuts = at::min(atInput, dim, false);
    // Note: cnnl_min/at::min is asynchronous
    camb::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), min);
    camb::aten::convertToRealLong(std::get<1>(atOuts), atIndices, at::ScalarType::Int);
    return diopiSuccess;
}

diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices,
        diopiConstTensorHandle_t input, int64_t dim) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atIndices = camb::aten::buildATen(max_indices);
    // Note: there is a problem in out version for 4d/5d tensor, so that we use here return version.
    // Note: cnnl_max_out use also copy_, so the performance doesn't change.
    auto atOuts = at::max(atInput, dim, false);
    camb::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), max);
    camb::aten::convertToRealLong(std::get<1>(atOuts), atIndices, at::ScalarType::Int);
    return diopiSuccess;
}

diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const int64_t* dim) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    // Note: cnnl has not out version
    auto atOut = at::any(atInput, *dim);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const int64_t* dim) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOut = camb::aten::buildATen(out);
    at::all_out(atOut, atInput, *dim);
    return diopiSuccess;
}

diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    // Note: cnnl has not out version
    auto atOut = at::softmax(atInput, dim);  // TODO(fengsibo): use default type instead
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    // Note: cnnl has not out version
    auto atOut = at::log_softmax(atInput, dim);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

// TOCHECK
diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atIndex = camb::aten::buildATen(index);
    auto atOut = at::index_select(atInput, dim, atIndex);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

// To Check
diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    camb::aten::setCurCtx(ctx);
    auto atIndex = camb::aten::buildATen(index);
    auto atInput = camb::aten::buildATen(input);
    auto atOut = at::gather(atInput, dim, atIndex, false);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    at::Tensor atOut = at::select(atInput, dim, index).contiguous();
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t source) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atMask = camb::aten::buildATen(mask);
    auto atSource = camb::aten::buildATen(source);
    camb::aten::invokeATenFuncRet(ctx, at::masked_scatter, out, atInput, atMask, atSource);
    return diopiSuccess;
}

diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets,
        diopiConstTensorHandle_t scores, double iouThreshold) {
    DIOPI_CHECK_PTR(out);
    camb::aten::setCurCtx(ctx);
    auto atDets = camb::aten::buildATen(dets);
    auto atScores = camb::aten::buildATen(scores);
    // auto atOut = vision::ops::nms_kernel(atDets, atScores, iouThreshold);
    // camb::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

// TOCHECK
diopiError_t diopiNonzero(diopiContextHandle_t ctx,
        diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    DIOPI_CHECK_PTR(out);
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOut = at::nonzero(atInput);
    camb::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOut = camb::aten::buildATen(out);
    auto atWeight = camb::aten::buildATen(weight);
    auto atBias = camb::aten::buildATen(bias);
    // Note: cnnl has not out version
    camb::aten::invokeATenFuncRet(ctx, at::linear, out, atInput, atWeight, atBias);
    return diopiSuccess;
}

diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t rois, double spatialScale, int64_t pooledHeight,
        int64_t pooledWidth, int64_t samplingRatio, bool aligned) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atRois = camb::aten::buildATen(rois);
    // auto atOut = vision::ops::roi_align_forward_kernel(atInput, atRois, spatialScale,
    //  pooledHeight, pooledWidth, samplingRatio, aligned);
    // camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf,
        double learningrate, double momentum, double dampening, double weightDecay, bool nesterov) {
    camb::aten::setCurCtx(ctx);
    auto atW = camb::aten::buildATen(w);
    auto atDw = camb::aten::buildATen(dw);
    auto atBuf = camb::aten::buildATen(buf);

    auto& p = atW;
    auto& d_p = atDw;
    if (weightDecay != 0) {
        d_p.add_(p, weightDecay);
    }
    if (momentum != 0) {
        atBuf.mul_(momentum).add_(d_p, 1 - dampening);
        if (nesterov) {
          d_p = d_p.add(atBuf, momentum);
        } else {
          d_p = atBuf;
        }
    }
    p.add_(d_p, -1 * learningrate);

    camb::aten::updateATen2Tensor(ctx, atW, w);
    camb::aten::updateATen2Tensor(ctx, atDw, dw);
    camb::aten::updateATen2Tensor(ctx, atBuf, buf);
    return diopiSuccess;
}

/**
 * @brief
 * @param errorIfNonfinite not supported
 */
diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t* parameters,
        int64_t parametersNum, double maxNorm, double normType, bool errorIfNonfinite) {
    DIOPI_CHECK(parameters != nullptr && out != nullptr,
                "Not supported: out or parameters is nullptr");
    camb::aten::setCurCtx(ctx);
    auto tensorList = camb::aten::buildATenList(parameters, parametersNum);
    *out = torch::nn::utils::clip_grad_norm_(tensorList, maxNorm, normType);
    return diopiSuccess;
}

// TOCHECK
diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx,
        diopiTensorHandle_t inout, diopiConstTensorHandle_t indices, double max_norm, double norm_type) {
    camb::aten::setCurCtx(ctx);
    auto atSelf = camb::aten::buildATen(inout);
    auto atIndices = camb::aten::buildATen(indices);
    at::embedding_renorm_(atSelf, atIndices, max_norm, norm_type);
    return diopiSuccess;
}

// TOCHECK
diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight,
        diopiConstTensorHandle_t indices, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    camb::aten::setCurCtx(ctx);
    auto atWeight = camb::aten::buildATen(weight);
    auto atIndices = camb::aten::buildATen(indices);
    camb::aten::invokeATenFuncRet(ctx, at::embedding, out, atWeight, atIndices, paddingIdx, scaleGradByFreq, sparse);
    return diopiSuccess;
}

diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t diagonal) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOut = camb::aten::buildATen(out);
    at::tril_out(atOut, atInput, diagonal);
    return diopiSuccess;
}

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t* tensors, int64_t insNum, int64_t dim) {
    DIOPI_CHECK_PTR(tensors);
    camb::aten::setCurCtx(ctx);
    auto tensorList = camb::aten::buildATenList(tensors, insNum);
    auto atOut = camb::aten::buildATen(out);
    at::cat_out(atOut, tensorList, dim);
    return diopiSuccess;
}

diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t outsNum,
        diopiConstTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim) {
    DIOPI_CHECK_PTR(outs);
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atSizes = camb::aten::buildAtIntArray(splitSizes);
    if (atInput.scalar_type() == at::ScalarType::Double) {
        NOT_SUPPORTED("float64 dtype");
        // auto atOuts = at::split(atInput.cpu(), atSizes, dim);
        // for (int i = 0; i < outsNum; ++i) {
        //     auto out = camb::aten::buildATen(outs[i]);
        //     ::cnrtMemcpy(out.data_ptr(), atOuts[i].contiguous().data_ptr(),
        //                  atOuts[i].nbytes(), CNRT_MEM_TRANS_DIR_HOST2DEV);
        // }
    } else {
        auto atOuts = at::split_with_sizes(atInput, atSizes, dim);
        for (int i = 0; i < outsNum; ++i) {
            camb::aten::updateATen2Tensor(ctx, atOuts[i].contiguous(), outs[i]);
        }
    }
    return diopiSuccess;
}

diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    DIOPI_CHECK_PTR(tensors);
    camb::aten::setCurCtx(ctx);
    auto tensorList = camb::aten::buildATenList(tensors, numTensors);

    std::vector<at::Tensor> a = camb::aten::buildATenList(tensors, numTensors);
    at::TensorList b = camb::aten::buildATenList(tensors, numTensors);

    auto atOut = camb::aten::buildATen(out);
    at::stack_out(atOut, tensorList, dim);
    return diopiSuccess;
}

// TODO(huqingqing) :indice not support int64
diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
        diopiConstTensorHandle_t input, int64_t dim, bool descending, const bool* stable) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atIndices = camb::aten::buildATen(indices);
#if TORCH_MM_VERSION <= TORCH_1_8_MM_VERSION
    auto atOuts = at::sort(atInput, dim, descending);
#else
    c10::optional<bool> atStable = stable ? c10::optional<bool>(*stable) : c10::optional<bool>(false);
    auto atOuts = at::sort(atInput, atStable, dim, descending);
#endif
    camb::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), values);
    camb::aten::convertToRealLong(std::get<1>(atOuts), atIndices, at::ScalarType::Int);
    return diopiSuccess;
}

diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
        diopiConstTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atValues = camb::aten::buildATen(values);
    auto atIndices = camb::aten::buildATen(indices);
    cnnl_topk_internal(atValues, atIndices, atInput, k, dim, largest, sorted);
    if  (!atValues.is_contiguous()) {
        atValues = atValues.contiguous();
        camb::aten::updateATen2Tensor(ctx, atValues, values);
    }
    if (atIndices.scalar_type() == at::ScalarType::Long) {
        camb::aten::convertToRealLong(atIndices, atIndices, at::ScalarType::Int);
    }
    return diopiSuccess;
}

diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncRet<at::Tensor (*)(at::Tensor const&, int64_t, int64_t)>
        (ctx, at::transpose, out, atInput, dim0, dim1);
    return diopiSuccess;
}

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t numClasses) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncRet(ctx, at::one_hot, out, atInput, numClasses);
    return diopiSuccess;
}

diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    auto atCondition = camb::aten::buildATen(condition);
    auto atInput = camb::aten::buildATen(input);
    auto atOther = camb::aten::buildATen(other);
    // Note: cnnl has not out version
    camb::aten::invokeATenFuncRet
        <at::Tensor (*)(at::Tensor const&, at::Tensor const&, at::Tensor const&)>
        (ctx, at::where, out, atCondition, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiSin(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::sin_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::sin_, atInput);
    return diopiSuccess;
}

diopiError_t diopiCos(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::cos_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::cos_, atInput);
    return diopiSuccess;
}

diopiError_t diopiAbs(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::abs_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::abs_, atInput);
    return diopiSuccess;
}

// Note: cnnl has no nan for negative input, the same behavior as log.
diopiError_t diopiSqrt(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    cnnl_sqrt_internal(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::sqrt_, atInput);
    return diopiSuccess;
}

diopiError_t diopiFloor(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::floor_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::floor_, atInput);
    return diopiSuccess;
}

diopiError_t diopiNeg(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::neg_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::neg_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSign(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::sign_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiTanh(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::tanh_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::tanh_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSigmoid(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::sigmoid_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::sigmoid_, atInput);
    return diopiSuccess;
}

diopiError_t diopiExp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::exp_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::exp_, atInput);
    return diopiSuccess;
}

// Note: cnnl has no nan for negative input, eg: log(-0.4890415) cnnl gets -19.764, pytorch gets nan.
diopiError_t diopiLog(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    cnnl_log_internal(atOut, atInput, CNNL_LOG_E);
    return diopiSuccess;
}

diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::log_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog2(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    cnnl_log_internal(atOut, atInput, CNNL_LOG_2);
    return diopiSuccess;
}

diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::log2_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog10(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    cnnl_log_internal(atOut, atInput, CNNL_LOG_10);;
    return diopiSuccess;
}

diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::log10_, atInput);
    return diopiSuccess;
}

diopiError_t diopiErf(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::erf_out(atOut, atInput);
    return diopiSuccess;
}

diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncInp(ctx, at::erf_, atInput);
    return diopiSuccess;
}

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atExponent = camb::aten::buildATen(exponent);
    at::Scalar atInput = camb::aten::buildAtScalar(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::pow_out(atOut, atInput, atExponent);
    return diopiSuccess;
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atExponent = camb::aten::buildAtScalar(exponent);
    // Note: fallback to CPU because camb doesn't support int32 input
    if (atInput.scalar_type() == at::ScalarType::Int) {
        auto atInCpu = atInput.to(at::kCPU);
        at::Tensor atOut = at::pow(atInCpu, atExponent);
        camb::aten::updateATen2Tensor(ctx, atOut, out);
    } else {
        at::Tensor atOut = camb::aten::buildATen(out);
        at::pow_out(atOut, atInput, atExponent);
    }
    return diopiSuccess;
}

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atExponent = camb::aten::buildATen(exponent);
    // Note: fallback to CPU because camb doesn't support int32 input
    if (atInput.scalar_type() == at::ScalarType::Int) {
        auto atInCpu = atInput.to(at::kCPU);
        auto atExCpu = atExponent.to(at::kCPU);
        at::Tensor atOut = at::pow(atInCpu, atExCpu);
        camb::aten::updateATen2Tensor(ctx, atOut, out);
    } else {
        at::Tensor atOut = camb::aten::buildATen(out);
        at::pow_out(atOut, atInput, atExponent);
    }
    return diopiSuccess;
}

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOther = camb::aten::buildATen(other);
    at::Scalar atAlpha = camb::aten::buildAtScalar(alpha);
    cnnl_add_out(atOut, atInput, atOther, atAlpha);
    return diopiSuccess;
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atOther = camb::aten::buildAtScalar(other);
    at::Scalar atAlpha = camb::aten::buildAtScalar(alpha);
    if (atInput.scalar_type() == at::ScalarType::Long) {
        NOT_SUPPORTED("long type");
        return diopiErrorOccurred;
    }
#if CNRT_VERSION >= 60002
    cnnl_add_out(atOut, atInput, wrapped_scalar_tensor(atOther), atAlpha);;
#else
    cnnl_add_out(atOut, atInput, at::scalar_tensor(atOther, at::device(at::kMLU).dtype(atInput.scalar_type())), atAlpha);
#endif
    return diopiSuccess;
}

diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOther = camb::aten::buildATen(other);
    at::Scalar atAlpha = camb::aten::buildAtScalar(alpha);
    at::sub_out(atOut, atInput, atOther, atAlpha);
    return diopiSuccess;
}

diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atOther = camb::aten::buildAtScalar(other);
    at::Scalar atAlpha = camb::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::sub(atInput, atOther, atAlpha);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOther = camb::aten::buildATen(other);
    cnnl_mul_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    // Note: scalar is double type
    at::Scalar atOther = camb::aten::buildAtScalar(other);
#if CNRT_VERSION >= 60002
    // Todo: to check
    cnnl_mul_out(atOut, atInput, wrapped_scalar_tensor(atOther));;
#else
    cnnl_mul_out(atOut, atInput, at::scalar_tensor(atOther, at::device(at::kMLU).dtype(atInput.scalar_type())));
#endif
    return diopiSuccess;
}

diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOther = camb::aten::buildATen(other);
    cnnl_ge_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atOther = camb::aten::buildAtScalar(other);
    at::ge_out(atOut, atInput, atOther);
    // cnnl_ge_out(atOut, atInput, wrapped_scalar_tensor(atOther));
    return diopiSuccess;
}

diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOther = camb::aten::buildATen(other);
    cnnl_gt_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atOther = camb::aten::buildAtScalar(other);
#if CNRT_VERSION >= 60002
    cnnl_gt_out(atOut, atInput, wrapped_scalar_tensor(atOther));
#else
    cnnl_gt_out(atOut, atInput, at::scalar_tensor(atOther, at::device(at::kMLU).dtype(atInput.scalar_type())));
#endif
    return diopiSuccess;
}

diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOther = camb::aten::buildATen(other);
    cnnl_le_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atOther = camb::aten::buildAtScalar(other);
#if CNRT_VERSION >= 60002
    cnnl_le_out(atOut, atInput, wrapped_scalar_tensor(atOther));
#else
    cnnl_le_out(atOut, atInput, at::scalar_tensor(atOther, at::device(at::kMLU).dtype(atInput.scalar_type())));
#endif
    return diopiSuccess;
}

diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOther = camb::aten::buildATen(other);
    cnnl_lt_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atOther = camb::aten::buildAtScalar(other);
#if CNRT_VERSION >= 60002
    cnnl_lt_out(atOut, atInput, wrapped_scalar_tensor(atOther));
#else
    cnnl_lt_out(atOut, atInput, at::scalar_tensor(atOther, at::device(at::kMLU).dtype(atInput.scalar_type())));
#endif
    return diopiSuccess;
}

diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOther = camb::aten::buildATen(other);
    cnnl_eq_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atOther = camb::aten::buildAtScalar(other);
#if CNRT_VERSION >= 60002
    cnnl_eq_out(atOut, atInput, wrapped_scalar_tensor(atOther));
#else
    cnnl_eq_out(atOut, atInput, at::scalar_tensor(atOther, at::device(at::kMLU).dtype(atInput.scalar_type())));
#endif
    return diopiSuccess;
}

diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOther = camb::aten::buildATen(other);
    cnnl_ne_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atOther = camb::aten::buildAtScalar(other);
#if CNRT_VERSION >= 60002
    cnnl_ne_out(atOut, atInput, wrapped_scalar_tensor(atOther));
#else
    cnnl_ne_out(atOut, atInput, at::scalar_tensor(atOther, at::device(at::kMLU).dtype(atInput.scalar_type())));
#endif
    return diopiSuccess;
}

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atTmpInput = camb::aten::buildATen(input);
    at::Tensor atTmpOther = camb::aten::buildATen(other);
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOther = atTmpOther.to(at::ScalarType::Bool);
    at::bitwise_and_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atTmpInput = camb::aten::buildATen(input);
    at::Scalar atTmpOther = camb::aten::buildAtScalar(other);
    at::Scalar atOther = atTmpOther.to<bool>();
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::bitwise_and_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atTmpInput = camb::aten::buildATen(input);
    at::Tensor atTmpOther = camb::aten::buildATen(other);
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOther = atTmpOther.to(at::ScalarType::Bool);
    at::bitwise_or_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atTmpInput = camb::aten::buildATen(input);
    at::Scalar atTmpOther = camb::aten::buildAtScalar(other);
    at::Scalar atOther = atTmpOther.to<bool>();
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::bitwise_or_out(atOut, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiScalar_t* min, const diopiScalar_t* max) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atMin = camb::aten::buildAtScalar(min);
    at::Scalar atMax = camb::aten::buildAtScalar(max);
    at::clamp_(atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atMin = camb::aten::buildAtScalar(min);
    at::Scalar atMax = camb::aten::buildAtScalar(max);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::clamp_out(atOut, atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiScalar_t* max) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atMax = camb::aten::buildAtScalar(max);
    at::clamp_max_(atInput, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atMax = camb::aten::buildAtScalar(max);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::clamp_max_out(atOut, atInput, atMax);
    return diopiSuccess;
}

#if TORCH_MM_VERSION > TORCH_1_9_MM_VERSION
diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atMin = camb::aten::buildATen(min);
    at::Tensor atMax = camb::aten::buildATen(max);
    at::clamp_(atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atMin = camb::aten::buildATen(min);
    at::Tensor atMax = camb::aten::buildATen(max);
    at::Tensor atOut = at::clamp(atInput, atMin, atMax);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        diopiConstTensorHandle_t max) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atMax = camb::aten::buildATen(max);
    at::clamp_max_(atInput, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atMax = camb::aten::buildATen(max);
    at::Tensor atOut = at::clamp_max(atInput, atMax);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        diopiConstTensorHandle_t min) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atMin = camb::aten::buildATen(min);
    at::clamp_(atInput, atMin);
    return diopiSuccess;
}

diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atMin = camb::aten::buildATen(min);
    at::Tensor atOut = at::clamp(atInput, atMin);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}
#endif

diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiScalar_t* min) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atMin = camb::aten::buildAtScalar(min);
    at::clamp_(atInput, atMin);
    return diopiSuccess;
}

diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Scalar atMin = camb::aten::buildAtScalar(min);
    at::Tensor atOut = at::clamp(atInput, atMin);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* value) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    auto atValue = camb::aten::buildAtScalar(value);
    at::fill_(atInput, atValue);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t output_size) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    auto atOutSize = camb::aten::buildAtIntArray(output_size);
    // adaptive_avg_pool2d_out can not have the same precision as adaptive_avg_pool2d
    at::Tensor atOut = at::adaptive_avg_pool2d(atInput, atOutSize);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t output_size) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    auto atOutSize = camb::aten::buildAtIntArray(output_size);
    // cnnl use nhwc memory format, return local indices
    auto atOuts = cnnl_adaptive_max_pool2d(atInput, atOutSize);
    auto atOut = (std::get<0>(atOuts)).contiguous();
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = camb::aten::buildATen(out);
    at::Tensor atIndices = camb::aten::buildATen(indices);
    auto atOutSize = camb::aten::buildAtIntArray(output_size);
    // cnnl use nhwc memory format, return local indices
    cnnl_adaptive_max_pool2d_out(atOut, atIndices, atInput, atOutSize);
    if (atIndices.scalar_type() == at::ScalarType::Long) {
        atIndices = atIndices.contiguous();
        camb::aten::updateATen2Tensor(ctx, atIndices, indices);
    }
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
        diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atGradInput = camb::aten::buildATen(grad_input);
    at::Tensor atGradOutput = camb::aten::buildATen(grad_output);
    at::Tensor atIndices = camb::aten::buildATen(indices);
    cnnl_adaptive_max_pool2d_backward_out(atGradInput, atGradOutput, atInput, atIndices);
    atGradInput = atGradInput.contiguous();
    camb::aten::updateATen2Tensor(ctx, atGradInput, grad_input);
    return diopiSuccess;
}

diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
        bool count_include_pad, const int64_t* divisor_override) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::IntArrayRef atKernelSize = camb::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = camb::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = camb::aten::buildAtIntArray(padding);
    c10::optional<int64_t> atDivisorOverride = divisor_override ? c10::optional<int64_t>(*divisor_override) : c10::nullopt;
    if (3 == atInput.dim()) {
       atInput = atInput.unsqueeze(0);
    }
    if (4 == atInput.dim()) {
        // cnnl has not out version and supports only 4d Tensor, atDivisorOverride is ununsed
        camb::aten::invokeATenFuncRet(ctx, at::avg_pool2d, out, atInput, atKernelSize, atStride,
            atPadding, ceil_mode, count_include_pad, atDivisorOverride);
    } else {
        NOT_SUPPORTED("dim < 3");
    }
    return diopiSuccess;
}

diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                    diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                    diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                    bool count_include_pad, const int64_t* divisor_override) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    at::IntArrayRef atKernelSize = camb::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = camb::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = camb::aten::buildAtIntArray(padding);
    c10::optional<int64_t> atDivisorOverride = divisor_override ? c10::optional<int64_t>(*divisor_override) : c10::nullopt;
    if (3 == atInput.dim()) {
        atInput = atInput.unsqueeze(0);
        atGradOutput = atGradOutput.unsqueeze(0);
    }
    if (4 == atInput.dim()) {
        // cnnl has not out version and supports only 4d Tensor
        camb::aten::invokeATenFuncRet(ctx, at::avg_pool2d_backward, grad_input, atGradOutput, atInput, atKernelSize,
                                      atStride, atPadding, ceil_mode, count_include_pad, atDivisorOverride);
    } else {
        NOT_SUPPORTED("dim < 3");
    }
    return diopiSuccess;
}

diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask,
        diopiConstTensorHandle_t input, double p, bool train) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    if (train) {
        diopi_tensor_list vecOut = {out, mask};
        camb::aten::invokeATenFuncRet(ctx, at::_fused_dropout, vecOut, atInput, 1 - p, c10::nullopt);
    }
    return diopiSuccess;
}

diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    if (train) {
        auto atOuts = at::_fused_dropout(atInput, 1 - p, c10::nullopt);
        camb::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), input);
        camb::aten::updateATen2Tensor(ctx, std::get<1>(atOuts), mask);
    }
    return diopiSuccess;
}

diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atTarget = camb::aten::buildATen(target);
    // TODO(huqingqing): try to use cnnl_mse_loss_internal with out argument
    camb::aten::invokeATenFuncRet(ctx, at::mse_loss, out, atInput, atTarget, reduction);
    return diopiSuccess;
}

diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs,
        diopiConstTensorHandle_t targets, float alpha, float gamma, diopiReduction_t reduction) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(inputs);
    at::Tensor atTarget = camb::aten::buildATen(targets);
    at::Tensor atP = at::sigmoid(atInput);
    at::Tensor atTerm1 = at::pow(1 - atP, gamma) * at::log(atP);
    at::Tensor atTerm2 = at::pow(atP, gamma) * at::log(1 - atP);
    at::Tensor atRes = -atTarget * atTerm1 * alpha - (1 - atTarget) * atTerm2 * (1- alpha);
    if (reduction == 0) {
        camb::aten::updateATen2Tensor(ctx, atRes, out);
    } else if (reduction == 1) {
        at::Tensor atOut = at::mean(atRes);
        camb::aten::updateATen2Tensor(ctx, atOut, out);
    } else if (reduction == 2) {
        at::Tensor atOut = at::sum(atRes);
        camb::aten::updateATen2Tensor(ctx, atOut, out);
    } else {
        NOT_SUPPORTED("sigmoid reduction type");
        return diopiErrorOccurred;
    }
    return diopiSuccess;
}

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
        diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
        diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean,
        diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atWeight = camb::aten::buildATen(weight);
    at::Tensor atBias = camb::aten::buildATen(bias);
    at::Tensor atRunningMean = camb::aten::buildATen(running_mean);
    at::Tensor atRunningVar = camb::aten::buildATen(running_var);
    diopi_tensor_list vecOut = {out, save_mean, save_invstd};
    // cnnl has not out version
    camb::aten::invokeATenFuncRet(ctx, at::native_batch_norm, vecOut, atInput, atWeight, atBias,
        atRunningMean, atRunningVar, training, momentum, eps);
    return diopiSuccess;
}

diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input,
        int64_t dim, int64_t start, int64_t end, int64_t step) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atOut = at::slice(atInput, dim, start, end, step).contiguous();
    camb::aten::updateATen2Tensor(ctx, atOut, null_out);
    return diopiSuccess;
}

// TOCHECK
diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t* indices, int64_t nums) {
    DIOPI_CHECK(out != nullptr && indices != nullptr,
                "Not supported: out or indices is nullptr");
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    c10::List<c10::optional<at::Tensor>> vecIdx;
    vecIdx.reserve(nums);
    for (size_t i = 0; i < nums; ++i) {
        if (indices[i] == nullptr) {
            vecIdx.emplace_back(c10::nullopt);
        } else {
            at::Tensor atIndex = camb::aten::buildATen(indices[i]);
            vecIdx.emplace_back(atIndex);
        }
    }
    // at::Tensor atOut = at::index(atInput, vecIdx).contiguous();
    // camb::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
        diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atTarget = camb::aten::buildATen(target);
    c10::optional<at::Tensor> atWeight = weight
        ? c10::optional<at::Tensor>(camb::aten::buildATen(weight))
        : c10::nullopt;
    c10::optional<at::Tensor> atPosWeight = pos_weight
        ? c10::optional<at::Tensor>(camb::aten::buildATen(pos_weight))
        : c10::nullopt;

    // camb::aten::invokeATenFuncRet(ctx, at::binary_cross_entropy_with_logits, out, atInput, atTarget, atWeight,
    //       atPosWeight, reduction);
    return diopiSuccess;
}

diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                           const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atMin = camb::aten::buildAtScalar(min_val);
    auto atMax = camb::aten::buildAtScalar(max_val);
    auto atOut = camb::aten::buildATen(out);
    at::hardtanh_out(atOut, atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val,
                              const diopiScalar_t* max_val) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atMin = camb::aten::buildAtScalar(min_val);
    auto atMax = camb::aten::buildAtScalar(max_val);
    camb::aten::invokeATenFuncInp(ctx, at::hardtanh_, atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                            const diopiScalar_t* threshold, const diopiScalar_t* value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atThreshold = camb::aten::buildAtScalar(threshold);
    auto atValue = camb::aten::buildAtScalar(value);
    auto atOut = camb::aten::buildATen(out);
    at::threshold_out(atOut, atInput, atThreshold, atValue);
    return diopiSuccess;
}

diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold,
                               const diopiScalar_t* value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atThreshold = camb::aten::buildAtScalar(threshold);
    auto atValue = camb::aten::buildAtScalar(value);
    camb::aten::invokeATenFuncInp(ctx, at::threshold_, atInput, atThreshold, atValue);
    return diopiSuccess;
}

diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                       diopiConstTensorHandle_t input, const char* approximate) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOut = camb::aten::buildATen(out);
    cnnl_activation_internal(atOut, atInput, CNNL_ACTIVATION_GELU);
    return diopiSuccess;
}

diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                          diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignore_index) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    return camb::aten::nll_loss_internal(ctx, out, atInput, target, weight, reduction, ignore_index);
}

diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
        diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
    camb::aten::setCurCtx(ctx);
    at::IntArrayRef atInputSizes = camb::aten::buildAtIntArray(input_sizes);
    at::Tensor atGradOutput = camb::aten::buildATen(grad_output);
    // camb::aten::invokeATenFuncRet(ctx, at::slice_backward, grad_input, atGradOutput, atInputSizes, dim, start, end, step);
    return diopiSuccess;
}

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input,
        diopiConstTensorHandle_t* indices, int64_t nums, diopiConstTensorHandle_t grad) {
    DIOPI_CHECK_PTR(indices);
    camb::aten::setCurCtx(ctx);
    at::Tensor atZerosInput = camb::aten::buildATen(zeros_like_input);
    at::Tensor atGrad = camb::aten::buildATen(grad);
    c10::List<c10::optional<at::Tensor>> vecIdx;
    vecIdx.reserve(nums);
    for (size_t i = 0; i < nums; ++i) {
        if (indices[i] == nullptr) {
            vecIdx.emplace_back(c10::nullopt);
        } else {
            at::Tensor atIndex = camb::aten::buildATen(indices[i]);
            vecIdx.emplace_back(atIndex);
        }
    }
    // camb::aten::invokeATenFuncRet(ctx, at::_index_put_impl_, grad_input, atZerosInput, vecIdx, atGrad, true, true);
    return diopiSuccess;
}

diopiError_t diopiSigmoidFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
        diopiTensorHandle_t grad_input, float gamma, float alpha, diopiReduction_t reduction) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atTarget = camb::aten::buildATen(target);
    at::Tensor atGradOutput = camb::aten::buildATen(grad_output);
    at::Tensor atP = at::sigmoid(atInput);
    // (1-p)**g * (1 - p - g*p*log(p))
    at::Tensor atTerm1 = at::pow(1 - atP, gamma) * (1 - atP - gamma * atP * at::log(at::clamp_min(atP, FLT_MIN)));
    // (p**g) * (g*(1-p)*log(1-p) - p)
    at::Tensor atTerm2 = at::pow(atP, gamma) * (gamma * (1 - atP) * at::log(at::clamp_min(1 - atP, FLT_MIN)) - atP);
    at::Tensor atRes = - atTarget * atTerm1 * alpha - (1 - atTarget) * atTerm2 * (1- alpha);
    atGradOutput *= atRes;
    camb::aten::updateATen2Tensor(ctx, atGradOutput, grad_input);
    return diopiSuccess;
}

diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
        diopiConstTensorHandle_t rois, double spatialScale, int64_t pooledHeight, int64_t pooledWidth, int64_t batchSize,
        int64_t channels, int64_t height, int64_t width, int64_t samplingRatio, bool aligned) {
    camb::aten::setCurCtx(ctx);
    auto atGrad = camb::aten::buildATen(grad);
    auto atRois = camb::aten::buildATen(rois);
    // auto atOut = vision::ops::roi_align_backward_kernel(atGrad, atRois, spatialScale,
      //  pooledHeight, pooledWidth, batchSize, channels, height, width, samplingRatio, aligned);
    // camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
        diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
        diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atGrad = camb::aten::buildATen(grad_output);
    auto atWeight = camb::aten::buildATen(weight);
    auto atStride = camb::aten::buildAtIntArray(stride);
    auto atPadding = camb::aten::buildAtIntArray(padding);
    auto atDilation = camb::aten::buildAtIntArray(dilation);
    auto grad_input_mask = std::array<bool, 3>{true, true, true};
#if CNRT_VERSION < 60002
    if (dilation.data[0] != 1 || dilation.data[1] != 1) {
        NOT_SUPPORTED("dilation !=1");
        return diopiErrorOccurred;
    }
#endif
    auto atOuts = cnnl_convolution_backward_overrideable(atGrad, atInput, atWeight, atStride, atPadding, atDilation,
        false, at::IntArrayRef(0), groups, grad_input_mask);

    camb::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), grad_input);
    camb::aten::updateATen2Tensor(ctx, std::get<1>(atOuts), grad_weight);
    if (bias_sizes != nullptr && grad3 != nullptr) {
        camb::aten::updateATen2Tensor(ctx, std::get<2>(atOuts), grad3);
    }
    return diopiSuccess;
}

diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                    diopiConstTensorHandle_t indices, int64_t numWeights, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    camb::aten::setCurCtx(ctx);
    auto atGrad = camb::aten::buildATen(grad);
    auto atIndices = camb::aten::buildATen(indices);
    camb::aten::invokeATenFuncRet(ctx, at::embedding_backward, out, atGrad, atIndices, numWeights, paddingIdx, scaleGradByFreq, sparse);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                            diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput  = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncRet(ctx, at::_adaptive_avg_pool2d_backward, grad_input, atGradOutput, atInput);
    return diopiSuccess;
}

diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput  = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    auto atSlope = camb::aten::buildAtScalar(negative_slope);
    camb::aten::invokeATenFuncRet(ctx, at::leaky_relu_backward, grad_input, atGradOutput, atInput, atSlope, input_is_result);
    return diopiSuccess;
}

diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                   diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    auto atMin = camb::aten::buildAtScalar(min_val);
    auto atMax = camb::aten::buildAtScalar(max_val);
    camb::aten::invokeATenFuncRet(ctx, at::hardtanh_backward, grad_input, atGradOutput, atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                               diopiConstTensorHandle_t input, const char* approximate) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atGradInput = camb::aten::buildATen(grad_input);
    auto atInput = camb::aten::buildATen(input);
    cnnl_activation_backward_internal(atGradInput, atInput, atGradOutput, CNNL_ACTIVATION_GELU);
    return diopiSuccess;
}

diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    auto atTarget = camb::aten::buildATen(target);
    camb::aten::invokeATenFuncRet(ctx, at::mse_loss_backward, grad_input, atGradOutput, atInput, atTarget, reduction);
    return diopiSuccess;
}

diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                               diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    auto atGradInput = camb::aten::buildATen(grad_input);
    at::tanh_backward_out(atGradInput, atGradOutput, atInput);
    return diopiSuccess;
}

diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad,
                                      diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index) {
    camb::aten::setCurCtx(ctx);
    auto atGrad = camb::aten::buildATen(grad);
    at::IntArrayRef atInputSize = camb::aten::buildAtIntArray(input_sizes);
    auto atIndex = camb::aten::buildATen(index);
    // camb::aten::invokeATenFuncRet(ctx, at::index_select_backward, grad_input, atGrad, atInputSize, dim, atIndex);
    return diopiSuccess;
}

diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                 diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t index) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    at::IntArrayRef atInputSize = camb::aten::buildAtIntArray(input_sizes);
    // camb::aten::invokeATenFuncRet(ctx, at::select_backward, grad_input, atGradOutput, atInputSize, dim, index);
    return diopiSuccess;
}

diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atOutput = camb::aten::buildATen(output);
    // TODO(huqingqing): use default type instead
    camb::aten::invokeATenFuncRet(ctx, at::_softmax_backward_data, grad_input, atGradOutput, atOutput, dim, atOutput);
    return diopiSuccess;
}

diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                     diopiConstTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atOutput = camb::aten::buildATen(output);
    // TODO(huqingqing): use default type instead
    camb::aten::invokeATenFuncRet(ctx, at::_log_softmax_backward_data, grad_input, atGradOutput, atOutput, dim, atOutput);
    return diopiSuccess;
}

diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                  diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atOutput = camb::aten::buildATen(output);
    auto atGradInput = camb::aten::buildATen(grad_input);
    at::sigmoid_backward_out(atGradInput, atGradOutput, atOutput);
    return diopiSuccess;
}

diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    auto atThreshold = camb::aten::buildAtScalar(threshold);
    camb::aten::invokeATenFuncRet(ctx, at::threshold_backward, grad_input, atGradOutput, atInput, atThreshold);
    return diopiSuccess;
}

diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                  diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    auto atTarget = camb::aten::buildATen(target);
    c10::optional<at::Tensor> atWeight = weight
        ? c10::optional<at::Tensor>(camb::aten::buildATen(weight))
        : c10::nullopt;
    c10::optional<at::Tensor> atPosWeight = pos_weight
        ? c10::optional<at::Tensor>(camb::aten::buildATen(pos_weight))
        : c10::nullopt;

    // camb::aten::invokeATenFuncRet(ctx, at::binary_cross_entropy_with_logits_backward, grad_input, atGradOutput, atInput, atTarget, atWeight,
    //                             atPosWeight, reduction);
    return diopiSuccess;
                                                  }

diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                  diopiReduction_t reduction, int64_t ignore_index) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    return camb::aten::nll_loss_bp_internal(ctx, grad_input, grad_output, atInput, target,
                                            weight, reduction, ignore_index);
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                    diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    at::IntArrayRef atKernelSize = camb::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = camb::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = camb::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = camb::aten::buildAtIntArray(dilation);
    at::Tensor atIndices = camb::aten::buildATen(indices);

    if (3 == atInput.dim()) {
       atInput = atInput.unsqueeze(0);
       atGradOutput = atGradOutput.unsqueeze(0);
    }
    // cnnl has not out version and supports only 4d Tensor
    // not support dilation != 1
    if (4 != atInput.dim()) {
        NOT_SUPPORTED("dim < 3");
        return diopiErrorOccurred;
    }

    if (dilation.len != 0 && (dilation.data[0] != 1 || dilation.data[1] != 1)) {
        NOT_SUPPORTED("dilation != 1");
        return diopiErrorOccurred;
    }

    auto atGradInput = cnnl_max_pool2d_with_indices_backward(atGradOutput, atInput, atKernelSize,
        atStride, atPadding, atDilation, ceil_mode, atIndices);
    atGradInput = atGradInput.contiguous();
    camb::aten::updateATen2Tensor(ctx, atGradInput, grad_input);
    return diopiSuccess;
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
        diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
        diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean,
        diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    auto atWeight = camb::aten::buildATen(weight);
    at::Tensor atRunningMean = camb::aten::buildATen(running_mean);
    at::Tensor atRunningVar = camb::aten::buildATen(running_var);
    at::Tensor atSaveMean = camb::aten::buildATen(save_mean);
    at::Tensor atSaveVar = camb::aten::buildATen(save_invstd);
    diopi_tensor_list vecOut = {grad_input, grad_weight, grad_bias};
    auto grad_input_mask = std::array<bool, 3>{true, true, true};
    camb::aten::invokeATenFuncRet(ctx, at::native_batch_norm_backward, vecOut, atGradOutput,  atInput, atWeight, atRunningMean,
                                  atRunningVar, atSaveMean, atSaveVar, training, eps, grad_input_mask);
    return diopiSuccess;
}

diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start,
        const diopiScalar_t* end, const diopiScalar_t* step) {
    camb::aten::setCurCtx(ctx);
    auto atOut = camb::aten::buildATen(out);
    auto atStart = camb::aten::buildAtScalar(start);
    auto atEnd = camb::aten::buildAtScalar(end);
    auto atStep = camb::aten::buildAtScalar(step);
    if (atOut.scalar_type() == at::ScalarType::Long) {
        auto atOutCpu = at::empty({atOut.numel()}, atOut.options().device(at::Device::Type::CPU)).reshape_as(atOut);
        at::arange_out(atOutCpu, atStart, atEnd, atStep);
        camb::aten::updateATen2Tensor(ctx, atOutCpu, out);
    } else {
        at::arange_out(atOut, atStart, atEnd, atStep);
    }
    return diopiSuccess;
}

diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {
    camb::aten::setCurCtx(ctx);
    auto atOut = camb::aten::buildATen(out);
    auto atOutCpu = at::empty({n}, atOut.options().device(at::Device::Type::CPU));
    at::randperm_out(atOutCpu, n, c10::nullopt);
    camb::aten::updateATen2Tensor(ctx, atOutCpu, out);
    return diopiSuccess;
}

diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx) {
    camb::aten::setCurCtx(ctx);
    auto atOut = camb::aten::buildATen(inout);
    auto atOutput = at::native::uniform_(atOut, from, to, c10::nullopt);
    return diopiSuccess;
}

diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx) {
    camb::aten::setCurCtx(ctx);
    auto atOut = camb::aten::buildATen(inout);
    if (to == nullptr) {
        auto atOutput = cnnl_random_(atOut, from, c10::nullopt, c10::nullopt);
    } else {
        auto atOutput = cnnl_random_(atOut, from, *to, c10::nullopt);
    }
    return diopiSuccess;
}

diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t idx) {
    camb::aten::setCurCtx(ctx);
    auto atOut = camb::aten::buildATen(inout);
    auto atOutput = at::bernoulli(atOut, c10::nullopt);
    return diopiSuccess;
}

diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t idx) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOut = camb::aten::buildATen(out);
    auto atOutput = at::bernoulli_out(atOut, atInput, c10::nullopt);
    return diopiSuccess;
}

diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, int64_t idx) {
    camb::aten::setCurCtx(ctx);
    auto atOut = camb::aten::buildATen(out);
    auto atOutput = at::bernoulli(atOut, p, c10::nullopt);
    return diopiSuccess;
}

diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                             diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atMask = camb::aten::buildATen(mask);
    auto atValue = camb::aten::buildATen(value);
    auto atOut = at::masked_fill(atInput, atMask, atValue);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atMask = camb::aten::buildATen(mask);
    auto atValue = camb::aten::buildATen(value);
    cnnl_masked_fill_(atInput, atMask, atValue);
    return diopiSuccess;
}

diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                   diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atMask = camb::aten::buildATen(mask);
    auto atValue = camb::aten::buildAtScalar(value);
    auto atOut = at::masked_fill(atInput, atMask, atValue);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask,
                                      const diopiScalar_t* value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atMask = camb::aten::buildATen(mask);
    auto atValue = camb::aten::buildAtScalar(value);
    cnnl_masked_fill_(atInput, atMask, atValue);
    return diopiSuccess;
}

diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad,
                        diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq,
                        float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atGrad = camb::aten::buildATen(grad);
    auto atExpAvg = camb::aten::buildATen(exp_avg);
    auto atExpAvgSq = camb::aten::buildATen(exp_avg_sq);
    auto atMaxExpAvgSq = camb::aten::buildATen(max_exp_avg_sq);

    atInput = atInput.mul(1 - lr * weight_decay);
    auto& param = atInput;
    auto grad_d = atGrad.data();
    auto bias_correction1 = 1 - pow(beta1, step);
    auto bias_correction2 = 1 - pow(beta2, step);
    atExpAvg.mul_(beta1).add_(grad_d, 1 - beta1);
    atExpAvgSq.mul_(beta2).addcmul_(grad_d, grad_d, 1- beta2);

    at::Tensor denom;
    if (amsgrad) {
        // at::maximum_out(atMaxExpAvgSq, atMaxExpAvgSq, atExpAvgSq);
        denom = (atMaxExpAvgSq.sqrt() / sqrt(bias_correction2)).add_(eps);
    } else {
        denom = (atExpAvgSq.sqrt() / sqrt(bias_correction2)).add_(eps);
    }
    auto stepSize = lr / bias_correction1;
    param = param.addcdiv(atExpAvg, denom, -1 * stepSize);

    camb::aten::updateATen2Tensor(ctx, atInput, input);
    camb::aten::updateATen2Tensor(ctx, atGrad, grad);
    camb::aten::updateATen2Tensor(ctx, atExpAvg, exp_avg);
    camb::aten::updateATen2Tensor(ctx, atExpAvgSq, exp_avg_sq);
    camb::aten::updateATen2Tensor(ctx, atMaxExpAvgSq, max_exp_avg_sq);
    return diopiSuccess;
}

diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad,
                       diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq,
                       float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atGrad = camb::aten::buildATen(grad);
    auto atExpAvg = camb::aten::buildATen(exp_avg);
    auto atExpAvgSq = camb::aten::buildATen(exp_avg_sq);
    auto atMaxExpAvgSq = camb::aten::buildATen(max_exp_avg_sq);

    auto& param = atInput;
    auto grad_d = atGrad.data();
    auto bias_correction1 = 1 - pow(beta1, step);
    auto bias_correction2 = 1 - pow(beta2, step);

    if (weight_decay != 0) {
        grad_d = grad_d.add(param, weight_decay);
    }
    atExpAvg.mul_(beta1).add_(grad_d, 1 - beta1);
    atExpAvgSq.mul_(beta2).addcmul_(grad_d, grad_d.conj(), 1- beta2);

    at::Tensor denom;
    if (amsgrad) {
        // at::maximum_out(atMaxExpAvgSq, atMaxExpAvgSq, atExpAvgSq);
        denom = (atMaxExpAvgSq.sqrt() / sqrt(bias_correction2)).add_(eps);
    } else {
        denom = (atExpAvgSq.sqrt() / sqrt(bias_correction2)).add_(eps);
    }
    auto stepSize = lr / bias_correction1;
    param = param.addcdiv(atExpAvg, denom, -1 * stepSize);

    camb::aten::updateATen2Tensor(ctx, atInput, input);
    camb::aten::updateATen2Tensor(ctx, atGrad, grad);
    camb::aten::updateATen2Tensor(ctx, atExpAvg, exp_avg);
    camb::aten::updateATen2Tensor(ctx, atExpAvgSq, exp_avg_sq);
    camb::aten::updateATen2Tensor(ctx, atMaxExpAvgSq, max_exp_avg_sq);
    return diopiSuccess;
}

diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad,
                           diopiTensorHandle_t square_avg, diopiTensorHandle_t acc_delta, float lr,
                           float rho, float eps, float weight_decay) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atGrad = camb::aten::buildATen(grad);
    auto atSquareAvg = camb::aten::buildATen(square_avg);
    auto atAccDelta = camb::aten::buildATen(acc_delta);

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
    camb::aten::updateATen2Tensor(ctx, atInput, input);
    camb::aten::updateATen2Tensor(ctx, atGrad, grad);
    camb::aten::updateATen2Tensor(ctx, atSquareAvg, square_avg);
    camb::aten::updateATen2Tensor(ctx, atAccDelta, acc_delta);
    return diopiSuccess;
}

diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
                                  diopiSize_t padding, diopiSize_t output_padding, int64_t groups, diopiSize_t dilation) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atWeight = camb::aten::buildATen(weight);
    auto atBias = camb::aten::buildATen(bias);
    auto atStride = camb::aten::buildAtIntArray(stride);
    auto atPadding = camb::aten::buildAtIntArray(padding);
    auto atOutputPadding = camb::aten::buildAtIntArray(output_padding);
    auto atDilation = camb::aten::buildAtIntArray(dilation);
    camb::aten::invokeATenFuncRet(ctx, at::conv_transpose2d, out,
        atInput, atWeight, atBias, atStride, atPadding, atOutputPadding, groups, atDilation);
    return diopiSuccess;
}

diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                         int64_t dim, diopiDtype_t dtype) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOut = at::cumsum(atInput, dim);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2,
                                  double p, const int64_t* compute_mode) {
    camb::aten::setCurCtx(ctx);
    auto atInput1 = camb::aten::buildATen(input1);
    auto atInput2 = camb::aten::buildATen(input2);
    c10::optional<int64_t> atComputMode = compute_mode ? c10::optional<int64_t>(*compute_mode) : c10::nullopt;
    camb::aten::invokeATenFuncRet(ctx, at::cdist, out, atInput1, atInput2, p, atComputMode);
    return diopiSuccess;
}

diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput1 = camb::aten::buildATen(input1);
    auto atInput2 = camb::aten::buildATen(input2);
    auto atCdist = camb::aten::buildATen(cdist);
    camb::aten::invokeATenFuncRet(ctx, at::_cdist_backward, grad_input, atGradOutput, atInput1, atInput2, p, atCdist);
    return diopiSuccess;
}

diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncRet(ctx, at::reciprocal, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    at::reciprocal_(atInput);
    return diopiSuccess;
}

diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    camb::aten::invokeATenFuncRet(ctx, at::bitwise_not, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    c10::optional<int64_t> atDim = dim ? c10::optional<int64_t>(*dim) : c10::nullopt;
    camb::aten::invokeATenFuncRet(ctx, at::argmax, out, atInput, atDim, keepdim);
    return diopiSuccess;
}

diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                               diopiReduction_t reduction, double beta) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::Tensor atTarget = camb::aten::buildATen(target);
    // camb::aten::invokeATenFuncRet(ctx, at::smooth_l1_loss, out, atInput, atTarget, reduction, beta);
    return diopiSuccess;
}

diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                       diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction, double beta) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    auto atTarget = camb::aten::buildATen(target);
    // camb::aten::invokeATenFuncRet(ctx, at::smooth_l1_loss_backward, grad_input, atGradOutput, atInput, atTarget, reduction, beta);
    return diopiSuccess;
}

diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOther = camb::aten::buildATen(other);
    // camb::aten::invokeATenFuncRet(ctx, at::maximum, out, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atOther = camb::aten::buildATen(other);
    // camb::aten::invokeATenFuncRet(ctx, at::minimum, out, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atMat2 = camb::aten::buildATen(mat2);
    camb::aten::invokeATenFuncRet(ctx, at::mm, out, atInput, atMat2);
    return diopiSuccess;
}

diopiError_t diopiConvolution3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
        diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atWeight = camb::aten::buildATen(weight);
    auto atBias = camb::aten::buildATen(bias);
    auto atStride = camb::aten::buildAtIntArray(stride);
    auto atPadding = camb::aten::buildAtIntArray(padding);
    auto atDilation = camb::aten::buildAtIntArray(dilation);
    camb::aten::invokeATenFuncRet(ctx, at::convolution, out,
        atInput, atWeight, atBias, atStride, atPadding, atDilation, false, at::IntArrayRef(0), groups);
    return diopiSuccess;
}

diopiError_t diopiConvolution3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
        diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
        diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atGrad = camb::aten::buildATen(grad_output);
    auto atWeight = camb::aten::buildATen(weight);
    auto atStride = camb::aten::buildAtIntArray(stride);
    auto atPadding = camb::aten::buildAtIntArray(padding);
    auto atDilation = camb::aten::buildAtIntArray(dilation);
    diopi_tensor_list vecOut = {grad_input, grad_weight};
    auto grad_input_mask = std::array<bool, 2>{true, true};

    // camb::aten::invokeATenFuncRet(ctx, at::cudnn_convolution_backward, vecOut, atInput, atGrad,
     //   atWeight, atPadding, atStride, atDilation, groups, false, false, false, grad_input_mask);

    if (bias_sizes != nullptr && grad3 != nullptr) {
        auto atBias = camb::aten::buildATen(grad3);
        at::Tensor atTmp = atGrad;
        int64_t size = atGrad.dim();
        while (atBias.dim() != size) {
            atTmp = at::sum(atTmp, -1, false);
            size -= 1;
        }
        if (atBias.size(0) !=  atTmp.size(0)) {
            atTmp = at::sum(atTmp, -1, false);
        }
        camb::aten::updateATen2Tensor(ctx, atTmp, grad3);
    }
    return diopiSuccess;
}

diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atSize = camb::aten::buildAtIntArray(size);
    auto atOut = at::native::expand(atInput, atSize).clone();
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dimension, int64_t size, int64_t step) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    // must use contiguous rather than clone in this case
    auto atOut = at::native::unfold(atInput, dimension, size, step).contiguous();
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
        diopiSize_t input_sizes, int64_t dimension, int64_t size, int64_t step) {
    camb::aten::setCurCtx(ctx);
    auto atGrad = camb::aten::buildATen(grad_output);
    auto atInputSize = camb::aten::buildAtIntArray(input_sizes);
    camb::aten::invokeATenFuncRet(ctx, at::unfold_backward, grad_input, atGrad, atInputSize, dimension, size, step);
    return diopiSuccess;
}

diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out,
                               diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    DIOPI_CHECK_PTR(out);
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atMask = camb::aten::buildATen(mask);
    auto atOut = at::masked_select(atInput, atMask);
    camb::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                       diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    auto atMask = camb::aten::buildATen(mask);
    // camb::aten::invokeATenFuncRet(ctx, at::masked_select_backward, grad_input, atGradOutput, atInput, atMask);
    return diopiSuccess;
}

// TOCHECK
diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                  int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atIndex = camb::aten::buildATen(index);
    auto atValue = camb::aten::buildAtScalar(value);
    auto atOut = at::index_fill(atInput, dim, atIndex, atValue);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                            int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atIndex = camb::aten::buildATen(index);
    auto atValue = camb::aten::buildATen(value);
    auto atOut = at::index_fill(atInput, dim, atIndex, atValue);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input,
                                     int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atIndex = camb::aten::buildATen(index);
    auto atValue = camb::aten::buildAtScalar(value);
    atInput.index_fill_(dim, atIndex, atValue);
    return diopiSuccess;
}

diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input,
                               int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
    camb::aten::setCurCtx(ctx);
    auto atInput = camb::aten::buildATen(input);
    auto atIndex = camb::aten::buildATen(index);
    auto atValue = camb::aten::buildATen(value);
    atInput.index_fill_(dim, atIndex, atValue);
    return diopiSuccess;
}

diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    camb::aten::setCurCtx(ctx);
    auto atStart = camb::aten::buildAtScalar(start);
    auto atEnd = camb::aten::buildAtScalar(end);
    c10::optional<int64_t> atStep(steps);
    at::Tensor atOut = camb::aten::buildATen(out);
    // linspace_out(atOut, atStart, atEnd, atStep);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
        diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    camb::aten::setCurCtx(ctx);
    auto atGradOutput = camb::aten::buildATen(grad_output);
    auto atInput = camb::aten::buildATen(input);
    auto atWeight = camb::aten::buildATen(weight);
    auto atGradInput = at::matmul(atGradOutput, atWeight);

    int64_t dims = atInput.dim();
    auto atGradWeight = at::matmul(atInput.transpose(dims-2, dims-1), atGradOutput);
    atGradWeight = atGradWeight.transpose(dims-2, dims-1);
    if (dims > 2) {
        std::vector<int64_t> sumDim;
        for (int i = 0; i < dims-2; ++i) {
            sumDim.push_back(i);
        }
        at::IntArrayRef atSumDim(sumDim.data(), sumDim.size());
        atGradWeight = at::sum(atGradWeight, atSumDim);
    }
    camb::aten::updateATen2Tensor(ctx, atGradInput, grad_input);
    camb::aten::updateATen2Tensor(ctx, atGradWeight, grad_weight);

    if (grad_bias) {
        std::vector<int64_t> sumDim;
        for (int i = 0; i < dims-1; ++i) {
            sumDim.push_back(i);
        }
        at::IntArrayRef atSumDim(sumDim.data(), sumDim.size());
        auto atGradBias = at::sum(atGradOutput, atSumDim);
        camb::aten::updateATen2Tensor(ctx, atGradBias, grad_bias);
    }
    return diopiSuccess;
}

diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src,
                             diopiConstTensorHandle_t index, const char* reduce) {
    camb::aten::setCurCtx(ctx);
    return diopiSuccess;
}

diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                   diopiConstTensorHandle_t index, const char* reduce) {
    camb::aten::setCurCtx(ctx);
    return diopiSuccess;
}

diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                          diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
    camb::aten::setCurCtx(ctx);
    return diopiSuccess;
}

diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
    camb::aten::setCurCtx(ctx);
    return diopiSuccess;
}


diopiError_t diopiPad(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiSize_t pad, const char* mode, double* value) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    at::IntArrayRef atPad = camb::aten::buildAtIntArray(pad);
    torch::nn::functional::PadFuncOptions::mode_t pad_mode;
    double atValue = 0;
    if (strcmp(mode, "constant") == 0) {
        DIOPI_CHECK_PTR(value);
        atValue = *value;
        pad_mode = torch::kConstant;
    } else if (strcmp(mode, "reflect") == 0) {
        pad_mode = torch::kReflect;
    } else if (strcmp(mode, "replicate") == 0) {
        if (3 == atInput.dim()) {
            NOT_SUPPORTED("MLU doesn't support replication_pad1d for 3D tensors currently");
            return diopiErrorOccurred;
        }
        pad_mode = torch::kReplicate;
    } else if (strcmp(mode, "circular") == 0) {
        pad_mode = torch::kCircular;
    } else {
        NOT_SUPPORTED("padding mode");
        return diopiErrorOccurred;
    }
    at::Tensor atOut = torch::nn::functional::detail::pad(atInput, atPad, pad_mode, atValue);
    camb::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    camb::aten::setCurCtx(ctx);
    at::Tensor atInput = camb::aten::buildATen(input);
    auto atDims = camb::aten::buildAtIntArray(dims);
    camb::aten::invokeATenFuncRet(ctx, at::native::permute, out, atInput, atDims);

    return diopiSuccess;
}


}  // extern "C"
