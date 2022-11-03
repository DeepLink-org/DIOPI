
#include <diopi/functions.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <math.h>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include "helper.hpp"
#include "vision_kernel.h"

#define FLT_MIN  __FLT_MIN__

extern "C" {

static const char* name = "CudaDevice"; 
static char version[1024] = {0};

const char* diopiGetVendorName() {
    return name;  
}

const char* diopiGetImplVersion() {
    if (strlen(version) == 0) {
        const char* diopiVersion = diopiGetVersion();
        sprintf(version, "Cuda Version: %d; Cudnn Version: %d; %s",
                CUDART_VERSION, CUDNN_VERSION, diopiVersion);   
    }
    return version; 
}

const char* diopiGetLastErrorString() {
    return impl::aten::_get_last_error_string();
}

diopiError_t diopiRelu(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::relu, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::relu_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        const diopiScalar_t* negative_slope) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atSlope = impl::aten::buildAtScalar(negative_slope);
    impl::aten::invokeATenFuncRet(ctx, at::leaky_relu, out, atInput, atSlope);
    return diopiSuccess;
}

diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx,
        diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atSlope = impl::aten::buildAtScalar(negative_slope);
    impl::aten::invokeATenFuncInp(ctx, at::leaky_relu_, atInput, atSlope);
    return diopiSuccess;
}

diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
        diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    bool atCeilMode = ceil_mode;
    impl::aten::invokeATenFuncRet(ctx, at::max_pool2d, out,
        atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
        diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    bool atCeilMode = ceil_mode;
    at::max_pool2d_with_indices_out(atOut, atIndices, atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

/**
 * @brief
 * @param rounding_mode supported in pytorch>=1.8
 */
diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
#if TORCH_MM_VERSION < TORCH_1_8_MM_VERSION
    at::div_out(atOut, atInput, atOther);
#else
    auto roundingMode = impl::aten::getRoundingMode(rounding_mode);
    at::div_out(atOut, atInput, atOther);
#endif
    impl::aten::sync(ctx);
    return diopiSuccess;
}

/**
 * @brief 
 * @param rounding_mode supported in pytorch>=1.8.0
 */
diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildAtScalar(other);
#if TORCH_MM_VERSION < TORCH_1_8_MM_VERSION
    impl::aten::invokeATenFuncRet
        <at::Tensor (*)(at::Tensor const&, c10::Scalar)>
        (ctx, at::div, out, atInput, atOther);
#else
    auto roundingMode = impl::aten::getRoundingMode(rounding_mode);
    auto atOut = at::div(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
#endif
    return diopiSuccess;
}

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
        diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    auto atDilation = impl::aten::buildAtIntArray(dilation);
    impl::aten::invokeATenFuncRet(ctx, at::convolution, out,
        atInput, atWeight, atBias, atStride, atPadding, atDilation, false, at::IntArrayRef(0), groups);
    return diopiSuccess;
}

/**
 * @brief 
 * 
 * @param ignore_index supported in torch >= 1.10.0
 * @param label_smoothing supported in torch >= 1.10.0
 */
diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
        int64_t reduction, int64_t ignore_index, double label_smoothing) {
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
    return diopiSuccess;
}

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    auto atInput = impl::aten::buildATen(input);
    auto atMat2 = impl::aten::buildATen(mat2);
    auto atOut = impl::aten::buildATen(out);
    at::bmm_out(atOut, atInput, atMat2);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atTensor1 = impl::aten::buildATen(tensor1);
    auto atTensor2 = impl::aten::buildATen(tensor2);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncRet(ctx, at::addcmul, out, atInput, atTensor1, atTensor2, atValue);
    return diopiSuccess;
}

diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildATen(other);
    //Note(huqingqing): pytorch optimize the bmm case by folding the batch into the first dimension.
    //It changes the shape of output and causes warnning when using matmul_out.
    impl::aten::invokeATenFuncRet(ctx, at::matmul, out, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atTensor1 = impl::aten::buildATen(tensor1);
    auto atTensor2 = impl::aten::buildATen(tensor2);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = impl::aten::buildATen(out);
    at::addcdiv_out(atOut, atInput, atTensor1, atTensor2, atValue);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

// CAFFE2_API Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
        diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    auto atInput = impl::aten::buildATen(input);
    auto atMax1 = impl::aten::buildATen(mat1);
    auto atMax2 = impl::aten::buildATen(mat2);
    auto atBeta = impl::aten::buildAtScalar(beta);
    auto atAlpha = impl::aten::buildAtScalar(alpha);
    auto atOut = impl::aten::buildATen(out);
    at::addmm_out(atOut, atInput, atMax1, atMax2, atBeta, atAlpha);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

// NOTE(fengsibo): add int, short, bool test case
diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atDim = impl::aten::buildAtIntArray(dim);
    at::mean_out(atOut, atInput, atDim);  // TODO(fengsibo): use default type instead
    impl::aten::sync(ctx);
    return diopiSuccess;
}

// NOTE(fengsibo): add int, short, bool test case
diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atDim = impl::aten::buildAtIntArray(dim);
    at::sum_out(atOut, atInput, atDim);  // TODO(fengsibo): use default type instead
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atDim = impl::aten::buildAtIntArray(dim);
    at::std_out(atOut, atInput, atDim, unbiased);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices,
        diopiConstTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(min);
    auto atIndices = impl::aten::buildATen(min_indices);
    at::min_out(atOut, atIndices, atInput, dim);
    impl::aten::sync(ctx); 
    return diopiSuccess;
}

diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices,
        diopiConstTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(max);
    auto atIndices = impl::aten::buildATen(max_indices);
    at::max_out(atOut, atIndices, atInput, dim);
    impl::aten::sync(ctx); 
    return diopiSuccess;
}

diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::any_out(atOut, atInput, dim);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::all_out(atOut, atInput, dim);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::softmax(atInput, dim);  // TODO(fengsibo): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::log_softmax(atInput, dim);  // TODO(fengsibo): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atOut = impl::aten::buildATen(out);
    at::index_select_out(atOut, atInput, dim, atIndex);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    auto atInput = impl::aten::buildATen(input);
    at::Tensor atOut = at::select(atInput, dim, index).contiguous();
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t source) {
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atSource = impl::aten::buildATen(source);
    impl::aten::invokeATenFuncRet(ctx, at::masked_scatter, out, atInput, atMask, atSource);
    return diopiSuccess;
}

diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t dets,
        diopiConstTensorHandle_t scores, double iouThreshold) {
    auto atDets = impl::aten::buildATen(dets);
    auto atScores = impl::aten::buildATen(scores);
    auto atOut = vision::ops::nms_kernel(atDets, atScores, iouThreshold);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiNonzero(diopiContextHandle_t ctx,
        diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::nonzero(atInput);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias) {
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);
    impl::aten::invokeATenFuncRet(ctx, at::linear, out, atInput, atWeight, atBias);
    return diopiSuccess;
}

diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t rois, double spatialScale, int64_t pooledHeight,
        int64_t pooledWidth, int64_t samplingRatio, bool aligned) {
    auto atInput = impl::aten::buildATen(input);
    auto atRois = impl::aten::buildATen(rois);
    auto atOut = vision::ops::roi_align_forward_kernel(atInput, atRois, spatialScale,
        pooledHeight, pooledWidth, samplingRatio, aligned);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t w, diopiTensorHandle_t dw, diopiTensorHandle_t buf,
        double learningrate, double momentum, double dampening, double weightDecay, bool nesterov) {
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
        atBuf.mul_(momentum).add_(d_p, 1 - dampening);
        if (nesterov) {
          d_p = d_p.add(atBuf, momentum);
        } else {
          d_p = atBuf;
        }
    }
    p.data().add_(d_p, -1 * learningrate);

    impl::aten::updateATen2Tensor(ctx, atW, w);
    impl::aten::updateATen2Tensor(ctx, atDw, dw);
    impl::aten::updateATen2Tensor(ctx, atBuf, buf);

    return diopiSuccess;
}

/**
 * @brief 
 * @param errorIfNonfinite supported in pytorch ?
 * @return diopiError_t 
 */
diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t* parameters,
        int64_t parametersNum, double maxNorm, double normType, bool errorIfNonfinite) {
    auto tensorList = impl::aten::buildATenList(parameters, parametersNum);
    *out = torch::nn::utils::clip_grad_norm_(tensorList, maxNorm, normType);
    return diopiSuccess;
}

diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx,
        diopiTensorHandle_t inout, diopiConstTensorHandle_t indices, double max_norm, double norm_type) {
    auto atSelf = impl::aten::buildATen(inout);
    auto atIndices = impl::aten::buildATen(indices);
    at::embedding_renorm_(atSelf, atIndices, max_norm, norm_type);
    return diopiSuccess;
}

diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight,
        diopiConstTensorHandle_t indices, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    auto atWeight = impl::aten::buildATen(weight);
    auto atIndices = impl::aten::buildATen(indices);
    impl::aten::invokeATenFuncRet(ctx, at::embedding, out, atWeight, atIndices, paddingIdx, scaleGradByFreq, sparse);
    return diopiSuccess;
}

diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t diagonal) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::tril_out(atOut, atInput, diagonal);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t* tensors, int64_t insNum, int64_t dim) {
    auto tensorList = impl::aten::buildATenList(tensors, insNum);
    auto atOut = impl::aten::buildATen(out);
    at::cat_out(atOut, tensorList, dim);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t outsNum,
        diopiConstTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim) {
    auto atInput = impl::aten::buildATen(input);
    auto atSizes = impl::aten::buildAtIntArray(splitSizes);
    auto atOuts = at::split_with_sizes(atInput, atSizes, dim);
    for (int i = 0; i < outsNum; ++i) {
        impl::aten::updateATen2Tensor(ctx, atOuts[i].contiguous(), outs[i]);
    }
    return diopiSuccess;
}

diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    auto tensorList = impl::aten::buildATenList(tensors, numTensors);
    
    std::vector<at::Tensor> a = impl::aten::buildATenList(tensors, numTensors);
    at::TensorList b = impl::aten::buildATenList(tensors, numTensors);
    
    auto atOut = impl::aten::buildATen(out);
    at::stack_out(atOut, tensorList, dim);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
        diopiConstTensorHandle_t input, int64_t dim, bool descending, const bool* stable) {
    auto atInput = impl::aten::buildATen(input);
    auto atValues = impl::aten::buildATen(values);
    auto atIndices = impl::aten::buildATen(indices);
#if TORCH_MM_VERSION <= TORCH_1_8_MM_VERSION
    at::sort_out(atValues, atIndices, atInput, dim, descending);
#else
    c10::optional<bool> atStable = stable ? c10::optional<bool>(*stable) : c10::nullopt;
    at::sort_out(atValues, atIndices, atInput, atStable, dim, descending);
#endif
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
        diopiConstTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted) {
    auto atInput = impl::aten::buildATen(input);
    auto atValues = impl::aten::buildATen(values);
    auto atIndices = impl::aten::buildATen(indices);
    at::topk_out(atValues, atIndices, atInput, k, dim, largest, sorted);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t dim0, int64_t dim1) {
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet<at::Tensor (*)(at::Tensor const&, int64_t, int64_t)>
        (ctx, at::transpose, out, atInput, dim0, dim1);
    return diopiSuccess;
}

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, int64_t numClasses) {
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::one_hot, out, atInput, numClasses);
    return diopiSuccess;
}

diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    auto atCondition = impl::aten::buildATen(condition);
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildATen(other);
    impl::aten::invokeATenFuncRet
        <at::Tensor (*)(at::Tensor const&, at::Tensor const&, at::Tensor const&)>
        (ctx, at::where, out, atCondition, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiSin(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sin_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::sin_, atInput);
    return diopiSuccess;
}

diopiError_t diopiCos(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::cos_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::cos_, atInput);
    return diopiSuccess;
}

diopiError_t diopiAbs(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::abs_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAbsInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::abs_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSqrt(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sqrt_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::sqrt_, atInput);
    return diopiSuccess;
}

diopiError_t diopiFloor(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::floor_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::floor_, atInput);
    return diopiSuccess;
}

diopiError_t diopiNeg(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::neg_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiNegInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::neg_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSign(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sign_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiTanh(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::tanh_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiTanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::tanh_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSigmoid(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sigmoid_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::sigmoid_, atInput);
    return diopiSuccess;
}

diopiError_t diopiExp(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::exp_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiExpInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::exp_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::log_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiLogInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::log_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog2(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::log2_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::log2_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog10(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::log10_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::log10_, atInput);
    return diopiSuccess;
}

diopiError_t diopiErf(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::erf_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiErfInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::erf_, atInput);
    return diopiSuccess;
}

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    at::Tensor atExponent = impl::aten::buildATen(exponent);
    at::Scalar atInput = impl::aten::buildAtScalar(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::pow_out(atOut, atInput, atExponent);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atExponent = impl::aten::buildAtScalar(exponent);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::pow_out(atOut, atInput, atExponent);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atExponent = impl::aten::buildATen(exponent);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::pow_out(atOut, atInput, atExponent);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::add_out(atOut, atInput, atOther, atAlpha);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::add(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::sub_out(atOut, atInput, atOther, atAlpha);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::sub(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::mul_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::mul(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::ge_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::ge_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::gt_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::gt_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::le_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::le_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::lt_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::lt_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::eq_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::eq_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::ne_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::ne_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    at::Tensor atTmpInput = impl::aten::buildATen(input);
    at::Tensor atTmpOther = impl::aten::buildATen(other);
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOther = atTmpOther.to(at::ScalarType::Bool);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::bitwise_and_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}


diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atTmpInput = impl::aten::buildATen(input);
    at::Scalar atTmpOther = impl::aten::buildAtScalar(other);
    at::Scalar atOther = atTmpOther.to<bool>();
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::bitwise_and_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    at::Tensor atTmpInput = impl::aten::buildATen(input);
    at::Tensor atTmpOther = impl::aten::buildATen(other);
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOther = atTmpOther.to(at::ScalarType::Bool);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::bitwise_or_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atTmpInput = impl::aten::buildATen(input);
    at::Scalar atTmpOther = impl::aten::buildAtScalar(other);
    at::Scalar atOther = atTmpOther.to<bool>();
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::bitwise_or_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiScalar_t* min, const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::clamp_(atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_out(atOut, atInput, atMin, atMax);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::clamp_max_(atInput, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_max_out(atOut, atInput, atMax);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

#if TORCH_MM_VERSION > TORCH_1_9_MM_VERSION
diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMin = impl::aten::buildATen(min);
    at::Tensor atMax = impl::aten::buildATen(max);
    at::clamp_(atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMin = impl::aten::buildATen(min);
    at::Tensor atMax = impl::aten::buildATen(max);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_out(atOut, atInput, atMin, atMax);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        diopiConstTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMax = impl::aten::buildATen(max);
    at::clamp_max_(atInput, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMax = impl::aten::buildATen(max);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_max_out(atOut, atInput, atMax);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        diopiConstTensorHandle_t min) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMin = impl::aten::buildATen(min);
    at::clamp_(atInput, atMin);
    return diopiSuccess;
}

diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMin = impl::aten::buildATen(min);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_out(atOut, atInput, atMin);
    impl::aten::sync(ctx);
    return diopiSuccess;
}
#endif

diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiScalar_t* min) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::clamp_(atInput, atMin);
    return diopiSuccess;
}

diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::clamp_out(atOut, atInput, atMin);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const float value) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::fill_(atInput, value);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t output_size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::adaptive_avg_pool2d_out(atOut, atInput, atOutSize);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t output_size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    auto atOuts = at::adaptive_max_pool2d(atInput, atOutSize);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), out);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    at::adaptive_max_pool2d_out(atOut, atIndices, atInput, atOutSize);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
        diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atGradOutput = impl::aten::buildATen(grad_output);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    at::Tensor atGradInput = impl::aten::buildATen(grad_input);
    at::adaptive_max_pool2d_backward_out(atGradInput, atGradOutput, atInput, atIndices);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
        bool count_include_pad, const int64_t* divisor_override) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    c10::optional<int64_t> atDivisorOverride = divisor_override ? c10::optional<int64_t>(*divisor_override) : c10::nullopt;
    at::Tensor atOut = impl::aten::buildATen(out);
    at::avg_pool2d_out(atOut, atInput, atKernelSize, atStride, atPadding, 
                       ceil_mode, count_include_pad, atDivisorOverride);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, double p, bool train) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::dropout, out, atInput, p, train);
    return diopiSuccess;

}

diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, double p, bool train) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::dropout_, atInput, p, train);
    return diopiSuccess;
}

diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t target, int64_t reduction) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    //Note(huqingqing): at::mse_loss_out reduce in the 0 dimension, which is different from at::mse_loss.
    //at::mse_loss reduce over all the dimensions.
    if (reduction == 0) {
        at::Tensor atOut = impl::aten::buildATen(out);
        at::mse_loss_out(atOut, atInput, atTarget, reduction);
        impl::aten::sync(ctx);
    } else {
        impl::aten::invokeATenFuncRet(ctx, at::mse_loss, out, atInput, atTarget, reduction);
    }
    return diopiSuccess;
}

diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t inputs,
        diopiConstTensorHandle_t targets, float alpha, float gamma, int64_t reduction) {
    at::Tensor atInput = impl::aten::buildATen(inputs);
    at::Tensor atTarget = impl::aten::buildATen(targets);
    at::Tensor atP = at::sigmoid(atInput);
    at::Tensor atTerm1 = at::pow(1 - atP, gamma) * at::log(atP);
    at::Tensor atTerm2 = at::pow(atP, gamma) * at::log(1 - atP);
    at::Tensor atRes = -atTarget * atTerm1 * alpha - (1 - atTarget) * atTerm2 * (1- alpha);
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
    return diopiSuccess;
}

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
        diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
        diopiConstTensorHandle_t bias, diopiConstTensorHandle_t running_mean,
        diopiConstTensorHandle_t running_var, bool training, double momentum, double eps) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atWeight = impl::aten::buildATen(weight);
    at::Tensor atBias = impl::aten::buildATen(bias);
    at::Tensor atRunningMean = impl::aten::buildATen(running_mean);
    at::Tensor atRunningVar = impl::aten::buildATen(running_var);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atSaveMean = impl::aten::buildATen(save_mean);
    at::Tensor atSaveInvstd = impl::aten::buildATen(save_invstd);
    at::native_batch_norm_out(atOut, atSaveMean, atSaveInvstd, atInput, atWeight, atBias,
                          atRunningMean, atRunningVar, training, momentum, eps);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, diopiConstTensorHandle_t input,
        int64_t dim, int64_t start, int64_t end, int64_t step) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = at::slice(atInput, dim, start, end, step).contiguous();
    impl::aten::updateATen2Tensor(ctx, atOut, null_out);
    return diopiSuccess;
}

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t* indices, int64_t nums) {
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
    return diopiSuccess;
}

diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
        diopiConstTensorHandle_t pos_weight, int64_t reduction) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    c10::optional<at::Tensor> atWeight = weight
        ? c10::optional<at::Tensor>(impl::aten::buildATen(weight))
        : c10::nullopt;
    c10::optional<at::Tensor> atPosWeight = pos_weight
        ? c10::optional<at::Tensor>(impl::aten::buildATen(pos_weight))
        : c10::nullopt;

    impl::aten::invokeATenFuncRet(ctx, at::binary_cross_entropy_with_logits, out, atInput, atTarget, atWeight,
            atPosWeight, reduction);
    return diopiSuccess;
}

diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                           const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    auto atInput = impl::aten::buildATen(input);
    auto atMin = impl::aten::buildAtScalar(min_val);
    auto atMax = impl::aten::buildAtScalar(max_val);
    auto atOut = impl::aten::buildATen(out);
    at::hardtanh_out(atOut, atInput, atMin, atMax);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, 
                              const diopiScalar_t* max_val) {
    auto atInput = impl::aten::buildATen(input);
    auto atMin = impl::aten::buildAtScalar(min_val);
    auto atMax = impl::aten::buildAtScalar(max_val);
    impl::aten::invokeATenFuncInp(ctx, at::hardtanh_, atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                            const diopiScalar_t* threshold, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atThreshold = impl::aten::buildAtScalar(threshold);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = impl::aten::buildATen(out);
    at::threshold_out(atOut, atInput, atThreshold, atValue);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold,
                               const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atThreshold = impl::aten::buildAtScalar(threshold);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncInp(ctx, at::threshold_, atInput, atThreshold, atValue);
    return diopiSuccess;
}

diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                       diopiConstTensorHandle_t input, const char* approximate) {
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::gelu, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                               diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, 
                               diopiConstTensorHandle_t weight, int64_t reduction, int64_t ignore_index) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atTarget = impl::aten::buildATen(target);
    auto atWeight = impl::aten::buildATen(weight);
    auto dim = atInput.dim();
    assert(dim > 1);
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
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
        diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
    at::IntArrayRef atInputSizes = impl::aten::buildAtIntArray(input_sizes);
    at::Tensor atGradOutput = impl::aten::buildATen(grad_output);
    impl::aten::invokeATenFuncRet(ctx, at::slice_backward, grad_input, atGradOutput, atInputSizes, dim, start, end, step);   
    return diopiSuccess;                                     
}

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input,
        diopiConstTensorHandle_t* indices, int64_t nums, diopiConstTensorHandle_t grad) {
    at::Tensor atZerosInput = impl::aten::buildATen(zeros_like_input);
    at::Tensor atGrad = impl::aten::buildATen(grad);
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
    impl::aten::invokeATenFuncRet(ctx, at::_index_put_impl_, grad_input, atZerosInput, vecIdx, atGrad, true, true);   
    return diopiSuccess;                                     
}

diopiError_t diopiSigmoidFocalLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_output,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
        diopiTensorHandle_t grad_input, float gamma, float alpha, int64_t reduction) {
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
    at::Tensor atRes = - atTarget * atTerm1 * alpha - (1 - atTarget) * atTerm2 * (1- alpha);
    atGradOutput *= atRes;
    impl::aten::updateATen2Tensor(ctx, atGradOutput, grad_input);
    return diopiSuccess;
}

diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
        diopiConstTensorHandle_t rois, double spatialScale, int64_t pooledHeight, int64_t pooledWidth, int64_t batchSize,
        int64_t channels, int64_t height, int64_t width, int64_t samplingRatio, bool aligned) {
    auto atGrad = impl::aten::buildATen(grad);
    auto atRois = impl::aten::buildATen(rois);
    auto atOut = vision::ops::roi_align_backward_kernel(atGrad, atRois, spatialScale,
        pooledHeight, pooledWidth, batchSize, channels, height, width, samplingRatio, aligned);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
        diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
        diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad_output);
    auto atWeight = impl::aten::buildATen(weight);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    auto atDilation = impl::aten::buildAtIntArray(dilation);
    diopi_tensor_list vecOut = {grad_input, grad_weight};
    auto grad_input_mask = std::array<bool, 2>{true, true};
    impl::aten::invokeATenFuncRet(ctx, at::cudnn_convolution_backward, vecOut, atInput, atGrad,
        atWeight, atPadding, atStride, atDilation, groups, false, false, false, grad_input_mask);
    if (bias_sizes != nullptr && grad3 != nullptr) {
        auto atBias = impl::aten::buildATen(grad3);
        at::Tensor atTmp = atGrad;
        int64_t size = atGrad.dim() - 1;
        while (atBias.dim() != size) {
            atTmp = at::sum(atTmp, -1, false);
            size -= 1;
        }
        if (atBias.size(0) !=  atTmp.size(0)) {
            atTmp = at::sum(atTmp, 0, false);
        }
        impl::aten::updateATen2Tensor(ctx, atTmp, grad3);
    }
    return diopiSuccess;
} 

diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad,
                                    diopiConstTensorHandle_t indices, int64_t numWeights, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    auto atGrad = impl::aten::buildATen(grad);
    auto atIndices = impl::aten::buildATen(indices);
    impl::aten::invokeATenFuncRet(ctx, at::embedding_backward, out, atGrad, atIndices, numWeights, paddingIdx, scaleGradByFreq, sparse);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                            diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    auto atGradOutput  = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::_adaptive_avg_pool2d_backward, grad_input, atGradOutput, atInput);
    return diopiSuccess;
}

diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
    auto atGradOutput  = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atSlope = impl::aten::buildAtScalar(negative_slope);
    impl::aten::invokeATenFuncRet(ctx, at::leaky_relu_backward, grad_input, atGradOutput, atInput, atSlope, input_is_result);
    return diopiSuccess;
}

diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                   diopiConstTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atMin = impl::aten::buildAtScalar(min_val);
    auto atMax = impl::aten::buildAtScalar(max_val);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::hardtanh_backward_out(atGradInput, atGradOutput, atInput, atMin, atMax);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                               diopiConstTensorHandle_t input, const char* approximate) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::gelu_backward, grad_input, atGradOutput, atInput);
    return diopiSuccess;
}

diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                    diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                    diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                    bool count_include_pad, const int64_t* divisor_override) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    c10::optional<int64_t> atDivisorOverride = divisor_override ? c10::optional<int64_t>(*divisor_override) : c10::nullopt;
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::avg_pool2d_backward_out(atGradInput, atGradOutput, atInput, atKernelSize, atStride, atPadding, 
                                ceil_mode, count_include_pad, atDivisorOverride);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, int64_t reduction) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::mse_loss_backward_out(atGradInput, atGradOutput, atInput, atTarget, reduction);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                               diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::tanh_backward_out(atGradInput, atGradOutput, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess; 
}

diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad,
                                      diopiSize_t input_sizes, int64_t dim, diopiConstTensorHandle_t index) {
    auto atGrad = impl::aten::buildATen(grad);
    at::IntArrayRef atInputSize = impl::aten::buildAtIntArray(input_sizes);
    auto atIndex = impl::aten::buildATen(index);
    impl::aten::invokeATenFuncRet(ctx, at::index_select_backward, grad_input, atGrad, atInputSize, dim, atIndex);
    return diopiSuccess;
}

diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                 diopiConstTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t index) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    at::IntArrayRef atInputSize = impl::aten::buildAtIntArray(input_sizes);
    impl::aten::invokeATenFuncRet(ctx, at::select_backward, grad_input, atGradOutput, atInputSize, dim, index);
    return diopiSuccess;
}

diopiError_t diopiSoftmaxBackwardData(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                      diopiConstTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atOutput = impl::aten::buildATen(output);
    // TODO(huqingqing): use default type instead
    impl::aten::invokeATenFuncRet(ctx, at::_softmax_backward_data, grad_input, atGradOutput, atOutput, dim, atOutput);
    return diopiSuccess;
}

diopiError_t diopiLogSoftmaxBackwardData(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                         diopiConstTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atOutput = impl::aten::buildATen(output);
    // TODO(huqingqing): use default type instead
    impl::aten::invokeATenFuncRet(ctx, at::_log_softmax_backward_data, grad_input, atGradOutput, atOutput, dim, atOutput);
    return diopiSuccess;
}

diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                  diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t output) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atOutput = impl::aten::buildATen(output);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::sigmoid_backward_out(atGradInput, atGradOutput, atOutput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, const diopiScalar_t* threshold) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atThreshold = impl::aten::buildAtScalar(threshold);
    impl::aten::invokeATenFuncRet(ctx, at::threshold_backward, grad_input, atGradOutput, atInput, atThreshold);
    return diopiSuccess;
}

diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
        diopiConstTensorHandle_t pos_weight, int64_t reduction) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    c10::optional<at::Tensor> atWeight = weight
        ? c10::optional<at::Tensor>(impl::aten::buildATen(weight))
        : c10::nullopt;
    c10::optional<at::Tensor> atPosWeight = pos_weight
        ? c10::optional<at::Tensor>(impl::aten::buildATen(pos_weight))
        : c10::nullopt;

    impl::aten::invokeATenFuncRet(ctx, at::binary_cross_entropy_with_logits_backward, grad_input, atGradOutput, atInput, atTarget, atWeight,
                                  atPosWeight, reduction);
    return diopiSuccess;
}

diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                       diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                       int64_t reduction, int64_t ignore_index) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    auto atWeight = impl::aten::buildATen(weight);
    auto atTotalWeight = impl::aten::buildATen(input).resize_({1}).fill_(atTarget.numel());
    
    auto dim = atInput.dim();
    assert(dim > 1);
    if (dim !=2 && dim != 4) {
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
        if (at::Reduction::None == reduction) {
            atGradOutput = atGradOutput.reshape(targetShape);
        }
    }
    auto atGradInput = impl::aten::buildATen(grad_input);
    if (dim == 2) {
        at::nll_loss_backward_out(atGradInput, atGradOutput, atInput, atTarget, atWeight, reduction,
                                  ignore_index, atTotalWeight);
    } else {
        at::nll_loss2d_backward_out(atGradInput, atGradOutput, atInput, atTarget, atWeight, reduction,
                                    ignore_index, atTotalWeight);
    }
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                    diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    auto atIndices = impl::aten::buildATen(indices);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::max_pool2d_with_indices_backward_out(atGradInput, atGradOutput, atInput, atKernelSize, 
                                             atStride, atPadding, atDilation, ceil_mode, atIndices);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
        diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
        diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean, 
        diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    c10::optional<at::Tensor> atRunningMean = running_mean
        ? c10::optional<at::Tensor>(impl::aten::buildATen(running_mean))
        : c10::nullopt;
    c10::optional<at::Tensor> atRunningVar = running_var
        ? c10::optional<at::Tensor>(impl::aten::buildATen(running_var))
        : c10::nullopt;
    c10::optional<at::Tensor> atSaveMean = save_mean
        ? c10::optional<at::Tensor>(impl::aten::buildATen(save_mean))
        : c10::nullopt;
    c10::optional<at::Tensor> atSaveVar = save_invstd
        ? c10::optional<at::Tensor>(impl::aten::buildATen(save_invstd))
        : c10::nullopt;
    auto grad_input_mask = std::array<bool, 3>{true, true, true};
    auto atOut = at::native_batch_norm_backward(atGradOutput, atInput, atWeight, atRunningMean,  atRunningVar, atSaveMean, 
        atSaveVar, training, eps, grad_input_mask);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOut), grad_input);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(atOut), grad_weight);
    impl::aten::updateATen2Tensor(ctx, std::get<2>(atOut), grad_bias);
    return diopiSuccess;
}

diopiError_t diopiArange(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start,
        const diopiScalar_t* end, const diopiScalar_t* step) {
    auto atOut = impl::aten::buildATen(out);
    auto atStart = impl::aten::buildAtScalar(start);
    auto atEnd = impl::aten::buildAtScalar(end);
    auto atStep = impl::aten::buildAtScalar(step);
    at::arange_out(atOut, atStart, atEnd, atStep);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {
    auto atOut = impl::aten::buildATen(out);
    at::randperm_out(atOut, n);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx) {
    auto atInOut = impl::aten::buildATen(inout);
    at::native::uniform_(atInOut, from, to, c10::nullopt);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx) {
    auto atInOut = impl::aten::buildATen(inout);
    if (to == nullptr) {
        at::native::random_(atInOut, from, c10::nullopt, c10::nullopt);
    } else {
        at::native::random_(atInOut, from, *to, c10::nullopt);
    }
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t idx) {
    auto atInOut = impl::aten::buildATen(inout);
    at::bernoulli(atInOut, c10::nullopt);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t idx) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::bernoulli_out(atOut, atInput, c10::nullopt);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, int64_t idx) {
    auto atOut = impl::aten::buildATen(out);
    at::bernoulli(atOut, p, c10::nullopt);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, 
                             diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value){
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildATen(value);
    auto atOut = at::masked_fill(atInput, atMask, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, 
                                diopiConstTensorHandle_t value){
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildATen(value);
    atInput.masked_fill_(atMask, atValue);
    return diopiSuccess;
}

diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, 
                                   diopiConstTensorHandle_t mask, const diopiScalar_t* value){
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = at::masked_fill(atInput, atMask, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;        
}

diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask, 
                                      const diopiScalar_t* value){
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildAtScalar(value);
    atInput.masked_fill_(atMask, atValue);
    return diopiSuccess;
}

diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad,
                        diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq,
                        float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad){
    
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
    atExpAvgSq.mul_(beta2).addcmul_(grad_d, grad_d, 1- beta2);

    at::Tensor denom;
    if(amsgrad) {
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
    return diopiSuccess;
}

diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, 
                       diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq, 
                       float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad) {
    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad);
    auto atExpAvg = impl::aten::buildATen(exp_avg);
    auto atExpAvgSq = impl::aten::buildATen(exp_avg_sq);
    auto atMaxExpAvgSq = impl::aten::buildATen(max_exp_avg_sq);

    auto& param = atInput;
    auto grad_d = atGrad.data();
    auto bias_correction1 = 1 - pow(beta1, step);
    auto bias_correction2 = 1 - pow(beta2, step);

    if(weight_decay != 0){
        grad_d = grad_d.add(param, weight_decay);
    }
    atExpAvg.mul_(beta1).add_(grad_d, 1 - beta1); 
    atExpAvgSq.mul_(beta2).addcmul_(grad_d, grad_d.conj(), 1- beta2);

    at::Tensor denom;
    if(amsgrad) {
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
    return diopiSuccess;
}

diopiError_t diopiAdadelta(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad, 
                           diopiTensorHandle_t square_avg, diopiTensorHandle_t acc_delta, float lr, 
                           float rho, float eps, float weight_decay) {
    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad);
    auto atSquareAvg = impl::aten::buildATen(square_avg);
    auto atAccDelta = impl::aten::buildATen(acc_delta);

    auto& param = atInput;
    auto grad_d = atGrad.data();
    if(weight_decay != 0){
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
    return diopiSuccess;
}

diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride, 
                                  diopiSize_t padding, diopiSize_t output_padding, int64_t groups, diopiSize_t dilation) {
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    auto atOutputPadding = impl::aten::buildAtIntArray(output_padding);
    auto atDilation = impl::aten::buildAtIntArray(dilation);
    impl::aten::invokeATenFuncRet(ctx, at::conv_transpose2d, out,
        atInput, atWeight, atBias, atStride, atPadding, atOutputPadding, groups, atDilation);
    return diopiSuccess;
}

diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                         int64_t dim, diopiDtype_t dtype){
    auto atInput = impl::aten::buildATen(input);
    auto atOut  = impl::aten::buildATen(out);
    at::cumsum_out(atOut, atInput, dim);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2,
                                  double p, const int64_t* compute_mode){
    auto atInput1 = impl::aten::buildATen(input1);
    auto atInput2 = impl::aten::buildATen(input2);
    c10::optional<int64_t> atComputMode = compute_mode ? c10::optional<int64_t>(*compute_mode) : c10::nullopt;
    impl::aten::invokeATenFuncRet(ctx, at::cdist, out, atInput1, atInput2, p, atComputMode);
    return diopiSuccess;
}

diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput1 = impl::aten::buildATen(input1);
    auto atInput2 = impl::aten::buildATen(input2);
    auto atCdist = impl::aten::buildATen(cdist);
    impl::aten::invokeATenFuncRet(ctx, at::_cdist_backward, grad_input, atGradOutput, atInput1, atInput2, p, atCdist);
    return diopiSuccess;
}

diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::reciprocal_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input){
    auto atInput = impl::aten::buildATen(input);
    at::reciprocal_(atInput);
    return diopiSuccess;
}

diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    at::bitwise_not_out(atOut, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const int64_t* dim, bool keepdim) {
    auto atInput = impl::aten::buildATen(input);
    c10::optional<int64_t> atDim = dim ? c10::optional<int64_t>(*dim) : c10::nullopt;
    impl::aten::invokeATenFuncRet(ctx, at::argmax, out, atInput, atDim, keepdim);
    return diopiSuccess;
}

diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                               int64_t reduction, double beta) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    if (reduction == 0) {
        at::Tensor atOut = impl::aten::buildATen(out);
        at::smooth_l1_loss_out(atOut, atInput, atTarget, reduction, beta);
        impl::aten::sync(ctx);
    } else {
        impl::aten::invokeATenFuncRet(ctx, at::smooth_l1_loss, out, atInput, atTarget, reduction, beta);
    }
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                       diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, int64_t reduction, double beta) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    auto atGradInput  = impl::aten::buildATen(grad_input);
    at::smooth_l1_loss_backward_out(atGradInput, atGradOutput, atInput, atTarget, reduction, beta);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiMaximum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    auto atInput = impl::aten::buildATen(input);
    auto atOther= impl::aten::buildATen(other);
    auto atOut = impl::aten::buildATen(out);
    at::maximum_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiMinimum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    auto atInput = impl::aten::buildATen(input);
    auto atOther= impl::aten::buildATen(other);
    auto atOut = impl::aten::buildATen(out);
    at::minimum_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    auto atInput = impl::aten::buildATen(input);
    auto atMat2= impl::aten::buildATen(mat2);
    auto atOut = impl::aten::buildATen(out);
    at::mm_out(atOut, atInput, atMat2);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiConvolution3d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t stride,
        diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    auto atDilation = impl::aten::buildAtIntArray(dilation);
    impl::aten::invokeATenFuncRet(ctx, at::convolution, out,
        atInput, atWeight, atBias, atStride, atPadding, atDilation, false, at::IntArrayRef(0), groups);
    return diopiSuccess;
}

diopiError_t diopiConvolution3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
        diopiTensorHandle_t grad3, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
        diopiSize_t dilation, bool transposed, diopiSize_t output_padding, int64_t groups) {
    auto atInput = impl::aten::buildATen(input);
    auto atGrad = impl::aten::buildATen(grad_output);
    auto atWeight = impl::aten::buildATen(weight);
    auto atStride = impl::aten::buildAtIntArray(stride);
    auto atPadding = impl::aten::buildAtIntArray(padding);
    auto atDilation = impl::aten::buildAtIntArray(dilation);
    diopi_tensor_list vecOut = {grad_input, grad_weight};
    auto grad_input_mask = std::array<bool, 2>{true, true};

    impl::aten::invokeATenFuncRet(ctx, at::cudnn_convolution_backward, vecOut, atInput, atGrad,
        atWeight, atPadding, atStride, atDilation, groups, false, false, false, grad_input_mask);

    if (bias_sizes != nullptr && grad3 != nullptr) {
        auto atBias = impl::aten::buildATen(grad3);
        at::Tensor atTmp = atGrad;
        int64_t size = atGrad.dim() - 1;
        while (atBias.dim() != size) {
            atTmp = at::sum(atTmp, -1, false);
            size -= 1;
        }
        if (atBias.size(0) !=  atTmp.size(0)) {
            atTmp = at::sum(atTmp, 0, false);
        }
        impl::aten::updateATen2Tensor(ctx, atTmp, grad3);
    }
    return diopiSuccess;
}

diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size, bool implicit) {
    auto atInput = impl::aten::buildATen(input);
    auto atSize = impl::aten::buildAtIntArray(size);
    auto atOut = at::native::expand(atInput, atSize, implicit).clone();
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiUnfold(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t size, int64_t step) {
    auto atInput = impl::aten::buildATen(input);
    // must use contiguous rather than clone in this case
    auto atOut = at::native::unfold(atInput, dim, size, step).contiguous();
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiUnfoldBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
        diopiSize_t input_sizes, int64_t dim, int64_t size, int64_t step) {
    auto atGrad = impl::aten::buildATen(grad_output);
    auto atInputSize = impl::aten::buildAtIntArray(input_sizes);
    impl::aten::invokeATenFuncRet(ctx, at::unfold_backward, grad_input, atGrad, atInputSize, dim, size, step);
    return diopiSuccess;
}

diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out,
                               diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atOut = at::masked_select(atInput, atMask);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                       diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    impl::aten::invokeATenFuncRet(ctx, at::masked_select_backward, grad_input, atGradOutput, atInput, atMask);
    return diopiSuccess;
}

diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                  int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = at::index_fill(atInput, dim, atIndex, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess; 
}

diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                            int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildATen(value);
    auto atOut = at::index_fill(atInput, dim, atIndex, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess; 
}

diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, diopiConstTensorHandle_t input,
                                     int64_t dim, diopiConstTensorHandle_t index, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildAtScalar(value);
    atInput.index_fill_(dim, atIndex, atValue);
    return diopiSuccess;  
}

diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t input,
                               int64_t dim, diopiConstTensorHandle_t index, diopiConstTensorHandle_t value) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildATen(value);
    atInput.index_fill_(dim, atIndex, atValue);
    return diopiSuccess;
}

diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    auto atStart = impl::aten::buildAtScalar(start);
    auto atEnd = impl::aten::buildAtScalar(end);
    c10::optional<int64_t> atStep(steps);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::linspace_out(atOut, atStart, atEnd, atStep);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiRoll(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t shifts, diopiSize_t dims) {
    auto atInput = impl::aten::buildATen(input);
    at::IntArrayRef atShifts = impl::aten::buildAtIntArray(shifts);
    at::IntArrayRef atDims = impl::aten::buildAtIntArray(dims);
    auto atOut = at::roll(atInput, atShifts, atDims);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atP = impl::aten::buildAtScalar(p);
    at::IntArrayRef atDim = impl::aten::buildAtIntArray(dim);
    at::norm_out(atOut, atInput, atP, atDim);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiGroupNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, int64_t num_groups, double eps) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atWeight = impl::aten::buildATen(weight);
    at::Tensor atBias = impl::aten::buildATen(bias);
    const int64_t N = atInput.size(0);
    const int64_t C = atInput.size(1);
    const auto input_shape = atInput.sizes();
    const int64_t HxW = c10::multiply_integers(input_shape.cbegin() + 2, input_shape.cend());
    diopi_tensor_list vecOut = {out, save_mean, save_invstd};
    impl::aten::invokeATenFuncRet(ctx, at::native_group_norm, vecOut, atInput, atWeight, atBias, N, C, HxW, num_groups, eps);  
    return diopiSuccess;
}

diopiError_t diopiGroupNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                    diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t mean,
                                    diopiConstTensorHandle_t rstd, int64_t num_groups) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atSaveMean = impl::aten::buildATen(mean);
    auto atSaveVar = impl::aten::buildATen(rstd);
    const int64_t N = atInput.size(0);
    const int64_t C = atInput.size(1);
    const auto input_shape = atInput.sizes();
    const int64_t HxW = c10::multiply_integers(input_shape.cbegin() + 2, input_shape.cend());
    auto grad_input_mask = std::array<bool, 3>{true, true, true};
    auto atOut = at::native_group_norm_backward(atGradOutput, atInput, atSaveMean, atSaveVar, 
                                                atWeight,  N, C, HxW, num_groups, grad_input_mask);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOut), grad_input);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(atOut), grad_weight);
    impl::aten::updateATen2Tensor(ctx, std::get<2>(atOut), grad_bias);
    return diopiSuccess;
}

diopiError_t diopiBCELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
        diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, int64_t reduction) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    c10::optional<at::Tensor> atWeight = weight
        ? c10::optional<at::Tensor>(impl::aten::buildATen(weight))
        : c10::nullopt;
    if (reduction == 0) {
        at::Tensor atOut = impl::aten::buildATen(out);
        at::binary_cross_entropy_out(atOut, atInput, atTarget, atWeight, reduction);
        impl::aten::sync(ctx);
    } else {
        impl::aten::invokeATenFuncRet(ctx, at::binary_cross_entropy, out, atInput, atTarget, atWeight, reduction);
    }
    return diopiSuccess;
}

diopiError_t diopiBCELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, int64_t reduction) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    c10::optional<at::Tensor> atWeight = weight
        ? c10::optional<at::Tensor>(impl::aten::buildATen(weight))
        : c10::nullopt;
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::binary_cross_entropy_backward_out(atGradInput, atGradOutput, 
                                          atInput, atTarget, atWeight, reduction);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
        diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalized_shape, double eps) {
    at::Tensor atInput = impl::aten::buildATen(input);
    c10::optional<at::Tensor> atWeight = weight
        ? c10::optional<at::Tensor>(impl::aten::buildATen(weight))
        : c10::nullopt;
    c10::optional<at::Tensor> atBias = bias
        ? c10::optional<at::Tensor>(impl::aten::buildATen(bias))
        : c10::nullopt;
    auto atNormalizedShape = impl::aten::buildAtIntArray(normalized_shape);
    diopi_tensor_list vecOut = {out, save_mean, save_invstd};
    impl::aten::invokeATenFuncRet(ctx, at::native_layer_norm, vecOut, atInput, atNormalizedShape, atWeight, atBias, eps);  
    return diopiSuccess;
}

diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                    diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias,
                                    diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atNormalizedShape = impl::aten::buildAtIntArray(normalized_shape);
    c10::optional<at::Tensor> atWeight;
    c10::optional<at::Tensor> atBias;
    auto grad_input_mask = std::array<bool, 3>{true, false, false};
    if (weight != nullptr) {
        atWeight = c10::optional<at::Tensor>(impl::aten::buildATen(weight));
        grad_input_mask.at(1)=true;
    } 
    if (bias != nullptr) {
        atBias = c10::optional<at::Tensor>(impl::aten::buildATen(bias));
        grad_input_mask.at(2)=true;
    } 

    auto atSaveMean = impl::aten::buildATen(mean);
    auto atSaveVar = impl::aten::buildATen(rstd);
    auto atOut = at::native_layer_norm_backward(atGradOutput, atInput, atNormalizedShape,
        atSaveMean, atSaveVar, atWeight, atBias, grad_input_mask);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOut), grad_input);
    if (grad_weight != nullptr) {
        impl::aten::updateATen2Tensor(ctx, std::get<1>(atOut), grad_weight);
    }
    if (grad_bias != nullptr) {
        impl::aten::updateATen2Tensor(ctx, std::get<2>(atOut), grad_bias);
    }
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t output_size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    auto atOut = impl::aten::buildATen(out);
    at::adaptive_avg_pool3d_out(atOut, atInput, atOutSize);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                            diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input) {
    auto atGradOutput  = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::adaptive_avg_pool3d_backward_out(atGradInput, atGradOutput, atInput);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t output_size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    auto atOuts = at::adaptive_max_pool3d(atInput, atOutSize);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), out);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t output_size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    auto atOut = impl::aten::buildATen(out);
    auto atIndices = impl::aten::buildATen(indices);
    at::adaptive_max_pool3d_out(atOut, atIndices, atInput, atOutSize);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
        diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t indices) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atGradOutput = impl::aten::buildATen(grad_output);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    at::Tensor atGradInput = impl::aten::buildATen(grad_input);
    at::adaptive_max_pool3d_backward_out(atGradInput, atGradOutput, atInput, atIndices);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiMaxPool3d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
        diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    bool atCeilMode = ceil_mode;
    impl::aten::invokeATenFuncRet(ctx, at::max_pool3d, out,
        atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
    return diopiSuccess;
}

diopiError_t diopiMaxPool3dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t indices, diopiConstTensorHandle_t input, diopiSize_t kernel_size,
        diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    bool atCeilMode = ceil_mode;
    at::Tensor atOut = impl::aten::buildATen(out);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    at::max_pool3d_with_indices_out(atOut, atIndices, atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiMaxPool3dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                    diopiSize_t dilation, bool ceil_mode, diopiConstTensorHandle_t indices) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    auto atIndices = impl::aten::buildATen(indices);
    auto atGradInput = impl::aten::buildATen(grad_input);
    at::max_pool3d_with_indices_backward_out(atGradInput, atGradOutput, atInput, atKernelSize, 
                                             atStride, atPadding, atDilation, ceil_mode, atIndices);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiPermute(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atDims = impl::aten::buildAtIntArray(dims);
    impl::aten::invokeATenFuncRet(ctx, at::permute, out, atInput, atDims);
    return diopiSuccess;
}

diopiError_t diopiCopyInp(diopiContextHandle_t ctx, diopiConstTensorHandle_t src, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atSrc = impl::aten::buildATen(src);
    at::native::copy_(atInput, atSrc, false);
    return diopiSuccess;
}

diopiError_t diopiGather(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atOut = impl::aten::buildATen(out);
    at::gather_out(atOut, atInput, dim, atIndex);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiGatherBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                 diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    bool sparse_grad = false;
    auto atOut = at::gather_backward(atGradOutput, atInput, dim, atIndex, sparse_grad);
    impl::aten::updateATen2Tensor(ctx, atOut, grad_input);
    return diopiSuccess;
}

diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildATen(other);
    auto atOut = impl::aten::buildATen(out);
    at::remainder_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiScalar_t* other) {
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildAtScalar(other);
    auto atOut = impl::aten::buildATen(out);
    at::remainder_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiScalar_t* input, diopiConstTensorHandle_t other) {
    auto atInputScalar = impl::aten::buildAtScalar(input);
    auto atInput = impl::aten::buildATen(other).clone().fill_(atInputScalar);
    auto atOther = impl::aten::buildATen(other);
    auto atOut = impl::aten::buildATen(out);
    at::remainder_out(atOut, atInput, atOther);
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiCTCLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t neg_log_likelihood, diopiTensorHandle_t log_alpha, 
                          diopiConstTensorHandle_t log_probs, diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths,
                          diopiConstTensorHandle_t target_lengths, int64_t blank, int64_t reduction, bool zero_infinity) {
    auto atLogProbs = impl::aten::buildATen(log_probs);
    auto atTarget = impl::aten::buildATen(targets);
    auto atInputLength = impl::aten::buildATen(input_lengths).to(at::Device(at::kCPU), at::kLong).contiguous();
    auto atTargetLength = impl::aten::buildATen(target_lengths).to(at::Device(at::kCPU), at::kLong).contiguous();
    std::vector<int64_t> inputL;
    std::vector<int64_t> targetL;
    for(int i = 0; i < atInputLength.numel(); ++i) {
        inputL.push_back(atInputLength.data_ptr<int64_t>()[i]);
        targetL.push_back(atTargetLength.data_ptr<int64_t>()[i]);
    }
    at::IntArrayRef il(inputL);
    at::IntArrayRef tl(targetL);
    auto atOut = at::native::ctc_loss_gpu(atLogProbs, atTarget, il, tl, blank, zero_infinity);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOut), neg_log_likelihood);
    impl::aten::updateATen2Tensor(ctx, std::get<1>(atOut), log_alpha);
    auto atRes = std::get<0>(atOut);
    if (zero_infinity) {
        atRes = at::where(atRes == at::Scalar(std::numeric_limits<double>::infinity()),
                          at::zeros({}, atRes.options()), atRes);
    }
    if (reduction == at::Reduction::Mean) {
        auto target_lengths_t = at::tensor(tl, atRes.options()).clamp_min(1);
        atRes = (atRes / target_lengths_t).mean();
    } else if (reduction == at::Reduction::Sum) {
        atRes = atRes.sum();
    }
    impl::aten::updateATen2Tensor(ctx, atRes, out);
    return diopiSuccess;
}

diopiError_t diopiCTCLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t log_probs,
                                  diopiConstTensorHandle_t targets, diopiConstTensorHandle_t input_lengths, diopiConstTensorHandle_t target_lengths, 
                                  diopiConstTensorHandle_t neg_log_likelihood, diopiConstTensorHandle_t log_alpha, int64_t blank, int64_t reduction, bool zero_infinity) {
    auto atLogProbs = impl::aten::buildATen(log_probs);
    auto atTarget = impl::aten::buildATen(targets);
    auto atInputLength = impl::aten::buildATen(input_lengths).to(at::Device(at::kCPU), at::kLong).contiguous();
    auto atTargetLength = impl::aten::buildATen(target_lengths).to(at::Device(at::kCPU), at::kLong).contiguous();
    std::vector<int64_t> inputL;
    std::vector<int64_t> targetL;
    for(int i = 0; i < atInputLength.numel(); ++i) {
        inputL.push_back(atInputLength.data_ptr<int64_t>()[i]);
        targetL.push_back(atTargetLength.data_ptr<int64_t>()[i]);
    }
    at::IntArrayRef il(inputL);
    at::IntArrayRef tl(targetL);

    int64_t batch_size = atLogProbs.size(1);
    std::vector<int64_t> expand_shape = {batch_size};
    at::IntArrayRef shape(expand_shape.data(), expand_shape.size());
    auto atGrad = impl::aten::buildATen(grad_output);
    if (reduction == at::Reduction::Mean) {
        atGrad = at::native::expand(atGrad, shape).clone();
        auto target_lengths_t = at::tensor(tl, atGrad.options()).clamp_min(1);;
        atGrad = atGrad/target_lengths_t;
        atGrad.mul_(1./batch_size);
    } else if (reduction == at::Reduction::Sum) {
        atGrad = at::native::expand(atGrad, shape);
    }
    auto atNegLogLikehood = impl::aten::buildATen(neg_log_likelihood);
    auto atLogAlpha = impl::aten::buildATen(log_alpha);
    auto atOut = at::native::ctc_loss_backward_gpu(atGrad, atLogProbs, atTarget, il, tl, atNegLogLikehood, atLogAlpha, blank, zero_infinity);
    impl::aten::updateATen2Tensor(ctx, atOut, grad_input);
    return diopiSuccess;
}

diopiError_t diopiIndexPutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values,
                              diopiConstTensorHandle_t* indices, bool accumulate){
    auto atInput = impl::aten::buildATen(input);
    auto atValues = impl::aten::buildATen(values);
    auto indices1 = c10::optional<at::Tensor>(impl::aten::buildATen(indices[0]));
    auto indices2 = c10::optional<at::Tensor>(impl::aten::buildATen(indices[1]));
    torch::List<c10::optional<at::Tensor>> atIndicesList({indices1, indices2});
    at::Tensor atOut = at::index_put(atInput, atIndicesList, atValues, accumulate);
    impl::aten::updateATen2Tensor(ctx, atOut, input);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiIndexPut(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                     diopiConstTensorHandle_t values, diopiConstTensorHandle_t* indices, bool accumulate) {
    auto atInput = impl::aten::buildATen(input);
    auto atValues = impl::aten::buildATen(values);
    auto indices1 = c10::optional<at::Tensor>(impl::aten::buildATen(indices[0]));
    auto indices2 = c10::optional<at::Tensor>(impl::aten::buildATen(indices[1]));
    torch::List<c10::optional<at::Tensor>> atIndicesList({indices1, indices2});
    at::Tensor atOut = at::index_put(atInput, atIndicesList, atValues, accumulate);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, 
                             diopiConstTensorHandle_t index, const char* reduce) {
    auto atInput = impl::aten::buildATen(input);
    auto atSrc = impl::aten::buildATen(src);
    auto atIndex = impl::aten::buildATen(index);
    at::Tensor atOut;
    if(strlen(reduce) != 0) {
        c10::string_view atReduce(reduce, strlen(reduce));
        atOut = at::scatter(atInput, dim, atIndex, atSrc, atReduce);
    } else {
        atOut = at::scatter(atInput, dim, atIndex, atSrc);
    }  
    impl::aten::updateATen2Tensor(ctx, atOut, input);
    return diopiSuccess;
}

diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                   diopiConstTensorHandle_t index, const char* reduce) {
    auto atInput = impl::aten::buildATen(input);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atIndex = impl::aten::buildATen(index);
    at::Tensor atOut; 
    if(strlen(reduce) != 0) {
        c10::string_view atReduce(reduce, strlen(reduce));
        atOut = at::scatter(atInput, dim, atIndex, atValue, atReduce);
    } else {
        atOut = at::scatter(atInput, dim, atIndex, atValue);
    }
    impl::aten::updateATen2Tensor(ctx, atOut, input);
    return diopiSuccess;
}

diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim,
                          diopiConstTensorHandle_t src, diopiConstTensorHandle_t index, const char* reduce) {
    auto atInput = impl::aten::buildATen(input);
    auto atSrc = impl::aten::buildATen(src);
    auto atIndex = impl::aten::buildATen(index);
    at::Tensor atOut;
    if(strlen(reduce) != 0) {
        c10::string_view atReduce(reduce, strlen(reduce));
        atOut = at::scatter(atInput, dim, atIndex, atSrc, atReduce);
    } else {
        atOut = at::scatter(atInput, dim, atIndex, atSrc);
    }
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index, const char* reduce) {
    auto atInput = impl::aten::buildATen(input);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atIndex = impl::aten::buildATen(index);
    at::Tensor atOut;
    if(strlen(reduce) != 0) {
        c10::string_view atReduce(reduce, strlen(reduce));
        atOut = at::scatter(atInput, dim, atIndex, atValue, atReduce);
    } else {
        atOut = at::scatter(atInput, dim, atIndex, atValue);
    }  
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::IntArrayRef atSize = impl::aten::buildAtIntArray(size);
    if (atInput.dim() == 3) {
        at::upsample_nearest1d_out(atOut, atInput, atSize);
    } else if (atInput.dim() == 4) {
        at::upsample_nearest2d_out(atOut, atInput, atSize);
    } else if (atInput.dim() == 5) {
        at::upsample_nearest3d_out(atOut, atInput, atSize); 
    }
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
        diopiSize_t out_size, diopiSize_t in_size) {
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
    }
    impl::aten::sync(ctx);
    return diopiSuccess;
}

//todo (huqingqing): not find UpsampleNearestExact op
diopiError_t diopiUpsampleNearestExact(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atSize = impl::aten::buildAtIntArray(size);
    if (atInput.dim() == 3) {
        //impl::aten::invokeATenFuncRet(ctx, at::_upsample_nearest_exact1d, out, atInput, atSize);
    } else if (atInput.dim() == 4) {
        //impl::aten::invokeATenFuncRet(ctx, at::_upsample_nearest_exact2d, out, atInput, atSize);
    } else if (atInput.dim() == 5) {
        //impl::aten::invokeATenFuncRet(ctx, at::_upsample_nearest_exact3d, out, atInput, atSize);
    }
    return diopiSuccess;
}

diopiError_t diopiUpsampleLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size,
        bool align_corners, const char* mode) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = impl::aten::buildATen(out);
    at::IntArrayRef atSize = impl::aten::buildAtIntArray(size);
    if ( 3 == atInput.dim() && 0 == strcmp(mode, "linear")) {
        at::upsample_linear1d_out(atOut, atInput, atSize, align_corners);
    } else if ( 4 == atInput.dim()) {
        if (0 == strcmp(mode, "bilinear")) {
            at::upsample_bilinear2d_out(atOut, atInput, atSize, align_corners);
        } else if (0 == strcmp(mode, "bicubic")) {
            at::upsample_bicubic2d_out(atOut, atInput, atSize, align_corners);
        } else {
            NOT_SUPPORTED("interpolate mode type");
            return diopiErrorOccurred;
        }
    } else if ( 5 == atInput.dim() && 0 == strcmp(mode, "trilinear")) {
        at::upsample_trilinear3d_out(atOut, atInput, atSize, align_corners); 
    } else {
        NOT_SUPPORTED("interpolate mode type");
        return diopiErrorOccurred;
    }
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiUpsampleLinearBackward(diopiContextHandle_t ctx,  diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
        diopiSize_t out_size, diopiSize_t in_size, bool align_corners, const char* mode) {
    at::Tensor atGradOut = impl::aten::buildATen(grad_output);
    at::Tensor atGradInput = impl::aten::buildATen(grad_input);
    at::IntArrayRef atOutSize = impl::aten::buildAtIntArray(out_size);
    at::IntArrayRef atInSize = impl::aten::buildAtIntArray(in_size);
    if ( 3 == atGradInput.dim() && 0 == strcmp(mode, "linear")) {
        at::upsample_linear1d_backward_out(atGradInput, atGradOut, atOutSize, atInSize, align_corners);
    } else if ( 4 == atGradInput.dim()) {
        if (0 == strcmp(mode, "bilinear")) {
            at::upsample_bilinear2d_backward_out(atGradInput, atGradOut, atOutSize, atInSize, align_corners);
        } else if (0 == strcmp(mode, "bicubic")) {
            at::upsample_bicubic2d_backward_out(atGradInput, atGradOut, atOutSize, atInSize, align_corners);
        } else {
            NOT_SUPPORTED("interpolate mode type");
            return diopiErrorOccurred;
        }
    } else if ( 5 == atGradInput.dim() && 0 == strcmp(mode, "trilinear")) {
        at::upsample_trilinear3d_backward_out(atGradInput, atGradOut, atOutSize, atInSize, align_corners); 
    } else {
        NOT_SUPPORTED("interpolate mode type");
        return diopiErrorOccurred;

    }
    impl::aten::sync(ctx);
    return diopiSuccess;
}

diopiError_t diopiPad(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, 
                      diopiSize_t pad, const char* mode, double* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atPad = impl::aten::buildAtIntArray(pad);
    TORCH_CHECK(atPad.size() % 2 == 0, "Padding length must be divisible by 2");
    TORCH_CHECK(((int64_t)(atPad.size() / 2)) <= atInput.dim(), "Padding length too large");
    if (strcmp(mode, "constant") == 0) {
        auto atOut = at::constant_pad_nd(atInput, atPad, *value);
        impl::aten::updateATen2Tensor(ctx, atOut, out);
    } else {
        auto atOut = impl::aten::buildATen(out);
        if (atPad.size() == 2 && (atInput.dim() == 2 || atInput.dim() == 3)) {
            if (strcmp(mode, "reflect") == 0) {
                at::reflection_pad1d_out(atOut, atInput, atPad);
            } else if (strcmp(mode, "replicate") == 0) {
                at::replication_pad1d_out(atOut, atInput, atPad);
            } else if (strcmp(mode, "circular") == 0) {
                //TODO(ht): not implement in aten
                // _pad_circular(input, pad);
            } else {
                TORCH_CHECK(false, "NotImplementedError");
            }
        } else if(atPad.size() == 4 && (atInput.dim() == 3 || atInput.dim() == 4)) {
            if (strcmp(mode, "reflect") == 0) {
                at::reflection_pad2d_out(atOut, atInput, atPad);
            } else if (strcmp(mode, "replication") == 0) {
                at::replication_pad2d_out(atOut, atInput, atPad);
            } else if (strcmp(mode, "circular") == 0) {
                //TODO(ht): not implement in aten
                // _pad_circular(input, pad);
            } else {
                TORCH_CHECK(false, "NotImplementedError");
            }
        } else if (atPad.size() == 6 && (atInput.dim() == 4 || atInput.dim() == 5)) {
            if (strcmp(mode, "reflection") == 0) {
                at::reflection_pad3d_out(atOut, atInput, atPad);
            } else if (strcmp(mode, "replication") == 0) {
                at::replication_pad3d_out(atOut, atInput, atPad);
            } else if (strcmp(mode, "circular") == 0) {
                //TODO(ht): not implement in aten
                // _pad_circular(input, pad);
            } else {
                TORCH_CHECK(false, "NotImplementedError");
            }
        } else {
            TORCH_CHECK(false, "Only 2D, 3D, 4D, 5D padding with non-constant padding are supported for now");
        }
        impl::aten::sync(ctx);
    }
    return diopiSuccess;
}

diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, int64_t* dim,
                         bool sorted, bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t* counts) {
    auto atInput = impl::aten::buildATen(input);
    bool return_inverse = false;
    if(indices) {
        return_inverse = true;
    }
    std::tuple<at::Tensor, at::Tensor, at::Tensor> atOuts;
    if(!dim) {
        atOuts = at::_unique2(atInput, sorted, return_inverse, return_counts);
    } else {
        atOuts = at::unique_dim(atInput, *dim, sorted, return_inverse, return_counts);
    }
    impl::aten::buildDiopiTensor(ctx, std::get<0>(atOuts), out);
    if(return_inverse) {
        impl::aten::updateATen2Tensor(ctx, std::get<1>(atOuts), indices);
    }
    if(return_counts) {
        impl::aten::buildDiopiTensor(ctx, std::get<2>(atOuts), counts);
    }
    return diopiSuccess;
}

diopiError_t diopiProd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t* dim, diopiDtype_t type) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    if (dim == nullptr) {
        auto atTmp = at::prod(atInput);
        impl::aten::updateATen2Tensor(ctx, atTmp, out);
    } else {
        at::prod_out(atOut, atInput, *dim);
        impl::aten::sync(ctx);
    }
    return diopiSuccess;
}

}  // extern "C"
