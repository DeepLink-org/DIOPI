
#include <diopi/functions.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <iostream>
#include <math.h>

#include "helper.hpp"
#include "vision_kernel.h"

#define FLT_MIN		__FLT_MIN__

extern "C" {

diopiError_t diopiRelu(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
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
        diopiTensorHandle_t out, const diopiTensorHandle_t input,
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
        const diopiTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
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
        diopiTensorHandle_t indices, const diopiTensorHandle_t input, diopiSize_t kernel_size,
        diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    bool atCeilMode = ceil_mode;
    diopi_tensor_list vecOut = {out, indices};
    impl::aten::invokeATenFuncRet(ctx, at::max_pool2d_with_indices, vecOut,
        atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
    return diopiSuccess;
}

/**
 * @brief
 * @param rounding_mode supported in pytorch>=1.8
 */
diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
#if TORCH_MM_VERSION < TORCH_1_8_MM_VERSION
    impl::aten::invokeATenFuncRet
        <at::Tensor (*)(at::Tensor const&, at::Tensor const&)>
        (ctx, at::div, out, atInput, atOther);
#else
    auto roundingMode = impl::aten::getRoundingMode(rounding_mode);
    auto atOut = at::div(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
#endif
    return diopiSuccess;
}

/**
 * @brief 
 * @param rounding_mode supported in pytorch>=1.8.0
 */
diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
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

diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t weight, const diopiTensorHandle_t bias, diopiSize_t stride,
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
        const diopiTensorHandle_t input, const diopiTensorHandle_t target, const diopiTensorHandle_t weight,
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
        const diopiTensorHandle_t input, const diopiTensorHandle_t mat2) {
    auto atInput = impl::aten::buildATen(input);
    auto atMat2= impl::aten::buildATen(mat2);
    impl::aten::invokeATenFuncRet(ctx, at::bmm, out, atInput, atMat2);
    return diopiSuccess;
}

diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t tensor1, const diopiTensorHandle_t tensor2, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atTensor1 = impl::aten::buildATen(tensor1);
    auto atTensor2 = impl::aten::buildATen(tensor2);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncRet(ctx, at::addcmul, out, atInput, atTensor1, atTensor2, atValue);
    return diopiSuccess;
}

diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildATen(other);
    impl::aten::invokeATenFuncRet(ctx, at::matmul, out, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t tensor1, const diopiTensorHandle_t tensor2, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atTensor1 = impl::aten::buildATen(tensor1);
    auto atTensor2 = impl::aten::buildATen(tensor2);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncRet(ctx, at::addcdiv, out, atInput, atTensor1, atTensor2, atValue);
    return diopiSuccess;
}

// CAFFE2_API Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t mat1,
        const diopiTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    auto atInput = impl::aten::buildATen(input);
    auto atMax1 = impl::aten::buildATen(mat1);
    auto atMax2 = impl::aten::buildATen(mat2);
    auto atBeta = impl::aten::buildAtScalar(beta);
    auto atAlpha = impl::aten::buildAtScalar(alpha);
    impl::aten::invokeATenFuncRet(
        ctx, at::addmm, out, atInput, atMax1, atMax2, atBeta, atAlpha);
    return diopiSuccess;
}

// NOTE(fengsibo): add int, short, bool test case
diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildATen(input);
    auto atDim = impl::aten::buildAtIntArray(dim);
    auto atOut = at::mean(atInput, atDim);  // TODO(fengsibo): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

// NOTE(fengsibo): add int, short, bool test case
diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildATen(input);
    auto atDim = impl::aten::buildAtIntArray(dim);
    auto atOut = at::sum(atInput, atDim);  // TODO(fengsibo): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    auto atInput = impl::aten::buildATen(input);
    auto atDim = impl::aten::buildAtIntArray(dim);
    auto atOut = at::std(atInput, atDim, unbiased);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices,
        const diopiTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildATen(input);
    auto atOuts = at::min(atInput, dim);
    diopi_tensor_list outs = {min, min_indices};
    impl::aten::updateATen2Tensor(ctx, atOuts, outs); 
    return diopiSuccess;
}

diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices,
        const diopiTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildATen(input);
    auto atOuts = at::max(atInput, dim);
    diopi_tensor_list outs = {max, max_indices};
    impl::aten::updateATen2Tensor(ctx, atOuts, outs); 
    return diopiSuccess;
}

diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::any(atInput, dim);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::all(atInput, dim);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::softmax(atInput, dim);  // TODO(fengsibo): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::log_softmax(atInput, dim);  // TODO(fengsibo): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim, const diopiTensorHandle_t index) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atOut = at::index_select(atInput, dim, atIndex);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim, int64_t index) {
    auto atInput = impl::aten::buildATen(input);
    at::Tensor atOut = at::select(atInput, dim, index).contiguous();
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t mask, const diopiTensorHandle_t source) {
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atSource = impl::aten::buildATen(source);
    impl::aten::invokeATenFuncRet(ctx, at::masked_scatter, out, atInput, atMask, atSource);
    return diopiSuccess;
}

diopiError_t diopiNms(diopiContextHandle_t ctx, diopiTensorHandle_t* out, const diopiTensorHandle_t dets,
        const diopiTensorHandle_t scores, double iouThreshold) {
    auto atDets = impl::aten::buildATen(dets);
    auto atScores = impl::aten::buildATen(scores);
    auto atOut = vision::ops::nms_kernel(atDets, atScores, iouThreshold);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
}

diopiError_t diopiNonzero(diopiContextHandle_t ctx,
        diopiTensorHandle_t* out, const diopiTensorHandle_t input) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::nonzero(atInput);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t weight, const diopiTensorHandle_t bias) {
    auto atInput = impl::aten::buildATen(input);
    auto atWeight = impl::aten::buildATen(weight);
    auto atBias = impl::aten::buildATen(bias);
    impl::aten::invokeATenFuncRet(ctx, at::linear, out, atInput, atWeight, atBias);
    return diopiSuccess;
}

diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t rois, double spatialScale, int64_t pooledHeight,
        int64_t pooledWidth, int64_t samplingRatio, bool aligned) {
    auto atInput = impl::aten::buildATen(input);
    auto atRois = impl::aten::buildATen(rois);
    auto atOut = vision::ops::roi_align_forward_kernel(atInput, atRois, spatialScale,
        pooledHeight, pooledWidth, samplingRatio, aligned);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
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
        diopiTensorHandle_t inout, const diopiTensorHandle_t indices, double max_norm, double norm_type) {
    auto atSelf = impl::aten::buildATen(inout);
    auto atIndices = impl::aten::buildATen(indices);
    at::embedding_renorm_(atSelf, atIndices, max_norm, norm_type);
    return diopiSuccess;
}

diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t weight,
        const diopiTensorHandle_t indices, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    auto atWeight = impl::aten::buildATen(weight);
    auto atIndices = impl::aten::buildATen(indices);
    impl::aten::invokeATenFuncRet(ctx, at::embedding, out, atWeight, atIndices, paddingIdx, scaleGradByFreq, sparse);
    return diopiSuccess;
}

diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t diagonal) {
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::tril, out, atInput, diagonal);
    return diopiSuccess;
}

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t* tensors, int64_t insNum, int64_t dim) {
    auto tensorList = impl::aten::buildATenList(tensors, insNum);
    impl::aten::invokeATenFuncRet
        <at::Tensor (*)(at::TensorList, int64_t)>(ctx, at::cat, out, tensorList, dim);
    return diopiSuccess;
}

diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t outsNum,
        const diopiTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim) {
    auto atInput = impl::aten::buildATen(input);
    auto atSizes = impl::aten::buildAtIntArray(splitSizes);
    auto atOuts = at::split_with_sizes(atInput, atSizes, dim);
    for (int i = 0; i < outsNum; ++i) {
        impl::aten::updateATen2Tensor(ctx, atOuts[i].contiguous(), outs[i]);
    }
    return diopiSuccess;
}

diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    auto tensorList = impl::aten::buildATenList(tensors, numTensors);
    
    std::vector<at::Tensor> a = impl::aten::buildATenList(tensors, numTensors);
    at::TensorList b = impl::aten::buildATenList(tensors, numTensors);
    
    impl::aten::invokeATenFuncRet(ctx, at::stack, out, tensorList, dim);
    return diopiSuccess;
}

diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
        const diopiTensorHandle_t input, int64_t dim, bool descending, const bool* stable) {
    auto atInput = impl::aten::buildATen(input);
    diopi_tensor_list vecOut = {values, indices};
#if TORCH_MM_VERSION <= TORCH_1_8_MM_VERSION
    impl::aten::invokeATenFuncRet
        <std::tuple<at::Tensor, at::Tensor> (*)(at::Tensor const &, int64_t, bool)>
        (ctx, at::sort, vecOut, atInput, dim, descending);
#else
    c10::optional<bool> atStable = stable ? c10::optional<bool>(*stable) : c10::nullopt;
    impl::aten::invokeATenFuncRet
        <std::tuple<at::Tensor, at::Tensor> (*)(at::Tensor const &, c10::optional<bool>, int64_t, bool)>
        (ctx, at::sort, vecOut, atInput, atStable, dim, descending);
#endif
    return diopiSuccess;
}

diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
        const diopiTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted) {
    auto atInput = impl::aten::buildATen(input);
    diopi_tensor_list vecOut = {values, indices};
    impl::aten::invokeATenFuncRet(ctx, at::topk, vecOut, atInput, k, dim, largest, sorted);
    return diopiSuccess;
}

diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim0, int64_t dim1) {
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet<at::Tensor (*)(at::Tensor const&, int64_t, int64_t)>
        (ctx, at::transpose, out, atInput, dim0, dim1);
    return diopiSuccess;
}

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t numClasses) {
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::one_hot, out, atInput, numClasses);
    return diopiSuccess;
}

diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t condition,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    auto atCondition = impl::aten::buildATen(condition);
    auto atInput = impl::aten::buildATen(input);
    auto atOther = impl::aten::buildATen(other);
    impl::aten::invokeATenFuncRet
        <at::Tensor (*)(at::Tensor const&, at::Tensor const&, at::Tensor const&)>
        (ctx, at::where, out, atCondition, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiSin(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::sin, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiSinInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::sin_, atInput);
    return diopiSuccess;
}

diopiError_t diopiCos(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::cos, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiCosInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::cos_, atInput);
    return diopiSuccess;
}

diopiError_t diopiAbs(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::abs, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiAbsInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::abs_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSqrt(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::sqrt, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::sqrt_, atInput);
    return diopiSuccess;
}

diopiError_t diopiFloor(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::floor, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiFloorInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::floor_, atInput);
    return diopiSuccess;
}

diopiError_t diopiNeg(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::neg, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiNegInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::neg_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSign(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::sign, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiTanh(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::tanh, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiTanhInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::tanh_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSigmoid(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::sigmoid, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::sigmoid_, atInput);
    return diopiSuccess;
}

diopiError_t diopiExp(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::exp, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiExpInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::exp_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::log, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiLogInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::log_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog2(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::log2, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::log2_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog10(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::log10, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::log10_, atInput);
    return diopiSuccess;
}

diopiError_t diopiErf(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::erf, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiErfInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::erf_, atInput);
    return diopiSuccess;
}

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiScalar_t* input, const diopiTensorHandle_t exponent) {
    at::Tensor atExponent = impl::aten::buildATen(exponent);
    at::Scalar atInput = impl::aten::buildAtScalar(input);
    at::Tensor atOut = at::pow(atInput, atExponent);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* exponent) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atExponent = impl::aten::buildAtScalar(exponent);
    at::Tensor atOut = at::pow(atInput, atExponent);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t exponent) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atExponent = impl::aten::buildATen(exponent);
    at::Tensor atOut = at::pow(atInput, atExponent);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::add(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::add(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::sub(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::sub(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = at::mul(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::mul(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = at::ge(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::ge(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = at::gt(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::gt(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = at::le(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::le(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = at::lt(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::lt(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = at::eq(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::eq(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOther = impl::aten::buildATen(other);
    at::Tensor atOut = at::ne(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::ne(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atTmpInput = impl::aten::buildATen(input);
    at::Tensor atTmpOther = impl::aten::buildATen(other);
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOther = atTmpOther.to(at::ScalarType::Bool);
    at::Tensor atOut = at::bitwise_and(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}


diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atTmpInput = impl::aten::buildATen(input);
    at::Scalar atTmpOther = impl::aten::buildAtScalar(other);
    at::Scalar atOther = atTmpOther.to<bool>();
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOut = at::bitwise_and(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atTmpInput = impl::aten::buildATen(input);
    at::Tensor atTmpOther = impl::aten::buildATen(other);
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOther = atTmpOther.to(at::ScalarType::Bool);
    at::Tensor atOut = at::bitwise_or(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atTmpInput = impl::aten::buildATen(input);
    at::Scalar atTmpOther = impl::aten::buildAtScalar(other);
    at::Scalar atOther = atTmpOther.to<bool>();
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOut = at::bitwise_or(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
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
        diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::Tensor atOut = at::clamp(atInput, atMin, atMax);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
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
        diopiTensorHandle_t input, const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::Tensor atOut = at::clamp_max(atInput, atMax);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

#if TORCH_MM_VERSION > TORCH_1_9_MM_VERSION
diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiTensorHandle_t min, const diopiTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMin = impl::aten::buildATen(min);
    at::Tensor atMax = impl::aten::buildATen(max);
    at::clamp_(atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        diopiTensorHandle_t input, const diopiTensorHandle_t min, const diopiTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMin = impl::aten::buildATen(min);
    at::Tensor atMax = impl::aten::buildATen(max);
    at::Tensor atOut = at::clamp(atInput, atMin, atMax);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMax = impl::aten::buildATen(max);
    at::clamp_max_(atInput, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t input, const diopiTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMax = impl::aten::buildATen(max);
    at::Tensor atOut = at::clamp_max(atInput, atMax);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiTensorHandle_t min) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMin = impl::aten::buildATen(min);
    at::clamp_(atInput, atMin);
    return diopiSuccess;
}

diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t input, const diopiTensorHandle_t min) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atMin = impl::aten::buildATen(min);
    at::Tensor atOut = at::clamp(atInput, atMin);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
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
        diopiTensorHandle_t input, const diopiScalar_t* min) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::Tensor atOut = at::clamp(atInput, atMin);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiFill(diopiContextHandle_t ctx, diopiTensorHandle_t input, const float value) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::fill_(atInput, value);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, diopiSize_t output_size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    at::Tensor atOut = at::adaptive_avg_pool2d(atInput, atOutSize);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, diopiSize_t output_size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    auto atOuts = at::adaptive_max_pool2d(atInput, atOutSize);
    impl::aten::updateATen2Tensor(ctx, std::get<0>(atOuts), out);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t indices, const diopiTensorHandle_t input, diopiSize_t output_size) {
    at::Tensor atInput = impl::aten::buildATen(input);
    auto atOutSize = impl::aten::buildAtIntArray(output_size);
    diopi_tensor_list vecOut = {out, indices};
    impl::aten::invokeATenFuncRet(ctx, at::adaptive_max_pool2d, vecOut, atInput, atOutSize);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
        const diopiTensorHandle_t grad_output, const diopiTensorHandle_t input, const diopiTensorHandle_t indices) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atGradOutput = impl::aten::buildATen(grad_output);
    at::Tensor atIndices = impl::aten::buildATen(indices);
    impl::aten::invokeATenFuncRet(ctx, at::adaptive_max_pool2d_backward, grad_input, atGradOutput, atInput, atIndices);
    return diopiSuccess;
}

diopiError_t diopiAvgPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
        bool count_include_pad, const int64_t* divisor_override) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    c10::optional<int64_t> atDivisorOverride = divisor_override ? c10::optional<int64_t>(*divisor_override) : c10::nullopt;
    impl::aten::invokeATenFuncRet(ctx, at::avg_pool2d, out, atInput, atKernelSize, atStride,
            atPadding, ceil_mode, count_include_pad, atDivisorOverride);
    return diopiSuccess;
}

diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, double p, bool train) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::dropout, out, atInput, p, train);
    return diopiSuccess;

}

diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, double p, bool train) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::dropout_, atInput, p, train);
    return diopiSuccess;
}

diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t target, int64_t reduction) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    impl::aten::invokeATenFuncRet(ctx, at::mse_loss, out, atInput, atTarget, reduction);
    return diopiSuccess;
}

diopiError_t diopiSigmoidFocalLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t inputs,
        const diopiTensorHandle_t targets, float alpha, float gamma, int64_t reduction) {
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
        diopiTensorHandle_t save_invstd, const diopiTensorHandle_t input, const diopiTensorHandle_t weight,
        const diopiTensorHandle_t bias, const diopiTensorHandle_t running_mean,
        const diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atWeight = impl::aten::buildATen(weight);
    at::Tensor atBias = impl::aten::buildATen(bias);
    at::Tensor atRunningMean = impl::aten::buildATen(running_mean);
    at::Tensor atRunningVar = impl::aten::buildATen(running_var);
    diopi_tensor_list vecOut = {out, save_mean, save_invstd};
    impl::aten::invokeATenFuncRet(ctx, at::native_batch_norm, vecOut, atInput, atWeight, atBias,
        atRunningMean, atRunningVar, training, momentum, eps);
    return diopiSuccess;
}

diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out, const diopiTensorHandle_t input,
        int64_t dim, int64_t start, int64_t end, int64_t step) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atOut = at::slice(atInput, dim, start, end, step).contiguous();
    impl::aten::updateATen2Tensor(ctx, atOut, null_out);
    return diopiSuccess;
}

diopiError_t diopiIndex(diopiContextHandle_t ctx, diopiTensorHandle_t* out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t* indices, int64_t nums) {
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

diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t target, const diopiTensorHandle_t weight,
        const diopiTensorHandle_t pos_weight, int64_t reduction) {
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

diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
                           const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    auto atInput = impl::aten::buildATen(input);
    auto atMin = impl::aten::buildAtScalar(min_val);
    auto atMax = impl::aten::buildAtScalar(max_val);
    impl::aten::invokeATenFuncRet(ctx, at::hardtanh, out, atInput, atMin, atMax);
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

diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
                            const diopiScalar_t* threshold, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atThreshold = impl::aten::buildAtScalar(threshold);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncRet(ctx, at::threshold, out, atInput, atThreshold, atValue);
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
                       const diopiTensorHandle_t input, const char* approximate) {
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::gelu, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiCrossNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                               const diopiTensorHandle_t input, const diopiTensorHandle_t target, 
                               const diopiTensorHandle_t weight, int64_t reduction, int64_t ignore_index) {
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    auto atWeight = impl::aten::buildATen(weight);
    auto dim = atInput.dim();
    assert(dim > 1);
    if (dim == 2) {
        impl::aten::invokeATenFuncRet(ctx, at::nll_loss, out, atInput, atTarget, atWeight, reduction, ignore_index);
    } else if (dim == 4) {
        impl::aten::invokeATenFuncRet(ctx, at::nll_loss2d, out, atInput, atTarget, atWeight, reduction, ignore_index);
    } else {
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
        auto atOut = at::nll_loss2d(atInput, atTarget, atWeight, reduction, ignore_index);
        impl::aten::updateATen2Tensor(ctx, atOut, out);
    }
    return diopiSuccess;
}

diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
        diopiSize_t input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step) {
    at::IntArrayRef atInputSizes = impl::aten::buildAtIntArray(input_sizes);
    at::Tensor atGradOutput = impl::aten::buildATen(grad_output);
    impl::aten::invokeATenFuncRet(ctx, at::slice_backward, grad_input, atGradOutput, atInputSizes, dim, start, end, step);   
    return diopiSuccess;                                     
}

diopiError_t diopiIndexBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input,
        const diopiTensorHandle_t* indices, int64_t nums, const diopiTensorHandle_t grad) {
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
        const diopiTensorHandle_t input, const diopiTensorHandle_t target,
        const diopiTensorHandle_t weight,  diopiTensorHandle_t grad_input, float gamma, float alpha) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    at::Tensor atGradOutput = impl::aten::buildATen(grad_output);
    at::Tensor atWeight = impl::aten::buildATen(weight);
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

diopiError_t diopiRoiAlignBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t grad,
        const diopiTensorHandle_t rois, double spatialScale, int64_t pooledHeight, int64_t pooledWidth, int64_t batchSize,
        int64_t channels, int64_t height, int64_t width, int64_t samplingRatio, bool aligned) {
    auto atGrad = impl::aten::buildATen(grad);
    auto atRois = impl::aten::buildATen(rois);
    auto atOut = vision::ops::roi_align_backward_kernel(atGrad, atRois, spatialScale,
        pooledHeight, pooledWidth, batchSize, channels, height, width, samplingRatio, aligned);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
}

diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
        diopiTensorHandle_t grad3, const diopiTensorHandle_t grad_output, const diopiTensorHandle_t input,
        const diopiTensorHandle_t weight, diopiSize_t *bias_sizes, diopiSize_t stride, diopiSize_t padding,
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
        int64_t size = atGrad.dim();
        while (atBias.dim() != size) {
            atTmp = at::sum(atTmp, -1, false);
            size -= 1;
        }
        if (atBias.size(0) !=  atTmp.size(0)) {
            atTmp = at::sum(atTmp, -1, false);
        }
        impl::aten::updateATen2Tensor(ctx, atTmp, grad3);
    }
    return diopiSuccess;
} 

diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t grad,
                                    const diopiTensorHandle_t indices, int64_t numWeights, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    auto atGrad = impl::aten::buildATen(grad);
    auto atIndices = impl::aten::buildATen(indices);
    impl::aten::invokeATenFuncRet(ctx, at::embedding_backward, out, atGrad, atIndices, numWeights, paddingIdx, scaleGradByFreq, sparse);
    return diopiSuccess;
}

diopiError_t diopiAdaptiveAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                            const diopiTensorHandle_t grad_output, const diopiTensorHandle_t input) {
    auto atGradOutput  = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::_adaptive_avg_pool2d_backward, grad_input, atGradOutput, atInput);
    return diopiSuccess;
}

diopiError_t diopiLeakyReluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                    const diopiTensorHandle_t input, const diopiScalar_t* negative_slope, bool input_is_result) {
    auto atGradOutput  = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atSlope = impl::aten::buildAtScalar(negative_slope);
    impl::aten::invokeATenFuncRet(ctx, at::leaky_relu_backward, grad_input, atGradOutput, atInput, atSlope, input_is_result);
    return diopiSuccess;
}

diopiError_t diopiHardtanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                   const diopiTensorHandle_t input, const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atMin = impl::aten::buildAtScalar(min_val);
    auto atMax = impl::aten::buildAtScalar(max_val);
    impl::aten::invokeATenFuncRet(ctx, at::hardtanh_backward, grad_input, atGradOutput, atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiGeluBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                               const diopiTensorHandle_t input, const char* approximate) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::gelu_backward, grad_input, atGradOutput, atInput);
    return diopiSuccess;
}

diopiError_t diopiAvgPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                    const diopiTensorHandle_t grad_output, const diopiTensorHandle_t input,
                                    diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode,
                                    bool count_include_pad, const int64_t* divisor_override) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    c10::optional<int64_t> atDivisorOverride = divisor_override ? c10::optional<int64_t>(*divisor_override) : c10::nullopt;
    impl::aten::invokeATenFuncRet(ctx, at::avg_pool2d_backward, grad_input, atGradOutput, atInput, atKernelSize, atStride, atPadding, 
                                  ceil_mode, count_include_pad, atDivisorOverride);
    return diopiSuccess;
}

diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                  const diopiTensorHandle_t input, const diopiTensorHandle_t target, int64_t reduction) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    impl::aten::invokeATenFuncRet(ctx, at::mse_loss_backward, grad_input, atGradOutput, atInput, atTarget, reduction);
    return diopiSuccess;
}

diopiError_t diopiTanhBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                               const diopiTensorHandle_t grad_output, const diopiTensorHandle_t input) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::tanh_backward, grad_input, atGradOutput, atInput);
    return diopiSuccess; 
}

diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad,
                                      diopiSize_t input_sizes, int64_t dim, const diopiTensorHandle_t index) {
    auto atGrad = impl::aten::buildATen(grad);
    at::IntArrayRef atInputSize = impl::aten::buildAtIntArray(input_sizes);
    auto atIndex = impl::aten::buildATen(index);
    impl::aten::invokeATenFuncRet(ctx, at::index_select_backward, grad_input, atGrad, atInputSize, dim, atIndex);
    return diopiSuccess;
}

diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                 const diopiTensorHandle_t grad_output, diopiSize_t input_sizes, int64_t dim, int64_t index) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    at::IntArrayRef atInputSize = impl::aten::buildAtIntArray(input_sizes);
    impl::aten::invokeATenFuncRet(ctx, at::select_backward, grad_input, atGradOutput, atInputSize, dim, index);
    return diopiSuccess;
}

diopiError_t diopiSoftmaxBackwardData(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                      const diopiTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atOutput = impl::aten::buildATen(output);
    // TODO(huqingqing): use default type instead
    impl::aten::invokeATenFuncRet(ctx, at::_softmax_backward_data, grad_input, atGradOutput, atOutput, dim, atOutput);
    return diopiSuccess;
}

diopiError_t diopiLogSoftmaxBackwardData(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                         const diopiTensorHandle_t output, int64_t dim, diopiDtype_t input_dtype) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atOutput = impl::aten::buildATen(output);
    // TODO(huqingqing): use default type instead
    impl::aten::invokeATenFuncRet(ctx, at::_log_softmax_backward_data, grad_input, atGradOutput, atOutput, dim, atOutput);
    return diopiSuccess;
}

diopiError_t diopiSigmoidBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input,
                                  const diopiTensorHandle_t grad_output, const diopiTensorHandle_t output) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atOutput = impl::aten::buildATen(output);
    impl::aten::invokeATenFuncRet(ctx, at::sigmoid_backward, grad_input, atGradOutput, atOutput);
    return diopiSuccess;
}

diopiError_t diopiThresholdBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                    const diopiTensorHandle_t input, const diopiScalar_t* threshold) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atThreshold = impl::aten::buildAtScalar(threshold);
    impl::aten::invokeATenFuncRet(ctx, at::threshold_backward, grad_input, atGradOutput, atInput, atThreshold);
    return diopiSuccess;
}

diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                                  const diopiTensorHandle_t input, const diopiTensorHandle_t target, const diopiTensorHandle_t weight,
                                                  const diopiTensorHandle_t pos_weight, int64_t reduction) {
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

diopiError_t diopiCrossNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                       const diopiTensorHandle_t input, const diopiTensorHandle_t target, const diopiTensorHandle_t weight,
                                       int64_t reduction, int64_t ignore_index) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    auto atWeight = impl::aten::buildATen(weight);
    auto atTotalWeight = impl::aten::buildATen(input).resize_({1}).fill_(atTarget.numel());
    
    auto dim = atInput.dim();
    assert(dim > 1);
    if (dim == 2) {
        impl::aten::invokeATenFuncRet(ctx, at::nll_loss_backward, grad_input, atGradOutput, atInput, atTarget, atWeight, reduction,
                                      ignore_index, atTotalWeight);
    } else if (dim == 4) {
        impl::aten::invokeATenFuncRet(ctx, at::nll_loss2d_backward, grad_input, atGradOutput, atInput, atTarget, atWeight, reduction,
                                      ignore_index, atTotalWeight);
    } else {
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
        auto atGradInput = at::nll_loss2d_backward(atGradOutput, atInput, atTarget, atWeight, reduction, ignore_index, atTotalWeight);
        impl::aten::updateATen2Tensor(ctx, atGradInput, grad_input);
    }
    return diopiSuccess;
}

diopiError_t diopiMaxPool2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                    const diopiTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding,
                                    diopiSize_t dilation, bool ceil_mode, const diopiTensorHandle_t indices) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    at::IntArrayRef atKernelSize = impl::aten::buildAtIntArray(kernel_size);
    at::IntArrayRef atStride = impl::aten::buildAtIntArray(stride);
    at::IntArrayRef atPadding = impl::aten::buildAtIntArray(padding);
    at::IntArrayRef atDilation = impl::aten::buildAtIntArray(dilation);
    auto atIndices = impl::aten::buildATen(indices);
    impl::aten::invokeATenFuncRet(ctx, at::max_pool2d_with_indices_backward, grad_input, atGradOutput, atInput, atKernelSize, 
                                  atStride, atPadding, atDilation, ceil_mode, atIndices);
    return diopiSuccess;
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
        diopiTensorHandle_t grad_bias, const diopiTensorHandle_t grad_output, const diopiTensorHandle_t input, const diopiTensorHandle_t weight,
        const diopiTensorHandle_t running_mean, const diopiTensorHandle_t running_var, const diopiTensorHandle_t save_mean, 
        const diopiTensorHandle_t save_invstd, bool training, double eps) {
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
    auto reserve = at::empty({0}, atInput.options().dtype(at::kByte));
    diopi_tensor_list vecOut = {grad_input, grad_weight, grad_bias};
    auto grad_input_mask = std::array<bool, 3>{true, true, true};
    // impl::aten::invokeATenFuncRet(ctx, at::cudnn_batch_norm_backward, vecOut, atInput, atGradOutput,  atWeight, atRunningMean,
    //                               atRunningVar, atSaveMean, atSaveVar, eps, reserve);
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
    auto atOutput = at::arange_out(atOut, atStart, atEnd, atStep);
    impl::aten::updateATen2Tensor(ctx, atOutput, out);
    return diopiSuccess;
}

diopiError_t diopiRandperm(diopiContextHandle_t ctx, diopiTensorHandle_t out, int64_t n, int64_t idx) {
    auto atOut = impl::aten::buildATen(out);
    at::Device device("cuda");
    auto atOutput = at::randperm(n, device);
    impl::aten::updateATen2Tensor(ctx, atOutput, out);
    return diopiSuccess;
}

diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx) {
    auto atOut = impl::aten::buildATen(inout);
    auto atOutput = at::native::uniform_(atOut, from, to, c10::nullopt);
    return diopiSuccess;
}

diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx) {
    auto atOut = impl::aten::buildATen(inout);
    if (to==nullptr) {
        auto atOutput = at::native::random_(atOut, from, c10::nullopt, c10::nullopt);
    } else {
        auto atOutput = at::native::random_(atOut, from, *to, c10::nullopt);
    }
    return diopiSuccess;
}

diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t idx) {
    auto atOut = impl::aten::buildATen(inout);
    auto atOutput = at::bernoulli(atOut, c10::nullopt);
    return diopiSuccess;
}

diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input, int64_t idx) {
    auto atInput = impl::aten::buildATen(input);
    auto atOut = impl::aten::buildATen(out);
    auto atOutput = at::bernoulli_out(atOut, atInput, c10::nullopt);
    return diopiSuccess;
}

diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, int64_t idx) {
    auto atOut = impl::aten::buildATen(out);
    auto atOutput = at::bernoulli(atOut, p, c10::nullopt);
    return diopiSuccess;
}

diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input, 
                             const diopiTensorHandle_t mask, const diopiTensorHandle_t value){
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildATen(value);
    auto atOut = at::masked_fill(atInput, atMask, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input, const diopiTensorHandle_t mask, 
                                const diopiTensorHandle_t value){
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildATen(value);
    auto atOut = at::masked_fill(atInput, atMask, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, input);
    return diopiSuccess;
}

diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input, 
                                   const diopiTensorHandle_t mask, const diopiScalar_t* value){
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = at::masked_fill(atInput, atMask, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;        
}

diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, const diopiTensorHandle_t input, const diopiTensorHandle_t mask, 
                                      const diopiScalar_t* value){
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atValue = impl::aten::buildAtScalar(value);
    at::masked_fill(atInput, atMask, atValue);
    auto atOut = at::masked_fill(atInput, atMask, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, input);
    return diopiSuccess;
}

diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t grad,
                        diopiTensorHandle_t exp_avg, diopiTensorHandle_t exp_avg_sq, diopiTensorHandle_t max_exp_avg_sq,
                        float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad, bool maximize){
    
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
                       float lr, float beta1, float beta2, float eps, float weight_decay, int64_t step, bool amsgrad, bool maximize) {
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

diopiError_t diopiConvTranspose2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
                                  const diopiTensorHandle_t weight, const diopiTensorHandle_t bias, diopiSize_t stride, 
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

diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
                         int64_t dim, diopiDtype_t dtype){
    auto atInput = impl::aten::buildATen(input);
    auto atOut = at::cumsum(atInput, dim);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input1, const diopiTensorHandle_t input2,
                                  double p, const int64_t* compute_mode){
    auto atInput1 = impl::aten::buildATen(input1);
    auto atInput2 = impl::aten::buildATen(input2);
    c10::optional<int64_t> atComputMode = compute_mode ? c10::optional<int64_t>(*compute_mode) : c10::nullopt;
    impl::aten::invokeATenFuncRet(ctx, at::cdist, out, atInput1, atInput2, p, atComputMode);
    return diopiSuccess;
}

diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                const diopiTensorHandle_t input1, const diopiTensorHandle_t input2, double p, const diopiTensorHandle_t cdist) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput1 = impl::aten::buildATen(input1);
    auto atInput2 = impl::aten::buildATen(input2);
    auto atCdist = impl::aten::buildATen(cdist);
    impl::aten::invokeATenFuncRet(ctx, at::_cdist_backward, grad_input, atGradOutput, atInput1, atInput2, p, atCdist);
    return diopiSuccess;
}

diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::reciprocal, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input){
    auto atInput = impl::aten::buildATen(input);
    at::reciprocal_(atInput);
    return diopiSuccess;
}

diopiError_t diopiBitwiseNot(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    auto atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::bitwise_not, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiArgmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input, const int64_t* dim, bool keepdim) {
    auto atInput = impl::aten::buildATen(input);
    c10::optional<int64_t> atDim = dim ? c10::optional<int64_t>(*dim) : c10::nullopt;
    impl::aten::invokeATenFuncRet(ctx, at::argmax, out, atInput, atDim, keepdim);
    return diopiSuccess;
}

diopiError_t diopiSmoothL1Loss(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input, const diopiTensorHandle_t target,
                               int64_t reduction, double beta) {
    at::Tensor atInput = impl::aten::buildATen(input);
    at::Tensor atTarget = impl::aten::buildATen(target);
    impl::aten::invokeATenFuncRet(ctx, at::smooth_l1_loss, out, atInput, atTarget, reduction, beta);
    return diopiSuccess;
}

diopiError_t diopiSmoothL1LossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                       const diopiTensorHandle_t input, const diopiTensorHandle_t target, int64_t reduction, double beta) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atTarget = impl::aten::buildATen(target);
    impl::aten::invokeATenFuncRet(ctx, at::smooth_l1_loss_backward, grad_input, atGradOutput, atInput, atTarget, reduction, beta);
    return diopiSuccess;
}

diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t* out,
                               const diopiTensorHandle_t input, const diopiTensorHandle_t mask) {
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    auto atOut = at::masked_select(atInput, atMask);
    impl::aten::buildDiopiTensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, const diopiTensorHandle_t grad_output,
                                       const diopiTensorHandle_t input, const diopiTensorHandle_t mask) {
    auto atGradOutput = impl::aten::buildATen(grad_output);
    auto atInput = impl::aten::buildATen(input);
    auto atMask = impl::aten::buildATen(mask);
    impl::aten::invokeATenFuncRet(ctx, at::masked_select_backward, grad_input, atGradOutput, atInput, atMask);
    return diopiSuccess;
}

diopiError_t diopiIndexFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
                                  int64_t dim, const diopiTensorHandle_t index, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = at::index_fill(atInput, dim, atIndex, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess; 
}

diopiError_t diopiIndexFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
                            int64_t dim, const diopiTensorHandle_t index, const diopiTensorHandle_t value) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildATen(value);
    auto atOut = at::index_fill(atInput, dim, atIndex, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess; 
}

diopiError_t diopiIndexFillInpScalar(diopiContextHandle_t ctx, const diopiTensorHandle_t input,
                                     int64_t dim, const diopiTensorHandle_t index, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildAtScalar(value);
    auto atOut = at::index_fill(atInput, dim, atIndex, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, input);
    return diopiSuccess;  
}

diopiError_t diopiIndexFillInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input,
                               int64_t dim, const diopiTensorHandle_t index, const diopiTensorHandle_t value) {
    auto atInput = impl::aten::buildATen(input);
    auto atIndex = impl::aten::buildATen(index);
    auto atValue = impl::aten::buildATen(value);
    auto atOut = at::index_fill(atInput, dim, atIndex, atValue);
    impl::aten::updateATen2Tensor(ctx, atOut, input);
    return diopiSuccess;
}

diopiError_t diopiLinspace(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* start, const diopiScalar_t* end, int64_t steps) {
    auto atStart = impl::aten::buildAtScalar(start);
    auto atEnd = impl::aten::buildAtScalar(end);
    c10::optional<int64_t> atStep(steps);
    at::Tensor atOut = impl::aten::buildATen(out);
    linspace_out(atOut, atStart, atEnd, atStep);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}                          
}  // extern "C"
