/**
 * @file functions.cpp
 * @author fengsibo@sensetime.com
 * @brief 
 * @version 0.1
 * @date 2022-09-05
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <diopi/functions.h>
#include <torch/nn.h>
#include <torch/optim.h>

#include "helper.hpp"

#if defined(__cplusplus)
extern "C" {
#endif // __cplusplus

diopiError_t diopiRelu(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::relu, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::relu_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiScalar_t* negative_slope) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atSlope = impl::aten::buildAtScalar(negative_slope);
    impl::aten::invokeATenFuncRet(ctx, at::leaky_relu, out, atInput, atSlope);
    return diopiSuccess;
}

diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx,
        diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atSlope = impl::aten::buildAtScalar(negative_slope);
    impl::aten::invokeATenFuncInp(ctx, at::leaky_relu_, atInput, atSlope);
    return diopiSuccess;
}

diopiError_t diopiMaxPool2d(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride,
        diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
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
    at::Tensor atInput = impl::aten::buildAtTensor(input);
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
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
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
    auto atInput = impl::aten::buildAtTensor(input);
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
    auto atInput = impl::aten::buildAtTensor(input);
    auto atWeight = impl::aten::buildAtTensor(weight);
    auto atBias = impl::aten::buildAtTensor(bias);
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
    auto atInput = impl::aten::buildAtTensor(input);
    auto atTarget = impl::aten::buildAtTensor(target);
    auto atWeight = impl::aten::buildAtTensor(weight);
    // auto atReduction = impl::aten::getEntropyReduction(reduction);
    // auto atOut = torch::nn::functional(atInput, atTarget, atWeight, ignore_index, atReduction, label_smoothing);
    // impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t mat2) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atMat2= impl::aten::buildAtTensor(mat2);
    impl::aten::invokeATenFuncRet(ctx, at::bmm, out, atInput, atMat2);
    return diopiSuccess;
}

diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t tensor1, const diopiTensorHandle_t tensor2, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atTensor1 = impl::aten::buildAtTensor(tensor1);
    auto atTensor2 = impl::aten::buildAtTensor(tensor2);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncRet(ctx, at::addcmul, out, atInput, atTensor1, atTensor2, atValue);
    return diopiSuccess;
}

diopiError_t diopiMatmul(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOther = impl::aten::buildAtTensor(other);
    impl::aten::invokeATenFuncRet(ctx, at::matmul, out, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t tensor1, const diopiTensorHandle_t tensor2, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atTensor1 = impl::aten::buildAtTensor(tensor1);
    auto atTensor2 = impl::aten::buildAtTensor(tensor2);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncRet(ctx, at::addcdiv, out, atInput, atTensor1, atTensor2, atValue);
    return diopiSuccess;
}

// CAFFE2_API Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t mat1, 
        const diopiTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atMax1 = impl::aten::buildAtTensor(mat1);
    auto atMax2 = impl::aten::buildAtTensor(mat2);
    auto atBeta = impl::aten::buildAtScalar(beta);
    auto atAlpha = impl::aten::buildAtScalar(alpha);
    impl::aten::invokeATenFuncRet(
        ctx, at::addmm, out, atInput, atMax1, atMax2, atBeta, atAlpha);
    return diopiSuccess;
}

// NOTE(fengsibo@sensetime.com): add int, short, bool test case
diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atDim = impl::aten::buildAtIntArray(dim);
    auto atOut = at::mean(atInput, atDim);  // TODO(fengsibo@sensetime.com): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

// NOTE(fengsibo@sensetime.com): add int, short, bool test case
diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, diopiSize_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atDim = impl::aten::buildAtIntArray(dim);
    auto atOut = at::sum(atInput, atDim);  // TODO(fengsibo@sensetime.com): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiStd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, diopiSize_t dim, bool unbiased) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atDim = impl::aten::buildAtIntArray(dim);
    auto atOut = at::std(atInput, atDim, unbiased);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t min_indices,
        const diopiTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOuts = at::min(atInput, dim);
    diopi_tensor_list outs = {min, min_indices};
    impl::aten::updateATen2Tensor(ctx, atOuts, outs); 
    return diopiSuccess;
}

diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t max_indices,
        const diopiTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOuts = at::max(atInput, dim);
    diopi_tensor_list outs = {max, max_indices};
    impl::aten::updateATen2Tensor(ctx, atOuts, outs); 
    return diopiSuccess;
}

diopiError_t diopiAny(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOut = at::any(atInput, dim);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAll(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOut = at::all(atInput, dim);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOut = at::softmax(atInput, dim);  // TODO(fengsibo@sensetime.com): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOut = at::log_softmax(atInput, dim);  // TODO(fengsibo@sensetime.com): use default type instead
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim, const diopiTensorHandle_t index) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atIndex = impl::aten::buildAtTensor(index);
    auto atOut = at::index_select(atInput, dim, atIndex);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

// diopiError_t diopiSelectCopy(diopiContextHandle_t ctx, diopiTensorHandle_t out,
//         const diopiTensorHandle_t input, int64_t dim, int64_t index) {
//     auto atInput = impl::aten::buildAtTensor(input);
//     impl::aten::invokeATenFuncRet
//         <at::Tensor (*)(at::Tensor const &, int64_t, int64_t)>(ctx, at::select, out, atInput, dim, index);
//     return diopiSuccess;
// }

diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t null_out,
        const diopiTensorHandle_t input, int64_t dim, int64_t start, int64_t end, int64_t step) {
    auto atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::slice, null_out, atInput, dim, start, end, step);
    return diopiSuccess;
}

diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t mask, const diopiTensorHandle_t source) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atMask = impl::aten::buildAtTensor(mask);
    auto atSource = impl::aten::buildAtTensor(source);
    impl::aten::invokeATenFuncRet(ctx, at::masked_scatter, out, atInput, atMask, atSource);
    return diopiSuccess;
}

diopiError_t diopiNonzero(diopiContextHandle_t ctx,
        diopiTensorHandle_t* out, const diopiTensorHandle_t input) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOut = at::nonzero(atInput);
    impl::aten::buildDIOPITensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiTensorHandle_t weight, const diopiTensorHandle_t bias) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atWeight = impl::aten::buildAtTensor(weight);
    auto atBias = impl::aten::buildAtTensor(bias);
    impl::aten::invokeATenFuncRet(ctx, at::linear, out, atInput, atWeight, atBias);
    return diopiSuccess;
}

// diopiError_t diopiRoiAlign(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
//         const diopiTensorHandle_t rois, double spatialScale, int64_t pooledHeight,
//         int64_t pooledWidth, int64_t samplingRatio, bool aligned) {
    
// }

// // TODO(fengsibo@sensetime.com) not implement
// diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
//         const diopiTensorHandle_t w, const diopiTensorHandle_t dw,
//         double learningrate, double momentum, double dampening, double weightDecay, bool nesterov) {
//     auto atW = impl::aten::buildAtTensor(w);
//     auto atDw = impl::aten::buildAtTensor(dw);
//     std::vector<at::Tensor> params = {atW, atDw};

//     torch::optim::SGD sgd(
//         params,
//         torch::optim::SGDOptions(learningrate)
//         .momentum(momentum)
//         .nesterov(nesterov)
//         .weight_decay(weightDecay));
//     sgd.step();
//     return diopiSuccess;
// }

/**
 * @brief 
 * @param errorIfNonfinite supported in pytorch ?
 * @return diopiError_t 
 */
diopiError_t diopiClipGradNorm(diopiContextHandle_t ctx, double* out, diopiTensorHandle_t* parameters,
        int64_t parametersNum, double maxNorm, double normType, bool errorIfNonfinite) {
    auto tensorList = impl::aten::buildAtTensorList(parameters, parametersNum);
    *out = torch::nn::utils::clip_grad_norm_(tensorList, maxNorm, normType);
    return diopiSuccess;
}

diopiError_t diopiEmbeddingRenorm_(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t indices, double max_norm, double norm_type) {
    auto atSelf = impl::aten::buildAtTensor(out);
    auto atIndices = impl::aten::buildAtTensor(indices);
    impl::aten::invokeATenFuncRet(
        ctx, at::embedding_renorm_, out, atSelf, atIndices, max_norm, norm_type);
    return diopiSuccess;
}

diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t weight,
        const diopiTensorHandle_t indices, int64_t paddingIdx, bool scaleGradByFreq, bool sparse) {
    auto atWeight = impl::aten::buildAtTensor(weight);
    auto atIndices = impl::aten::buildAtTensor(indices);
    impl::aten::invokeATenFuncRet(ctx, at::embedding, out, atWeight, atIndices, paddingIdx, scaleGradByFreq, sparse);
    return diopiSuccess;
}

diopiError_t diopiTril(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t diagonal) {
    auto atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::tril, out, atInput, diagonal);
    return diopiSuccess;
}

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t* tensors, int64_t insNum, int64_t dim) {
    auto tensorList = impl::aten::buildAtTensorList(tensors, insNum);
    impl::aten::invokeATenFuncRet
        <at::Tensor (*)(at::TensorList, int64_t)>(ctx, at::cat, out, tensorList, dim);
    return diopiSuccess;
}

diopiError_t diopiSplitWithSizes(diopiContextHandle_t ctx, diopiTensorHandle_t* outs, int64_t outsNum,
        const diopiTensorHandle_t input, const diopiSize_t* splitSizes, int64_t dim) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atSizes = impl::aten::buildAtIntArray(splitSizes);
    diopi_tensor_list vecOut;
    for (size_t i = 0; i < outsNum; ++i) {
        vecOut.emplace_back(outs[i]);
    }
    impl::aten::invokeATenFuncRet(ctx, at::split_with_sizes, vecOut, atInput, atSizes, dim);
    return diopiSuccess;
}

diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    auto tensorList = impl::aten::buildAtTensorList(tensors, numTensors);
    
    std::vector<at::Tensor> a = impl::aten::buildAtTensorList(tensors, numTensors);
    at::TensorList b = impl::aten::buildAtTensorList(tensors, numTensors);
    
    impl::aten::invokeATenFuncRet(ctx, at::stack, out, tensorList, dim);
    return diopiSuccess;
}

diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
        const diopiTensorHandle_t input, int64_t dim, bool descending, const bool stable) {
    auto atInput = impl::aten::buildAtTensor(input);
    diopi_tensor_list vecOut = {values, indices};
#if TORCH_MM_VERSION < TORCH_1_9_MM_VERSION
    impl::aten::invokeATenFuncRet
        <std::tuple<at::Tensor, at::Tensor> (*)(at::Tensor const &, int64_t, bool)>
        (ctx, at::sort, vecOut, atInput, dim, descending);
#else
    impl::aten::invokeATenFuncRet
        <std::tuple<at::Tensor, at::Tensor> (*)(at::Tensor const &, c10::optional<bool>, int64_t, bool)>
        (ctx, at::sort, vecOut, atInput, stable, dim, descending);
#endif
    return diopiSuccess;
}

diopiError_t diopiTopk(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
        const diopiTensorHandle_t input, int64_t k, int64_t dim, bool largest, bool sorted) {
    auto atInput = impl::aten::buildAtTensor(input);
    diopi_tensor_list vecOut = {values, indices};
    impl::aten::invokeATenFuncRet(ctx, at::topk, vecOut, atInput, k, dim, largest, sorted);
    return diopiSuccess;
}

diopiError_t diopiTranspose(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t dim0, int64_t dim1) {
    auto atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet<at::Tensor (*)(at::Tensor const&, int64_t, int64_t)>
        (ctx, at::transpose, out, atInput, dim0, dim1);
    return diopiSuccess;
}

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, int64_t numClasses) {
    auto atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::one_hot, out, atInput, numClasses);
    return diopiSuccess;
}

diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t condition,
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    auto atCondition = impl::aten::buildAtTensor(condition);
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOther = impl::aten::buildAtTensor(other);
    impl::aten::invokeATenFuncRet
        <at::Tensor (*)(at::Tensor const&, at::Tensor const&, at::Tensor const&)>
        (ctx, at::where, out, atCondition, atInput, atOther);
    return diopiSuccess;
}

diopiError_t diopiSin(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::sin, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiSinInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::sin_, atInput);
    return diopiSuccess;
}

diopiError_t diopiCos(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::cos, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiCosInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::cos_, atInput);
    return diopiSuccess;
}

diopiError_t diopiAbs(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::abs, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiAbsInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::abs_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSqrt(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::sqrt, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiSqrtInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::sqrt_, atInput);
    return diopiSuccess;
}

diopiError_t diopiFloor(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::floor, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiFloorInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::floor_, atInput);
    return diopiSuccess;
}

diopiError_t diopiNeg(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::neg, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiNegInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::neg_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSign(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::sign, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiTanh(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::tanh, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiTanhInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::tanh_, atInput);
    return diopiSuccess;
}

diopiError_t diopiSigmoid(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::sigmoid, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiSigmoidInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::sigmoid_, atInput);
    return diopiSuccess;
}

diopiError_t diopiExp(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::exp, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiExpInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::exp_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::log, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiLogInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::log_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog2(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::log2, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog2Inp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::log2_, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog10(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::log10, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiLog10Inp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::log10_, atInput);
    return diopiSuccess;
}

diopiError_t diopiErf(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::erf, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiErfInp(diopiContextHandle_t ctx, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncInp(ctx, at::erf_, atInput);
    return diopiSuccess;
}

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiScalar_t* input, const diopiTensorHandle_t exponent) {
    at::Tensor atExponent = impl::aten::buildAtTensor(exponent);
    at::Scalar atInput = impl::aten::buildAtScalar(input);
    at::Tensor atOut = at::pow(atInput, atExponent);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* exponent) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atExponent = impl::aten::buildAtScalar(exponent);
    at::Tensor atOut = at::pow(atInput, atExponent);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiTensorHandle_t exponent) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atExponent = impl::aten::buildAtTensor(exponent);
    at::Tensor atOut = at::pow(atInput, atExponent);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::add(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::add(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::sub(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(alpha);
    at::Tensor atOut = at::sub(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Tensor atOut = at::mul(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::mul(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiGe(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Tensor atOut = at::ge(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiGeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::ge(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiGt(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Tensor atOut = at::gt(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiGtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::gt(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLe(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Tensor atOut = at::le(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::le(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLt(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Tensor atOut = at::lt(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiLtScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::lt(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiEq(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Tensor atOut = at::eq(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiEqScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::eq(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiNe(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Tensor atOut = at::ne(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiNeScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(other);
    at::Tensor atOut = at::ne(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atTmpInput = impl::aten::buildAtTensor(input);
    at::Tensor atTmpOther = impl::aten::buildAtTensor(other);
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOther = atTmpOther.to(at::ScalarType::Bool);
    at::Tensor atOut = at::bitwise_and(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}


diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atTmpInput = impl::aten::buildAtTensor(input);
    at::Scalar atTmpOther = impl::aten::buildAtScalar(other);
    at::Scalar atOther = atTmpOther.to<bool>();
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOut = at::bitwise_and(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atTmpInput = impl::aten::buildAtTensor(input);
    at::Tensor atTmpOther = impl::aten::buildAtTensor(other);
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOther = atTmpOther.to(at::ScalarType::Bool);
    at::Tensor atOut = at::bitwise_or(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atTmpInput = impl::aten::buildAtTensor(input);
    at::Scalar atTmpOther = impl::aten::buildAtScalar(other);
    at::Scalar atOther = atTmpOther.to<bool>();
    at::Tensor atInput = atTmpInput.to(at::ScalarType::Bool);
    at::Tensor atOut = at::bitwise_or(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiScalar_t* min, const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::clamp_(atInput, atMin, atMax);
    return diopiSuccess;
}

// todo: pytorch 1.7 don't support Tensor min and tensor max
diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiTensorHandle_t min, const diopiTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atMin = impl::aten::buildAtTensor(min);
    at::Tensor atMax = impl::aten::buildAtTensor(max);
    //at::clamp_(atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::Tensor atOut = at::clamp(atInput, atMin, atMax);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        diopiTensorHandle_t input, const diopiTensorHandle_t min, const diopiTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atMin = impl::aten::buildAtTensor(min);
    at::Tensor atMax = impl::aten::buildAtTensor(max);
    //at::Tensor atOut = at::clamp(atInput, atMin, atMax);
    //impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::clamp_(atInput, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atMax = impl::aten::buildAtTensor(max);
    //at::clamp_(atInput, atMax);
    return diopiSuccess;
}

diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        diopiTensorHandle_t input, const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atMax = impl::aten::buildAtScalar(max);
    at::Tensor atOut = at::clamp(atInput, atMax);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        diopiTensorHandle_t input, const diopiTensorHandle_t max) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atMax = impl::aten::buildAtTensor(max);
    //at::Tensor atOut = at::clamp(atInput, atMax);
    //impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiScalar_t* min) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::clamp_(atInput, atMin);
    return diopiSuccess;
}

diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiTensorHandle_t min) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atMin = impl::aten::buildAtTensor(min);
    //at::clamp_(atInput, atMin);
    return diopiSuccess;
}

diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        diopiTensorHandle_t input, const diopiScalar_t* min) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atMin = impl::aten::buildAtScalar(min);
    at::Tensor atOut = at::clamp(atInput, atMin);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        diopiTensorHandle_t input, const diopiTensorHandle_t min) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atMin = impl::aten::buildAtTensor(min);
    //at::Tensor atOut = at::clamp(atInput, atMin);
    //impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiFill(diopiContextHandle_t ctx, 
        diopiTensorHandle_t input, const float value) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::fill_(atInput, value);
    return diopiSuccess;
}


diopiError_t diopiHardtanh(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
                           const diopiScalar_t* min_val, const diopiScalar_t* max_val) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atMin = impl::aten::buildAtScalar(min_val);
    auto atMax = impl::aten::buildAtScalar(max_val);
    impl::aten::invokeATenFuncRet(ctx, at::hardtanh, out, atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiHardtanhInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min_val, 
                              const diopiScalar_t* max_val) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atMin = impl::aten::buildAtScalar(min_val);
    auto atMax = impl::aten::buildAtScalar(max_val);
    impl::aten::invokeATenFuncInp(ctx, at::hardtanh_, atInput, atMin, atMax);
    return diopiSuccess;
}

diopiError_t diopiThreshold(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input,
                                     const diopiScalar_t* threshold, const diopiScalar_t* value) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atThreshold = impl::aten::buildAtScalar(threshold);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncRet(ctx, at::threshold, out, atInput, atThreshold, atValue);
    return diopiSuccess;
}
diopiError_t diopiThresholdInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* threshold,
                               const diopiScalar_t* value) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atThreshold = impl::aten::buildAtScalar(threshold);
    auto atValue = impl::aten::buildAtScalar(value);
    impl::aten::invokeATenFuncInp(ctx, at::threshold_, atInput, atThreshold, atValue);
    return diopiSuccess;
}

diopiError_t diopiGelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
                       const diopiTensorHandle_t input, const char* approximate) {
    auto atInput = impl::aten::buildAtTensor(input);
    impl::aten::invokeATenFuncRet(ctx, at::gelu, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiCrossNLLLoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
                               const diopiTensorHandle_t input, const diopiTensorHandle_t target, 
                               const diopiTensorHandle_t weight, int64_t reduction, int64_t ignore_index) {
    auto buildAtInput = impl::aten::buildAtTensor(input);
    auto buildAtTarget = impl::aten::buildAtTensor(target);
    auto atWeight = impl::aten::buildAtTensor(weight);
    auto atInput = buildAtInput;
    auto atTarget = buildAtTarget;
    if(buildAtInput.dim() > 2){
        auto channel = buildAtInput.size(1);
        int64_t totalSize = 1;
        for(int i=0; i<buildAtInput.dim(); ++i){
            totalSize *= buildAtInput.size(i);
        }
        std::vector<int64_t> toShape {channel, totalSize/channel};
        at::IntArrayRef intArrayShape = impl::aten::buildAtIntArray(diopiSize_t(toShape.data(), toShape.size()));
        atInput = buildAtInput.transpose(0, 1).reshape(intArrayShape).transpose(0, 1);
    }
    if(buildAtTarget.dim() > 1){
        int64_t totalSize = 1;
        for(int i=0; i<buildAtTarget.dim(); ++i){
            totalSize *= buildAtTarget.size(i);
        }
        atTarget = buildAtTarget.reshape(totalSize);
    }
    auto atOut = at::nll_loss(atInput, atTarget, atWeight, reduction, ignore_index);
    if(reduction == at::Reduction::None && buildAtTarget.dim() > 1){
        std::vector<int64_t> toShape;
        for(int i=0; i<buildAtTarget.dim(); ++i){
            toShape.push_back(buildAtTarget.size(i));
        }
        at::IntArrayRef intArrayShape = impl::aten::buildAtIntArray(diopiSize_t(toShape.data(), toShape.size()));
        atOut = atOut.reshape(intArrayShape);
    }
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

#if defined(__cplusplus)
}
#endif // __cplusplus
