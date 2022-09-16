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
    at::Scalar atSlope = impl::aten::buildAtScalar(input, negative_slope);
    impl::aten::invokeATenFuncRet(ctx, at::leaky_relu, out, atInput, atSlope);
    return diopiSuccess;
}

diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx,
        diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atSlope = impl::aten::buildAtScalar(input, negative_slope);
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
    impl::aten::invokeATenFuncRet
        <at::Tensor (*)(at::Tensor const&, at::Tensor const&)>
        (ctx, at::div, out, atInput, atOther);
    return diopiSuccess;
}

/**
 * @brief 
 * @param rounding_mode supported in pytorch>=1.8.0
 */
diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOther = impl::aten::buildAtScalar(input, other);
    impl::aten::invokeATenFuncRet
        <at::Tensor (*)(at::Tensor const&, c10::Scalar)>
        (ctx, at::div, out, atInput, atOther);
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
    impl::aten::invokeATenFuncRet(ctx, at::conv2d, out,
        atInput, atWeight, atBias, atStride, atPadding, atDilation, groups);
    return diopiSuccess;
}

diopiError_t diopiNonzero(diopiContextHandle_t ctx,
        diopiTensorHandle_t* out, const diopiTensorHandle_t input) {
    auto atInput = impl::aten::buildAtTensor(input);
    auto atOut = at::nonzero(atInput);
    at::IntArrayRef atSize = atOut.sizes();
    at::IntArrayRef atStride = atOut.strides();
    diopiSize_t size(const_cast<int64_t*>(atStride.data()), atStride.size());
    diopiSize_t stride(const_cast<int64_t*>(atStride.data()), atStride.size());
    diopiDtype_t dtype;
    diopiGetTensorDtype(*out, &dtype);
    diopiRequireTensor(ctx, out, &size, &stride, dtype, diopi_device);
    impl::aten::updateATen2Tensor(ctx, atOut, *out);
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

diopiError_t diopiSgd(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t w, const diopiTensorHandle_t dw,
        float lr, float momentum, float dampening, float weightDecay, bool nesterov) {
    auto atW = impl::aten::buildAtTensor(w);
    auto atDw = impl::aten::buildAtTensor(dw);
    std::vector<at::Tensor> params = {atW, atDw};

    torch::optim::SGD sgd(
          params,
          torch::optim::SGDOptions(lr)
            .momentum(momentum)
            .nesterov(nesterov)
            .weight_decay(weightDecay));
    auto atOut = sgd.step();
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

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
    impl::aten::invokeATenFuncRet(ctx, at::stack, out, tensorList, dim);
    return diopiSuccess;
}

/**
 * @brief 
 * 
 * @param stable supported in pytorch>=1.8.0
 * @return diopiError_t 
 */
diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices,
        const diopiTensorHandle_t input, int64_t dim, bool descending, const bool* stable) {
    auto atInput = impl::aten::buildAtTensor(input);
    diopi_tensor_list vecOut = {values, indices};
    impl::aten::invokeATenFuncRet
        <std::tuple<at::Tensor, at::Tensor> (*)(at::Tensor const &, int64_t, bool)>
        (ctx, at::sort, vecOut, atInput, dim, descending);
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

#if defined(__cplusplus)
}
#endif // __cplusplus
