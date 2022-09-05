/**
 * @file functions.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-09-05
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <diopi/functions.h>
#include <iostream>

<<<<<<< HEAD
#include "helper.hpp"

extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = buildAtTensor(input);
    invokeATenFuncRet(ctx, at::relu, out, atInput);
    return diopiSuccess;
}

extern "C" diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = buildAtTensor(input);
    invokeATenFuncInp(ctx, at::relu_, atInput);
    return diopiSuccess;
}

extern "C" diopiError_t diopiLeakyRelu(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input,
        const diopiScalar_t* negative_slope) {
    at::Tensor atInput = buildAtTensor(input);
    at::Scalar atSlope = buildAtScalar(input, negative_slope);
    invokeATenFuncRet(ctx, at::leaky_relu, out, atInput, atSlope);
    return diopiSuccess;
}

extern "C" diopiError_t diopiLeakyReluInp(diopiContextHandle_t ctx,
        diopiTensorHandle_t input, const diopiScalar_t* negative_slope) {
    at::Tensor atInput = buildAtTensor(input);
    at::Scalar atSlope = buildAtScalar(input, negative_slope);
    invokeATenFuncInp(ctx, at::leaky_relu_, atInput, atSlope);
    return diopiSuccess;
}

// inline ::std::tuple<at::Tensor, at::Tensor> at::max_pool2d_with_indices(const at::Tensor &self, at::IntArrayRef kernel_size, at::IntArrayRef stride = {}, at::IntArrayRef padding = 0, at::IntArrayRef dilation = 1, bool ceil_mode = false)
extern "C" diopiError_t diopiMaxPool2dWithIndices(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        diopiTensorHandle_t indices, const diopiTensorHandle_t input, diopiSize_t kernel_size,
        diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, bool ceil_mode) {
    at::Tensor atInput = buildAtTensor(input);
    at::IntArrayRef atKernelSize(kernel_size.data, kernel_size.len);
    at::IntArrayRef atStride(stride.data, stride.len);
    at::IntArrayRef atPadding(padding.data, padding.len);
    at::IntArrayRef atDilation(dilation.data, dilation.len);
    bool atCeilMode = ceil_mode;
    std::vector<diopiTensorHandle_t> vecOut = {out, indices};
    invokeATenFuncRet(ctx, at::max_pool2d_with_indices, vecOut,
        atInput, atKernelSize, atStride, atPadding, atDilation, atCeilMode);
}

extern "C" diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
    const diopiTensorHandle_t input, int64_t dim, diopiDtype_t dtype) {

}
=======
#include <torch/torch.h>

extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input)
{
    torch::Tensor tensor = torch::rand({2, 3});
    namespace F = torch::nn::functional;
    F::relu(tensor, F::ReLUFuncOptions().inplace(true));
    std::cout << tensor << std::endl;
}
>>>>>>> 2b42d4e (init lib torch)
