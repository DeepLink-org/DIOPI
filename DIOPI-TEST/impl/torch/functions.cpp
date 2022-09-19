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

#include "helper.hpp"

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
    at::Scalar atInput = impl::aten::buildAtScalar(exponent, input);
    at::Tensor atOut = at::pow(atInput, atExponent);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out,
        const diopiTensorHandle_t input, const diopiScalar_t* exponent) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atExponent = impl::aten::buildAtScalar(input, exponent);
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
    at::Scalar atAlpha = impl::aten::buildAtScalar(input, alpha);
    at::Tensor atOut = at::add(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(input, alpha);
    at::Tensor atOut = at::add(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSub(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(input, alpha);
    at::Tensor atOut = at::sub(atInput, atOther, atAlpha);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiSubScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
    at::Scalar atAlpha = impl::aten::buildAtScalar(input, alpha);
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
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
    at::Tensor atOut = at::mul(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Tensor atOut = at::div(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
    // todo: convert rounding_mode to string
    at::Tensor atOut = at::div(atInput, atOther);
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
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
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
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
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
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
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
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
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
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
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
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
    at::Tensor atOut = at::ne(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseAnd(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Tensor atOut = at::bitwise_and(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseAndScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
    // todo: cast input to bool
    at::Tensor atOut = at::bitwise_and(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOr(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiTensorHandle_t other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Tensor atOther = impl::aten::buildAtTensor(other);
    at::Tensor atOut = at::bitwise_or(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiBitwiseOrScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, 
        const diopiTensorHandle_t input, const diopiScalar_t* other) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atOther = impl::aten::buildAtScalar(input, other);
    at::Tensor atOut = at::bitwise_or(atInput, atOther);
    impl::aten::updateATen2Tensor(ctx, atOut, out);
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input,
        const diopiScalar_t* min, const diopiScalar_t* max) {
    at::Tensor atInput = impl::aten::buildAtTensor(input);
    at::Scalar atMin = impl::aten::buildAtScalar(input, min);
    at::Scalar atMax = impl::aten::buildAtScalar(input, max);
    at::clamp_(atInput, atMin, atMax);
    return diopiSuccess;
}

// pytorch 1.7 don't support
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
    at::Scalar atMin = impl::aten::buildAtScalar(input, min);
    at::Scalar atMax = impl::aten::buildAtScalar(input, max);
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
    at::Scalar atMax = impl::aten::buildAtScalar(input, max);
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
    at::Scalar atMax = impl::aten::buildAtScalar(input, max);
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
    at::Scalar atMin = impl::aten::buildAtScalar(input, min);
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
    at::Scalar atMin = impl::aten::buildAtScalar(input, min);
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