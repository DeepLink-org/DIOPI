/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

namespace {
diopiError_t softmaxForward(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor output, int64_t dim, bool isLog = false) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputCasted = input;
    DiopiTensor outputCasted = output;

    std::vector<DiopiTensor*> tensors{&inputCasted, &outputCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
    std::vector<int> srcInputShape{inputCasted.shape().begin(), inputCasted.shape().end()};
    std::vector<int> srcOutputShape{outputCasted.shape().begin(), outputCasted.shape().end()};

    const int inputRank = inputCasted.shape().size();
    int mode = dim;
    mode = (mode < 0) ? (mode + inputRank) : mode;
    const size_t inputDim = 3;
    std::vector<int> inputShape(inputDim, 1);
    if (inputRank != 0) {
        if (inputRank <= 3) {
            inputShape[2] = srcInputShape[inputRank - 1];
            inputShape[1] = (inputRank == 1) ? 1 : srcInputShape[inputRank - 2];
            inputShape[0] = (inputRank == 3) ? srcInputShape[0] : 1;
        } else {
            auto reduceDim = [](const std::vector<int>& data, int from, int to) -> int {
                to = std::min<int>(to, data.size());
                from = std::max<int>(0, from);
                return std::accumulate(data.cbegin() + from, data.cbegin() + to + 1, 1LL, std::multiplies<int64_t>());
            };
            const bool flag = (mode == inputRank - 1);
            inputShape[0] = reduceDim(srcInputShape, 0, flag ? (mode - 2) : (mode - 1));
            inputShape[1] = srcInputShape[flag ? (mode - 1) : mode];
            inputShape[2] = reduceDim(srcInputShape, flag ? mode : (mode + 1), (inputRank - 1));
        }
    }
    cnnlSoftmaxMode_t modeTmp;
    if (inputRank == 3 && mode == 0) {
        modeTmp = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
    } else if (mode == inputRank - 1) {
        modeTmp = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
    } else {
        modeTmp = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
    }

    const float alpha = 1;
    const float beta = 0;

    CnnlTensorDesc xDesc, yDesc;
    DIOPI_CALL(xDesc.set(inputCasted, CNNL_LAYOUT_ARRAY, inputShape));
    DIOPI_CALL(yDesc.set(outputCasted, CNNL_LAYOUT_ARRAY, inputShape));
    DIOPI_CALLCNNL(cnnlSoftmaxForward_v2(handle,
                                         isLog ? CNNL_SOFTMAX_LOG : CNNL_SOFTMAX_ACCURATE,
                                         modeTmp,
                                         CNNL_COMPUTATION_FAST,
                                         &alpha,
                                         xDesc.get(),
                                         inputCasted.data(),
                                         &beta,
                                         yDesc.get(),
                                         outputCasted.data()));

    DIOPI_CALL(dataTypeCast(ctx, output, outputCasted));
    return diopiSuccess;
}

diopiError_t softmaxBackward(diopiContextHandle_t ctx, DiopiTensor gradInputTensor, DiopiTensor gradOutputTensor, DiopiTensor outputTensor, int64_t dim,
                              bool isLog = false) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor gradInputCasted = gradInputTensor;
    DiopiTensor gradOutputCasted = gradOutputTensor;
    DiopiTensor outputCasted = outputTensor;

    std::vector<DiopiTensor*> tensors{&gradInputCasted, &gradOutputCasted, &outputCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    std::vector<int> srcOutputShape{outputCasted.shape().begin(), outputCasted.shape().end()};

    const int inputRank = gradInputCasted.shape().size();

    const size_t inputDim = 3;
    int mode = dim;
    std::vector<int> outputShape(inputDim, 1);
    if (inputRank != 0) {
        if (inputRank <= 3) {
            outputShape[2] = srcOutputShape[inputRank - 1];
            outputShape[1] = (inputRank == 1) ? 1 : srcOutputShape[inputRank - 2];
            outputShape[0] = (inputRank == 3) ? srcOutputShape[0] : 1;
        } else {
            auto reduceDim = [](const std::vector<int>& data, int from, int to) -> int {
                to = std::min<int>(to, data.size());
                from = std::max<int>(0, from);
                return std::accumulate(data.cbegin() + from, data.cbegin() + to + 1, 1LL, std::multiplies<int64_t>());
            };
            const bool flag = (mode == inputRank - 1);
            outputShape[0] = reduceDim(srcOutputShape, 0, flag ? (mode - 2) : (mode - 1));
            outputShape[1] = srcOutputShape[flag ? (mode - 1) : mode];
            outputShape[2] = reduceDim(srcOutputShape, flag ? mode : (mode + 1), (inputRank - 1));
        }
    }

    mode = (mode < 0) ? (mode + inputRank) : mode;

    cnnlSoftmaxMode_t modeTmp;
    if (inputRank == 3 && mode == 0) {
        modeTmp = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
    } else if (mode == inputRank - 1) {
        modeTmp = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
    } else {
        modeTmp = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
    }

    CnnlTensorDesc gradInputDesc, gradOutputDesc, outputDesc;
    DIOPI_CALL(gradInputDesc.set(gradInputCasted, CNNL_LAYOUT_ARRAY, outputShape));
    DIOPI_CALL(gradOutputDesc.set(gradOutputCasted, CNNL_LAYOUT_ARRAY, outputShape));
    DIOPI_CALL(outputDesc.set(outputCasted, CNNL_LAYOUT_ARRAY, outputShape));

    DIOPI_CALLCNNL(cnnlSoftmaxBackward(handle,
                                       isLog ? CNNL_SOFTMAX_LOG : CNNL_SOFTMAX_ACCURATE,
                                       modeTmp,
                                       NULL,
                                       outputDesc.get(),
                                       outputCasted.data(),
                                       gradOutputDesc.get(),
                                       gradOutputCasted.data(),
                                       NULL,
                                       gradInputDesc.get(),
                                       gradInputCasted.data()));
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputCasted));
    return diopiSuccess;
}

}  // namespace

extern "C" diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    DIOPI_CALL(softmaxForward(ctx, inputTensor, outputTensor, dim));
    return diopiSuccess;
}

extern "C" diopiError_t diopiSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                             diopiConstTensorHandle_t output, int64_t dim) {
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor outputTensor(output);
    DIOPI_CALL(softmaxBackward(ctx, gradInputTensor, gradOutputTensor, outputTensor, dim));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLogSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);
    DIOPI_CALL(softmaxForward(ctx, inputTensor, outputTensor, dim, true));
    return diopiSuccess;
}

extern "C" diopiError_t diopiLogSoftmaxBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                diopiConstTensorHandle_t output, int64_t dim) {
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor outputTensor(output);
    DIOPI_CALL(softmaxBackward(ctx, gradInputTensor, gradOutputTensor, outputTensor, dim, true));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
