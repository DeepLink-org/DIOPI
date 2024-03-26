/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <diopi/functions_ext.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../mlu_helper.hpp"

namespace impl {
namespace camb {

diopiError_t diopiRotaryEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t x, diopiConstTensorHandle_t cos,
                                  diopiConstTensorHandle_t sin, const bool conj, const bool interleaved) {
    // TODO: currently input,cos,sin are contiguous, we do not know "how strides best accelerate kernel"
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(x);
    DiopiTensor cosTensor(cos);
    DiopiTensor sinTensor(sin);
    DiopiTensor outputTensor(out);

    if (shapeHasZero(inputTensor.shape())) {
        // zero shape protection
        return diopiSuccess;
    }

    // change input data type, (in the camb first input method, support half, bfloat16 and float)
    std::vector<DiopiTensor*> pTensors{&inputTensor, &cosTensor, &sinTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_float16, diopi_dtype_bfloat16};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor outTmpTr = outputTensor;
    if (outputTensor.dtype() != inputTensor.dtype()) {
        outTmpTr = requiresTensor(ctx, outputTensor.shape(), outputTensor.stride(), inputTensor.dtype());
    }

    // split input into two part, via offset and stride without new storage;
    // interleaved == false: abcdefgh -> abcd,efgh in last dimension;
    // interleaved == true: abcdefgh ->aceg,bdfh in last dimension;
    int64_t storageOffset = 0;
    std::vector<int64_t> shape = inputTensor.shape();
    shape[shape.size() - 1] = shape[shape.size() - 1] >> 1;
    std::vector<int64_t> stride;

    if (interleaved) {
        // currently the cpu/cuda reference do not support interleaved
        // but camb is correct
        stride = inputTensor.stride();
        stride[stride.size() - 1] = stride[stride.size() - 1] << 1;
        storageOffset = 1;
    } else {
        stride = inputTensor.stride();
        storageOffset = shape[shape.size() - 1];
    }
    void* input2Ptr;
    void* output2Ptr;
    if (inputTensor.dtype() == diopi_dtype_float32) {
        input2Ptr = static_cast<void*>((static_cast<float*>(inputTensor.data())) + storageOffset);
        output2Ptr = static_cast<void*>((static_cast<float*>(outTmpTr.data())) + storageOffset);
    } else {
        // short 和bfloat和float都是16位，内存移动相同
        input2Ptr = static_cast<void*>((static_cast<short*>(inputTensor.data())) + storageOffset);
        output2Ptr = static_cast<void*>((static_cast<short*>(outTmpTr.data())) + storageOffset);
    }

    // set Tensors' decriptor
    CnnlTensorDesc inputDesc;
    CnnlTensorDesc cosDesc;
    CnnlTensorDesc sinDesc;
    CnnlTensorDesc outTmpDesc;

    if (shape.size() == 1) {
        // input:[head_dim]
        std::vector<int64_t> inputShape = {1, 1, 1, shape[0]};
        std::vector<int64_t> inputStride = {stride[0], stride[0], stride[0], stride[0]};
        inputDesc.set(inputTensor.dtype(), inputShape, inputStride, CNNL_LAYOUT_ARRAY);
        outTmpDesc.set(inputTensor.dtype(), inputShape, inputStride, CNNL_LAYOUT_ARRAY);
    } else if (shape.size() == 2) {
        // input:[seqLen,head_dim]
        std::vector<int64_t> inputShape = {1, shape[0], 1, shape[1]};
        std::vector<int64_t> inputStride = {stride[0], stride[0], stride[1], stride[1]};
        inputDesc.set(inputTensor.dtype(), inputShape, inputStride, CNNL_LAYOUT_ARRAY);
        outTmpDesc.set(inputTensor.dtype(), inputShape, inputStride, CNNL_LAYOUT_ARRAY);
    } else if (shape.size() == 3) {
        // input:[seqLen,head_num,head_dim]
        std::vector<int64_t> inputShape = {1, shape[0], shape[1], shape[2]};
        std::vector<int64_t> inputStride = {stride[0], stride[0], stride[1], stride[2]};
        inputDesc.set(inputTensor.dtype(), inputShape, inputStride, CNNL_LAYOUT_ARRAY);
        outTmpDesc.set(inputTensor.dtype(), inputShape, inputStride, CNNL_LAYOUT_ARRAY);
    } else if (shape.size() == 4) {
        inputDesc.set(inputTensor.dtype(), shape, stride, CNNL_LAYOUT_ARRAY);
        outTmpDesc.set(inputTensor.dtype(), shape, stride, CNNL_LAYOUT_ARRAY);
    } else {
        DIOPI_CHECK(true, "camb currently only support 1<= input.dim() <= 4");
    }

    if (cosTensor.dim() == 1) {
        std::vector<int64_t> cosShape = {1, 1, cosTensor.shape()[0]};
        std::vector<int64_t> cosStride = calContiguousStride(cosShape);
        cosDesc.set(cosTensor.dtype(), cosShape, cosStride, CNNL_LAYOUT_ARRAY);
        sinDesc.set(sinTensor.dtype(), cosShape, cosStride, CNNL_LAYOUT_ARRAY);
    } else if (cosTensor.dim() == 2) {
        std::vector<int64_t> cosShape = {cosTensor.shape()[0], 1, cosTensor.shape()[1]};
        std::vector<int64_t> cosStride = calContiguousStride(cosShape);
        cosDesc.set(cosTensor.dtype(), cosShape, cosStride, CNNL_LAYOUT_ARRAY);
        sinDesc.set(sinTensor.dtype(), cosShape, cosStride, CNNL_LAYOUT_ARRAY);
    } else if (cosTensor.dim() == 3) {
        std::vector<int64_t> cosShape = {cosTensor.shape()[0], 1, cosTensor.shape()[2]};
        std::vector<int64_t> cosStride = calContiguousStride(cosShape);
        cosDesc.set(cosTensor.dtype(), cosShape, cosStride, CNNL_LAYOUT_ARRAY);
        sinDesc.set(sinTensor.dtype(), cosShape, cosStride, CNNL_LAYOUT_ARRAY);
    } else {
        DIOPI_CHECK(true, "camb currently only support 1<= cos.dim() <= 3");
    }

    // set embedding Descriptor
    CnnlResourceGuard<cnnlRotaryEmbeddingDescriptor_t, cnnlCreateRotaryEmbeddingDescriptor, cnnlDestroyRotaryEmbeddingDescriptor> embeddingDesc;
    cnnlSeqDataLayout_t seqLayout = CNNL_SEQDATA_NTBC;
    DIOPI_CALL_CNNL(cnnlSetRotaryEmbeddingDescriptor(embeddingDesc.get(), conj, seqLayout));

    size_t workspaceSize = 0;
    void* workspace = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALL_CNNL(cnnlRotaryEmbedding(handle,
                                        embeddingDesc.get(),
                                        inputDesc.get(),
                                        inputTensor.data(),
                                        inputDesc.get(),
                                        input2Ptr,
                                        cosDesc.get(),
                                        cosTensor.data(),
                                        sinDesc.get(),
                                        sinTensor.data(),
                                        workspace,
                                        workspaceSize,
                                        outTmpDesc.get(),
                                        outTmpTr.data(),
                                        outTmpDesc.get(),
                                        output2Ptr));

    if (outTmpTr.dtype() != outputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outputTensor, outTmpTr));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
