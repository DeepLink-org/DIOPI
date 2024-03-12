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
    // TODO: currently do not know "how strides best accelerate kernel"
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(x);
    DiopiTensor cosTensor(cos);
    DiopiTensor sinTensor(sin);
    DiopiTensor outputTensor(out);

    // change input data type, cos & sin only support float32, input support float16 & float32
    std::vector<DiopiTensor*> pTensors{&inputTensor, &cosTensor, &sinTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    diopiDtype_t calType = diopi_dtype_float32;
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor outTmpTr = outputTensor;
    if (outputTensor.dtype() != inputTensor.dtype()) {
        outTmpTr = requiresTensor(ctx, outputTensor.shape(), outputTensor.stride(), calType);
    }

    // split input into two part, via offset and stride without new storage;
    // interleaved == true: abcdefgh -> abcd,efgh in last dimension;
    // interleaved == false: abcdefgh ->aceg,bdfh in last dimension;
    int64_t storageOffset = 0;
    std::vector<int64_t> shape = inputTensor.shape();
    shape[shape.size() - 1] = shape[shape.size() - 1] >> 1;
    std::vector<int64_t> stride;

    // set Tensors' decriptor
    CnnlTensorDesc inputDesc;
    CnnlTensorDesc cosDesc;
    CnnlTensorDesc sinDesc;
    CnnlTensorDesc outTmpDesc;

    if (interleaved) {
        // currently the cpu/cuda reference do not support interleaved
        // camb is correct
        stride = inputTensor.stride();
        stride[stride.size() - 1] = stride[stride.size() - 1] << 1;
        storageOffset = sizeof(float);
    } else {
        stride = inputTensor.stride();
        storageOffset = sizeof(float) * shape[shape.size() - 1];
    }

    if (shape.size() == 1) {
        // input:[head_size]
        std::vector<int64_t> inputShape = {1, 1, 1, shape[0]};
        std::vector<int64_t> inputStride = {1, 1, 1, stride[0]};
        inputDesc.set(calType, inputShape, inputStride, CNNL_LAYOUT_ARRAY);
        outTmpDesc.set(calType, inputShape, inputStride, CNNL_LAYOUT_ARRAY);
    } else if (shape.size() == 2) {
        // input:[seqLen,head_size]
        std::vector<int64_t> inputShape = {1, shape[0], 1, shape[1]};
        std::vector<int64_t> inputStride = {1, stride[0], stride[0], stride[1]};
        inputDesc.set(calType, inputShape, inputStride, CNNL_LAYOUT_ARRAY);
        outTmpDesc.set(calType, inputShape, inputStride, CNNL_LAYOUT_ARRAY);
    } else if (shape.size() == 3) {
        // input:[seqLen,head_num,head_size]
        std::vector<int64_t> inputShape = {1, shape[0], shape[1], shape[2]};
        std::vector<int64_t> inputStride = {1, stride[0], stride[1], stride[2]};
        inputDesc.set(calType, inputShape, inputStride, CNNL_LAYOUT_ARRAY);
        outTmpDesc.set(calType, inputShape, inputStride, CNNL_LAYOUT_ARRAY);
    } else if (shape.size() == 4) {
        std::vector<int64_t> inputShape = shape;
        std::vector<int64_t> inputStride = stride;
        inputDesc.set(calType, inputShape, inputStride, CNNL_LAYOUT_ARRAY);
        outTmpDesc.set(calType, inputShape, inputStride, CNNL_LAYOUT_ARRAY);
    } else {
        DIOPI_CHECK(true, "camb currently only support input.dim() <= 4");
    }

    if (cosTensor.dim() == 1) {
        std::vector<int64_t> cosShape = {1, 1, cosTensor.shape()[0]};
        std::vector<int64_t> cosStride = calContiguousStride(cosShape);
        cosDesc.set(calType, cosShape, cosStride, CNNL_LAYOUT_ARRAY);
        sinDesc.set(calType, cosShape, cosStride, CNNL_LAYOUT_ARRAY);
    } else if (cosTensor.dim() == 2) {
        std::vector<int64_t> cosShape = {cosTensor.shape()[0], 1, cosTensor.shape()[1]};
        std::vector<int64_t> cosStride = calContiguousStride(cosShape);
        cosDesc.set(calType, cosShape, cosStride, CNNL_LAYOUT_ARRAY);
        sinDesc.set(calType, cosShape, cosStride, CNNL_LAYOUT_ARRAY);
    } else if (cosTensor.dim() == 3) {
        std::vector<int64_t> cosShape = {cosTensor.shape()[0], 1, cosTensor.shape()[2]};
        std::vector<int64_t> cosStride = calContiguousStride(cosShape);
        cosDesc.set(calType, cosShape, cosStride, CNNL_LAYOUT_ARRAY);
        sinDesc.set(calType, cosShape, cosStride, CNNL_LAYOUT_ARRAY);
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
                                        inputTensor.data() + storageOffset,
                                        cosDesc.get(),
                                        cosTensor.data(),
                                        sinDesc.get(),
                                        sinTensor.data(),
                                        workspace,
                                        workspaceSize,
                                        outTmpDesc.get(),
                                        outTmpTr.data(),
                                        outTmpDesc.get(),
                                        outTmpTr.data() + storageOffset));

    if (outTmpTr.dtype() != outputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outputTensor, outTmpTr));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
