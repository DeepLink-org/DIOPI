
/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include <diopi/functions_ext.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiFlashAttention(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t softmaxLse, diopiGeneratorHandle_t gen,
                                 diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t alibiSlopes,
                                 float pDropout, float softmaxScale, bool isCausal, int32_t windowSizeLeft, int32_t windowSizeRight) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DIOPI_CHECK(alibiSlopes == nullptr, "For camb, flash attention currently does not support Attention with Linear Biases (ALiBi)!");

    DiopiTensor qTensor(q);
    DiopiTensor kTensor(k);
    DiopiTensor vTensor(v);
    DiopiTensor attentionOutTensor(attentionOut);
    DiopiTensor softmaxLseTensor(softmaxLse);

    std::vector<int64_t> oriShapeQ = qTensor.shape();
    const int64_t batchSize = oriShapeQ[0];
    const int64_t seqLenQ = oriShapeQ[1];
    const int64_t headNumQ = oriShapeQ[2];
    const int64_t headDim = oriShapeQ[3];

    std::vector<int64_t> oriShapeK = kTensor.shape();
    const int64_t seqLenK = oriShapeK[1];
    const int64_t headNumK = oriShapeK[2];
    const int64_t totalSeqQ = batchSize * seqLenQ;
    const int64_t totalSeqK = batchSize * seqLenK;

    DIOPI_CHECK(batchSize > 0, "Batch size must be positive.");
    DIOPI_CHECK(headDim <= 256, "For camb, flash attention only supports head dimension at most 256.");
    DIOPI_CHECK(headNumQ % headNumK == 0, "Number of heads in key/value must divide number of heads in query.");

    std::vector<int64_t> shapeQ = {totalSeqQ, headNumQ, headDim};
    std::vector<int64_t> strideQ = calContiguousStride(shapeQ);
    std::vector<int64_t> shapeKV = {totalSeqK, headNumK, headDim};
    std::vector<int64_t> strideKV = calContiguousStride(shapeKV);
    std::vector<int64_t> cumSeqShape = {batchSize + 1};
    std::vector<int64_t> cumSeqStride = {1};
    std::vector<int64_t> softmaxLseShape = {headNumQ, totalSeqQ};
    std::vector<int64_t> softmaxLseStride = calContiguousStride(softmaxLseShape);

    // get cumSeqQ and cumSeqKV on cpu
    DiopiTensor cumSeqQ = requiresTensor(ctx, cumSeqShape, diopi_dtype_int32, diopiDevice_t::diopi_host);
    DiopiTensor cumSeqKV = requiresTensor(ctx, cumSeqShape, diopi_dtype_int32, diopiDevice_t::diopi_host);
    int* cumSeqQPtr = static_cast<int*>(cumSeqQ.data());
    int* cumSeqKVPtr = static_cast<int*>(cumSeqKV.data());
    for (int32_t i = 0; i < batchSize + 1; i++) {
        cumSeqQPtr[i] = i * seqLenQ;
        cumSeqKVPtr[i] = i * seqLenK;
    }

    // require cumSeqQTensor & cumSeqKVTensor on device
    DiopiTensor cumSeqQTensor = requiresTensor(ctx, cumSeqShape, cumSeqStride, diopi_dtype_int32);
    DiopiTensor cumSeqKVTensor = requiresTensor(ctx, cumSeqShape, cumSeqStride, diopi_dtype_int32);
    uint64_t bytes = sizeof(int32_t) * (batchSize + 1);
    cnrtMemcpyAsync_V2(cumSeqQTensor.data(), static_cast<void*>(cumSeqQPtr), bytes, getStream(ctx), cnrtMemcpyHostToDev);
    cnrtMemcpyAsync_V2(cumSeqKVTensor.data(), static_cast<void*>(cumSeqKVPtr), bytes, getStream(ctx), cnrtMemcpyHostToDev);

    // convert dtype
    DIOPI_CALL(autoCastTensorType(ctx, {&qTensor, &kTensor, &vTensor}, {diopi_dtype_float16, diopi_dtype_bfloat16}));

    DiopiTensor attentionOutTensorTmp = attentionOutTensor;
    if (attentionOutTensor.dtype() != qTensor.dtype()) {
        attentionOutTensorTmp = requiresTensor(ctx, attentionOutTensor.shape(), attentionOutTensor.stride(), qTensor.dtype());
    }

    DiopiTensor softmaxLseTensorTmp = softmaxLseTensor;
    if (softmaxLseTensor.dtype() != diopi_dtype_float32) {
        softmaxLseTensorTmp = requiresTensor(ctx, softmaxLseTensor.shape(), softmaxLseTensor.stride(), diopi_dtype_float32);
    }

    // set descriptor
    CnnlResourceGuard<cnnlFlashAttentionDescriptor_t, cnnlCreateFlashAttentionDescriptor, cnnlDestroyFlashAttentionDescriptor> flashAttentionDesc;
    cnnlAttentionMaskMode_t maskMode = isCausal ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE;
    DIOPI_CALL_CNNL(cnnlSetFlashAttentionDescriptor(
        flashAttentionDesc.get(), CNNL_DTYPE_FLOAT, CNNL_ACTIVATION_HIGH_PRECISION, maskMode, true, false, false, seqLenQ, seqLenK, pDropout, softmaxScale));

    DIOPI_CALL_CNNL(cnnlSetFlashAttentionSlidingWindowSize(flashAttentionDesc.get(), windowSizeLeft, windowSizeRight, 1));

    CnnlTensorDesc qDesc;
    CnnlTensorDesc kDesc;
    CnnlTensorDesc vDesc;
    CnnlTensorDesc cumSeqQTensorDesc;
    CnnlTensorDesc cumSeqKVTensorDesc;
    CnnlTensorDesc attentionOutTmpDesc;
    CnnlTensorDesc softmaxLseTmpDesc;

    qDesc.set(qTensor.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    kDesc.set(kTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);
    vDesc.set(vTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);
    cumSeqQTensorDesc.set(cumSeqQTensor.dtype(), cumSeqShape, cumSeqStride, CNNL_LAYOUT_ARRAY);
    cumSeqKVTensorDesc.set(cumSeqKVTensor.dtype(), cumSeqShape, cumSeqStride, CNNL_LAYOUT_ARRAY);
    attentionOutTmpDesc.set(attentionOutTensorTmp.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    softmaxLseTmpDesc.set(softmaxLseTensorTmp.dtype(), softmaxLseShape, softmaxLseStride, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetFlashAttentionForwardWorkspaceSize(handle, flashAttentionDesc.get(), qDesc.get(), kDesc.get(), vDesc.get(), &workspaceSize));
    void* workspace = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    // get rng state
    size_t rngState[2];
    rngState[0] = 0;
    rngState[1] = 0;
    if (pDropout > 0.0) {
        DIOPI_CALL(diopiGeneratorGetSeedAndOffset(gen, &(rngState[0]), &(rngState[1])));
    }

    DIOPI_CALL_CNNL(cnnlFlashAttentionForward(handle,
                                              flashAttentionDesc.get(),
                                              qDesc.get(),
                                              qTensor.data(),
                                              kDesc.get(),
                                              kTensor.data(),
                                              vDesc.get(),
                                              vTensor.data(),
                                              cumSeqQTensorDesc.get(),
                                              cumSeqQTensor.data(),
                                              cumSeqKVTensorDesc.get(),
                                              cumSeqKVTensor.data(),
                                              rngState,
                                              workspace,
                                              workspaceSize,
                                              nullptr,
                                              nullptr,
                                              softmaxLseTmpDesc.get(),
                                              softmaxLseTensorTmp.data(),
                                              attentionOutTmpDesc.get(),
                                              attentionOutTensorTmp.data()))

    if (attentionOutTensorTmp.dtype() != attentionOutTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, attentionOutTensor, attentionOutTensorTmp));
    }

    if (softmaxLseTensor.dtype() != diopi_dtype_float32) {
        DIOPI_CALL(dataTypeCast(ctx, softmaxLseTensor, softmaxLseTensorTmp));
    }

    return diopiSuccess;
}

diopiError_t diopiFlashAttentionBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                         diopiConstTensorHandle_t gradOutput, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q,
                                         diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t alibiSlopes,
                                         diopiConstTensorHandle_t attentionOut, diopiConstTensorHandle_t softmaxLse, float pDropout, float softmaxScale,
                                         bool isCausal, int32_t windowSizeLeft, int32_t windowSizeRight) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DIOPI_CHECK(alibiSlopes == nullptr, "For camb, flash attention currently does not support Attention with Linear Biases (ALiBi)!");

    DiopiTensor qTensor(q);
    DiopiTensor kTensor(k);
    DiopiTensor vTensor(v);
    DiopiTensor gradOutTensor(gradOutput);
    DiopiTensor attentionOutTensor(attentionOut);
    DiopiTensor softmaxLseTensor(softmaxLse);

    DiopiTensor gradQTensor(gradQ);
    DiopiTensor gradKTensor(gradK);
    DiopiTensor gradVTensor(gradV);

    std::vector<int64_t> oriShapeQ = qTensor.shape();
    const int64_t batchSize = oriShapeQ[0];
    const int64_t seqLenQ = oriShapeQ[1];
    const int64_t headNumQ = oriShapeQ[2];
    const int64_t headDim = oriShapeQ[3];

    std::vector<int64_t> oriShapeK = kTensor.shape();
    const int64_t seqLenK = oriShapeK[1];
    const int64_t headNumK = oriShapeK[2];
    const int64_t totalSeqQ = batchSize * seqLenQ;
    const int64_t totalSeqK = batchSize * seqLenK;

    DIOPI_CHECK(batchSize > 0, "Batch size must be positive.");
    DIOPI_CHECK(headDim <= 256, "For camb, flash attention only supports head dimension at most 256.");
    DIOPI_CHECK(headNumQ % headNumK == 0, "Number of heads in key/value must divide number of heads in query.");

    std::vector<int64_t> shapeQ = {totalSeqQ, headNumQ, headDim};
    std::vector<int64_t> strideQ = calContiguousStride(shapeQ);
    std::vector<int64_t> shapeKV = {totalSeqK, headNumK, headDim};
    std::vector<int64_t> strideKV = calContiguousStride(shapeKV);
    std::vector<int64_t> cumSeqShape = {batchSize + 1};
    std::vector<int64_t> cumSeqStride = {1};
    std::vector<int64_t> softmaxLseShape = {headNumQ, totalSeqQ};
    std::vector<int64_t> softmaxLseStride = calContiguousStride(softmaxLseShape);

    // get cumSeqQ and cumSeqKV on cpu
    DiopiTensor cumSeqQ = requiresTensor(ctx, cumSeqShape, diopi_dtype_int32, diopiDevice_t::diopi_host);
    DiopiTensor cumSeqKV = requiresTensor(ctx, cumSeqShape, diopi_dtype_int32, diopiDevice_t::diopi_host);
    int* cumSeqQPtr = static_cast<int*>(cumSeqQ.data());
    int* cumSeqKVPtr = static_cast<int*>(cumSeqKV.data());
    for (int32_t i = 0; i < batchSize + 1; i++) {
        cumSeqQPtr[i] = i * seqLenQ;
        cumSeqKVPtr[i] = i * seqLenK;
    }

    // require cumSeqQTensor & cumSeqKVTensor on device
    DiopiTensor cumSeqQTensor = requiresTensor(ctx, cumSeqShape, cumSeqStride, diopi_dtype_int32);
    DiopiTensor cumSeqKVTensor = requiresTensor(ctx, cumSeqShape, cumSeqStride, diopi_dtype_int32);
    uint64_t bytes = sizeof(int32_t) * (batchSize + 1);
    cnrtMemcpyAsync_V2(cumSeqQTensor.data(), static_cast<void*>(cumSeqQPtr), bytes, getStream(ctx), cnrtMemcpyHostToDev);
    cnrtMemcpyAsync_V2(cumSeqKVTensor.data(), static_cast<void*>(cumSeqKVPtr), bytes, getStream(ctx), cnrtMemcpyHostToDev);

    // convert dtype
    DIOPI_CALL(autoCastTensorType(ctx, {&qTensor, &kTensor, &vTensor, &gradOutTensor, &attentionOutTensor}, {diopi_dtype_float16, diopi_dtype_bfloat16}));
    DIOPI_CALL(autoCastTensorType(ctx, {&softmaxLseTensor}, {diopi_dtype_float32}));

    DiopiTensor gradQTensorTmp = gradQTensor;
    if (gradQTensor.dtype() != qTensor.dtype()) {
        gradQTensorTmp = requiresTensor(ctx, gradQTensor.shape(), gradQTensor.stride(), qTensor.dtype());
    }

    DiopiTensor gradKTensorTmp = gradKTensor;
    if (gradKTensor.dtype() != qTensor.dtype()) {
        gradKTensorTmp = requiresTensor(ctx, gradKTensor.shape(), gradKTensor.stride(), qTensor.dtype());
    }

    DiopiTensor gradVTensorTmp = gradVTensor;
    if (gradVTensor.dtype() != qTensor.dtype()) {
        gradVTensorTmp = requiresTensor(ctx, gradVTensor.shape(), gradVTensor.stride(), qTensor.dtype());
    }

    // set descriptor
    CnnlResourceGuard<cnnlFlashAttentionDescriptor_t, cnnlCreateFlashAttentionDescriptor, cnnlDestroyFlashAttentionDescriptor> flashAttentionDesc;
    cnnlAttentionMaskMode_t maskMode = isCausal ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE;
    DIOPI_CALL_CNNL(cnnlSetFlashAttentionBackwardDescriptor(
        flashAttentionDesc.get(), CNNL_DTYPE_FLOAT, CNNL_ACTIVATION_HIGH_PRECISION, maskMode, true, false, false, seqLenQ, seqLenK, pDropout, softmaxScale));

    DIOPI_CALL_CNNL(cnnlSetFlashAttentionSlidingWindowSize(flashAttentionDesc.get(), windowSizeLeft, windowSizeRight, 1));

    CnnlTensorDesc qDesc;
    CnnlTensorDesc kDesc;
    CnnlTensorDesc vDesc;
    CnnlTensorDesc cumSeqQTensorDesc;
    CnnlTensorDesc cumSeqKVTensorDesc;
    CnnlTensorDesc gradOutDesc;
    CnnlTensorDesc attentionOutDesc;
    CnnlTensorDesc softmaxLseDesc;
    CnnlTensorDesc gradQDesc;
    CnnlTensorDesc gradKDesc;
    CnnlTensorDesc gradVDesc;

    qDesc.set(qTensor.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    kDesc.set(kTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);
    vDesc.set(vTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);
    cumSeqQTensorDesc.set(cumSeqQTensor.dtype(), cumSeqShape, cumSeqStride, CNNL_LAYOUT_ARRAY);
    cumSeqKVTensorDesc.set(cumSeqKVTensor.dtype(), cumSeqShape, cumSeqStride, CNNL_LAYOUT_ARRAY);

    gradOutDesc.set(gradOutTensor.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    attentionOutDesc.set(attentionOutTensor.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    softmaxLseDesc.set(softmaxLseTensor.dtype(), softmaxLseShape, softmaxLseStride, CNNL_LAYOUT_ARRAY);
    gradQDesc.set(qTensor.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    gradKDesc.set(kTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);
    gradVDesc.set(vTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);

    // get workspace
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetFlashAttentionBackwardWorkspaceSize(handle, flashAttentionDesc.get(), qDesc.get(), kDesc.get(), vDesc.get(), &workspaceSize));
    void* workspace = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    // get rng state
    size_t rngState[2];
    rngState[0] = 0;
    rngState[1] = 0;
    if (pDropout > 0.0) {
        DIOPI_CALL(diopiGeneratorGetSeedAndOffset(gen, &(rngState[0]), &(rngState[1])));
    }

    DIOPI_CALL_CNNL(cnnlFlashAttentionBackward(handle,
                                               flashAttentionDesc.get(),
                                               gradOutDesc.get(),
                                               gradOutTensor.data(),
                                               qDesc.get(),
                                               qTensor.data(),
                                               kDesc.get(),
                                               kTensor.data(),
                                               vDesc.get(),
                                               vTensor.data(),
                                               attentionOutDesc.get(),
                                               attentionOutTensor.data(),
                                               softmaxLseDesc.get(),
                                               softmaxLseTensor.data(),
                                               cumSeqQTensorDesc.get(),
                                               cumSeqQTensor.data(),
                                               cumSeqKVTensorDesc.get(),
                                               cumSeqKVTensor.data(),
                                               rngState,
                                               workspace,
                                               workspaceSize,
                                               gradQDesc.get(),
                                               gradQTensorTmp.data(),
                                               gradKDesc.get(),
                                               gradKTensorTmp.data(),
                                               gradVDesc.get(),
                                               gradVTensorTmp.data(),
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr))

    if (gradQTensorTmp.dtype() != gradQTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradQTensor, gradQTensorTmp));
    }

    if (gradKTensorTmp.dtype() != gradKTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradKTensor, gradKTensorTmp));
    }

    if (gradVTensorTmp.dtype() != gradVTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradVTensor, gradVTensorTmp));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
