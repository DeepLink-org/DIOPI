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

diopiError_t diopiFlashAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t softmaxLse, diopiGeneratorHandle_t gen,
                                       diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t cumSeqQ,
                                       diopiConstTensorHandle_t cumSeqKV, diopiConstTensorHandle_t alibiSlopes, int32_t maxSeqLenQ, int32_t maxSeqLenKV,
                                       float pDropout, float softmaxScale, bool isCausal, int32_t windowSizeLeft, int32_t windowSizeRight) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DIOPI_CHECK(alibiSlopes == nullptr, "For camb, flash attention currently does not support Attention with Linear Biases (ALiBi)!");

    DiopiTensor qTensor(q);
    DiopiTensor kTensor(k);
    DiopiTensor vTensor(v);
    DiopiTensor cumSeqQTensor(cumSeqQ);
    DiopiTensor cumSeqKVTensor(cumSeqKV);
    DiopiTensor softmaxLseTensor(softmaxLse);
    DiopiTensor attentionOutTensor(attentionOut);

    DIOPI_CHECK(qTensor.dim() == 3 && kTensor.dim() == 3 && vTensor.dim() == 3, "cnnlFlashAttention should have 3-D qkv");

    const int64_t totalSeqQ = qTensor.shape()[0];
    const int64_t headNumQ = qTensor.shape()[1];
    const int64_t headDim = qTensor.shape()[2];
    const int64_t headNumK = kTensor.shape()[1];
    DIOPI_CHECK(headDim <= 256, "For camb, flash attention only supports head dimension at most 256.");
    DIOPI_CHECK(headNumQ % headNumK == 0, "Number of heads in key/value must divide number of heads in query.");

    // convert dtype
    DIOPI_CALL(autoCastTensorType(ctx, {&qTensor, &kTensor, &vTensor}, {diopi_dtype_float16}));
    DIOPI_CALL(autoCastTensorType(ctx, {&cumSeqQTensor, &cumSeqKVTensor}, {diopi_dtype_int32}));

    DiopiTensor attentionOutTensorTmp = attentionOutTensor;
    if (attentionOutTensor.dtype() != qTensor.dtype()) {
        attentionOutTensorTmp = requiresTensor(ctx, attentionOutTensor.shape(), attentionOutTensor.stride(), qTensor.dtype());
    }

    DiopiTensor softmaxLseTensorTmp = softmaxLseTensor;
    if (softmaxLseTensorTmp.dtype() != diopi_dtype_float32) {
        softmaxLseTensorTmp = requiresTensor(ctx, softmaxLseTensor.shape(), softmaxLseTensor.stride(), diopi_dtype_float32);
    }

    // set descriptor
    CnnlResourceGuard<cnnlFlashAttentionDescriptor_t, cnnlCreateFlashAttentionDescriptor, cnnlDestroyFlashAttentionDescriptor> flashAttentionDesc;
    cnnlAttentionMaskMode_t maskMode = isCausal ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE;
    DIOPI_CALL_CNNL(cnnlSetFlashAttentionDescriptor(flashAttentionDesc.get(),
                                                    CNNL_DTYPE_FLOAT,
                                                    CNNL_ACTIVATION_HIGH_PRECISION,
                                                    maskMode,
                                                    true,
                                                    false,
                                                    false,
                                                    maxSeqLenQ,
                                                    maxSeqLenKV,
                                                    pDropout,
                                                    softmaxScale));

    DIOPI_CALL_CNNL(cnnlSetFlashAttentionSlidingWindowSize(flashAttentionDesc.get(), windowSizeLeft, windowSizeRight, 1));

    CnnlTensorDesc qDesc(qTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc kDesc(kTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc vDesc(vTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc cumSeqQDesc(cumSeqQTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc cumSeqKVDesc(cumSeqKVTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc attentionOutTmpDesc(attentionOutTensorTmp, CNNL_LAYOUT_ARRAY);

    std::vector<int64_t> softmaxLseShape = {headNumQ, totalSeqQ};
    std::vector<int64_t> softmaxLseStride = calContiguousStride(softmaxLseShape);
    CnnlTensorDesc softmaxLseTmpDesc;
    softmaxLseTmpDesc.set(softmaxLseTensorTmp.dtype(), softmaxLseShape, softmaxLseStride, CNNL_LAYOUT_ARRAY);

    // get workspace
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

    DIOPI_CALL_CNNL(cnnlFlashAttentionForward_v2(handle,
                                                 flashAttentionDesc.get(),
                                                 qDesc.get(),
                                                 qTensor.data(),
                                                 kDesc.get(),
                                                 kTensor.data(),
                                                 vDesc.get(),
                                                 vTensor.data(),
                                                 cumSeqQDesc.get(),
                                                 cumSeqQTensor.data(),
                                                 cumSeqKVDesc.get(),
                                                 cumSeqKVTensor.data(),
                                                 nullptr,  //[seqused_k desc],for future used
                                                 nullptr,  //[seqused_k],for future used
                                                 nullptr,  //[attn mask desc ],for future used
                                                 nullptr,  //[attn mask],for future used
                                                 nullptr,  //[alibi slopes desc], for future used
                                                 nullptr,  //[alibi slopes], for future used
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

    if (softmaxLseTensorTmp.dtype() != softmaxLseTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, softmaxLseTensor, softmaxLseTensorTmp));
    }

    return diopiSuccess;
}

diopiError_t diopiFlashAttentionVarLenBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV,
                                               diopiConstTensorHandle_t gradOutput, diopiGeneratorHandle_t gen, diopiConstTensorHandle_t q,
                                               diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t cumSeqQ,
                                               diopiConstTensorHandle_t cumSeqKV, diopiConstTensorHandle_t alibiSlopes, diopiConstTensorHandle_t attentionOut,
                                               diopiConstTensorHandle_t softmaxLse, int32_t maxSeqLenQ, int32_t maxSeqLenKV, float pDropout, float softmaxScale,
                                               bool isCausal, int32_t windowSizeLeft, int32_t windowSizeRight) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DIOPI_CHECK(alibiSlopes == nullptr, "For camb, flash attention currently does not support Attention with Linear Biases (ALiBi)!");

    DiopiTensor qTensor(q);
    DiopiTensor kTensor(k);
    DiopiTensor vTensor(v);
    DiopiTensor cumSeqQTensor(cumSeqQ);
    DiopiTensor cumSeqKVTensor(cumSeqKV);
    DiopiTensor gradOutTensor(gradOutput);
    DiopiTensor attentionOutTensor(attentionOut);
    DiopiTensor softmaxLseTensor(softmaxLse);

    DiopiTensor gradQTensor(gradQ);
    DiopiTensor gradKTensor(gradK);
    DiopiTensor gradVTensor(gradV);

    DIOPI_CHECK(qTensor.dim() == 3 && kTensor.dim() == 3 && vTensor.dim() == 3, "cnnlFlashAttention should have 3-D qkv");

    const int64_t totalSeqQ = qTensor.shape()[0];
    const int64_t headNumQ = qTensor.shape()[1];
    const int64_t headDim = qTensor.shape()[2];
    const int64_t headNumK = kTensor.shape()[1];
    DIOPI_CHECK(headDim <= 256, "For camb, flash attention only supports head dimension at most 256.");
    DIOPI_CHECK(headNumQ % headNumK == 0, "Number of heads in key/value must divide number of heads in query.");

    // convert dtype
    DIOPI_CALL(autoCastTensorType(ctx, {&qTensor, &kTensor, &vTensor, &attentionOutTensor, &gradOutTensor}, {diopi_dtype_float16}));
    DIOPI_CALL(autoCastTensorType(ctx, {&cumSeqQTensor, &cumSeqKVTensor}, {diopi_dtype_int32}));
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
    DIOPI_CALL_CNNL(cnnlSetFlashAttentionBackwardDescriptor(flashAttentionDesc.get(),
                                                            CNNL_DTYPE_FLOAT,
                                                            CNNL_ACTIVATION_HIGH_PRECISION,
                                                            maskMode,
                                                            true,
                                                            false,
                                                            false,
                                                            maxSeqLenQ,
                                                            maxSeqLenKV,
                                                            pDropout,
                                                            softmaxScale));

    DIOPI_CALL_CNNL(cnnlSetFlashAttentionSlidingWindowSize(flashAttentionDesc.get(), windowSizeLeft, windowSizeRight, 1));

    CnnlTensorDesc qDesc(qTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc kDesc(kTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc vDesc(vTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc cumSeqQDesc(cumSeqQTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc cumSeqKVDesc(cumSeqKVTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutDesc(gradOutTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc attentionOutDesc(attentionOutTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradQDesc(gradQTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradKDesc(gradKTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradVDesc(gradVTensorTmp, CNNL_LAYOUT_ARRAY);

    std::vector<int64_t> softmaxLseShape = {headNumQ, totalSeqQ};
    std::vector<int64_t> softmaxLseStride = calContiguousStride(softmaxLseShape);
    CnnlTensorDesc softmaxLseDesc;
    softmaxLseDesc.set(softmaxLseTensor.dtype(), softmaxLseShape, softmaxLseStride, CNNL_LAYOUT_ARRAY);

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

    DIOPI_CALL_CNNL(cnnlFlashAttentionBackward_v2(handle,
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
                                                  cumSeqQDesc.get(),
                                                  cumSeqQTensor.data(),
                                                  cumSeqKVDesc.get(),
                                                  cumSeqKVTensor.data(),
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
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
                                                  nullptr));

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
