/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cnmlrt.h>
#include <diopi/functions_ext.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../mlu_helper.hpp"

namespace impl {
namespace camb {

diopiError_t diopiFlashAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t attentionOut, diopiTensorHandle_t softmaxLse, diopiGeneratorHandle_t gen,
                                       diopiConstTensorHandle_t q, diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t cumSeqQ,
                                       diopiConstTensorHandle_t cumSeqKV, diopiConstTensorHandle_t alibiSlopes, int maxSeqLenQ, int maxSeqLenKV, float pDropout,
                                       float softmaxScale, bool isCausal, int windowSizeLeft, int windowSizeRight) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor qTensor(q);
    DiopiTensor kTensor(k);
    DiopiTensor vTensor(v);
    DiopiTensor cumSeqQTensor(cumSeqQ);
    DiopiTensor cumSeqKVTensor(cumSeqKV);
    DiopiTensor softmaxLseTensor(softmaxLse);
    DiopiTensor attentionOutTensor(attentionOut);

    // change dtype of input and output
    std::vector<DiopiTensor*> qkvTensors{&qTensor, &kTensor, &vTensor};
    std::set<diopiDtype_t> supportedQKVDtypes{diopi_dtype_float16, diopi_dtype_bfloat16};
    DIOPI_CALL(autoCastTensorType(ctx, qkvTensors, supportedQKVDtypes));

    std::vector<DiopiTensor*> cumSeqTensors{&cumSeqQTensor, &cumSeqKVTensor};
    std::set<diopiDtype_t> supportedCumSeqDtypes{diopi_dtype_int32};
    DIOPI_CALL(autoCastTensorType(ctx, cumSeqTensors, supportedCumSeqDtypes));

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

    CnnlTensorDesc qDesc(qTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc kDesc(kTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc vDesc(vTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc cumSeqQDesc(cumSeqQTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc cumSeqKVDesc(cumSeqKVTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc attentionOutTmpDesc(attentionOutTensorTmp, CNNL_LAYOUT_ARRAY);

    const int64_t totalSeqQ = qTensor.shape()[0];
    const int64_t headNum = qTensor.shape()[1];
    std::vector<int64_t> softmaxLseShape = {headNum, totalSeqQ};
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

    DIOPI_CALL_CNNL(cnnlFlashAttentionForward(handle,
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
                                              pDropout > 0.0 ? rngState : nullptr,
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
                                               diopiConstTensorHandle_t softmaxLse, int maxSeqLenQ, int maxSeqLenKV, float pDropout, float softmaxScale,
                                               bool isCausal, int windowSizeLeft, int windowSizeRight) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

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

    // change dtype
    std::vector<DiopiTensor*> qkvTensors{&qTensor, &kTensor, &vTensor, &attentionOutTensor, &gradOutTensor};
    std::set<diopiDtype_t> supportedQKVDtypes{diopi_dtype_float16, diopi_dtype_bfloat16};
    DIOPI_CALL(autoCastTensorType(ctx, qkvTensors, supportedQKVDtypes));
    DIOPI_CALL(autoCastTensorType(ctx, {&cumSeqQTensor, &cumSeqKVTensor}, {diopi_dtype_int32}));
    DIOPI_CALL(autoCastTensorType(ctx, {&softmaxLseTensor}, {diopi_dtype_float32}));

    DiopiTensor gradQTmpTr = gradQTensor;
    if (gradQTensor.dtype() != qTensor.dtype()) {
        gradQTmpTr = requiresTensor(ctx, gradQTmpTr.shape(), gradQTmpTr.stride(), qTensor.dtype());
    }

    DiopiTensor gradKTmpTr = gradKTensor;
    if (gradKTensor.dtype() != qTensor.dtype()) {
        gradKTmpTr = requiresTensor(ctx, gradKTmpTr.shape(), gradKTmpTr.stride(), qTensor.dtype());
    }

    DiopiTensor gradVTmpTr = gradVTensor;
    if (gradVTensor.dtype() != qTensor.dtype()) {
        gradVTmpTr = requiresTensor(ctx, gradVTmpTr.shape(), gradVTmpTr.stride(), qTensor.dtype());
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

    DIOPI_CALL_CNNL(cnnlSetFlashAttentionSlidingWindowSize(flashAttentionDesc.get(), -1, -1, 1));

    CnnlTensorDesc qDesc(qTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc kDesc(kTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc vDesc(vTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc cumSeqQDesc(cumSeqQTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc cumSeqKVDesc(cumSeqKVTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutDesc(gradOutTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(attentionOutTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradQDesc(gradQTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradKDesc(gradKTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradVDesc(gradVTmpTr, CNNL_LAYOUT_ARRAY);

    const int64_t totalSeqQ = qTensor.shape()[0];
    const int64_t headNum = qTensor.shape()[1];
    std::vector<int64_t> softmaxLseShape = {headNum, totalSeqQ};
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
                                               outputDesc.get(),
                                               attentionOutTensor.data(),
                                               softmaxLseDesc.get(),
                                               softmaxLseTensor.data(),
                                               cumSeqQDesc.get(),
                                               cumSeqQTensor.data(),
                                               cumSeqKVDesc.get(),
                                               cumSeqKVTensor.data(),
                                               pDropout > 0.0 ? rngState : nullptr,
                                               workspace,
                                               workspaceSize,
                                               gradQDesc.get(),
                                               gradQTmpTr.data(),
                                               gradKDesc.get(),
                                               gradKTmpTr.data(),
                                               gradVDesc.get(),
                                               gradVTmpTr.data(),
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               nullptr));

    if (gradQTmpTr.dtype() != gradQTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradQTensor, gradQTmpTr));
    }

    if (gradKTmpTr.dtype() != gradKTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradKTensor, gradKTmpTr));
    }

    if (gradVTmpTr.dtype() != gradVTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradVTensor, gradVTmpTr));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
