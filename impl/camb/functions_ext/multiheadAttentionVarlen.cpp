/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */
#include <cnmlrt.h>
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

DIOPI_API diopiError_t diopiMultiHeadAttentionVarLen(diopiContextHandle_t ctx, diopiTensorHandle_t q, diopiTensorHandle_t k, diopiTensorHandle_t v,
                                                     diopiConstTensorHandle_t cumSeqQ, diopiConstTensorHandle_t cumSeqK, int64_t maxQ, int64_t maxK,
                                                     double dropoutP, bool isCausal, bool returnDebugMask, double scale, diopiTensorHandle_t out,
                                                     diopiTensorHandle_t softmaxLse, diopiGeneratorHandle_t gen, diopiTensorHandle_t debugAttnMask) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor qTensor(q);
    DiopiTensor kTensor(k);
    DiopiTensor vTensor(v);
    DiopiTensor cumSeqQTensor(cumSeqQ);
    DiopiTensor cumSeqKTensor(cumSeqK);
    DiopiTensor outTensor(out);
    DiopiTensor softmaxLseTensor(softmaxLse);
    DiopiTensor outputTensor(out);
    DiopiTensor dropoutMask(debugAttnMask);

    if (qTensor.numel() == 0) {
        return diopiSuccess;
    }
    // get sizes
    const int totalQ = qTensor.shape()[0];
    const int numHeadsQ = qTensor.shape()[1];

    const int seqlenQ = maxQ;
    const int seqlenK = maxK;

    std::vector<int64_t> softmaxShape = {numHeadsQ, totalQ};
    std::vector<int64_t> softmaxStride = calContiguousStride(softmaxShape);

    // change input,output data type
    std::vector<DiopiTensor*> qkvTensors{&qTensor, &kTensor, &vTensor};
    std::set<diopiDtype_t> supportedQKVDtypes{diopi_dtype_float16, diopi_dtype_bfloat16};
    DIOPI_CALL(autoCastTensorType(ctx, qkvTensors, supportedQKVDtypes));
    std::vector<DiopiTensor*> cumTensors{&cumSeqQTensor, &cumSeqKTensor};
    std::set<diopiDtype_t> supportedCumDtypes{diopi_dtype_int32};
    DIOPI_CALL(autoCastTensorType(ctx, cumTensors, supportedCumDtypes));

    DiopiTensor outTmpTr = outputTensor;
    if (outputTensor.dtype() != qTensor.dtype()) {
        outTmpTr = requiresTensor(ctx, outputTensor.shape(), outputTensor.stride(), qTensor.dtype());
    }

    DiopiTensor softmaxLseTmpTensor = softmaxLseTensor;
    if (softmaxLseTmpTensor.dtype() != diopi_dtype_float32) {
        softmaxLseTmpTensor = requiresTensor(ctx, softmaxLseTensor.shape(), softmaxLseTensor.stride(), diopi_dtype_float32);
    }

    // set descriptor
    CnnlResourceGuard<cnnlFlashAttentionDescriptor_t, cnnlCreateFlashAttentionDescriptor, cnnlDestroyFlashAttentionDescriptor> flashATTDesc;

    bool returnDropout = false;
    float qkScale = static_cast<float>(scale);
    cnnlAttentionMaskMode_t attenMask = isCausal ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE;
    DIOPI_CALL_CNNL(cnnlSetFlashAttentionDescriptor(flashATTDesc.get(),
                                                    CNNL_DTYPE_FLOAT,
                                                    CNNL_ACTIVATION_HIGH_PRECISION,
                                                    attenMask,
                                                    true,
                                                    false,
                                                    returnDropout,
                                                    seqlenQ,
                                                    seqlenK,
                                                    static_cast<float>(dropoutP),
                                                    qkScale));

    CnnlTensorDesc qDesc(qTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc kDesc(kTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc vDesc(vTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc qLenDesc(cumSeqQTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc kLenDesc(cumSeqKTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTmpDesc(outTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc softmaxLseTmpDesc;
    softmaxLseTmpDesc.set(softmaxLseTmpTensor.dtype(), softmaxShape, softmaxStride, CNNL_LAYOUT_ARRAY);

    // CnnlTensorDesc dropoutDesc(dropoutTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc dropoutDesc;
    if (returnDebugMask) {
        dropoutDesc.set(dropoutMask, CNNL_LAYOUT_ARRAY);
    }

    // get workspace
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetFlashAttentionForwardWorkspaceSize(handle, flashATTDesc.get(), qDesc.get(), kDesc.get(), vDesc.get(), &workspaceSize));
    void* workspace = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    // get random size
    size_t randomNum[2];
    randomNum[0] = 0;
    randomNum[1] = 0;

    if (dropoutP > 0.0) {
        DIOPI_CALL(diopiGeneratorGetSeedAndOffset(gen, &(randomNum[0]), &(randomNum[1])));
    }

    DIOPI_CALL_CNNL(cnnlFlashAttentionForward(handle,
                                              flashATTDesc.get(),
                                              qDesc.get(),
                                              qTensor.data(),
                                              kDesc.get(),
                                              kTensor.data(),
                                              vDesc.get(),
                                              vTensor.data(),
                                              qLenDesc.get(),
                                              cumSeqQTensor.data(),
                                              kLenDesc.get(),
                                              cumSeqKTensor.data(),
                                              dropoutP > 0.0 ? randomNum : nullptr,
                                              workspace,
                                              workspaceSize,
                                              returnDebugMask ? dropoutDesc.get() : nullptr,
                                              returnDebugMask ? dropoutMask.data() : nullptr,
                                              softmaxLseTmpDesc.get(),
                                              softmaxLseTmpTensor.data(),
                                              outTmpDesc.get(),
                                              outTmpTr.data()))

    if (outTmpTr.dtype() != outputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outputTensor, outTmpTr));
    }

    if (softmaxLseTmpTensor.dtype() != softmaxLseTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, softmaxLseTensor, softmaxLseTmpTensor));
    }

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMultiHeadAttentionVarLenBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q,
                                                             diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                                             diopiConstTensorHandle_t softmaxLse, diopiConstTensorHandle_t cumSeqQ,
                                                             diopiConstTensorHandle_t cumSeqK, int64_t maxQ, int64_t maxK, double dropoutP, bool isCausal,
                                                             diopiGeneratorHandle_t gen, double scale, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK,
                                                             diopiTensorHandle_t gradV) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor qTensor(q);
    DiopiTensor kTensor(k);
    DiopiTensor vTensor(v);
    DiopiTensor cumSeqQTensor(cumSeqQ);
    DiopiTensor cumSeqKTensor(cumSeqK);
    DiopiTensor gradOutTensor(gradOut);
    DiopiTensor outputTensor(out);
    DiopiTensor softmaxLseTensor(softmaxLse);

    DiopiTensor gradQTensor(gradQ);
    DiopiTensor gradKTensor(gradK);
    DiopiTensor gradVTensor(gradV);

    if (qTensor.numel() == 0) {
        return diopiSuccess;
    }

    // get sizes
    const int totalQ = qTensor.shape()[0];
    const int numHeads = qTensor.shape()[1];

    std::vector<int64_t> softmaxShape = {numHeads, totalQ};
    std::vector<int64_t> softmaxStride = calContiguousStride(softmaxShape);

    // change dtype
    std::vector<DiopiTensor*> qkvTensors{&qTensor, &kTensor, &vTensor, &gradOutTensor, &outputTensor};
    std::set<diopiDtype_t> supportedQKVDtypes{diopi_dtype_float16, diopi_dtype_bfloat16};
    DIOPI_CALL(autoCastTensorType(ctx, qkvTensors, supportedQKVDtypes));
    std::vector<DiopiTensor*> cumTensors{&cumSeqQTensor, &cumSeqKTensor};
    std::set<diopiDtype_t> supportedCumDtypes{diopi_dtype_int32};
    DIOPI_CALL(autoCastTensorType(ctx, cumTensors, supportedCumDtypes));
    std::vector<DiopiTensor*> softmaxTensors{&softmaxLseTensor};
    std::set<diopiDtype_t> supportedSoftmaxDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, softmaxTensors, supportedSoftmaxDtypes));

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
    CnnlResourceGuard<cnnlFlashAttentionDescriptor_t, cnnlCreateFlashAttentionDescriptor, cnnlDestroyFlashAttentionDescriptor> flashAttBckDesc;

    bool returnDropout = false;
    float qkScale = (float)scale;
    cnnlAttentionMaskMode_t attenMask = (isCausal == true) ? CNNL_ATTN_MASK_CAUSAL : CNNL_ATTN_MASK_NONE;
    DIOPI_CALL_CNNL(cnnlSetFlashAttentionBackwardDescriptor(flashAttBckDesc.get(),
                                                            CNNL_DTYPE_FLOAT,
                                                            CNNL_ACTIVATION_HIGH_PRECISION,
                                                            attenMask,
                                                            true,
                                                            false,
                                                            returnDropout,
                                                            (int)maxQ,
                                                            (int)maxK,
                                                            (float)dropoutP,
                                                            qkScale));

    DIOPI_CALL_CNNL(cnnlSetFlashAttentionSlidingWindowSize(flashAttBckDesc.get(), -1, -1, 1));

    CnnlTensorDesc qDesc(qTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc kDesc(kTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc vDesc(vTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc qLenDesc(cumSeqQTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc kLenDesc(cumSeqKTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutDesc(gradOutTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradQDesc(gradQTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradKDesc(gradKTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradVDesc(gradVTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc softmaxLseDesc;
    softmaxLseDesc.set(softmaxLseTensor.dtype(), softmaxShape, softmaxStride, CNNL_LAYOUT_ARRAY);

    // get workspace
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetFlashAttentionBackwardWorkspaceSize(handle, flashAttBckDesc.get(), qDesc.get(), kDesc.get(), vDesc.get(), &workspaceSize));
    void* workspace = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    // get random size
    size_t randomNum[2];
    randomNum[0] = 0;
    randomNum[1] = 0;

    if (dropoutP > 0.0) {
        DIOPI_CALL(diopiGeneratorGetSeedAndOffset(gen, &(randomNum[0]), &(randomNum[1])));
    }

    DIOPI_CALL_CNNL(cnnlGetFlashAttentionGeneratedRandomNumbers(handle, flashAttBckDesc.get(), qDesc.get(), vDesc.get(), qLenDesc.get(), randomNum))

    DIOPI_CALL_CNNL(cnnlFlashAttentionBackward(handle,
                                               flashAttBckDesc.get(),
                                               gradOutDesc.get(),
                                               gradOutTensor.data(),
                                               qDesc.get(),
                                               qTensor.data(),
                                               kDesc.get(),
                                               kTensor.data(),
                                               vDesc.get(),
                                               vTensor.data(),
                                               outputDesc.get(),
                                               outputTensor.data(),
                                               softmaxLseDesc.get(),
                                               softmaxLseTensor.data(),
                                               qLenDesc.get(),
                                               cumSeqQTensor.data(),
                                               kLenDesc.get(),
                                               cumSeqKTensor.data(),
                                               randomNum,
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
