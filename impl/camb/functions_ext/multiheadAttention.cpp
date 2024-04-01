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

DIOPI_API diopiError_t diopiMultiHeadAttention(diopiContextHandle_t ctx, diopiTensorHandle_t q, diopiTensorHandle_t k, diopiTensorHandle_t v,
                                               double dropoutP, bool isCausal, bool returnDebugMask, double scale, diopiTensorHandle_t out,
                                               diopiTensorHandle_t softmaxLse, diopiGeneratorHandle_t gen, diopiTensorHandle_t debugAttnMask) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor qTensor(q);
    DiopiTensor kTensor(k);
    DiopiTensor vTensor(v);
    DiopiTensor outTensor(out);
    DiopiTensor softmaxLseTensor(softmaxLse);
    DiopiTensor dropoutMask(debugAttnMask);

    if (qTensor.numel() == 0) {
        return diopiSuccess;
    }

    // get sizes
    const int batchSize = qTensor.shape()[0];
    const int seqlenQ = qTensor.shape()[1];
    const int numHeadsQ = qTensor.shape()[2];
    const int headSize = qTensor.shape()[3];
    const int seqlenK = kTensor.shape()[1];
    const int numHeadsK = kTensor.shape()[2];
    int totalQ = batchSize * seqlenQ;
    int totalK = batchSize * seqlenK;

    std::vector<int64_t> shapeQ = {totalQ, numHeadsQ, headSize};
    std::vector<int64_t> strideQ = calContiguousStride(shapeQ);
    std::vector<int64_t> shapeKV = {totalK, numHeadsK, headSize};
    std::vector<int64_t> strideKV = calContiguousStride(shapeKV);
    std::vector<int64_t> csShape = {batchSize + 1};
    std::vector<int64_t> csStride = {1};
    std::vector<int64_t> softmaxShape = {numHeadsQ, totalQ};
    std::vector<int64_t> softmaxStride = calContiguousStride(softmaxShape);

    DIOPI_CHECK(batchSize > 0, "batch size must be postive");
    DIOPI_CHECK(headSize <= 256, "FlashAttention forward only supports head dimension at most 256");
    DIOPI_CHECK(numHeadsQ % numHeadsK == 0, "Number of heads in key/value must divide number of heads in query");

    // get accumulateQ
    int32_t cuSeqlensQ[batchSize + 1];
    int32_t cuSeqlensK[batchSize + 1];
    for (int32_t i = 0; i < batchSize + 1; i++) {
        cuSeqlensQ[i] = i * seqlenQ;
        cuSeqlensK[i] = i * seqlenK;
    }

    // require tensor csq & csk on device
    DiopiTensor csq = requiresTensor(ctx, csShape, csStride, diopi_dtype_int32);
    DiopiTensor csk = requiresTensor(ctx, csShape, csStride, diopi_dtype_int32);

    void* csqPtr = csq.data();
    void* cskPtr = csk.data();
    diopiStreamHandle_t streamHandle;
    diopiGetStream(ctx, &streamHandle);
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    uint64_t bytes = sizeof(int32_t) * (batchSize + 1);
    cnrtMemcpyAsync(csqPtr, static_cast<void*>(cuSeqlensQ), bytes, phStream, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpyAsync(cskPtr, static_cast<void*>(cuSeqlensK), bytes, phStream, CNRT_MEM_TRANS_DIR_HOST2DEV);

    // change input,output data type
    std::vector<DiopiTensor*> qkvTensors{&qTensor, &kTensor, &vTensor};
    std::set<diopiDtype_t> supportedQKVDtypes{diopi_dtype_float16};
    DIOPI_CALL(autoCastTensorType(ctx, qkvTensors, supportedQKVDtypes));

    DiopiTensor outTmpTr = outTensor;
    if (outTensor.dtype() != qTensor.dtype()) {
        outTmpTr = requiresTensor(ctx, outTensor.shape(), outTensor.stride(), qTensor.dtype());
    }

    DiopiTensor softmaxLseTmpTensor = softmaxLseTensor;
    if (softmaxLseTensor.dtype() != diopi_dtype_float32) {
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

    CnnlTensorDesc qDesc;
    CnnlTensorDesc kDesc;
    CnnlTensorDesc vDesc;
    CnnlTensorDesc csqDesc;
    CnnlTensorDesc cskDesc;
    CnnlTensorDesc outTmpDesc;
    CnnlTensorDesc softmaxLseTmpDesc;
    CnnlTensorDesc dropoutDesc;

    qDesc.set(qTensor.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    kDesc.set(kTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);
    vDesc.set(vTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);
    csqDesc.set(diopi_dtype_int32, csShape, csStride, CNNL_LAYOUT_ARRAY);
    cskDesc.set(diopi_dtype_int32, csShape, csStride, CNNL_LAYOUT_ARRAY);
    outTmpDesc.set(outTmpTr.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    softmaxLseTmpDesc.set(softmaxLseTmpTensor.dtype(), softmaxShape, softmaxStride, CNNL_LAYOUT_ARRAY);  // will not be the real result

    // get workspace, currently not used, just for future use
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetFlashAttentionForwardWorkspaceSize(handle, flashATTDesc.get(), qDesc.get(), kDesc.get(), vDesc.get(), &workspaceSize));
    void* workspace = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    // get random size
    size_t randomNum[2];
    randomNum[0] = 0;
    randomNum[1] = 0;
    if (dropoutP > 0.0) {
        DIOPI_CALL(diopiGeneratorGetSeedAndOffset(gen, randomNum[0], randomNum[1]));
    }

    if (returnDebugMask) {
        std::vector<int64_t> dropoutShape = {batchSize, numHeadsQ, seqlenQ, seqlenK};
        std::vector<int64_t> dropoutStride = calContiguousStride(dropoutShape);
        // dropoutMask = requiresTensor(ctx, dropoutShape, dropoutStride, diopi_dtype_float32);
        dropoutDesc.set(dropoutMask.dtype(), dropoutShape, dropoutStride, CNNL_LAYOUT_ARRAY);
    }

    DIOPI_CALL_CNNL(cnnlFlashAttentionForward(handle,
                                              flashATTDesc.get(),
                                              qDesc.get(),
                                              qTensor.data(),
                                              kDesc.get(),
                                              kTensor.data(),
                                              vDesc.get(),
                                              vTensor.data(),
                                              csqDesc.get(),
                                              csqPtr,
                                              cskDesc.get(),
                                              cskPtr,
                                              dropoutP > 0.0 ? randomNum : nullptr,
                                              workspace,
                                              workspaceSize,
                                              returnDebugMask ? dropoutDesc.get() : nullptr,
                                              returnDebugMask ? dropoutMask.data() : nullptr,
                                              softmaxLseTmpDesc.get(),
                                              softmaxLseTmpTensor.data(),
                                              outTmpDesc.get(),
                                              outTmpTr.data()))

    if (outTmpTr.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTmpTr));
    }

    if (softmaxLseTensor.dtype() != diopi_dtype_float32) {
        DIOPI_CALL(dataTypeCast(ctx, softmaxLseTensor, softmaxLseTmpTensor));
    }

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMultiHeadAttentionBackward(diopiContextHandle_t ctx, diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t q,
                                                       diopiConstTensorHandle_t k, diopiConstTensorHandle_t v, diopiConstTensorHandle_t out,
                                                       diopiConstTensorHandle_t softmaxLse, double dropoutP, bool isCausal, diopiGeneratorHandle_t gen,
                                                       double scale, diopiTensorHandle_t gradQ, diopiTensorHandle_t gradK, diopiTensorHandle_t gradV) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor qTensor(q);
    DiopiTensor kTensor(k);
    DiopiTensor vTensor(v);
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
    const int batchSize = qTensor.shape()[0];
    const int seqlenQ = qTensor.shape()[1];
    const int numHeadsQ = qTensor.shape()[2];
    const int headSize = qTensor.shape()[3];
    const int seqlenK = kTensor.shape()[1];
    const int numHeadsK = kTensor.shape()[2];
    int totalQ = batchSize * seqlenQ;
    int totalK = batchSize * seqlenQ;

    std::vector<int64_t> shapeQ = {totalQ, numHeadsQ, headSize};
    std::vector<int64_t> strideQ = calContiguousStride(shapeQ);
    std::vector<int64_t> shapeKV = {totalK, numHeadsK, headSize};
    std::vector<int64_t> strideKV = calContiguousStride(shapeKV);
    std::vector<int64_t> csShape = {batchSize + 1};
    std::vector<int64_t> csStride = {1};
    std::vector<int64_t> softmaxShape = {numHeadsQ, totalQ};
    std::vector<int64_t> softmaxStride = calContiguousStride(softmaxShape);

    // change dtype
    std::vector<DiopiTensor*> qkvTensors{&qTensor, &kTensor, &vTensor, &gradOutTensor, &outputTensor};
    std::set<diopiDtype_t> supportedQKVDtypes{diopi_dtype_float16, diopi_dtype_bfloat16};
    DIOPI_CALL(autoCastTensorType(ctx, qkvTensors, supportedQKVDtypes));
    std::vector<DiopiTensor*> softmaxTensors{&softmaxLseTensor};
    std::set<diopiDtype_t> supportedSoftmaxDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, softmaxTensors, supportedSoftmaxDtypes));

    // get accumulateQ
    int32_t cuSeqlensQ[batchSize + 1];
    int32_t cuSeqlensK[batchSize + 1];
    for (int32_t i = 0; i < batchSize + 1; i++) {
        cuSeqlensQ[i] = i * seqlenQ;
        cuSeqlensK[i] = i * seqlenK;
    }

    // require tensor csq & csk on device
    DiopiTensor csq = requiresTensor(ctx, csShape, csStride, diopi_dtype_int32);
    DiopiTensor csk = requiresTensor(ctx, csShape, csStride, diopi_dtype_int32);

    void* csqPtr = csq.data();
    void* cskPtr = csk.data();
    diopiStreamHandle_t streamHandle;
    diopiGetStream(ctx, &streamHandle);
    cnrtQueue_t phStream = (cnrtQueue_t)streamHandle;
    uint64_t bytes = sizeof(int32_t) * (batchSize + 1);
    cnrtMemcpyAsync(csqPtr, static_cast<void*>(cuSeqlensQ), bytes, phStream, CNRT_MEM_TRANS_DIR_HOST2DEV);
    cnrtMemcpyAsync(cskPtr, static_cast<void*>(cuSeqlensK), bytes, phStream, CNRT_MEM_TRANS_DIR_HOST2DEV);

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
                                                            (int)seqlenQ,
                                                            (int)seqlenK,
                                                            (float)dropoutP,
                                                            qkScale));

    DIOPI_CALL_CNNL(cnnlSetFlashAttentionSlidingWindowSize(flashAttBckDesc.get(), -1, -1, 1));

    CnnlTensorDesc qDesc;
    CnnlTensorDesc kDesc;
    CnnlTensorDesc vDesc;
    CnnlTensorDesc csqDesc;
    CnnlTensorDesc cskDesc;
    CnnlTensorDesc gradOutDesc;
    CnnlTensorDesc outputDesc;
    CnnlTensorDesc softmaxLseDesc;
    CnnlTensorDesc gradQDesc;
    CnnlTensorDesc gradKDesc;
    CnnlTensorDesc gradVDesc;

    qDesc.set(qTensor.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    kDesc.set(kTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);
    vDesc.set(vTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);
    csqDesc.set(diopi_dtype_int32, csShape, csStride, CNNL_LAYOUT_ARRAY);
    cskDesc.set(diopi_dtype_int32, csShape, csStride, CNNL_LAYOUT_ARRAY);
    gradOutDesc.set(gradOutTensor.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    outputDesc.set(outputTensor.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    softmaxLseDesc.set(softmaxLseTensor.dtype(), softmaxShape, softmaxStride, CNNL_LAYOUT_ARRAY);
    gradQDesc.set(qTensor.dtype(), shapeQ, strideQ, CNNL_LAYOUT_ARRAY);
    gradKDesc.set(kTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);
    gradVDesc.set(vTensor.dtype(), shapeKV, strideKV, CNNL_LAYOUT_ARRAY);

    // get workspace
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetFlashAttentionBackwardWorkspaceSize(handle, flashAttBckDesc.get(), qDesc.get(), kDesc.get(), vDesc.get(), &workspaceSize));
    void* workspace = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    // get random size
    size_t randomNum[2];
    randomNum[0] = 0;
    randomNum[1] = 0;

    if (dropoutP > 0.0) {
        DIOPI_CALL(diopiGeneratorGetSeedAndOffset(gen, randomNum[0], randomNum[1]));
    }

    DIOPI_CALL_CNNL(cnnlGetFlashAttentionGeneratedRandomNumbers(handle, flashAttBckDesc.get(), qDesc.get(), vDesc.get(), csqDesc.get(), randomNum))

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
                                               csqDesc.get(),
                                               csqPtr,
                                               cskDesc.get(),
                                               cskPtr,
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
                                               nullptr))

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
