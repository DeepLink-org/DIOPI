/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

#define REQUIRES_TENSOR_BY_DTYPE_OR_NOT(tensor1, tensor2, targetDtype, memoryFormat) \
    DiopiTensor tensor1 = tensor2;                                                   \
    if (tensor2.defined() && tensor1.dtype() != targetDtype) {                       \
        tensor1 = requiresTensor(ctx, tensor1.shape(), targetDtype, memoryFormat);   \
    }

diopiError_t diopiBatchNormBackwardReduce(diopiContextHandle_t ctx, diopiTensorHandle_t sumDy, diopiTensorHandle_t sumDyXmu, diopiTensorHandle_t gradWeight,
                                          diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, diopiConstTensorHandle_t weight, bool inputG,
                                          bool weightG, bool biasG) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    // input
    DiopiTensor gradOutTr(gradOut);  // MLU-dz
    DiopiTensor inputTr(input);      // MLU-x
    DiopiTensor meanTr(mean);        // MLU-mean
    DiopiTensor invstdTr(invstd);    // MLU-ivstd
    // output
    DiopiTensor sumDyTr(sumDy);            // MLU-sumDy
    DiopiTensor sumDyXmuTr(sumDyXmu);      // MLU-sumDyXmu
    DiopiTensor gradWeightTr(gradWeight);  // MLU-dfilter
    DiopiTensor gradBiasTr(gradBias);      // MLU-dbias

    auto dim = inputTr.dim();
    cnnlTensorLayout_t layout;
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");
    DIOPI_CHECK(dim == gradOutTr.dim(), "Input dim != out dim");

    // check the input dimension
    if (2 == dim) {
        layout = CNNL_LAYOUT_NC;
    } else if (3 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "inputTensor's memory format should be channelsLast1d");
        DIOPI_CHECK(gradOutTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "inputTensor's memory format should be channelsLast1d");
        layout = CNNL_LAYOUT_NLC;
    } else if (4 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "inputTensor's memory format should be channelsLast2d");
        DIOPI_CHECK(gradOutTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "inputTensor's memory format should be channelsLast2d");
        layout = CNNL_LAYOUT_NHWC;
    } else if (5 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "inputTensor's memory format should be channelsLast3d");
        DIOPI_CHECK(gradOutTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "inputTensor's memory format should be channelsLast3d");
        layout = CNNL_LAYOUT_NDHWC;
    } else {
        DIOPI_CHECK(false, "Dim of input tensor should be in [2,3,4,5].");
    }

    // check the input dtype
    std::vector<DiopiTensor*> pTensors{&gradOutTr, &inputTr, &meanTr, &invstdTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    // check the output dtype
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(sumDyTmpTr, sumDyTr, diopi_dtype_float32, diopiMemoryFormat_t::Contiguous);
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(sumDyXmuTmpTr, sumDyXmuTr, diopi_dtype_float32, diopiMemoryFormat_t::Contiguous);
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(gradWeightTmpTr, gradWeightTr, diopi_dtype_float32, diopiMemoryFormat_t::Contiguous);
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(gradBiasTmpTr, gradBiasTr, diopi_dtype_float32, diopiMemoryFormat_t::Contiguous);

    // get descriptor
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc gradOutDesc(gradOutTr, layout);
    CnnlTensorDesc meanDesc(meanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invstdDesc(invstdTr, CNNL_LAYOUT_ARRAY);

    /*Note: meanTr only for generating CnnlTensorDesc, which will not be used*/
    CnnlTensorDesc gradWeightDesc(weightG ? gradWeightTmpTr : meanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradBiasDesc(biasG ? gradBiasTmpTr : meanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc sumDyDesc(inputG ? sumDyTmpTr : meanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc sumDyXmuDesc(inputG ? sumDyXmuTmpTr : meanTr, CNNL_LAYOUT_ARRAY);

    /* Get Workspace */
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetSyncBatchnormBackwardReduceWorkspaceSize(handle, inputDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALL_CNNL(cnnlSyncBatchnormBackwardReduce_v2(handle,
                                                       gradOutDesc.get(),
                                                       gradOutTr.data(),
                                                       inputDesc.get(),
                                                       inputTr.data(),
                                                       meanDesc.get(),
                                                       meanTr.data(),
                                                       invstdDesc.get(),
                                                       invstdTr.data(),
                                                       workspacePtr,
                                                       workspaceSize,
                                                       weightG ? gradWeightDesc.get() : nullptr,
                                                       weightG ? gradWeightTmpTr.data() : nullptr,
                                                       biasG ? gradBiasDesc.get() : nullptr,
                                                       biasG ? gradBiasTmpTr.data() : nullptr,
                                                       inputG ? sumDyDesc.get() : nullptr,
                                                       inputG ? sumDyTmpTr.data() : nullptr,
                                                       inputG ? sumDyXmuDesc.get() : nullptr,
                                                       inputG ? sumDyXmuTmpTr.data() : nullptr,
                                                       inputG,
                                                       weightG,
                                                       biasG))

    if (inputG) {
        DIOPI_CALL(dataTypeCast(ctx, sumDyTr, sumDyTmpTr));
        DIOPI_CALL(dataTypeCast(ctx, sumDyXmuTr, sumDyXmuTmpTr));
    }
    if (weightG) {
        DIOPI_CALL(dataTypeCast(ctx, gradWeightTr, gradWeightTmpTr));
    }
    if (biasG) {
        DIOPI_CALL(dataTypeCast(ctx, gradBiasTr, gradBiasTmpTr));
    }

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormElemt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, float eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    // input
    DiopiTensor inputTr(input);
    DiopiTensor weightTr(weight);
    DiopiTensor biasTr(bias);
    DiopiTensor meanTr(mean);
    DiopiTensor invstdTr(invstd);
    // output
    DiopiTensor outTr(out);

    auto dim = inputTr.dim();
    cnnlTensorLayout_t layout;
    diopiMemoryFormat_t memoryFormat;

    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");
    DIOPI_CHECK(dim == outTr.dim(), "Input dim != out dim");

    if (2 == dim) {
        memoryFormat = diopiMemoryFormat_t::Contiguous;
        layout = CNNL_LAYOUT_NC;
    } else if (3 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "inputTensor's memory format should be channelsLast");
        DIOPI_CHECK(outTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "outputTensor's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NLC;
        memoryFormat = diopiMemoryFormat_t::ChannelsLast1d;
    } else if (4 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "inputTensor's memory format should be channelsLast");
        DIOPI_CHECK(outTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "outputTensor's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NHWC;
        memoryFormat = diopiMemoryFormat_t::ChannelsLast;
    } else if (5 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "inputTensor's memory format should be channelsLast");
        DIOPI_CHECK(outTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "outputTensor's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NDHWC;
        memoryFormat = diopiMemoryFormat_t::ChannelsLast3d;
    } else {
        DIOPI_CHECK(false, "Dim of input tensor should be in [2,3,4,5].");
    }

    // check the input dtype
    std::vector<DiopiTensor*> pTensors{&inputTr, &weightTr, &biasTr, &meanTr, &invstdTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    // check the output dtype
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(outTmpTr, outTr, inputTr.dtype(), memoryFormat);

    // get descriptor
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc outputDesc(outTmpTr, layout);
    CnnlTensorDesc weightDesc(weightTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc biasDesc(biasTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc meanDesc(meanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invstdDesc(invstdTr, CNNL_LAYOUT_ARRAY);

    DIOPI_CALL_CNNL(cnnlSyncBatchNormElemt(handle,
                                           inputDesc.get(),
                                           inputTr.data(),
                                           meanDesc.get(),
                                           meanTr.data(),
                                           invstdDesc.get(),
                                           invstdTr.data(),
                                           weightDesc.get(),
                                           weightTr.data(),
                                           biasDesc.get(),
                                           biasTr.data(),
                                           outputDesc.get(),
                                           outTmpTr.data()))

    // Copy back to origin, if required
    DIOPI_CALL(dataTypeCast(ctx, outTr, outTmpTr));

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormBackwardElemt(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOut,
                                                   diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd,
                                                   diopiConstTensorHandle_t weight, diopiConstTensorHandle_t sumDy, diopiConstTensorHandle_t sumDyXmu,
                                                   diopiConstTensorHandle_t count) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    // input
    DiopiTensor gradOutTr(gradOut);
    DiopiTensor inputTr(input);
    DiopiTensor meanTr(mean);
    DiopiTensor invstdTr(invstd);
    DiopiTensor weightTr(weight);
    DiopiTensor sumDyTr(sumDy);
    DiopiTensor sumDyXmuTr(sumDyXmu);
    DiopiTensor countTr(count);
    // output
    DiopiTensor gradInputTr(gradInput);

    auto dim = inputTr.dim();
    cnnlTensorLayout_t layout;
    diopiMemoryFormat_t memoryFormat;

    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");
    DIOPI_CHECK(dim == gradOutTr.dim(), "Input dim != gradOutTr dim");
    DIOPI_CHECK(dim == gradInputTr.dim(), "Input dim != gradInputTr dim");

    if (2 == dim) {
        memoryFormat = diopiMemoryFormat_t::Contiguous;
        layout = CNNL_LAYOUT_NC;
    } else if (3 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "inputTensor's memory format should be channelsLast");
        DIOPI_CHECK(gradOutTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "gradOutTr's memory format should be channelsLast");
        DIOPI_CHECK(gradInputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "gradInputTr's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NLC;
        memoryFormat = diopiMemoryFormat_t::ChannelsLast1d;
    } else if (4 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "inputTensor's memory format should be channelsLast");
        DIOPI_CHECK(gradOutTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "gradOutTr's memory format should be channelsLast");
        DIOPI_CHECK(gradInputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "gradInputTr's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NHWC;
        memoryFormat = diopiMemoryFormat_t::ChannelsLast;
    } else if (5 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "inputTensor's memory format should be channelsLast");
        DIOPI_CHECK(gradOutTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "gradOutTr's memory format should be channelsLast");
        DIOPI_CHECK(gradInputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "gradInputTr's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NDHWC;
        memoryFormat = diopiMemoryFormat_t::ChannelsLast3d;
    } else {
        DIOPI_CHECK(false, "Dim of input tensor should be in [2,3,4,5].");
    }

    // check the input dtype
    std::vector<DiopiTensor*> pTensors{&inputTr, &gradOutTr, &meanTr, &invstdTr, &weightTr, &sumDyTr, &sumDyXmuTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    std::vector<DiopiTensor*> pCountTensors{&countTr};
    std::set<diopiDtype_t> supportedCountDtypes{diopi_dtype_int32};
    DIOPI_CALL(autoCastTensorType(ctx, pCountTensors, supportedCountDtypes));

    // check the output dtype
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(gradInputTmpTr, gradInputTr, inputTr.dtype(), memoryFormat);

    // get descriptor
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc gradOutDesc(gradOutTr, layout);
    CnnlTensorDesc meanDesc(meanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invstdDesc(invstdTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc weightDesc(weightTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc sumDyDesc(sumDyTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc sumDyXmuDesc(sumDyXmuTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc countDesc(countTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradInputDesc(gradInputTmpTr, layout);

    DIOPI_CALL_CNNL(cnnlSyncBatchNormBackwardElemtV2(handle,
                                                     gradOutDesc.get(),
                                                     gradOutTr.data(),
                                                     inputDesc.get(),
                                                     inputTr.data(),
                                                     meanDesc.get(),
                                                     meanTr.data(),
                                                     invstdDesc.get(),
                                                     invstdTr.data(),
                                                     weightDesc.get(),
                                                     weightTr.data(),
                                                     sumDyDesc.get(),
                                                     sumDyTr.data(),
                                                     sumDyXmuDesc.get(),
                                                     sumDyXmuTr.data(),
                                                     countDesc.get(),
                                                     countTr.data(),
                                                     gradInputDesc.get(),
                                                     gradInputTmpTr.data()))

    // Copy back to origin, if required
    DIOPI_CALL(dataTypeCast(ctx, gradInputTr, gradInputTmpTr));

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormStats(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input,
                                           double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    // input
    DiopiTensor inputTr(input);
    float epsValue = static_cast<float>(eps);
    // output
    DiopiTensor meanTr(mean);
    DiopiTensor invstdTr(invstd);

    auto dim = inputTr.dim();
    cnnlTensorLayout_t layout;
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");

    if (2 == dim) {
        layout = CNNL_LAYOUT_NC;
    } else if (3 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast1d), "inputTensor's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NLC;
    } else if (4 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast), "inputTensor's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NHWC;
    } else if (5 == dim) {
        DIOPI_CHECK(inputTr.isContiguous(diopiMemoryFormat_t::ChannelsLast3d), "inputTensor's memory format should be channelsLast");
        layout = CNNL_LAYOUT_NDHWC;
    } else {
        DIOPI_CHECK(false, "Dim of input tensor should be in [2,3,4,5].");
    }

    // check the input dtype
    std::vector<DiopiTensor*> pTensors{&inputTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    // check the output dtype
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(invstdTmpTr, invstdTr, diopi_dtype_float32, diopiMemoryFormat_t::Contiguous);
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(meanTmpTr, meanTr, diopi_dtype_float32, diopiMemoryFormat_t::Contiguous);

    // get descriptor
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc meanDesc(meanTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invstdDesc(invstdTmpTr, CNNL_LAYOUT_ARRAY);

    /* Get Workspace */
    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetSyncBatchNormStatsWorkspaceSize(handle, inputDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALL_CNNL(cnnlSyncBatchNormStats_v2(
        handle, inputDesc.get(), inputTr.data(), workspacePtr, workspaceSize, epsValue, meanDesc.get(), meanTmpTr.data(), invstdDesc.get(), invstdTmpTr.data()))

    // Copy back to origin, if required
    DIOPI_CALL(dataTypeCast(ctx, meanTr, meanTmpTr));
    DIOPI_CALL(dataTypeCast(ctx, invstdTr, invstdTmpTr));

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormGatherStatsWithCounts(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd,
                                                           diopiConstTensorHandle_t input, diopiConstTensorHandle_t meanAll, diopiConstTensorHandle_t invstdAll,
                                                           diopiTensorHandle_t runningMean, diopiTensorHandle_t runningVar, float momentum, float eps,
                                                           diopiConstTensorHandle_t counts) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    // input
    DiopiTensor inputTr(input);
    DiopiTensor meanAllTr(meanAll);
    DiopiTensor invstdAllTr(invstdAll);
    DiopiTensor countsTr(counts);
    DiopiTensor runningMeanTr(runningMean);
    DiopiTensor runningVarTr(runningVar);
    // output
    DiopiTensor meanTr(mean);
    DiopiTensor invstdTr(invstd);
    DiopiTensor runningMeanTrOrigin(runningMean);
    DiopiTensor runningVarTrOrigin(runningVar);

    DIOPI_CHECK(meanAllTr.dim() == 2, "meanAll dim is out of range");
    DIOPI_CHECK(invstdAllTr.dim() == 2, "invstdAll dim is out of range");

    // check the input dtype
    std::vector<DiopiTensor*> pTensors{&meanAllTr, &invstdAllTr, &countsTr};
    if (runningMeanTr.defined()) {
        pTensors.push_back(&runningMeanTr);
    }
    if (runningVarTr.defined()) {
        pTensors.push_back(&runningVarTr);
    }
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    // check the output dtype
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(invstdTmpTr, invstdTr, diopi_dtype_float32, diopiMemoryFormat_t::Contiguous);
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(meanTmpTr, meanTr, diopi_dtype_float32, diopiMemoryFormat_t::Contiguous);
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(runningMeanTmpTr, runningMeanTr, diopi_dtype_float32, diopiMemoryFormat_t::Contiguous);
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(runningVarTmpTr, runningVarTr, diopi_dtype_float32, diopiMemoryFormat_t::Contiguous);
    DiopiTensor countIntTr = requiresTensor(ctx, countsTr.shape(), countsTr.dtype(), diopiMemoryFormat_t::Contiguous);

    // get descriptor
    CnnlTensorDesc meanAllDesc(meanAllTr, CNNL_LAYOUT_NC);
    CnnlTensorDesc invstdAllDesc(invstdAllTr, CNNL_LAYOUT_NC);
    CnnlTensorDesc countsDesc(countsTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc countsIntDesc(countIntTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc meanDesc(meanTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invstdDesc(invstdTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc runningMeanDesc(runningMeanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc runningVarDesc(runningVarTr, CNNL_LAYOUT_ARRAY);

    DIOPI_CALL_CNNL(cnnlTrunc(handle, countsDesc.get(), countsTr.data(), countsIntDesc.get(), countIntTr.data()))

    // check whether the counts are all positive
    DiopiTensor min = requiresTensor(ctx, {1}, countsTr.dtype(), diopiMemoryFormat_t::Contiguous);
    DIOPI_CALL(diopiMinAll(ctx, min.tensorHandle(), countIntTr.tensorHandle()));
    std::unique_ptr<float> ptr(new float[sizeof(float)]);
    cnrtMemcpyAsync(ptr.get(), min.data(), sizeof(float), getStream(ctx), cnrtMemcpyDevToHost);
    syncStreamInCtx(ctx);
    int minCount = static_cast<float>(reinterpret_cast<float*>(ptr.get())[0]);
    DIOPI_CHECK(minCount >= 0.0, "counts should be positive");

    DIOPI_CALL_CNNL(cnnlSyncBatchNormGatherStatsWithCounts(handle,
                                                           meanAllDesc.get(),
                                                           meanAllTr.data(),
                                                           invstdAllDesc.get(),
                                                           invstdAllTr.data(),
                                                           runningMeanDesc.get(),
                                                           runningMeanTr.defined() ? runningMeanTr.data() : nullptr,
                                                           runningVarDesc.get(),
                                                           runningVarTr.defined() ? runningVarTr.data() : nullptr,
                                                           momentum,
                                                           eps,
                                                           countsIntDesc.get(),
                                                           countIntTr.data(),
                                                           meanDesc.get(),
                                                           meanTmpTr.data(),
                                                           invstdDesc.get(),
                                                           invstdTmpTr.data()))
    // Copy back to origin, if required
    DIOPI_CALL(diopiCopyInp(ctx, runningMeanTr.tensorHandle(), runningMeanTrOrigin.tensorHandle()));
    DIOPI_CALL(diopiCopyInp(ctx, runningVarTr.tensorHandle(), runningVarTrOrigin.tensorHandle()));
    DIOPI_CALL(dataTypeCast(ctx, meanTr, meanTmpTr));
    DIOPI_CALL(dataTypeCast(ctx, invstdTr, invstdTmpTr));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
