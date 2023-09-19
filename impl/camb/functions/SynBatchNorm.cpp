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

    DIOPI_CALLCNNL(cnnlSyncBatchNormElemt(handle,
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

DIOPI_API diopiError_t diopiBatchNormStats(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input,
                                           double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    // input
    DiopiTensor inputTr(input);
    DiopiTensor meanTr(mean);
    float epsValue = static_cast<float>(eps);
    // output
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
    std::vector<DiopiTensor*> pTensors{&inputTr, &meanTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    // check the output dtype
    DiopiTensor invstdTmpTr = invstdTr;
    if (invstdTr.dtype() != meanTr.dtype()) {
        invstdTmpTr = requiresTensor(ctx, invstdTr.shape(), meanTr.dtype());
    }

    // get descriptor
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc meanDesc(meanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invstdDesc(invstdTmpTr, CNNL_LAYOUT_ARRAY);

    /* Get Workspace */
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetSyncBatchNormStatsWorkspaceSize(handle, inputDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALLCNNL(cnnlSyncBatchNormStats_v2(
        handle, inputDesc.get(), inputTr.data(), workspacePtr, workspaceSize, epsValue, meanDesc.get(), meanTr.data(), invstdDesc.get(), invstdTmpTr.data()))

    // Copy back to origin, if required
    DIOPI_CALL(diopiCopyInp(ctx, invstdTmpTr.tensorHandle(), invstdTr.tensorHandle()));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
