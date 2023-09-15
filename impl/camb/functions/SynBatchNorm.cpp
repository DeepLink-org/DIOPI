/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
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
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");
    DIOPI_CHECK(dim == outTr.dim(), "Input dim != out dim");

    // check the input dimension
    if (3 == dim) {
        inputTr.unsqueeze(3);
        outTr.view(inputTr.shape());
    }
    if (2 == dim) {
        inputTr.unsqueeze(2);
        inputTr.unsqueeze(3);
        outTr.view(inputTr.shape());
    }

    // check the input dtype
    std::vector<DiopiTensor*> pTensors{&inputTr, &weightTr, &biasTr, &meanTr, &invstdTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    /* Transpose to channels last */
    diopiMemoryFormat_t memoryFormat = inputTr.dim() == 4 ? diopiMemoryFormat_t::ChannelsLast : diopiMemoryFormat_t::ChannelsLast3d;
    DIOPI_CALL(contiguous(ctx, inputTr, memoryFormat));
    DiopiTensor outTmpTr = requiresTensor(ctx, outTr.shape(), inputTr.dtype(), memoryFormat);

    // get descriptor
    cnnlTensorLayout_t layout = inputTr.dim() == 4 ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NDHWC;
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

    // channels last -> contiguous
    DIOPI_CALL(contiguous(ctx, outTmpTr, diopiMemoryFormat_t::Contiguous));
    // Copy back to origin
    DIOPI_CALL(diopiCopyInp(ctx, outTmpTr.tensorHandle(), outTr.tensorHandle()));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
