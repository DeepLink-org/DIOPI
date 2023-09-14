/**
 * file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
diopiError_t diopiBatchNormBackwardReduce(diopiContextHandle_t ctx, diopiTensorHandle_t sumDy, diopiTensorHandle_t sumDyXmu, diopiTensorHandle_t gradWeight,
                                          diopiTensorHandle_t gradBias, diopiConstTensorHandle_t gradOut, diopiConstTensorHandle_t input,
                                          diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, diopiConstTensorHandle_t weight, bool inputG,
                                          bool weightG, bool biasG) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    // output
    DiopiTensor sumDyTr(sumDy);            // MLU-sumDy
    DiopiTensor sumDyXmuTr(sumDyXmu);    // MLU-sumDyXmu
    DiopiTensor gradWeightTr(gradWeight);  // MLU-dfilter
    DiopiTensor gradBiasTr(gradBias);      // MLU-dbias

    // input
    DiopiTensor gradOutTr(gradOut);  // MLU-dz
    DiopiTensor inputTr(input);        // MLU-x
    DiopiTensor meanTr(mean);          // MLU-mean
    DiopiTensor invstdTr(invstd);      // MLU-ivstd
    DiopiTensor weightTr(weight);      // no found in MLU

    auto dim = inputTr.dim();
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");
    DIOPI_CHECK(dim == gradOutTr.dim(), "Input dim != out dim");

    // check the input dimension
    if (3 == dim) {
        inputTr.unsqueeze(3);
        gradOutTr.unsqueeze(3);
    } else if (2 == dim) {
        inputTr.unsqueeze(2);
        inputTr.unsqueeze(3);
        gradOutTr.unsqueeze(2);
        gradOutTr.unsqueeze(3);
    }

    // check the input dtype
    std::vector<DiopiTensor*> pTensors{&gradOutTr, &inputTr, &meanTr, &invstdTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    /* Transpose to channels last */
    diopiMemoryFormat_t memoryFormat = inputTr.dim() == 4 ? diopiMemoryFormat_t::ChannelsLast : diopiMemoryFormat_t::ChannelsLast3d;
    DIOPI_CALL(contiguous(ctx, inputTr, memoryFormat));
    DIOPI_CALL(contiguous(ctx, gradOutTr, memoryFormat));

    /*Get output tmp tensor*/
    DiopiTensor gradWeightTmpTr = gradWeightTr;
    DiopiTensor gradBiasTmpTr = gradBiasTr;
    DiopiTensor sumDyTmpTr = sumDyTr;
    DiopiTensor sumDyXmuTmpTr = sumDyXmuTr;
    if (gradWeightTr.dtype() != inputTr.dtype() || gradBiasTr.dtype() != inputTr.dtype() || sumDyTr.dtype() != inputTr.dtype() ||
        sumDyXmuTr.dtype() != inputTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradWeightTmpTr, inputTr.dtype()));
        DIOPI_CALL(dataTypeCast(ctx, gradBiasTmpTr, inputTr.dtype()));
        DIOPI_CALL(dataTypeCast(ctx, sumDyTmpTr, inputTr.dtype()));
        DIOPI_CALL(dataTypeCast(ctx, sumDyXmuTmpTr, inputTr.dtype()));
    }

    // get descriptor
    cnnlTensorLayout_t layout = inputTr.dim() == 4 ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NDHWC;
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc grad_outDesc(gradOutTr, layout);
    CnnlTensorDesc meanDesc(meanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invstdDesc(invstdTr, CNNL_LAYOUT_ARRAY);

    CnnlTensorDesc gradWeightDesc(gradWeightTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradBiasDesc(gradBiasTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc sumDyDesc(sumDyTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc sumDyXmuDesc(sumDyXmuTmpTr, CNNL_LAYOUT_ARRAY);

    /* Get Workspace */
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetSyncBatchnormBackwardReduceWorkspaceSize(handle, inputDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALLCNNL(cnnlSyncBatchnormBackwardReduce_v2(handle,
                                                      grad_outDesc.get(),
                                                      gradOutTr.data(),
                                                      inputDesc.get(),
                                                      inputTr.data(),
                                                      meanDesc.get(),
                                                      meanTr.data(),
                                                      invstdDesc.get(),
                                                      invstdTr.data(),
                                                      workspacePtr,
                                                      workspaceSize,
                                                      gradWeightDesc.get(),
                                                      gradWeightTmpTr.data(),
                                                      gradBiasDesc.get(),
                                                      gradBiasTmpTr.data(),
                                                      sumDyDesc.get(),
                                                      sumDyTmpTr.data(),
                                                      sumDyXmuDesc.get(),
                                                      sumDyXmuTmpTr.data(),
                                                      inputG,
                                                      weightG,
                                                      biasG))

    if (gradWeightTmpTr.dtype() != gradWeightTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradWeightTr, gradWeightTmpTr));
        DIOPI_CALL(dataTypeCast(ctx, gradBiasTr, gradBiasTmpTr));
        DIOPI_CALL(dataTypeCast(ctx, sumDyTr, sumDyTmpTr));
        DIOPI_CALL(dataTypeCast(ctx, sumDyXmuTr, sumDyXmuTmpTr));
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
}

// DIOPI_API diopiError_t diopiBatchNormStats(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input,
//                                            double eps);{
//     cnnlHandle_t handle = cnnlHandlePool.get(ctx);

// }

}  // namespace camb
}  // namespace impl

