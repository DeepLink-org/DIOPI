/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
diopiError_t diopiBatchNormBackwardReduce(diopiContextHandle_t ctx, diopiTensorHandle_t sum_dy, diopiTensorHandle_t sum_dy_xmu,
                                        diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_out,
                                        diopiConstTensorHandle_t input, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd,
                                        diopiConstTensorHandle_t weight, bool input_g, bool weight_g, bool bias_g){

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    //output
    DiopiTensor sum_dyTr(sum_dy);//MLU-sum_dy
    DiopiTensor sum_dy_xmuTr(sum_dy_xmu);//MLU-sum_dy_xmu
    DiopiTensor grad_weightTr(grad_weight); //MLU-dfilter
    DiopiTensor grad_biasTr(grad_bias);//MLU-dbias
    
    //input
    DiopiTensor grad_outTr(grad_out);//MLU-dz
    DiopiTensor inputTr(input); //MLU-x
    DiopiTensor meanTr(mean); //MLU-mean
    DiopiTensor invstdTr(invstd); //MLU-ivstd
    DiopiTensor weightTr(weight); //no found in MLU

    auto dim = inputTr.dim();
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");
    DIOPI_CHECK(dim == grad_outTr.dim(), "Input dim != out dim");


    //check the input dimension
    if (3 == dim) {
        inputTr.unsqueeze(3);
        grad_outTr.unsqueeze(3);
    }else if (2 == dim) {
        inputTr.unsqueeze(2);
        inputTr.unsqueeze(3);
        grad_outTr.unsqueeze(2);
        grad_outTr.unsqueeze(3);
    }

    //check the input dtype
    std::vector<DiopiTensor*> pTensors{&grad_outTr,&inputTr,&meanTr,&invstdTr};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    /* Transpose to channels last */
    diopiMemoryFormat_t memoryFormat = inputTr.dim() == 4 ? diopiMemoryFormat_t::ChannelsLast : diopiMemoryFormat_t::ChannelsLast3d;
    DIOPI_CALL(contiguous(ctx, inputTr, memoryFormat));
    DIOPI_CALL(contiguous(ctx, grad_outTr, memoryFormat));

    /*Get output tmp tensor*/
    DiopiTensor grad_weightTmpTr = grad_weightTr;
    DiopiTensor grad_biasTmpTr = grad_biasTr;
    DiopiTensor sum_dyTmpTr = sum_dyTr;
    DiopiTensor sum_dy_xmuTmpTr = sum_dy_xmuTr;
    if (grad_weightTr.dtype() != inputTr.dtype() ||grad_biasTr.dtype() != inputTr.dtype() ||
        sum_dyTr.dtype() != inputTr.dtype() ||sum_dy_xmuTr.dtype() != inputTr.dtype()
    ) {
        DIOPI_CALL(dataTypeCast(ctx, grad_weightTmpTr, inputTr.dtype()));
        DIOPI_CALL(dataTypeCast(ctx, grad_biasTmpTr, inputTr.dtype()));
        DIOPI_CALL(dataTypeCast(ctx, sum_dyTmpTr, inputTr.dtype()));
        DIOPI_CALL(dataTypeCast(ctx, sum_dy_xmuTmpTr, inputTr.dtype()));
    }


    //get descriptor
    cnnlTensorLayout_t layout = inputTr.dim() == 4 ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NDHWC;
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc grad_outDesc(grad_outTr, layout);
    CnnlTensorDesc meanDesc(meanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invstdDesc(invstdTr, CNNL_LAYOUT_ARRAY);

    CnnlTensorDesc grad_weightDesc(grad_weightTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_biasDesc(grad_biasTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc sum_dyDesc(sum_dyTmpTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc sum_dy_xmuDesc(sum_dy_xmuTmpTr, CNNL_LAYOUT_ARRAY);

    /* Get Workspace */
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetSyncBatchnormBackwardReduceWorkspaceSize(handle, inputDesc.get(), &workspaceSize));
    void* workspacePtr = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALLCNNL(cnnlSyncBatchnormBackwardReduce_v2(handle,grad_outDesc,grad_outTr,
                                                    inputDesc,inputTr,meanDesc,meanTr,invstdDesc,invstdTr,
                                                    workspacePtr,workspaceSize,
                                                    grad_weightDesc,grad_weightTmpTr,grad_biasDesc,grad_biasTmpTr,
                                                    sum_dyDesc,sum_dyTmpTr,sum_dy_xmuDesc,sum_dy_xmuTmpTr,
                                                    input_g,weight_g,bias_g
    ))

    if (grad_weightTmpTr.dtype() != grad_weightTr.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, grad_weightTr, grad_weightTmpTr));
        DIOPI_CALL(dataTypeCast(ctx, grad_biasTr, grad_biasTmpTr));
        DIOPI_CALL(dataTypeCast(ctx, sum_dyTr, sum_dyTmpTr));
        DIOPI_CALL(dataTypeCast(ctx, sum_dy_xmuTr, sum_dy_xmuTmpTr));
    }

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBatchNormElemt(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t invstd, float eps){

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    //input
    DiopiTensor inputTr(input);
    DiopiTensor weightTr(weight);
    DiopiTensor biasTr(bias);
    DiopiTensor meanTr(mean);
    DiopiTensor invstdTr(invstd);
    //output
    DiopiTensor outTr(out);

    auto dim = inputTr.dim();
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");
    DIOPI_CHECK(dim == outTr.dim(), "Input dim != out dim");


    //check the input dimension
    if (3 == dim) {
        inputTr.unsqueeze(3);
        outTr.view(inputTr.shape());
    }
    if (2 == dim) {
        inputTr.unsqueeze(2);
        inputTr.unsqueeze(3);
        outTr.view(inputTr.shape());
    }

    //check the input dtype
    std::vector<DiopiTensor*> pTensors{&input,&weight,&bias,&mean,&invstd};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    /* Transpose to channels last */
    diopiMemoryFormat_t memoryFormat = inputTr.dim() == 4 ? diopiMemoryFormat_t::ChannelsLast : diopiMemoryFormat_t::ChannelsLast3d;
    DIOPI_CALL(contiguous(ctx, inputTr, memoryFormat));
    DiopiTensor outTmpTr = requiresTensor(ctx, outTr.shape(), inputTr.dtype(), memoryFormat);

    //get descriptor
    cnnlTensorLayout_t layout = inputTr.dim() == 4 ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NDHWC;
    CnnlTensorDesc inputDesc(inputTr, layout);
    CnnlTensorDesc outputDesc(outTmpTr, layout);
    CnnlTensorDesc weightDesc(weightTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc biasDesc(biasTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc meanDesc(meanTr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc invstdDesc(invstdTr, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlSyncBatchNormElemt(handle,inputDesc,inputTr,meanDesc,meanTr,invstdDesc,invstdTr,
                                        weightDesc,weightTr,biasDesc,biasTr,outputDesc,outTmpTr
    ))

    // channels last -> contiguous
    DIOPI_CALL(contiguous(ctx, outputTmpTr, diopiMemoryFormat_t::Contiguous));
    // Copy back to origin
    DIOPI_CALL(diopiCopyInp(ctx, outTmpTr.tensorHandle(), outTr.tensorHandle()));




}

// DIOPI_API diopiError_t diopiBatchNormStats(diopiContextHandle_t ctx, diopiTensorHandle_t mean, diopiTensorHandle_t invstd, diopiConstTensorHandle_t input,
//                                            double eps);{
//     cnnlHandle_t handle = cnnlHandlePool.get(ctx);


                                        
// }


}  // namespace camb
}  // namespace impl
