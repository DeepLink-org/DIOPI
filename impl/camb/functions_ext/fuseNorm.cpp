// /**
//  * @file
//  * @author DeepLink
//  * @copyright  (c) 2023, DeepLink.
//  */

// #include <diopi/functions_ext.h>

// #include "../cnnl_helper.hpp"
// #include "../common/common.hpp"
// #include "../common/debug.hpp"

// #include "../mlu_helper.hpp"

// namespace impl {
// namespace camb {
// diopiError_t diopiRMSNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t invRms, diopiConstTensorHandle_t input,
//                                     diopiSize_t normalizedShape, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, double eps){
//     cnnlHandle_t handle = cnnlHandlePool.get(ctx);
//     DiopiTensor inputTensor(input);
//     DiopiTensor weightTensor(weight);
//     DiopiTensor biasTensor(bias);

//     DiopiTensor outputTensor(out);
//     DiopiTensor invRmsTensor(invRms);

//     std::vector<DiopiTensor*> pTensors{&inputTensor, &weightTensor, &biasTensor};
//     std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32};

//     DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

//     //set tensor descriptor
//     CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
//     CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_ARRAY);
//     CnnlTensorDesc biasDesc(biasTensor, CNNL_LAYOUT_ARRAY);
//     CnnlTensorDesc outputDesc(outputTensor, CNNL_LAYOUT_ARRAY);
//     CnnlTensorDesc invRmsDesc;

//     std::vector<int64_t> shape = {1,5};
//     std::vector<int64_t> stride = {1,1};
//     invRmsDesc.set(invRmsTensor.dtype(), shape, stride, CNNL_LAYOUT_ARRAY);

//     printDevData(ctx,inputTensor);
//     printDevData(ctx,weightTensor);
//     printDevData(ctx,outputTensor);
//     printDevData(ctx,invRmsTensor);

//     //set op descriptor
//     CnnlResourceGuard<cnnlFuseNormDescriptor_t, cnnlCreateFuseNormDescriptor, cnnlDestroyFuseNormDescriptor> fuseNormDesc;
//     bool hasNormScale = true;
//     bool hasNormBias = true;
//     bool hasBias = false;
//     bool hasResidual = false;
//     // bool storeOutputBeforeLayernorm = invRmsTensor.defined()?true:false;
//     bool storeOutputBeforeLayernorm = false;
//     cnnlDataType_t dataType = CNNL_DTYPE_FLOAT;
//     cnnlTransformerNormType_t normType = CNNL_TRANSFORMER_RMSNORM;
//     DIOPI_CALL_CNNL(cnnlSetFuseNormDescriptor(fuseNormDesc.get(),static_cast<float>(eps),1.0, hasNormScale, hasNormBias,hasBias,hasResidual,storeOutputBeforeLayernorm,dataType,normType));

//     //get workspace
//     size_t workspaceSize = 0;
//     DIOPI_CALL_CNNL(cnnlGetFuseNormWorkspaceSize(handle,fuseNormDesc.get(),inputDesc.get(), &workspaceSize));
//     void* workspace = workspaceSize == 0 ? nullptr : requiresBuffer(ctx, workspaceSize).data();

//     DIOPI_CALL_CNNL(cnnlFuseNorm(handle,
//                     fuseNormDesc.get(),
//                     inputDesc.get(),
//                     inputTensor.data(),
//                     weightDesc.get(),
//                     weightTensor.data(),
//                     biasDesc.get(),
//                     biasTensor.data(),
//                     nullptr,
//                     nullptr,
//                     nullptr,
//                     nullptr,
//                     workspace,
//                     workspaceSize,
//                     outputDesc.get(),
//                     outputTensor.data(),
//                     storeOutputBeforeLayernorm?invRmsDesc.get():nullptr,
//                     storeOutputBeforeLayernorm?invRmsTensor.data():nullptr
//                     ))
// }


// }  // namespace camb
// }  // namespace impl
