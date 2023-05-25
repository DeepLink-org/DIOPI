/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiAddmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat1,
                                  diopiConstTensorHandle_t mat2, const diopiScalar_t* beta, const diopiScalar_t* alpha) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor mat1Tensor(mat1);
    DiopiTensor mat2Tensor(mat2);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    std::vector<DiopiTensor*> pTensors{&inputTensor, &mat1Tensor, &mat2Tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor inputTensorTmp = *pTensors[0];
    DiopiTensor mat1TensorTmp = *pTensors[1];
    DiopiTensor mat2TensorTmp = *pTensors[2];
    DiopiTensor outTensorTmp = outTensor;
    DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, inputTensorTmp.dtype()));

    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mat1Desc(mat1TensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mat2Desc(mat2TensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);
    DiopiTensor mmResultTensor = requiresTensor(ctx, vec2diopiSizeT(outTensor.shape()), inputTensorTmp.dtype());
    CnnlTensorDesc mmResultDesc(mmResultTensor, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> cnnlMatMulDesc;
    cnnlMatMulDescriptor_t matmulDesc = cnnlMatMulDesc.get();

    int32_t isTransa = 0;
    int32_t isTransb = 0;
    int32_t allowTf32I32 = 1;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc, CNNL_MATMUL_DESC_TRANSA, &(isTransa), sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc, CNNL_MATMUL_DESC_TRANSB, &(isTransb), sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc, CNNL_MATMUL_ALLOW_TF32, &(allowTf32I32), sizeof(int32_t)));

    size_t workspaceSize = 0;
    int requestedAlgoCount = 1;
    int returnAlgoCount = 0;
    CnnlResourceGuard<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> cnnlMatMulHeuristic;
    cnnlMatMulHeuristicResult_t heuristicResult = cnnlMatMulHeuristic.get();
    CnnlResourceGuard<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> cnnlMatMulAlgo;
    cnnlMatMulAlgo_t matmulAlgo = cnnlMatMulAlgo.get();

    DIOPI_CALLCNNL(cnnlGetMatMulAlgoHeuristic(handle,
                                              matmulDesc,
                                              mat1Desc.get(),
                                              mat2Desc.get(),
                                              mmResultDesc.get(),
                                              mmResultDesc.get(),
                                              nullptr,
                                              requestedAlgoCount,
                                              &heuristicResult,
                                              &returnAlgoCount));
    DIOPI_CALLCNNL(cnnlGetMatMulHeuristicResult(heuristicResult, matmulAlgo, &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    float alphaTmp;
    if (alpha->stype <= 7) {
        alphaTmp = alpha->ival;
    } else {
        alphaTmp = alpha->fval;
    }

    float betaTmp;
    if (beta->stype <= 7) {
        betaTmp = beta->ival;
    } else {
        betaTmp = beta->fval;
    }

    float alphaDefault = 1;
    float betaDefault = 0;

    DIOPI_CALLCNNL(cnnlMatMul_v2(handle,
                                 matmulDesc,
                                 matmulAlgo,
                                 &alphaDefault,
                                 mat1Desc.get(),
                                 mat1TensorTmp.data(),
                                 mat2Desc.get(),
                                 mat2TensorTmp.data(),
                                 &betaDefault,
                                 mmResultDesc.get(),
                                 mmResultTensor.data(),
                                 workspace,
                                 workspaceSize,
                                 mmResultDesc.get(),
                                 mmResultTensor.data()));

    CnnlResourceGuard<cnnlOpTensorDescriptor_t, cnnlCreateOpTensorDescriptor, cnnlDestroyOpTensorDescriptor> cnnlOpTensorDesc;
    cnnlOpTensorDescriptor_t optensorDesc = cnnlOpTensorDesc.get();
    workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetOpTensorWorkspaceSize(handle, mmResultDesc.get(), inputDesc.get(), outDesc.get(), &workspaceSize));
    workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlOpTensor(handle,
                                optensorDesc,
                                &alphaTmp,
                                mmResultDesc.get(),
                                mmResultTensor.data(),
                                &betaTmp,
                                inputDesc.get(),
                                inputTensorTmp.data(),
                                workspace,
                                workspaceSize,
                                &betaDefault,
                                outDesc.get(),
                                outTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));

    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
