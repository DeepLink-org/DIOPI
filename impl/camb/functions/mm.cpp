/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor aTensor(input);
    DiopiTensor bTensor(mat2);
    DiopiTensor outTensor(out);

    DiopiTensor aCasted = aTensor;
    DiopiTensor bCasted = bTensor;
    DiopiTensor outCasted = outTensor;

    std::vector<DiopiTensor*> tensors{&aCasted, &bCasted, &outCasted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    CnnlTensorDesc aDesc(aCasted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc bDesc(bCasted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outCasted, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> matmulDesc;

    cnnlDataType_t compType;
    if (outCasted.dtype() == diopi_dtype_float32) {
        compType = CNNL_DTYPE_FLOAT;
    } else if (outCasted.dtype() == diopi_dtype_float16) {
        compType = CNNL_DTYPE_HALF;
    } else {
        return diopiDtypeNotSupported;
    }
    DIOPI_CALL_CNNL(cnnlSetMatMulDescAttr(matmulDesc.get(), CNNL_MATMUL_DESC_COMPUTE_TYPE, &(compType), sizeof(cnnlDataType_t)));
    int32_t isTransa = 0;
    DIOPI_CALL_CNNL(cnnlSetMatMulDescAttr(matmulDesc.get(), CNNL_MATMUL_DESC_TRANSA, &(isTransa), sizeof(int32_t)));
    int32_t isTransb = 0;
    DIOPI_CALL_CNNL(cnnlSetMatMulDescAttr(matmulDesc.get(), CNNL_MATMUL_DESC_TRANSB, &(isTransb), sizeof(int32_t)));
    int32_t allowTf32I32 = 1;
    DIOPI_CALL_CNNL(cnnlSetMatMulDescAttr(matmulDesc.get(), CNNL_MATMUL_ALLOW_TF32, &(allowTf32I32), sizeof(int32_t)));

    int32_t useBeta = 0;
    float beta = 0.0;

    DIOPI_CALL_CNNL(cnnlSetMatMulDescAttr(matmulDesc.get(), CNNL_MATMUL_USE_BETA, &(useBeta), sizeof(int32_t)));

    size_t workspaceSize = 0;
    int requestedAlgoCount = 1;
    int returnAlgoCount = 0;
    CnnlResourceGuard<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> heuristicResult;
    CnnlResourceGuard<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> algo;
    DIOPI_CALL_CNNL(cnnlGetMatMulAlgoHeuristic(handle,
                                               matmulDesc.get(),
                                               aDesc.get(),
                                               bDesc.get(),
                                               outDesc.get(),
                                               outDesc.get(),
                                               nullptr,
                                               requestedAlgoCount,
                                               &heuristicResult.get(),
                                               &returnAlgoCount));
    DIOPI_CALL_CNNL(cnnlGetMatMulHeuristicResult(heuristicResult.get(), algo.get(), &workspaceSize));

    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    float alphaDefault = 1.0;

    DIOPI_CALL_CNNL(cnnlMatMul_v2(handle,
                                  matmulDesc.get(),
                                  algo.get(),
                                  &alphaDefault,
                                  aDesc.get(),
                                  aCasted.data(),
                                  bDesc.get(),
                                  bCasted.data(),
                                  &beta,
                                  outDesc.get(),
                                  outCasted.data(),
                                  workspace,
                                  workspaceSize,
                                  outDesc.get(),
                                  outCasted.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outCasted));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
