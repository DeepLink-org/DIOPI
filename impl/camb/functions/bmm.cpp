/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(mat2);
    DiopiTensor outTensor(out);

    DIOPI_CHECK(inputTensor.dim() >= 2 && otherTensor.dim() >= 2, "the dim of mats should be greater than /equal to 2");
    DIOPI_CHECK(inputTensor.shape()[inputTensor.dim() - 1] == otherTensor.shape()[otherTensor.dim() - 2], "the col of matA should equal to row of matB");

    int32_t isTransa = 0;
    int32_t isTransb = 0;

    int32_t useStride = 1;
    if (inputTensor.isContiguous() && otherTensor.isContiguous()) {
        useStride = 0;
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    int32_t allowTf32Int = 1;
    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> bmmDesc;
    DIOPI_CALL_CNNL(cnnlSetMatMulDescAttr(bmmDesc.get(), CNNL_MATMUL_ALLOW_TF32, &allowTf32Int, sizeof(allowTf32Int)));
    DIOPI_CALL_CNNL(cnnlSetMatMulDescAttr(bmmDesc.get(), CNNL_MATMUL_DESC_TRANSA, &(isTransa), sizeof(int32_t)));
    DIOPI_CALL_CNNL(cnnlSetMatMulDescAttr(bmmDesc.get(), CNNL_MATMUL_DESC_TRANSB, &(isTransb), sizeof(int32_t)));
    DIOPI_CALL_CNNL(cnnlSetMatMulDescAttr(bmmDesc.get(), CNNL_MATMUL_USE_STRIDE, &(useStride), sizeof(int32_t)));

    CnnlResourceGuard<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> bmmAlgo;
    CnnlResourceGuard<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> bmmHeuristicResult;
    int returnAlgoCount = 0;
    DIOPI_CALL_CNNL(cnnlGetBatchMatMulAlgoHeuristic(
        handle, bmmDesc.get(), inputDesc.get(), otherDesc.get(), outDesc.get(), nullptr, 1, &(bmmHeuristicResult.get()), &returnAlgoCount));

    size_t workspaceSize = 0;
    DIOPI_CALL_CNNL(cnnlGetBatchMatMulHeuristicResult(bmmHeuristicResult.get(), bmmAlgo.get(), &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    float alpha = 1;
    float beta = 0;
    DIOPI_CALL_CNNL(cnnlBatchMatMulBCast_v2(handle,
                                            bmmDesc.get(),
                                            bmmAlgo.get(),
                                            &alpha,
                                            inputDesc.get(),
                                            inputTensor.data(),
                                            otherDesc.get(),
                                            otherTensor.data(),
                                            &beta,
                                            outDesc.get(),
                                            outTensor.data(),
                                            workspace,
                                            workspaceSize));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
