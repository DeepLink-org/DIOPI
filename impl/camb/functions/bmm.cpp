/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiBmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(mat2);
    DiopiTensor outTensor(out);

    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, {&inputTensor, &otherTensor}, supportedDtypes));

    DiopiTensor outTensorTmp = outTensor;
    if (outTensor.dtype() != inputTensor.dtype()) {
        outTensorTmp = requiresTensor(ctx, outTensor.shape(), inputTensor.dtype());
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    int32_t allowTf32Int = 1;
    CnnlDescBase<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> bmmDesc;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(bmmDesc.get(), CNNL_MATMUL_ALLOW_TF32, &allowTf32Int, sizeof(allowTf32Int)));
    CnnlDescBase<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> bmmAlgo;
    CnnlDescBase<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> bmmHeuristicResult;
    int returnAlgoCount = 0;
    DIOPI_CALLCNNL(cnnlGetBatchMatMulAlgoHeuristic(
        handle, bmmDesc.get(), inputDesc.get(), otherDesc.get(), outDesc.get(), nullptr, 1, &(bmmHeuristicResult.get()), &returnAlgoCount));

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetBatchMatMulHeuristicResult(bmmHeuristicResult.get(), bmmAlgo.get(), &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    int alpha = 1;
    int beta = 0;
    DIOPI_CALLCNNL(cnnlBatchMatMulBCast_v2(handle,
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
    if (outTensor.dtype() != outTensorTmp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
