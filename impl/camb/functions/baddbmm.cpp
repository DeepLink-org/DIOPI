/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

/**
 * @brief Broadcast input add batch matmul.
 * @param[in] ctx Context environment.
 * @param input the broadcastable tensor to add, and it would be inout tensor for camb cnnl.
 * @param batch1 the first batch of matrices to be multiplied. type = [float16, float32, float64].
 * @param batch2 the second batch of matrices to be multiplied. type = [float16, float32, float64].
 * @param beta the offset coeff
 * @param alpha the scaling factor
 * @param[out] out the output tensor. type = [float16, float32, float64].
 */

static diopiError_t batchAddBatchMatmul(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor batch1, DiopiTensor batch2, float beta, float alpha) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlTensorDesc batch1Desc(batch1, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc batch2Desc(batch2, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(input, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> matmulDescObj;
    cnnlMatMulDescriptor_t matmulDesc = matmulDescObj.get();

    cnnlDataType_t dataType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dataType, batch1.dtype()));
    int32_t isTransa = 0;
    int32_t isTransb = 0;
    int32_t isBetaUse = 1;
    int32_t isTF32Allow = 1;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc, CNNL_MATMUL_DESC_COMPUTE_TYPE, &dataType, sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc, CNNL_MATMUL_DESC_TRANSA, &isTransa, sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc, CNNL_MATMUL_DESC_TRANSB, &isTransb, sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc, CNNL_MATMUL_USE_BETA, &isBetaUse, sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmulDesc, CNNL_MATMUL_ALLOW_TF32, &isTF32Allow, sizeof(int32_t)));

    CnnlResourceGuard<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> matmulHrObj;
    cnnlMatMulHeuristicResult_t matmulHr = matmulHrObj.get();

    CnnlResourceGuard<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> matmulAlgoObj;
    cnnlMatMulAlgo_t matmulAlgo = matmulAlgoObj.get();

    size_t workspaceSize = 0;
    int requestAlgoCount = 1;
    int returnAlgoCount = 0;
    DIOPI_CALLCNNL(cnnlGetBatchMatMulAlgoHeuristic(handle,
                                                   matmulDesc,
                                                   batch1Desc.get(),
                                                   batch2Desc.get(),
                                                   inputDesc.get(),
                                                   nullptr,  // preference not supported.
                                                   requestAlgoCount,
                                                   &matmulHr,
                                                   &returnAlgoCount));
    DIOPI_CALLCNNL(cnnlGetBatchMatMulHeuristicResult(matmulHr, matmulAlgo, &workspaceSize));

    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
        DIOPI_CHECK(workspace != nullptr, "[diopiBaddbmm] require buffers: size = %d, for workspace failed.", workspaceSize);
    }

    DIOPI_CALLCNNL(cnnlBatchMatMulBCast_v2(handle,
                                           matmulDesc,
                                           matmulAlgo,
                                           &alpha,
                                           batch1Desc.get(),
                                           batch1.data(),
                                           batch2Desc.get(),
                                           batch2.data(),
                                           &beta,
                                           inputDesc.get(),
                                           input.data(),
                                           workspace,
                                           workspaceSize));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2,
                                       double beta, double alpha) {
    DiopiTensor inputTensor(input);
    DiopiTensor batch1Tensor(batch1);
    DiopiTensor batch2Tensor(batch2);

    std::vector<DiopiTensor *> tensorsVecPtr{&batch1Tensor, &batch2Tensor, &inputTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64};
    DIOPI_CALL(autoCastTensorType(ctx, tensorsVecPtr, supportedDtypes));

    DiopiTensor batch1CastedTensor = *tensorsVecPtr[0];
    DiopiTensor batch2CastedTensor = *tensorsVecPtr[1];
    DiopiTensor inputCastedTensor = *tensorsVecPtr[2];

    DIOPI_CALL(batchAddBatchMatmul(ctx, inputCastedTensor, batch1CastedTensor, batch2CastedTensor, beta, alpha));
    DIOPI_CALL(dataTypeCast(ctx, inputTensor, inputCastedTensor));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1,
                                    diopiConstTensorHandle_t batch2, double beta, double alpha) {
    DiopiTensor batch1Tensor(batch1);
    DiopiTensor batch2Tensor(batch2);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    DIOPI_CALL(broadcastContiguous(ctx, outTensor, inputTensor));

    std::vector<DiopiTensor *> tensorsVecPtr{&batch1Tensor, &batch2Tensor, &outTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_float64};
    DIOPI_CALL(autoCastTensorType(ctx, tensorsVecPtr, supportedDtypes));

    DiopiTensor batch1CastedTensor = *tensorsVecPtr[0];
    DiopiTensor batch2CastedTensor = *tensorsVecPtr[1];
    DiopiTensor outCastedTensor = *tensorsVecPtr[2];

    DIOPI_CALL(batchAddBatchMatmul(ctx, outCastedTensor, batch1CastedTensor, batch2CastedTensor, beta, alpha));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outCastedTensor));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
