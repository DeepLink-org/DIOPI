/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <string.h>
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

    DiopiTensor mat1_tensor(mat1);
    DiopiTensor mat2_tensor(mat2);
    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);

    std::vector<DiopiTensor*> pTensors{&input_tensor, &mat1_tensor, &mat2_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor input_tensor_tmp = *pTensors[0];
    DiopiTensor mat1_tensor_tmp = *pTensors[1];
    DiopiTensor mat2_tensor_tmp = *pTensors[2];
    DiopiTensor out_tensor_tmp = out_tensor;
    DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, input_tensor_tmp.dtype()));

    CnnlTensorDesc input_desc(input_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mat1_desc(mat1_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mat2_desc(mat2_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);
    DiopiTensor mm_result_tensor = requiresTensor(ctx, vec2diopiSize_t(out_tensor.shape()), input_tensor_tmp.dtype());
    CnnlTensorDesc mm_result_desc(mm_result_tensor, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> CnnlMatMulDesc;
    cnnlMatMulDescriptor_t matmul_desc = CnnlMatMulDesc.get();

    int32_t is_transa = 0;
    int32_t is_transb = 0;
    int32_t allow_tf32_i32 = 1;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSA, &(is_transa), sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSB, &(is_transb), sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_ALLOW_TF32, &(allow_tf32_i32), sizeof(int32_t)));

    size_t workspace_size = 0;
    int requestedAlgoCount = 1;
    int returnAlgoCount = 0;
    CnnlResourceGuard<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> CnnlMatMulHeuristic;
    cnnlMatMulHeuristicResult_t heuristicResult = CnnlMatMulHeuristic.get();
    CnnlResourceGuard<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> CnnlMatMulAlgo;
    cnnlMatMulAlgo_t matmul_algo = CnnlMatMulAlgo.get();

    DIOPI_CALLCNNL(cnnlGetMatMulAlgoHeuristic(handle,
                                              matmul_desc,
                                              mat1_desc.get(),
                                              mat2_desc.get(),
                                              mm_result_desc.get(),
                                              mm_result_desc.get(),
                                              nullptr,
                                              requestedAlgoCount,
                                              &heuristicResult,
                                              &returnAlgoCount));
    DIOPI_CALLCNNL(cnnlGetMatMulHeuristicResult(heuristicResult, matmul_algo, &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    float alpha_;
    if (alpha->stype <= 7) {
        alpha_ = alpha->ival;
    } else {
        alpha_ = alpha->fval;
    }

    float beta_;
    if (beta->stype <= 7) {
        beta_ = beta->ival;
    } else {
        beta_ = beta->fval;
    }

    float alpha_default = 1;
    float beta_default = 0;

    DIOPI_CALLCNNL(cnnlMatMul_v2(handle,
                                 matmul_desc,
                                 matmul_algo,
                                 &alpha_default,
                                 mat1_desc.get(),
                                 mat1_tensor_tmp.data(),
                                 mat2_desc.get(),
                                 mat2_tensor_tmp.data(),
                                 &beta_default,
                                 mm_result_desc.get(),
                                 mm_result_tensor.data(),
                                 workspace,
                                 workspace_size,
                                 mm_result_desc.get(),
                                 mm_result_tensor.data()));

    CnnlResourceGuard<cnnlOpTensorDescriptor_t, cnnlCreateOpTensorDescriptor, cnnlDestroyOpTensorDescriptor> CnnlOpTensorDesc;
    cnnlOpTensorDescriptor_t optensor_desc = CnnlOpTensorDesc.get();
    size_t workspace_size_ = 0;
    DIOPI_CALLCNNL(cnnlGetOpTensorWorkspaceSize(handle, mm_result_desc.get(), input_desc.get(), out_desc.get(), &workspace_size_));
    void* workspace_ = nullptr;
    if (0 != workspace_size_) {
        workspace_ = requiresBuffer(ctx, workspace_size_).data();
    }

    DIOPI_CALLCNNL(cnnlOpTensor(handle,
                                optensor_desc,
                                &alpha_,
                                mm_result_desc.get(),
                                mm_result_tensor.data(),
                                &beta_,
                                input_desc.get(),
                                input_tensor_tmp.data(),
                                workspace_,
                                workspace_size_,
                                &beta_default,
                                out_desc.get(),
                                out_tensor_tmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));

    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
