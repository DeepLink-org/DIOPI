/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiMm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mat2) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor a_tensor(input);
    DiopiTensor b_tensor(mat2);
    DiopiTensor out_tensor(out);

    DiopiTensor a_casted = a_tensor;
    DiopiTensor b_casted = b_tensor;
    DiopiTensor out_casted = out_tensor;

    std::vector<DiopiTensor*> tensors{&a_casted, &b_casted, &out_casted};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    CnnlTensorDesc a_desc(a_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc b_desc(b_casted, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_casted, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> matmul_desc;

    cnnlDataType_t comp_type;
    if (out_casted.dtype() == diopi_dtype_float32) {
        comp_type = CNNL_DTYPE_FLOAT;
    } else if (out_casted.dtype() == diopi_dtype_float16) {
        comp_type = CNNL_DTYPE_HALF;
    } else {
        set_last_error_string("%s", "matmul on support float or half.");
        return diopiDtypeNotSupported;
    }
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc.get(), CNNL_MATMUL_DESC_COMPUTE_TYPE, &(comp_type), sizeof(cnnlDataType_t)));
    int32_t is_transa = 0;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc.get(), CNNL_MATMUL_DESC_TRANSA, &(is_transa), sizeof(int32_t)));
    int32_t is_transb = 0;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc.get(), CNNL_MATMUL_DESC_TRANSB, &(is_transb), sizeof(int32_t)));
    int32_t allow_tf32_i32 = 1;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc.get(), CNNL_MATMUL_ALLOW_TF32, &(allow_tf32_i32), sizeof(int32_t)));

    int32_t use_beta = 0;
    float beta = 0.0;

    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc.get(), CNNL_MATMUL_USE_BETA, &(use_beta), sizeof(int32_t)));

    size_t workspace_size = 0;
    int requestedAlgoCount = 1;
    int returnAlgoCount = 0;
    CnnlResourceGuard<cnnlMatMulHeuristicResult_t, cnnlCreateMatMulHeuristicResult, cnnlDestroyMatMulHeuristicResult> heuristic_result;
    CnnlResourceGuard<cnnlMatMulAlgo_t, cnnlMatMulAlgoCreate, cnnlMatMulAlgoDestroy> algo;
    DIOPI_CALLCNNL(cnnlGetMatMulAlgoHeuristic(handle,
                                              matmul_desc.get(),
                                              a_desc.get(),
                                              b_desc.get(),
                                              out_desc.get(),
                                              out_desc.get(),
                                              nullptr,
                                              requestedAlgoCount,
                                              &heuristic_result.get(),
                                              &returnAlgoCount));
    DIOPI_CALLCNNL(cnnlGetMatMulHeuristicResult(heuristic_result.get(), algo.get(), &workspace_size));

    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    float alpha_default = 1.0;

    DIOPI_CALLCNNL(cnnlMatMul_v2(handle,
                                 matmul_desc.get(),
                                 algo.get(),
                                 &alpha_default,
                                 a_desc.get(),
                                 a_casted.data(),
                                 b_desc.get(),
                                 b_casted.data(),
                                 &beta,
                                 out_desc.get(),
                                 out_casted.data(),
                                 workspace,
                                 workspace_size,
                                 out_desc.get(),
                                 out_casted.data()));
    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_casted));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
