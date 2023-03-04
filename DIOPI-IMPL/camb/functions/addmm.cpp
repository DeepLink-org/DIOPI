#include <diopi/functions.h>
#include <string.h>
#include <iostream>
#include <numeric>
#include "../cnnl_helper.hpp"

extern "C" {

DIOPI_API diopiError_t diopiAddmm(diopiContextHandle_t ctx,
                                  diopiTensorHandle_t out,
                                  diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t mat1,
                                  diopiConstTensorHandle_t mat2,
                                  const diopiScalar_t* beta,
                                  const diopiScalar_t* alpha) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> CnnlMatMulDesc;
    cnnlMatMulDescriptor_t matmul_desc = CnnlMatMulDesc.get();

    int32_t is_transa = 0;
    int32_t is_transb = 0;
    int32_t allow_tf32_i32 = 1;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSA, &(is_transa), sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_DESC_TRANSB, &(is_transb), sizeof(int32_t)));
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc, CNNL_MATMUL_ALLOW_TF32, &(allow_tf32_i32), sizeof(int32_t)));

    auto mat1_tensor = impl::camb::makeTensor(mat1);
    auto mat2_tensor = impl::camb::makeTensor(mat2);
    auto input_tensor = impl::camb::makeTensor(input);
    auto out_tensor = impl::camb::makeTensor(out);
    diopiTensorHandle_t mm_result;
    diopiTensorHandle_t tmpc;
    diopiSize_t out_shape;
    diopiGetTensorShape(out, &out_shape);
    diopiRequireTensor(ctx, &mm_result, &out_shape, nullptr, out_tensor.dtype(), diopi_device);
    auto mm_result_tensor = impl::camb::makeTensor(mm_result);
    diopiRequireTensor(ctx, &tmpc, &out_shape, nullptr, out_tensor.dtype(), diopi_device);
    auto tmpc_tensor = impl::camb::makeTensor(tmpc);

    CnnlTensorDesc tmpc_desc(tmpc_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mm_result_desc(mm_result_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mat1_desc(mat1_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mat2_desc(mat2_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    void* tmpc_ptr = tmpc_tensor.data();
    void* mm_result_ptr = mm_result_tensor.data();
    const void* mat1_ptr = mat1_tensor.data();
    const void* mat2_ptr = mat2_tensor.data();
    const void* input_ptr = input_tensor.data();
    void* out_ptr = out_tensor.data();

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
                                              tmpc_desc.get(),
                                              nullptr,
                                              requestedAlgoCount,
                                              &heuristicResult,
                                              &returnAlgoCount));
    DIOPI_CALLCNNL(cnnlGetMatMulHeuristicResult(heuristicResult, matmul_algo, &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = impl::camb::requiresBuffer(ctx, workspace_size).data();
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
                                 mat1_ptr,
                                 mat2_desc.get(),
                                 mat2_ptr,
                                 &beta_default,
                                 tmpc_desc.get(),
                                 tmpc_ptr,
                                 workspace,
                                 workspace_size,
                                 mm_result_desc.get(),
                                 mm_result_ptr));

    CnnlResourceGuard<cnnlOpTensorDescriptor_t, cnnlCreateOpTensorDescriptor, cnnlDestroyOpTensorDescriptor> CnnlOpTensorDesc;
    cnnlOpTensorDescriptor_t optensor_desc = CnnlOpTensorDesc.get();
    size_t workspace_size_ = 0;
    DIOPI_CALLCNNL(cnnlGetOpTensorWorkspaceSize(handle, mm_result_desc.get(), input_desc.get(), out_desc.get(), &workspace_size_));
    void* workspace_ = nullptr;
    if (0 != workspace_size_) {
        workspace_ = impl::camb::requiresBuffer(ctx, workspace_size_).data();
    }

    DIOPI_CALLCNNL(cnnlOpTensor(handle,
                                optensor_desc,
                                &alpha_,
                                mm_result_desc.get(),
                                mm_result_ptr,
                                &beta_,
                                input_desc.get(),
                                input_ptr,
                                workspace_,
                                workspace_size_,
                                &beta_default,
                                out_desc.get(),
                                out_ptr));

    return diopiSuccess;
}
}
