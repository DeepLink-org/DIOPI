/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

namespace {
diopiError_t flatten_to_2d(std::vector<int64_t> in_dims, std::vector<int>& out_dims) {
    out_dims.resize(2);
    if (in_dims.size() >= 2) {
        out_dims[0] = std::accumulate(in_dims.begin(), in_dims.end() - 1, 1, std::multiplies<int32_t>());
        out_dims[1] = in_dims[in_dims.size() - 1];
    } else {
        return diopiErrorOccurred;
    }
    return diopiSuccess;
}

diopiError_t matmul(diopiContextHandle_t ctx, DiopiTensor input_a, DiopiTensor input_b, DiopiTensor input_bias, DiopiTensor output, bool trans_a,
                    bool trans_b) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    std::vector<int> input_shape, weight_shape, output_shape;
    DIOPI_CALL(flatten_to_2d(input_a.shape(), input_shape));
    DIOPI_CALL(flatten_to_2d(input_b.shape(), weight_shape));
    DIOPI_CALL(flatten_to_2d(output.shape(), output_shape));

    CnnlTensorDesc a_desc, b_desc, bias_desc, output_desc;
    DIOPI_CALL(a_desc.set(input_a, CNNL_LAYOUT_ARRAY, input_shape));
    DIOPI_CALL(b_desc.set(input_b, CNNL_LAYOUT_ARRAY, weight_shape));
    DIOPI_CALL(output_desc.set(output, CNNL_LAYOUT_ARRAY, output_shape));

    CnnlResourceGuard<cnnlMatMulDescriptor_t, cnnlMatMulDescCreate, cnnlMatMulDescDestroy> matmul_desc;

    cnnlDataType_t comp_type;
    if (output.dtype() == diopi_dtype_float32) {
        comp_type = CNNL_DTYPE_FLOAT;
    } else if (output.dtype() == diopi_dtype_float16) {
        comp_type = CNNL_DTYPE_HALF;
    } else {
        set_last_error_string("%s", "matmul on support float or half.");
        return diopiDtypeNotSupported;
    }
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc.get(), CNNL_MATMUL_DESC_COMPUTE_TYPE, &(comp_type), sizeof(cnnlDataType_t)));

    int32_t is_transa = 0;
    if (trans_a) {
        is_transa = 1;
    }
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc.get(), CNNL_MATMUL_DESC_TRANSA, &(is_transa), sizeof(int32_t)));

    int32_t is_transb = 0;
    if (trans_b) {
        is_transb = 1;
    }
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc.get(), CNNL_MATMUL_DESC_TRANSB, &(is_transb), sizeof(int32_t)));

    int32_t allow_tf32_i32 = 0;
    DIOPI_CALLCNNL(cnnlSetMatMulDescAttr(matmul_desc.get(), CNNL_MATMUL_ALLOW_TF32, &(allow_tf32_i32), sizeof(int32_t)));

    int32_t use_beta = 0;
    float beta = 0.0;
    void* bias_ptr = nullptr;
    if (input_bias.defined()) {
        use_beta = 1;
        beta = 1.0;
        bias_ptr = input_bias.data();
        DIOPI_CALL(bias_desc.set(input_bias, CNNL_LAYOUT_ARRAY));
        DIOPI_CALLCNNL(cnnlExpand(handle, bias_desc.get(), input_bias.data(), output_desc.get(), output.data()));
    }
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
                                              output_desc.get(),
                                              output_desc.get(),
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
                                 input_a.data(),
                                 b_desc.get(),
                                 input_b.data(),
                                 &beta,
                                 output_desc.get(),
                                 output.data(),
                                 workspace,
                                 workspace_size,
                                 output_desc.get(),
                                 output.data()));

    return diopiSuccess;
}
}  // namespace

extern "C" diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t bias) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor weight_tensor(weight);
    DiopiTensor bias_tensor(bias);
    DiopiTensor output_tensor(out);
    DiopiTensor out_temp = output_tensor;

    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, out_temp, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, weight_tensor, diopi_dtype_float32));
        if (bias != nullptr) {
            DIOPI_CALL(dataTypeCast(ctx, bias_tensor, diopi_dtype_float32));
        }
    }

    DIOPI_CALL(matmul(ctx, input_tensor, weight_tensor, bias_tensor, out_temp, false, true));
    if (out_temp.dtype() != output_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output_tensor, out_temp));
    }
    return diopiSuccess;
}
extern "C" diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight,
                                            diopiTensorHandle_t grad_bias, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input,
                                            diopiConstTensorHandle_t weight) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_weight_tensor(grad_weight);
    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor input_tensor(input);
    DiopiTensor weight_tensor(weight);
    DiopiTensor grad_input_temp = grad_input_tensor;
    DiopiTensor grad_weight_temp = grad_weight_tensor;
    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, grad_input_temp, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, grad_weight_temp, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, grad_output_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, weight_tensor, diopi_dtype_float32));
    }
    DiopiTensor bias_tensor((diopiTensorHandle_t) nullptr);

    DIOPI_CALL(matmul(ctx, grad_output_tensor, input_tensor, bias_tensor, grad_weight_temp, true, false));
    if (grad_weight_temp.dtype() != grad_weight_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, grad_weight_tensor, grad_weight_temp));
    }
    DIOPI_CALL(matmul(ctx, grad_output_tensor, weight_tensor, bias_tensor, grad_input_temp, false, false));
    if (grad_input_temp.dtype() != grad_input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, grad_input_temp));
    }

    if (grad_bias != nullptr) {
        DiopiTensor bias_grad_tensor(grad_bias);
        DiopiTensor bias_grad_temp = bias_grad_tensor;
        if (bias_grad_temp.dtype() == diopi_dtype_float64) {
            DIOPI_CALL(dataTypeCast(ctx, bias_grad_temp, diopi_dtype_float32));
        }
        CnnlTensorDesc bias_grad_desc;
        DIOPI_CALL(bias_grad_desc.set(bias_grad_temp, CNNL_LAYOUT_ARRAY));

        std::vector<int> output_shape;
        DIOPI_CALL(flatten_to_2d(grad_output_tensor.shape(), output_shape));
        CnnlTensorDesc grad_output_desc;
        DIOPI_CALL(grad_output_desc.set(grad_output_tensor, CNNL_LAYOUT_ARRAY, output_shape));

        size_t workspace_size_bias;
        DIOPI_CALLCNNL(cnnlGetBiasAddBackwardWorkspaceSize(handle, grad_output_desc.get(), bias_grad_desc.get(), 3, &workspace_size_bias))

        void* workspace_bias = nullptr;
        if (0 != workspace_size_bias) {
            workspace_bias = requiresBuffer(ctx, workspace_size_bias).data();
        }
        DIOPI_CALLCNNL(cnnlBiasAddBackward_v2(
            handle, grad_output_desc.get(), grad_output_tensor.data(), 1, bias_grad_desc.get(), bias_grad_temp.data(), workspace_bias, workspace_size_bias));
        if (bias_grad_tensor.dtype() != bias_grad_temp.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, bias_grad_tensor, bias_grad_temp));
        }
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
