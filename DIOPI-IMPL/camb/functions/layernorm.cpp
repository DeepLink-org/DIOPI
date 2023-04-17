#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiLayerNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiSize_t normalized_shape,
                            double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);
    DiopiTensor save_mean_tensor(save_mean);
    DiopiTensor save_invstd_tensor(save_invstd);

    diopiDtype_t out_dtype = out_tensor.dtype();
    if (out_dtype != diopi_dtype_float32 && out_dtype != diopi_dtype_float16) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, save_mean_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, save_invstd_tensor, diopi_dtype_float32));
    }

    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc save_meanDesc(save_mean_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size(0);
    DIOPI_CALLCNNL(cnnlGetLayerNormOpWorkspaceSize(handle, normalized_shape.len, inputDesc.get(), &workspace_size));
    void *workspace = nullptr;
    if (workspace_size > 0) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    void *weight_ptr = nullptr;
    void *bias_ptr = nullptr;
    CnnlTensorDesc weight_biasDesc;
    cnnlTensorDescriptor_t weight_bias_desc = nullptr;
    if (weight != nullptr && bias != nullptr) {
        DiopiTensor weight_tensor(weight);
        DiopiTensor bias_tensor(bias);
        if (out_dtype != diopi_dtype_float32 && out_dtype != diopi_dtype_float16) {
            DIOPI_CALL(dataTypeCast(ctx, weight_tensor, diopi_dtype_float32));
            DIOPI_CALL(dataTypeCast(ctx, bias_tensor, diopi_dtype_float32));
        }
        weight_ptr = weight_tensor.data();
        bias_ptr = bias_tensor.data();
        weight_biasDesc.set(weight_tensor, CNNL_LAYOUT_ARRAY);
        weight_bias_desc = weight_biasDesc.get();
    }

    int axis = input_tensor.dim() - normalized_shape.len;
    DIOPI_CALLCNNL(cnnlLayerNormForward(handle,
                                        inputDesc.get(),
                                        input_tensor.data(),
                                        axis,
                                        weight_bias_desc,
                                        weight_ptr,
                                        bias_ptr,
                                        eps,
                                        workspace,
                                        workspace_size,
                                        outDesc.get(),
                                        out_tensor.data(),
                                        save_meanDesc.get(),
                                        save_mean_tensor.data(),
                                        save_invstd_tensor.data()));

    if (out_dtype != diopi_dtype_float32 && out_dtype != diopi_dtype_float16) {
        DiopiTensor out_tensor_(out);
        DiopiTensor save_mean_tensor_(save_mean);
        DiopiTensor save_invstd_tensor_(save_invstd);
        DIOPI_CALL(dataTypeCast(ctx, out_tensor_, out_tensor));
        DIOPI_CALL(dataTypeCast(ctx, save_mean_tensor_, save_mean_tensor));
        DIOPI_CALL(dataTypeCast(ctx, save_invstd_tensor_, save_invstd_tensor));
    }

    return diopiSuccess;
}

diopiError_t diopiLayerNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                    diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t bias, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t rstd, diopiSize_t normalized_shape) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor grad_input_tensor(grad_input);
    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor input_tensor(input);
    DiopiTensor mean_tensor(mean);
    DiopiTensor rstd_tensor(rstd);
    DiopiTensor weight_tensor(weight);
    DiopiTensor bias_tensor(bias);
    DiopiTensor grad_weight_tensor(grad_weight);
    DiopiTensor grad_bias_tensor(grad_bias);

    diopiDtype_t out_dtype = grad_input_tensor.dtype();
    if (out_dtype != diopi_dtype_float16 && out_dtype != diopi_dtype_float32) {
        DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, grad_output_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, mean_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, rstd_tensor, diopi_dtype_float32));
    }

    CnnlTensorDesc grad_inputDesc(grad_input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_outputDesc(grad_output_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc meanDesc(mean_tensor, CNNL_LAYOUT_ARRAY);

    void *weight_ptr = nullptr;
    CnnlTensorDesc weight_biasDesc;
    cnnlTensorDescriptor_t weight_bias_desc = nullptr;
    void *grad_weight_ptr = nullptr;
    void *grad_bias_ptr = nullptr;
    if (weight != nullptr && bias != nullptr) {
        if (out_dtype != diopi_dtype_float16 && out_dtype != diopi_dtype_float32) {
            DIOPI_CALL(dataTypeCast(ctx, weight_tensor, diopi_dtype_float32));
            DIOPI_CALL(dataTypeCast(ctx, grad_weight_tensor, diopi_dtype_float32));
            DIOPI_CALL(dataTypeCast(ctx, grad_bias_tensor, diopi_dtype_float32));
        }

        weight_ptr = weight_tensor.data();
        grad_weight_ptr = grad_weight_tensor.data();
        grad_bias_ptr = grad_bias_tensor.data();
        weight_biasDesc.set(weight_tensor, CNNL_LAYOUT_ARRAY);
        weight_bias_desc = weight_biasDesc.get();
    } else {
        weight_tensor = requiresTensor(ctx, normalized_shape, input_tensor.dtype());
        grad_weight_tensor = requiresTensor(ctx, normalized_shape, input_tensor.dtype());
        grad_bias_tensor = requiresTensor(ctx, normalized_shape, input_tensor.dtype());
        diopiScalar_t one = {diopi_dtype_float32, 1};
        diopiScalar_t zero = {diopi_dtype_float32, 0};
        DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(weight_tensor), &one));
        DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(grad_weight_tensor), &zero));
        DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(grad_bias_tensor), &zero));
        weight_ptr = weight_tensor.data();
        weight_biasDesc.set(weight_tensor, CNNL_LAYOUT_ARRAY);
        weight_bias_desc = weight_biasDesc.get();
        grad_weight_ptr = grad_weight_tensor.data();
        grad_bias_ptr = grad_bias_tensor.data();
    }

    int axis = input_tensor.dim() - normalized_shape.len;

    size_t workspace_size(0);
    DIOPI_CALLCNNL(cnnlGetLayerNormBackwardWorkspaceSize(handle, inputDesc.get(), axis, &workspace_size));
    void *workspace;
    if (workspace_size > 0) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlLayerNormBackward_v2(handle,
                                            inputDesc.get(),
                                            input_tensor.data(),
                                            axis,
                                            grad_outputDesc.get(),
                                            grad_output_tensor.data(),
                                            weight_bias_desc,
                                            weight_ptr,
                                            meanDesc.get(),
                                            mean_tensor.data(),
                                            rstd_tensor.data(),
                                            workspace,
                                            workspace_size,
                                            grad_inputDesc.get(),
                                            grad_input_tensor.data(),
                                            grad_weight_ptr,
                                            grad_bias_ptr));
    if (out_dtype != diopi_dtype_float16 && out_dtype != diopi_dtype_float32) {
        DiopiTensor grad_input_tensor_(grad_input);
        DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor_, grad_input_tensor));
        if (grad_bias != nullptr && grad_weight != nullptr) {
            DiopiTensor grad_weight_tensor_(grad_weight);
            DiopiTensor grad_bias_tensor_(grad_bias);
            DIOPI_CALL(dataTypeCast(ctx, grad_weight_tensor_, grad_weight_tensor));
            DIOPI_CALL(dataTypeCast(ctx, grad_bias_tensor_, grad_bias_tensor));
        }
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
