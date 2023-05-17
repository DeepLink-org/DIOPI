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

extern "C" {

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean, diopiTensorHandle_t save_invstd,
                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean,
                            diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor save_mean_tr(save_mean);
    DiopiTensor save_invstd_tr(save_invstd);
    DiopiTensor input_tr(input);
    DiopiTensor weight_tr(weight);
    DiopiTensor bias_tr(bias);
    DiopiTensor running_mean_tr(running_mean);
    DiopiTensor running_var_tr(running_var);
    DiopiTensor output_tr(out);

    /* Some basic check */
    if (running_mean_tr.defined() && running_var_tr.defined()) {
        DIOPI_CHECK(running_mean_tr.dtype() == running_var_tr.dtype(), "running_mean and running_var need to have the same data types");
    }
    auto dim = input_tr.dim();
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");
    DIOPI_CHECK(dim == output_tr.dim(), "Input dim != out dim");

    if (3 == dim) {
        input_tr.unsqueeze(3);
        output_tr.reshape(input_tr.shape());
    }
    if (2 == dim) {
        input_tr.unsqueeze(2);
        input_tr.unsqueeze(3);
        output_tr.reshape(input_tr.shape());
    }

    std::vector<DiopiTensor*> p_tensors{&input_tr, &weight_tr, &bias_tr};
    if (running_mean_tr.defined()) {
        p_tensors.push_back(&running_mean_tr);
    }
    if (running_var_tr.defined()) {
        p_tensors.push_back(&running_var_tr);
    }
    std::set<diopiDtype_t> supported_dtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, p_tensors, supported_dtypes));

    // Note: 1. output.dtype = input.dtype  2. channelsLast format
    MemoryFormat memory_format = input_tr.dim() == 4 ? MemoryFormat::ChannelsLast : MemoryFormat::ChannelsLast3d;
    DiopiTensor output_tmp_tr = requiresTensor(ctx, output_tr.shape(), input_tr.dtype(), memory_format);

    /* Transpose to channels last */
    DIOPI_CALL(contiguous_(ctx, input_tr, memory_format));

    CnnlTensorDesc weight_bias_mean_var_desc(weight_tr, CNNL_LAYOUT_ARRAY);
    cnnlTensorLayout_t layout = input_tr.dim() == 4 ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NDHWC;
    CnnlTensorDesc input_desc(input_tr, layout);
    CnnlTensorDesc output_desc(output_tmp_tr, layout);

    if (training) {
        size_t workspace_size = 0;
        DIOPI_CALLCNNL(cnnlGetBatchNormForwardWorkspaceSize(handle, input_desc.get(), &workspace_size));

        void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

        // set activition part to default
        cnnlActivationMode_t active_mode = CNNL_ACTIVATION_IDENTITY;
        cnnlActivationDescriptor_t activation_desc = nullptr;
        DIOPI_CALLCNNL(cnnlCreateActivationDescriptor(&activation_desc));
        cnnlSetActivationDescriptor_v5(activation_desc, active_mode, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 1.0, -1, 1.0, 1.0, false);
        DIOPI_CALLCNNL(cnnlBatchNormForwardTraining_v2(handle,
                                                       activation_desc,
                                                       CNNL_BATCHNORM_SPATIAL,
                                                       CNNL_BATCHNORM_OPS_BN,
                                                       nullptr,
                                                       nullptr,
                                                       input_desc.get(),
                                                       input_tr.data(),
                                                       NULL,
                                                       NULL,
                                                       weight_bias_mean_var_desc.get(),
                                                       weight_tr.data(),
                                                       bias_tr.data(),
                                                       running_mean_tr.defined() ? running_mean_tr.data() : nullptr,
                                                       running_var_tr.defined() ? running_var_tr.data() : nullptr,
                                                       static_cast<float>(eps),
                                                       static_cast<float>(momentum),
                                                       output_desc.get(),
                                                       output_tmp_tr.data(),
                                                       save_mean_tr.data(),
                                                       save_invstd_tr.data(),
                                                       workspace_ptr,
                                                       workspace_size,
                                                       NULL,
                                                       0));
    } else {
        DIOPI_CALLCNNL(cnnlBatchNormForwardInference(handle,
                                                     nullptr,
                                                     nullptr,
                                                     input_desc.get(),
                                                     input_tr.data(),
                                                     weight_bias_mean_var_desc.get(),
                                                     weight_tr.data(),
                                                     bias_tr.data(),
                                                     running_mean_tr.defined() ? running_mean_tr.data() : nullptr,
                                                     running_var_tr.defined() ? running_var_tr.data() : nullptr,
                                                     static_cast<float>(eps),
                                                     output_desc.get(),
                                                     output_tmp_tr.data()));
    }

    // channels last -> contiguous
    DIOPI_CALL(contiguous_(ctx, output_tmp_tr, MemoryFormat::Contiguous));
    // Copy back to origin
    DIOPI_CALL(diopiCopyInp(ctx, output_tmp_tr.tensorHandle(), output_tr.tensorHandle()));

    return diopiSuccess;
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t grad_weight, diopiTensorHandle_t grad_bias,
                                    diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t running_mean, diopiConstTensorHandle_t running_var, diopiConstTensorHandle_t save_mean,
                                    diopiConstTensorHandle_t save_invstd, bool training, double eps) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor grad_input_tr(grad_input);
    DiopiTensor grad_weight_tr(grad_weight);
    DiopiTensor grad_bias_tr(grad_bias);
    DiopiTensor input_tr(input);
    DiopiTensor weight_tr(weight);
    DiopiTensor running_mean_tr(running_mean);
    DiopiTensor running_var_tr(running_var);
    DiopiTensor save_mean_tr(save_mean);
    DiopiTensor save_invstd_tr(save_invstd);

    DiopiTensor grad_output_tr(grad_output);

    if (running_mean_tr.defined() && running_var_tr.defined()) {
        DIOPI_CHECK(running_mean_tr.dtype() == running_var_tr.dtype(), "running_mean and running_var need to have the same data types");
    }
    auto dim = input_tr.dim();
    DIOPI_CHECK(dim >= 2 && dim <= 5, "Input dim is out of range");

    if (3 == dim) {
        input_tr.unsqueeze(3);
        grad_output_tr.unsqueeze(3);
        grad_input_tr.reshape(input_tr.shape());
    }
    if (2 == dim) {
        input_tr.unsqueeze(2);
        input_tr.unsqueeze(3);
        grad_output_tr.unsqueeze(2);
        grad_output_tr.unsqueeze(3);
        grad_input_tr.reshape(input_tr.shape());
    }

    std::vector<DiopiTensor*> p_tensors{&grad_output_tr, &input_tr, &weight_tr};
    if (running_mean_tr.defined()) {
        p_tensors.push_back(&running_mean_tr);
    }
    if (running_var_tr.defined()) {
        p_tensors.push_back(&running_var_tr);
    }
    std::set<diopiDtype_t> supported_dtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, p_tensors, supported_dtypes));

    DiopiTensor grad_weight_tmp_tr = grad_weight_tr;
    if (grad_weight_tr.dtype() != grad_output_tr.dtype()) {
        grad_weight_tmp_tr = requiresTensor(ctx, grad_weight_tr.shape(), grad_output_tr.dtype());
    }
    DiopiTensor grad_bias_tmp_tr = grad_bias_tr;
    if (grad_bias_tr.dtype() != grad_output_tr.dtype()) {
        grad_bias_tmp_tr = requiresTensor(ctx, grad_bias_tr.shape(), grad_output_tr.dtype());
    }

    /* Transpose */
    MemoryFormat memory_format = input_tr.dim() == 4 ? MemoryFormat::ChannelsLast : MemoryFormat::ChannelsLast3d;
    DIOPI_CALL(contiguous_(ctx, input_tr, memory_format));
    DIOPI_CALL(contiguous_(ctx, grad_output_tr, memory_format));

    // Note: 1. output.dtype = input.dtype  2. channelsLast format
    DiopiTensor grad_input_tmp_tr = requiresTensor(ctx, grad_input_tr.shape(), grad_output_tr.dtype(), memory_format);

    cnnlTensorLayout_t layout = input_tr.dim() == 4 ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NDHWC;
    CnnlTensorDesc input_desc(input_tr, layout);
    CnnlTensorDesc grad_output_desc(grad_output_tr, layout);
    CnnlTensorDesc grad_input_desc(grad_input_tmp_tr, layout);
    CnnlTensorDesc weight_bias_mean_var_desc(weight_tr, CNNL_LAYOUT_ARRAY);

    // set activition part
    cnnlBatchNormMode_t mode = CNNL_BATCHNORM_SPATIAL;
    cnnlBatchNormOps_t bnOps = CNNL_BATCHNORM_OPS_BN;
    cnnlActivationMode_t active_mode = CNNL_ACTIVATION_IDENTITY;

    cnnlActivationDescriptor_t activation_desc = nullptr;
    DIOPI_CALLCNNL(cnnlCreateActivationDescriptor(&activation_desc));
    cnnlSetActivationDescriptor_v5(activation_desc, active_mode, CNNL_ACTIVATION_HIGH_PRECISION, CNNL_NOT_PROPAGATE_NAN, 1.0, -1, 1.0, 1.0, false);

    if (training) {
        // get workspace
        size_t workspace_size = 0;
        DIOPI_CALLCNNL(cnnlGetBatchNormBackwardWorkspaceSize(handle, input_desc.get(), &workspace_size));

        void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

        DIOPI_CALLCNNL(cnnlBatchNormBackward_v2(handle,
                                                activation_desc,
                                                mode,
                                                bnOps,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                input_desc.get(),
                                                input_tr.data(),
                                                NULL,
                                                NULL,
                                                grad_output_desc.get(),
                                                grad_output_tr.data(),
                                                weight_bias_mean_var_desc.get(),
                                                weight_tr.data(),
                                                NULL,
                                                save_mean_tr.defined() ? save_mean_tr.data() : nullptr,
                                                save_invstd_tr.defined() ? save_invstd_tr.data() : nullptr,
                                                static_cast<float>(eps),
                                                NULL,
                                                NULL,
                                                grad_input_desc.get(),
                                                grad_input_tmp_tr.data(),
                                                grad_weight_tmp_tr.data(),
                                                grad_bias_tmp_tr.data(),
                                                workspace_ptr,
                                                workspace_size,
                                                NULL,
                                                0));
    } else {
        size_t workspace_size = 0;
        DIOPI_CALLCNNL(cnnlGetFrozenBatchNormBackwardWorkspaceSize(handle, input_desc.get(), &workspace_size));

        void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

        DIOPI_CALLCNNL(cnnlFrozenBatchNormBackward_v2(handle,
                                                      activation_desc,
                                                      mode,
                                                      bnOps,
                                                      input_desc.get(),
                                                      input_tr.data(),
                                                      NULL,
                                                      NULL,
                                                      grad_output_desc.get(),
                                                      grad_output_tr.data(),
                                                      weight_bias_mean_var_desc.get(),
                                                      weight_tr.data(),
                                                      NULL,
                                                      running_mean_tr.defined() ? running_mean_tr.data() : nullptr,
                                                      running_var_tr.defined() ? running_var_tr.data() : nullptr,
                                                      static_cast<float>(eps),
                                                      workspace_ptr,
                                                      workspace_size,
                                                      NULL,
                                                      NULL,
                                                      grad_input_desc.get(),
                                                      grad_input_tmp_tr.data(),
                                                      grad_weight_tmp_tr.data(),
                                                      grad_bias_tmp_tr.data()));
    }

    // Channels last -> contiguous
    DIOPI_CALL(contiguous_(ctx, grad_input_tmp_tr, MemoryFormat::Contiguous));
    DIOPI_CALL(diopiCopyInp(ctx, grad_input_tmp_tr.tensorHandle(), grad_input_tr.tensorHandle()));
    DIOPI_CALL(diopiCopyInp(ctx, grad_weight_tmp_tr.tensorHandle(), grad_weight_tr.tensorHandle()));
    DIOPI_CALL(diopiCopyInp(ctx, grad_bias_tmp_tr.tensorHandle(), grad_bias_tr.tensorHandle()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
