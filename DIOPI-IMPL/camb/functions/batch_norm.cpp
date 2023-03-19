/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiBatchNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t save_mean,
                                      diopiTensorHandle_t save_invstd, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                      diopiConstTensorHandle_t bias, diopiTensorHandle_t running_mean,
                                      diopiTensorHandle_t running_var, bool training, double momentum, double eps) {
    /* Generate Tensors */
    auto save_mean_tr = DiopiTensor(save_mean);
    auto save_invstd_tr = DiopiTensor(save_invstd);
    auto input_tr = DiopiTensor(input);
    auto weight_tr = DiopiTensor(weight);
    auto bias_tr = DiopiTensor(bias);
    auto running_mean_tr = DiopiTensor(running_mean);
    auto running_var_tr = DiopiTensor(running_var);
    auto output_tr = DiopiTensor(out);

    /* Some basic check */
    if (running_mean_tr.defined() && running_var_tr.defined()) {
        DIOPI_CHECK(running_mean_tr.dtype() ==  running_var_tr.dtype(), "running_mean and running_var need to have the same data types");
    }
    // TODO(ywt): 2,3,5 dim support
    DIOPI_CHECK(input_tr.dim() >= 4 && input_tr.dim() <=4, "Input dim is out of range");
    DIOPI_CHECK(input_tr.dim() == output_tr.dim(), "Input dim != out dim");

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    /* Transpose NCHW to NHWC */
    auto memory_format = MemoryFormat::ChannelsLast;
    auto input_channel_last = input_tr.contiguous(ctx, memory_format);
    cnnl_transpose(ctx, handle, input_tr, input_channel_last, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    auto output_channel_last = output_tr.contiguous(ctx, memory_format);

    CnnlTensorDesc weight_bias_mean_var_desc(weight_tr, CNNL_LAYOUT_ARRAY);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc input_channel_last_desc(input_channel_last, layout);
    CnnlTensorDesc output_channel_last_desc(output_channel_last, layout);

    if (training) {
        // get workspace
        size_t workspace_size = 0;
        DIOPI_CHECKCNNL(cnnlGetBatchNormForwardWorkspaceSize(handle, input_channel_last_desc.get(), &workspace_size));

        void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

        // set activition part to default
        cnnlActivationMode_t active_mode = CNNL_ACTIVATION_IDENTITY;
        cnnlActivationDescriptor_t activation_desc = nullptr;
        cnnlCreateActivationDescriptor(&activation_desc);
        cnnlSetActivationDescriptor_v5(activation_desc, active_mode, CNNL_ACTIVATION_HIGH_PRECISION,
                                                            CNNL_NOT_PROPAGATE_NAN, 1.0, -1, 1.0, 1.0, false);
        DIOPI_CALLCNNL(cnnlBatchNormForwardTraining_v2(
            /* handle   */ handle,
            /*activation_desc */ activation_desc,
            /*mode */ CNNL_BATCHNORM_SPATIAL,
            /*bnOps */ CNNL_BATCHNORM_OPS_BN,
            /* alpha    */ nullptr,
            /* beta     */ nullptr,
            /* x_desc   */ input_channel_last_desc.get(),
            /* x        */ input_channel_last.data(),
            /* z_desc */ NULL,
            /* z */ NULL,
            /* wbmvd    */ weight_bias_mean_var_desc.get(),
            /* weight   */ weight_tr.data(),
            /* bias     */ bias_tr.data(),
            /* mov_mean */ running_mean_tr.defined() ? running_mean_tr.data() : nullptr,
            /* mov_var  */ running_var_tr.defined() ? running_var_tr.data() : nullptr,
            /* eps      */ static_cast<float>(eps),
            /* momentum */ static_cast<float>(momentum),
            /* y_desc   */ output_channel_last_desc.get(),
            /* y        */ output_channel_last.data(),
            /* save_mean*/ save_mean_tr.data(),
            /* save_std */ save_invstd_tr.data(),
            /* workspace */ workspace_ptr,
            /* workspace_size */ workspace_size,
            /* reservespace */ NULL,
            /* reservespace_size */ 0));
    } else {
        DIOPI_CALLCNNL(cnnlBatchNormForwardInference(
            /* handle   */ handle,
            /* alpha    */ nullptr,
            /* beta     */ nullptr,
            /* x_desc   */ input_channel_last_desc.get(),
            /* x        */ input_channel_last.data(),
            /* wbmvd    */ weight_bias_mean_var_desc.get(),
            /* weight   */ weight_tr.data(),
            /* bias     */ bias_tr.data(),
            /* mov_mean */ running_mean_tr.defined() ? running_mean_tr.data() : nullptr,
            /* mov_var  */ running_var_tr.defined() ? running_var_tr.data() : nullptr,
            /* eps      */ static_cast<float>(eps),
            /* z_desc   */ output_channel_last_desc.get(),
            /* z        */ output_channel_last.data()));
    }

    // NHWC -> NCHW
    cnnl_transpose(ctx, handle, output_channel_last, output_tr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);

    // cnrtQueueSync(stream);
    return diopiSuccess;
}

diopiError_t diopiBatchNormBackward(diopiContextHandle_t ctx,
                                    diopiTensorHandle_t grad_input,
                                    diopiTensorHandle_t grad_weight,
                                    diopiTensorHandle_t grad_bias,
                                    diopiConstTensorHandle_t grad_output,
                                    diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t weight,
                                    diopiConstTensorHandle_t running_mean,
                                    diopiConstTensorHandle_t running_var,
                                    diopiConstTensorHandle_t save_mean,
                                    diopiConstTensorHandle_t save_invstd,
                                    bool training, double eps) {
    /* Generate diopi Tensors and Handle*/
    auto grad_input_tr = DiopiTensor(grad_input);
    auto grad_weight_tr = DiopiTensor(grad_weight);
    auto grad_bias_tr = DiopiTensor(grad_bias);
    auto input_tr = DiopiTensor(input);
    auto weight_tr = DiopiTensor(weight);
    auto running_mean_tr = DiopiTensor(running_mean);
    auto running_var_tr = DiopiTensor(running_var);
    auto save_mean_tr = DiopiTensor(save_mean);
    auto save_invstd_tr = DiopiTensor(save_invstd);

    auto grad_output_tr = DiopiTensor(grad_output);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    /* Some basic check */
    if (running_mean_tr.defined() && running_var_tr.defined()) {
        DIOPI_CHECK(running_mean_tr.dtype() ==  running_var_tr.dtype(), "running_mean and running_var need to have the same data types");
    }
    // TODO(ywt): 2,3,5 dim support
    DIOPI_CHECK(input_tr.dim() >= 4 && input_tr.dim() <=4, "Input dim is out of range");

    /* Transpose */
    auto memory_format = MemoryFormat::ChannelsLast;
    auto input_channel_last = input_tr.contiguous(ctx, memory_format);
    cnnl_transpose(ctx, handle, input_tr, input_channel_last, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    auto grad_output_channel_last = grad_output_tr.contiguous(ctx, memory_format);
    cnnl_transpose(ctx, handle, grad_output_tr, grad_output_channel_last, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    auto grad_input_channel_last = grad_input_tr.contiguous(ctx, memory_format);

    /* Generate description */
    cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDesc input_desc(input_channel_last, layout);
    CnnlTensorDesc grad_output_desc(grad_output_channel_last, layout);
    CnnlTensorDesc grad_input_desc(grad_input_channel_last, layout);
    CnnlTensorDesc weight_bias_mean_var_desc(weight_tr, CNNL_LAYOUT_ARRAY);

    // set activition part
    cnnlBatchNormMode_t mode = CNNL_BATCHNORM_SPATIAL;
    cnnlBatchNormOps_t bnOps = CNNL_BATCHNORM_OPS_BN;
    cnnlActivationMode_t active_mode = CNNL_ACTIVATION_IDENTITY;

    cnnlActivationDescriptor_t activation_desc = nullptr;
    cnnlCreateActivationDescriptor(&activation_desc);
    cnnlSetActivationDescriptor_v5(activation_desc, active_mode, CNNL_ACTIVATION_HIGH_PRECISION,
                                                        CNNL_NOT_PROPAGATE_NAN, 1.0, -1, 1.0, 1.0, false);

    if (training) {
        // get workspace
        size_t workspace_size = 0;
        DIOPI_CHECKCNNL(cnnlGetBatchNormBackwardWorkspaceSize(handle, input_desc.get(), &workspace_size));

        void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

        DIOPI_CALLCNNL(cnnlBatchNormBackward_v2(
            handle,
            activation_desc,
            mode,
            bnOps,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            input_desc.get(),
            input_channel_last.data(),
            NULL,
            NULL,
            grad_output_desc.get(),
            grad_output_channel_last.data(),
            weight_bias_mean_var_desc.get(),
            weight_tr.data(),
            NULL,
            save_mean_tr.defined() ? save_mean_tr.data() : nullptr,
            save_invstd_tr.defined() ? save_invstd_tr.data() : nullptr,
            static_cast<float>(eps),
            NULL,
            NULL,
            grad_input_desc.get(),
            grad_input_channel_last.data(),
            grad_weight_tr.data(),
            grad_bias_tr.data(),
            workspace_ptr,
            workspace_size,
            NULL,
            0));
    } else {
        size_t workspace_size = 0;
        DIOPI_CHECKCNNL(cnnlGetFrozenBatchNormBackwardWorkspaceSize(handle, input_desc.get(), &workspace_size));

        void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

        DIOPI_CALLCNNL(cnnlFrozenBatchNormBackward_v2(
            handle,
            activation_desc,
            mode,
            bnOps,
            input_desc.get(),
            input_channel_last.data(),
            NULL,
            NULL,
            grad_output_desc.get(),
            grad_output_channel_last.data(),
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
            grad_input_channel_last.data(),
            grad_weight_tr.data(),
            grad_bias_tr.data()));
    }

    // NHWC -> NCHW
    cnnl_transpose(ctx, handle, grad_input_channel_last, grad_input_tr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
