/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <string.h>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiBCEWithLogits(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                                          diopiConstTensorHandle_t weight, diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor target_tensor(target);
    DiopiTensor weight_tensor(weight);
    DiopiTensor pos_weight_tensor(pos_weight);
    DiopiTensor out_tensor(out);

    bool weight_flag = true;
    bool pos_weight_flag = true;
    if (!weight) {
        weight_flag = false;
    }
    if (!pos_weight) {
        pos_weight_flag = false;
    }

    std::vector<DiopiTensor*> inTensors{&input_tensor, &target_tensor};
    DIOPI_CALL(autoCastTensorType(ctx, inTensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor input_tensor_tmp = *inTensors[0];
    DiopiTensor target_tensor_tmp = *inTensors[1];
    DiopiTensor out_tensor_tmp = out_tensor;
    DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, input_tensor_tmp.dtype()));

    CnnlTensorDesc input_desc(input_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc target_desc(target_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

    DiopiTensor weight_tensor_tmp;
    DiopiTensor pos_weight_tensor_tmp;
    CnnlTensorDesc weight_desc;
    CnnlTensorDesc pos_weight_desc;
    if (weight_flag) {
        std::vector<DiopiTensor*> wTensors{&weight_tensor};
        DIOPI_CALL(autoCastTensorType(ctx, wTensors, {diopi_dtype_float16, diopi_dtype_float32}));
        weight_tensor_tmp = *wTensors[0];
        weight_desc.set(weight_tensor_tmp, CNNL_LAYOUT_ARRAY);
    }
    if (pos_weight_flag) {
        std::vector<DiopiTensor*> poTensors{&pos_weight_tensor};
        DIOPI_CALL(autoCastTensorType(ctx, poTensors, {diopi_dtype_float16, diopi_dtype_float32}));
        pos_weight_tensor_tmp = *poTensors[0];
        pos_weight_desc.set(pos_weight_tensor_tmp, CNNL_LAYOUT_ARRAY);
    }

    cnnlBceWithLogitsReduction_t reduction_mode;
    switch (reduction) {
        case 0:
            reduction_mode = CNNL_BCE_WITH_LOGITS_NONE;
            break;
        case 1:
            reduction_mode = CNNL_BCE_WITH_LOGITS_MEAN;
            break;
        case 2:
            reduction_mode = CNNL_BCE_WITH_LOGITS_SUM;
            break;
        default:
            DIOPI_CHECK(false, "bce_with_logits reduction parameter is not avaliable");
            break;
    }

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetBceWithLogitsWorkspaceSize(
        handle, input_desc.get(), weight_flag ? weight_desc.get() : nullptr, pos_weight_flag ? pos_weight_desc.get() : nullptr, &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    cnnlComputationPreference_t mode = CNNL_COMPUTATION_FAST;
    DIOPI_CALLCNNL(cnnlBceWithLogits_v2(handle,
                                        mode,
                                        input_desc.get(),
                                        input_tensor_tmp.data(),
                                        target_desc.get(),
                                        target_tensor_tmp.data(),
                                        weight_flag ? weight_desc.get() : nullptr,
                                        weight_flag ? weight_tensor_tmp.data() : nullptr,
                                        pos_weight_flag ? pos_weight_desc.get() : nullptr,
                                        pos_weight_flag ? pos_weight_tensor_tmp.data() : nullptr,
                                        reduction_mode,
                                        workspace,
                                        workspace_size,
                                        out_desc.get(),
                                        out_tensor_tmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBCEWithLogitsBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                  diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight,
                                                  diopiConstTensorHandle_t pos_weight, diopiReduction_t reduction) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor grad_output_tensor(grad_output);
    DiopiTensor input_tensor(input);
    DiopiTensor target_tensor(target);
    DiopiTensor weight_tensor(weight);
    DiopiTensor pos_weight_tensor(pos_weight);
    DiopiTensor grad_input_tensor(grad_input);

    bool weight_flag = true;
    bool pos_weight_flag = true;
    if (!weight) {
        weight_flag = false;
    }
    if (!pos_weight) {
        pos_weight_flag = false;
    }

    std::vector<DiopiTensor*> inTensors{&grad_output_tensor, &input_tensor, &target_tensor};
    DIOPI_CALL(autoCastTensorType(ctx, inTensors, {diopi_dtype_float16, diopi_dtype_float32}));
    DiopiTensor grad_output_tensor_tmp = *inTensors[0];
    DiopiTensor input_tensor_tmp = *inTensors[1];
    DiopiTensor target_tensor_tmp = *inTensors[2];
    DiopiTensor grad_input_tensor_tmp = grad_input_tensor;
    DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor_tmp, input_tensor_tmp.dtype()));

    CnnlTensorDesc grad_output_desc(grad_output_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc input_desc(input_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc target_desc(target_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_input_desc(grad_input_tensor_tmp, CNNL_LAYOUT_ARRAY);

    DiopiTensor weight_tensor_tmp;
    DiopiTensor pos_weight_tensor_tmp;
    CnnlTensorDesc weight_desc;
    CnnlTensorDesc pos_weight_desc;
    if (weight_flag) {
        std::vector<DiopiTensor*> wTensors{&weight_tensor};
        DIOPI_CALL(autoCastTensorType(ctx, wTensors, {diopi_dtype_float16, diopi_dtype_float32}));
        weight_tensor_tmp = *wTensors[0];
        weight_desc.set(weight_tensor_tmp, CNNL_LAYOUT_ARRAY);
    }
    if (pos_weight_flag) {
        std::vector<DiopiTensor*> poTensors{&pos_weight_tensor};
        DIOPI_CALL(autoCastTensorType(ctx, poTensors, {diopi_dtype_float16, diopi_dtype_float32}));
        pos_weight_tensor_tmp = *poTensors[0];
        pos_weight_desc.set(pos_weight_tensor_tmp, CNNL_LAYOUT_ARRAY);
    }

    cnnlBceWithLogitsReduction_t reduction_mode;
    switch (reduction) {
        case 0:
            reduction_mode = CNNL_BCE_WITH_LOGITS_NONE;
            break;
        case 1:
            reduction_mode = CNNL_BCE_WITH_LOGITS_MEAN;
            break;
        case 2:
            reduction_mode = CNNL_BCE_WITH_LOGITS_SUM;
            break;
        default:
            DIOPI_CHECK(false, "bce_with_logits reduction parameter is not avaliable");
            break;
    }

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetBceWithLogitsBackwardWorkspaceSize(
        handle, target_desc.get(), weight_flag ? weight_desc.get() : nullptr, pos_weight_flag ? pos_weight_desc.get() : nullptr, &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlBceWithLogitsBackward(handle,
                                             grad_output_desc.get(),
                                             grad_output_tensor_tmp.data(),
                                             input_desc.get(),
                                             input_tensor_tmp.data(),
                                             target_desc.get(),
                                             target_tensor_tmp.data(),
                                             weight_flag ? weight_desc.get() : nullptr,
                                             weight_flag ? weight_tensor_tmp.data() : nullptr,
                                             pos_weight_flag ? pos_weight_desc.get() : nullptr,
                                             pos_weight_flag ? pos_weight_tensor_tmp.data() : nullptr,
                                             reduction_mode,
                                             workspace,
                                             workspace_size,
                                             grad_input_desc.get(),
                                             grad_input_tensor_tmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, grad_input_tensor, grad_input_tensor_tmp));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
