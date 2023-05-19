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
diopiError_t diopiEmbedding(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t weight, diopiConstTensorHandle_t indices,
                            int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
    DIOPI_CHECK(padding_idx >= std::numeric_limits<int32_t>::min() && padding_idx <= std::numeric_limits<int32_t>::max(),
                "out of the range of values for the INT32 data");
    int32_t padding_idx_casted = static_cast<int32_t>(padding_idx);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor out_tensor(out);
    DiopiTensor weight_tensor(weight);
    DiopiTensor indices_tensor(indices);

    DIOPI_CHECK(padding_idx >= -1 && padding_idx < weight_tensor.shape().front(), "padding_idx should be valid");

    std::vector<DiopiTensor *> tensors{&indices_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_int32, diopi_dtype_int64};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, supportedDtypes));

    DiopiTensor out_tensor_tmp = out_tensor;
    if (weight_tensor.dtype() != out_tensor.dtype()) {
        out_tensor_tmp = requiresTensor(ctx, out_tensor.shape(), weight_tensor.dtype());
    }

    // special case
    if (indices_tensor.dim() == 0 && indices_tensor.numel() == 1) {
        out_tensor_tmp.unsqueeze(0);
    }

    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc weight_desc(weight_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indices_desc(indices_tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlEmbeddingForward_v2(handle,
                                           weight_desc.get(),
                                           weight_tensor.data(),
                                           indices_desc.get(),
                                           static_cast<const int *>(indices_tensor.data()),
                                           padding_idx_casted,
                                           nullptr,
                                           nullptr,
                                           out_desc.get(),
                                           out_tensor_tmp.data()));
    if (out_tensor_tmp.dtype() != out_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));
    }
    return diopiSuccess;
}

diopiError_t diopiEmbeddingBackward(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t grad, diopiConstTensorHandle_t indices,
                                    int64_t num_weights, int64_t padding_idx, bool scale_grad_byfreq, bool sparse) {
    DIOPI_CHECK(padding_idx >= std::numeric_limits<int32_t>::min() && padding_idx <= std::numeric_limits<int32_t>::max(),
                "out of the range of values for the INT32 data");
    DIOPI_CHECK(num_weights >= std::numeric_limits<int32_t>::min() && num_weights <= std::numeric_limits<int32_t>::max(),
                "out of the range of values for the INT32 data");

    int32_t padding_idx_casted = static_cast<int32_t>(padding_idx);
    int32_t num_weights_casted = static_cast<int32_t>(num_weights);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor out_tensor(out);
    DiopiTensor grad_tensor(grad);
    DiopiTensor indices_tensor(indices);

    DIOPI_CHECK(out_tensor.shape().front() == num_weights_casted && out_tensor.shape().back() == grad_tensor.shape().back(), "mismatch of shape");

    std::vector<DiopiTensor *> tensors{&grad_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, supportedDtypes));

    std::vector<DiopiTensor *> tensors1{&indices_tensor};
    DIOPI_CALL(autoCastTensorType(ctx, tensors1, {diopi_dtype_int32}));

    DiopiTensor out_tensor_tmp = out_tensor;
    if (grad_tensor.dtype() != out_tensor.dtype()) {
        out_tensor_tmp = requiresTensor(ctx, out_tensor.shape(), grad_tensor.dtype());
    }

    // special case
    if (indices_tensor.dim() == 0 && indices_tensor.numel() == 1) {
        grad_tensor.unsqueeze(0);
    }

    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_desc(grad_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indices_desc(indices_tensor, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;

    DIOPI_CALLCNNL(cnnlGetEmbeddingBackwardWorkspaceSize(handle, grad_desc.get(), out_desc.get(), scale_grad_byfreq, &workspace_size));

    void *workspace = nullptr;
    if (workspace_size != 0) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlEmbeddingBackward(handle,
                                         padding_idx_casted,
                                         scale_grad_byfreq,
                                         indices_desc.get(),
                                         indices_tensor.data(),
                                         grad_desc.get(),
                                         grad_tensor.data(),
                                         workspace,
                                         workspace_size,
                                         out_desc.get(),
                                         out_tensor_tmp.data()));
    if (out_tensor_tmp.dtype() != out_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));
    }
    return diopiSuccess;
}
}  // extern "C"
}  // namespace camb
}  // namespace impl