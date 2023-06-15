#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {
diopiError_t diopiUnique(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, const int64_t *dim, bool sorted,
                         bool return_counts, diopiTensorHandle_t indices, diopiTensorHandle_t *counts) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    // input_tensor
    DiopiTensor input_tensor(input);
    int realDim = -1;  // If dim is set to -1, the unique of the flattened input is to apply in CNNL.
    if (dim != nullptr) {
        realDim = ((*dim) < 0) ? (*dim + input_tensor.dim()) : *dim;
    }

    // dtype cast
    diopiDtype_t origin_input_dtype = input_tensor.dtype();
    std::vector<DiopiTensor *> pTensors{&input_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_int32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    // output_tensor
    DiopiTensor output_tensor =
        (realDim != -1) ? requiresTensor(ctx, {input_tensor.shape()}, input_tensor.dtype()) : requiresTensor(ctx, {input_tensor.numel()}, input_tensor.dtype());
    // index_tensor
    DiopiTensor index_tensor = (realDim != -1) ? requiresTensor(ctx, {input_tensor.shape()[realDim]}, diopi_dtype_int32)
                                               : requiresTensor(ctx, {input_tensor.numel()}, diopi_dtype_int32);
    // counts_tensor
    DiopiTensor counts_tensor = (realDim != -1) ? requiresTensor(ctx, {output_tensor.shape()[realDim]}, diopi_dtype_int32)
                                                : requiresTensor(ctx, {output_tensor.numel()}, diopi_dtype_int32);

    // Tensor Desc
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(output_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(index_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc countsDesc(counts_tensor, CNNL_LAYOUT_ARRAY);

    CnnlResourceGuard<cnnlUniqueDescriptor_t, cnnlCreateUniqueDescriptor, cnnlDestroyUniqueDescriptor> uniqueDesc;

    // torch.unique always sort the tensor at the beginning
    // regardless of the sort argument when dim is specified
    if (*dim != -1) {
        sorted = true;
    }
    cnnlUniqueSort_t mode = sorted ? CNNL_SORT_ASCEND : CNNL_UNSORT_FORWARD;
    bool return_indices = (indices != nullptr) ? true : false;

    if (mode == CNNL_UNSORT_FORWARD) {
        DIOPI_CHECK((input_tensor.dim() == 1),
                    "the dimension of input must be one-dimensional "
                    "when mode is CNNL_UNSORT_FORWARD");
    }
    DIOPI_CALLCNNL(cnnlSetUniqueDescriptor(uniqueDesc.get(), mode, realDim, return_indices, return_counts));
    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetUniqueWorkspaceSize(handle, uniqueDesc.get(), inputDesc.get(), &workspace_size));
    void *workspace = nullptr;
    if (workspace_size != 0) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    std::vector<int64_t> temp{1, 1};
    DiopiTensor outlen_tensor = requiresTensor(ctx, temp, diopi_dtype_int32);

    DIOPI_CALLCNNL(cnnlUnique_v2(handle,
                                 uniqueDesc.get(),
                                 inputDesc.get(),
                                 input_tensor.data(),
                                 workspace,
                                 workspace_size,
                                 static_cast<int *>(outlen_tensor.data()),
                                 outputDesc.get(),
                                 output_tensor.data(),
                                 indexDesc.get(),
                                 return_indices ? index_tensor.data() : NULL,
                                 countsDesc.get(),
                                 return_counts ? counts_tensor.data() : NULL));

    DIOPI_CALL(dataTypeCast(ctx, output_tensor, origin_input_dtype));
    int32_t outlen_host = 0;
    cnrtMemcpyAsync(&outlen_host, outlen_tensor.data(), sizeof(int32_t), getStream(ctx), cnrtMemcpyDevToHost);
    cnrtQueueSync(getStream(ctx));

    std::vector<int64_t> true_out_shape = input_tensor.shape();
    if (realDim != -1) {
        true_out_shape[realDim] = outlen_host;
    } else {
        true_out_shape[0] = outlen_host;
    }
    DiopiTensor sliced_output_tensor = requiresTensor(ctx, true_out_shape, output_tensor.dtype());
    DiopiTensor sliced_counts_tensor = requiresTensor(ctx, {outlen_host}, diopi_dtype_int32);

    diopiSlice(ctx, diopiTensorHandle_t(sliced_output_tensor), diopiTensorHandle_t(output_tensor), realDim, 0, outlen_host, 1);
    diopiSlice(ctx, diopiTensorHandle_t(sliced_counts_tensor), diopiTensorHandle_t(counts_tensor), 0, 0, outlen_host, 1);

    *out = diopiTensorHandle_t(sliced_output_tensor);
    if (return_indices) {
        DiopiTensor true_index_tensor(indices);
        CnnlTensorDesc true_index_desc(true_index_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALL(dataTypeCast(ctx, index_tensor, diopi_dtype_int64));
        CnnlTensorDesc indexDesc64(index_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlCopy(handle, indexDesc64.get(), index_tensor.data(), true_index_desc.get(), true_index_tensor.data()));
    }
    if (return_counts) {
        *counts = diopiTensorHandle_t(sliced_counts_tensor);
    }
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl