/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiCat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t num_inputs, int64_t dim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    std::vector<CnnlTensorDesc> inputsDesc(num_inputs);
    std::vector<cnnlTensorDescriptor_t> inputs_desc(num_inputs);
    std::vector<const void *> inputs(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        DiopiTensor temp_tensor(tensors[i]);
        inputsDesc[i].set(temp_tensor, CNNL_LAYOUT_ARRAY);
        inputs_desc[i] = inputsDesc[i].get();
        inputs[i] = temp_tensor.data();
    }

    size_t workspace_size(0);
    DIOPI_CALLCNNL(cnnlGetConcatWorkspaceSize(handle, num_inputs, &workspace_size));
    void * workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DiopiTensor out_tensor(out);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlConcat(handle, num_inputs, dim, inputs_desc.data(), inputs.data(), workspace, workspace_size, out_desc.get(), out_tensor.data()));

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
