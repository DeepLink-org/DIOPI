#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {
extern "C" {
DIOPI_API diopiError_t diopiStack(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t* tensors, int64_t numTensors, int64_t dim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    std::vector<CnnlTensorDesc> inputsDesc(numTensors);
    std::vector<cnnlTensorDescriptor_t> inputs_desc(numTensors);
    std::vector<const void*> inputs_data(numTensors);

    // insert a new dim to input_tensors
    for (int i = 0; i < numTensors; i++) {
        DiopiTensor temp_tensor(tensors[i]);
        std::vector<int> cat_shape(temp_tensor.shape().begin(), temp_tensor.shape().end());
        cnnlDataType_t dtype;
        CnnlDataType::convertToCnnlType(&dtype, temp_tensor.dtype());
        if (dim == -1) {
            dim = temp_tensor.shape().size();
        }
        cat_shape.insert(cat_shape.begin() + dim, 1);
        int cat_dimNb = cat_shape.size();

        inputs_data[i] = temp_tensor.data();
        inputsDesc[i].set(temp_tensor, CNNL_LAYOUT_ARRAY);
        inputs_desc[i] = inputsDesc[i].get();
        DIOPI_CALLCNNL(cnnlSetTensorDescriptor(inputs_desc[i], CNNL_LAYOUT_ARRAY, dtype, cat_dimNb, cat_shape.data()));
    }
    size_t workspace_size(0);
    DIOPI_CALLCNNL(cnnlGetConcatWorkspaceSize(handle, numTensors, &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }
    DiopiTensor out_tensor(out);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlConcat(handle, numTensors, dim, inputs_desc.data(), inputs_data.data(), workspace, workspace_size, out_desc.get(), out_tensor.data()));
    return diopiSuccess;
}
}  // extern "C"
}  // namespace camb
}  // namespace impl
