#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);
    if (diopi_dtype_float64 == input_tensor.dtype()) {
        return diopiDtypeNotSupported;
    }

    const std::vector<int64_t> src_input_shape = input_tensor.shape();
    std::vector<int> perm_data{dims.data, dims.data + dims.len};
    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> trans_desc;

    int dim_num = src_input_shape.size();
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(trans_desc.get(), dim_num, perm_data.data()));
    size_t workspace_size;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, input_desc.get(), trans_desc.get(), &workspace_size));

    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(
        cnnlTranspose_v2(handle, trans_desc.get(), input_desc.get(), input_tensor.data(), output_desc.get(), output_tensor.data(), workspace, workspace_size));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
