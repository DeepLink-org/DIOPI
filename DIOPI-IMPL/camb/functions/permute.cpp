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

extern "C" diopiError_t diopiPermute(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);
    if (diopi_dtype_float64 == input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
    }
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);


    std::vector<int> perm_data{dims.data, dims.data + dims.len};
    for (int i = 0; i < perm_data.size(); i++) {
        if (perm_data[i] < 0) {
            perm_data[i] += input_tensor.dim();
        }
    }

    CnnlResourceGuard<cnnlTransposeDescriptor_t, cnnlCreateTransposeDescriptor, cnnlDestroyTransposeDescriptor> trans_desc;

    const std::vector<int64_t> src_input_shape = input_tensor.shape();
    int dim_num = src_input_shape.size();
    DIOPI_CALLCNNL(cnnlSetTransposeDescriptor(trans_desc.get(), dim_num, perm_data.data()));
    size_t workspace_size;
    DIOPI_CALLCNNL(cnnlGetTransposeWorkspaceSize(handle, input_desc.get(), trans_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    if (input_tensor.dtype() == output_tensor.dtype()) {
        DIOPI_CALLCNNL(cnnlTranspose_v2(
            handle, trans_desc.get(), input_desc.get(), input_tensor.data(), output_desc.get(), output_tensor.data(), workspace, workspace_size));
    } else {
        DiopiTensor out_temp = requiresTensor(ctx, output_tensor.shape(), input_tensor.dtype());
        CnnlTensorDesc out_temp_desc(out_temp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(
            cnnlTranspose_v2(handle, trans_desc.get(), input_desc.get(), input_tensor.data(), out_temp_desc.get(), out_temp.data(), workspace, workspace_size));
        DIOPI_CALL(dataTypeCast(ctx, output_tensor, out_temp));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
