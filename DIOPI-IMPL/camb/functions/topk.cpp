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

DIOPI_API diopiError_t diopiTopk(diopiContextHandle_t ctx,
                                 diopiTensorHandle_t values,
                                 diopiTensorHandle_t indices,
                                 diopiConstTensorHandle_t input,
                                 int64_t k,
                                 int64_t dim,
                                 bool largest,
                                 bool sorted) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto indices_tensor = DiopiTensor(indices);
    auto values_tensor = DiopiTensor(values);

    DiopiTensor values_tensor_temp = values_tensor;
    DiopiTensor input_tensor_temp = input_tensor;
    if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor_temp, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, values_tensor_temp, diopi_dtype_float32));
    } else if (input_tensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor_temp, diopi_dtype_int32));
        DIOPI_CALL(dataTypeCast(ctx, values_tensor_temp, diopi_dtype_int32));
    } else {
        input_tensor_temp = DiopiTensor(input);
        values_tensor_temp = DiopiTensor(values);
    }

    DiopiTensor indices_tensor_temp = indices_tensor;
    DIOPI_CALL(dataTypeCast(ctx, indices_tensor_temp, diopi_dtype_int32));
    CnnlTensorDesc input_desc(input_tensor_temp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc values_desc(values_tensor_temp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indices_desc(indices_tensor_temp, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetTopKTensorWorkspaceSize(handle, input_desc.get(), k, dim, largest, values_desc.get(), indices_desc.get(), &workspace_size));
    void *workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }
    const bool lower_index_first = true;
    DIOPI_CALLCNNL(cnnlTopKTensor_v3(handle,
                                     input_desc.get(),
                                     input_tensor_temp.data(),
                                     k,
                                     dim,
                                     largest,
                                     sorted,
                                     lower_index_first,
                                     workspace,
                                     workspace_size,
                                     values_desc.get(),
                                     values_tensor_temp.data(),
                                     indices_desc.get(),
                                     indices_tensor_temp.data()))
    if (values_tensor_temp.dtype() != values_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, values_tensor, values_tensor_temp));
    }

    if (indices_tensor_temp.dtype() != indices_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, indices_tensor, indices_tensor_temp));
    }

    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
