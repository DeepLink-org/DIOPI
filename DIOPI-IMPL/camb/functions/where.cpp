#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {
diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor other_tensor(other);
    DiopiTensor cond_tensor(condition);
    DiopiTensor out_tensor(out);

    std::vector<DiopiTensor*> inputs{&input_tensor, &other_tensor, &cond_tensor};
    std::set<diopiDtype_t> inputs_support_dtype{
        diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_int64, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, inputs, inputs_support_dtype));
    std::vector<DiopiTensor*> cond{&cond_tensor};
    std::set<diopiDtype_t> cond_support_dtype{diopi_dtype_uint8, diopi_dtype_bool};
    DIOPI_CALL(autoCastTensorType(ctx, cond, cond_support_dtype));

    DiopiTensor out_tensor_temp = out_tensor;
    if (out_tensor_temp.dtype() != input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor_temp, input_tensor.dtype()));
    }

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(other_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc cond_desc(cond_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_temp, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetSelectV2WorkspaceSize(handle, cond_desc.get(), input_desc.get(), other_desc.get(), &workspace_size));
    void* workspace = nullptr;
    workspace = requiresBuffer(ctx, workspace_size).data();

    DIOPI_CALLCNNL(cnnlSelectV2(handle,
                                cond_desc.get(),
                                cond_tensor.data(),
                                input_desc.get(),
                                input_tensor.data(),
                                other_desc.get(),
                                other_tensor.data(),
                                workspace,
                                workspace_size,
                                out_desc.get(),
                                out_tensor_temp.data()));

    if (out_tensor_temp.dtype() != out_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_temp));
    }
    return diopiSuccess;
}
}  // extern "C"
}  // namespace camb
}  // namespace impl
