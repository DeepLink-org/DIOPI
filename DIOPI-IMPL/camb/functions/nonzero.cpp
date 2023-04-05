
#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DiopiTensor nonzeroCount(diopiContextHandle_t ctx, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);

    std::vector<int64_t> shape = {1};
    auto num_true = requiresTensor(ctx, shape, diopi_dtype_int32);
    CnnlTensorDesc num_trueDesc(num_true, CNNL_LAYOUT_ARRAY);

    cnnlNumTrue_v2(handle, inputDesc.get(), input_tensor.data(), num_trueDesc.get(), num_true.data());
    return num_true;
}

diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    if (input_tensor.dtype() == diopi_dtype_uint8 || input_tensor.dtype() == diopi_dtype_int8 || input_tensor.dtype() == diopi_dtype_int16 ||
        input_tensor.dtype() == diopi_dtype_int64) {
        dataTypeCast(ctx, input_tensor, diopi_dtype_int32);
    }
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);

    DiopiTensor num_true = nonzeroCount(ctx, diopiTensorHandle_t(input_tensor));
    CnnlTensorDesc num_trueDesc(num_true, CNNL_LAYOUT_ARRAY);

    size_t workspace_size(0);
    DIOPI_CALLCNNL(cnnlGetWhereWorkspaceSize(handle, num_trueDesc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    int32_t count = 0;
    cnrtMemcpy(&count, num_true.data(), sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST);
    // ! copy again, otherwise copy might fail.
    cnrtMemcpy(&count, num_true.data(), sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST);

    std::vector<int64_t> shape(2);
    shape[0] = count;
    shape[1] = input_tensor.dim();
    auto out_tensor = requiresTensor(ctx, shape, diopi_dtype_int32);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlWhere_v2(
        handle, inputDesc.get(), input_tensor.data(), num_trueDesc.get(), num_true.data(), false, workspace, workspace_size, outDesc.get(), out_tensor.data()));
    *out = diopiTensorHandle_t(out_tensor);
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
