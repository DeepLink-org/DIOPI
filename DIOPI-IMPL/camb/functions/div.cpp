/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

DIOPI_API diopiError_t
diopiDiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto other_tensor = DiopiTensor(other);
    auto out_tensor = DiopiTensor(out);

    DiopiTensor out_tensor_temp;
    if ((out_tensor.dtype() != diopi_dtype_float16) && (out_tensor.dtype() != diopi_dtype_float32)) {
        out_tensor_temp = dataTypeCast(ctx, out_tensor, diopi_dtype_float32);
    } else {
        out_tensor_temp = DiopiTensor(out);
    }

    input_tensor = dataTypeCast(ctx, input_tensor, out_tensor_temp.dtype());
    other_tensor = dataTypeCast(ctx, other_tensor, out_tensor_temp.dtype());

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(other_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_temp, CNNL_LAYOUT_ARRAY);
    size_t workspace_size = 0;
    cnnlGetDivWorkspaceSize(handle, input_desc.get(), other_desc.get(), out_desc.get(), &workspace_size);
    void* workspace = nullptr;
    workspace = requiresBuffer(ctx, workspace_size).data();

    cnnlDiv_v2(handle,
               CNNL_COMPUTATION_HIGH_PRECISION,
               input_desc.get(),
               input_tensor.data(),
               other_desc.get(),
               other_tensor.data(),
               workspace,
               workspace_size,
               out_desc.get(),
               out_tensor.data());
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiDivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, diopiRoundMode_t rounding_mode) {
    diopiDiv(ctx, input, input, other, rounding_mode);
    return diopiSuccess;
}

DIOPI_API diopiError_t
diopiDivScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto other_tensor = makeTensorFromScalar(ctx, other);
    auto out_tensor = DiopiTensor(out);
    diopiDiv(ctx, out, input, diopiTensorHandle_t(other_tensor), rounding_mode);
    return diopiSuccess;
}
DIOPI_API diopiError_t diopiDivInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other, diopiRoundMode_t rounding_mode) {
    diopiDivScalar(ctx, input, input, other, rounding_mode);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
