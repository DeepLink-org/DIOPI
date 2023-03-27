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

DIOPI_API diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto other_tensor = DiopiTensor(other);
    auto out_tensor = DiopiTensor(out);

    DiopiTensor out_tensor_tmp;
    if ((out_tensor.dtype() != diopi_dtype_float16) && (out_tensor.dtype() != diopi_dtype_float32)) {
        out_tensor_tmp = dataTypeCast(ctx, out_tensor, diopi_dtype_float16);
    } else {
        out_tensor_tmp = DiopiTensor(out);
    }
    input_tensor = dataTypeCast(ctx, input_tensor, out_tensor_tmp.dtype());
    other_tensor = dataTypeCast(ctx, other_tensor, out_tensor_tmp.dtype());

    DiopiTensor bcast_input_tensor = broadcastHelper(ctx, input_tensor, out_tensor_tmp);
    DiopiTensor bcast_other_tensor = broadcastHelper(ctx, other_tensor, out_tensor_tmp);

    CnnlTensorDesc bcast_input_desc(bcast_input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc bcast_other_desc(bcast_other_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

    cnnlTensorDescriptor_t input_descs[] = {bcast_input_desc.get(), bcast_other_desc.get()};
    const void* inputs[] = {bcast_input_tensor.data(), bcast_other_tensor.data()};

    DIOPI_CALLCNNL(cnnlMulN(handle, input_descs, inputs, 2, out_desc.get(), out_tensor_tmp.data()))
    if (out_tensor_tmp.dtype() != out_tensor.dtype()) {
        dataTypeCast(ctx, out_tensor, out_tensor_tmp);
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    diopiMul(ctx, input, input, other);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto out_tensor = DiopiTensor(out);
    auto other_tensor = makeTensorFromScalar(ctx, other);
    diopiMul(ctx, out, input, diopiTensorHandle_t(other_tensor));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    diopiMulScalar(ctx, input, input, other);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
