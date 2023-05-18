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

diopiError_t diopiMul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor other_tensor(other);
    DiopiTensor out_tensor(out);


    DiopiTensor out_tensor_tmp = out_tensor;
    if ((out_tensor.dtype() != diopi_dtype_float16) && (out_tensor.dtype() != diopi_dtype_float32)) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, diopi_dtype_float32));
    }
    DIOPI_CALL(dataTypeCast(ctx, input_tensor, out_tensor_tmp.dtype()));
    DIOPI_CALL(dataTypeCast(ctx, other_tensor, out_tensor_tmp.dtype()));

    DiopiTensor bcast_input_tensor;
    broadcastHelper(ctx, input_tensor, out_tensor_tmp, &bcast_input_tensor);
    DiopiTensor bcast_other_tensor;
    broadcastHelper(ctx, other_tensor, out_tensor_tmp, &bcast_other_tensor);

    CnnlTensorDesc bcast_input_desc(bcast_input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc bcast_other_desc(bcast_other_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

    cnnlTensorDescriptor_t input_descs[] = {bcast_input_desc.get(), bcast_other_desc.get()};
    const void* inputs[] = {bcast_input_tensor.data(), bcast_other_tensor.data()};

    DIOPI_CALLCNNL(cnnlMulN(handle, input_descs, inputs, 2, out_desc.get(), out_tensor_tmp.data()))
    if (out_tensor_tmp.dtype() != out_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));
    }
    return diopiSuccess;
}

diopiError_t diopiMulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other) {
    DIOPI_CALL(diopiMul(ctx, input, input, other));
    return diopiSuccess;
}

diopiError_t diopiMulScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);
    DiopiTensor other_tensor_tmp;
    makeTensorFromScalar(ctx, other, other_tensor_tmp);
    auto other_tensor = other_tensor_tmp.tensorHandle();
    DIOPI_CALL(diopiMul(ctx, out, input, diopiTensorHandle_t(other_tensor)));
    return diopiSuccess;
}

diopiError_t diopiMulInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* other) {
    DIOPI_CALL(diopiMulScalar(ctx, input, input, other));
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
