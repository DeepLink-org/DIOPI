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

DIOPI_API diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto out_tensor = DiopiTensor(out);

    DIOPI_CHECK(((input_tensor.dtype() == diopi_dtype_float16) || (input_tensor.dtype() == diopi_dtype_float32)),
                "input datatype not supported, only float16, float32 supported");
    DIOPI_CHECK(((out_tensor.dtype() == diopi_dtype_float16) || (out_tensor.dtype() == diopi_dtype_float32)),
                "out datatype not supported, only float16, float32 supported");

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlReciprocal(handle, input_desc.get(), input_tensor.data(), out_desc.get(), out_tensor.data()));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiReciprocal(ctx, input, input);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
