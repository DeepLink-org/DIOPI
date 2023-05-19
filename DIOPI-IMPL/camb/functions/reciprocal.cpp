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

diopiError_t diopiReciprocal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);

    DIOPI_CHECK(((inputTensor.dtype() == diopi_dtype_float16) || (inputTensor.dtype() == diopi_dtype_float32)),
                "input datatype not supported, only float16, float32 supported");
    DIOPI_CHECK(((outTensor.dtype() == diopi_dtype_float16) || (outTensor.dtype() == diopi_dtype_float32)),
                "out datatype not supported, only float16, float32 supported");

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlReciprocal(handle, inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data()));
    return diopiSuccess;
}

diopiError_t diopiReciprocalInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiReciprocal(ctx, input, input);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
