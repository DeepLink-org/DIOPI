#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {
extern "C" {

diopiError_t getClampBoundPtr(diopiContextHandle_t ctx, diopiConstTensorHandle_t bound, diopiDtype_t desire_dtype, const char* bound_type, void** out) {
    if (nullptr != bound) {
        DiopiTensor bound_tensor(bound);
        DIOPI_CHECK(bound_tensor.numel() == 1, "only supported when %s is scalar or one element Tensor currently", bound_type);
        if (desire_dtype != bound_tensor.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, bound_tensor, desire_dtype));
        }
        *out = bound_tensor.data();
    }
    *out = nullptr;
    return diopiSuccess;
}

diopiError_t clampCommon(diopiContextHandle_t ctx, diopiConstTensorHandle_t input, diopiTensorHandle_t out, diopiConstTensorHandle_t min,
                         diopiConstTensorHandle_t max) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(out);
    DIOPI_CHECK(input_tensor.dtype() == output_tensor.dtype(), "the dtype of input and output must be the same")

    auto input32_tensor = input_tensor;
    auto output32_tensor = output_tensor;
    if (input_tensor.dtype() == diopi_dtype_int64 || input_tensor.dtype() == diopi_dtype_int16 || input_tensor.dtype() == diopi_dtype_int8) {
        DIOPI_CALL(dataTypeCast(ctx, input32_tensor, diopi_dtype_int32));
        DIOPI_CALL(dataTypeCast(ctx, input32_tensor, diopi_dtype_int32));
    }
    CnnlTensorDesc input32Desc(input32_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output32Desc(output32_tensor, CNNL_LAYOUT_ARRAY);

    void* min_ptr = nullptr;
    void* max_ptr = nullptr;
    getClampBoundPtr(ctx, min, input32_tensor.dtype(), "min", &min_ptr);
    getClampBoundPtr(ctx, max, input32_tensor.dtype(), "max", &max_ptr);

    DIOPI_CALLCNNL(
        cnnlClip_v2(handle, CNNL_POINTER_MODE_DEVICE, input32Desc.get(), input32_tensor.data(), min_ptr, max_ptr, output32Desc.get(), output32_tensor.data()));
    if (output_tensor.dtype() != output32_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, output_tensor, output32_tensor));
    }
    return diopiSuccess;
}

diopiError_t diopiClampInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min, const diopiScalar_t* max) {
    DiopiTensor min_tensor_tmp;
    DiopiTensor max_tensor_tmp;
    makeTensorFromScalar(ctx, min, min_tensor_tmp);
    makeTensorFromScalar(ctx, max, max_tensor_tmp);
    diopiTensorHandle_t min_tensor = min_tensor_tmp.tensorHandle();
    diopiTensorHandle_t max_tensor = max_tensor_tmp.tensorHandle();
    return clampCommon(ctx, input, input, min_tensor, max_tensor);
}

diopiError_t diopiClampInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min, diopiConstTensorHandle_t max) {
    return clampCommon(ctx, input, input, min, max);
}

diopiError_t diopiClampScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min,
                              const diopiScalar_t* max) {
    DiopiTensor min_tensor_tmp;
    DiopiTensor max_tensor_tmp;
    makeTensorFromScalar(ctx, min, min_tensor_tmp);
    makeTensorFromScalar(ctx, max, max_tensor_tmp);
    diopiTensorHandle_t min_tensor = min_tensor_tmp.tensorHandle();
    diopiTensorHandle_t max_tensor = max_tensor_tmp.tensorHandle();
    return clampCommon(ctx, input, out, min_tensor, max_tensor);
}

diopiError_t diopiClamp(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min,
                        diopiConstTensorHandle_t max) {
    return clampCommon(ctx, input, out, min, max);
}

diopiError_t diopiClampMaxInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* max) {
    DiopiTensor max_tensor_tmp;
    makeTensorFromScalar(ctx, max, max_tensor_tmp);
    diopiTensorHandle_t max_tensor = max_tensor_tmp.tensorHandle();
    return clampCommon(ctx, input, input, nullptr, max_tensor);
}

diopiError_t diopiClampMaxInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t max) {
    return clampCommon(ctx, input, input, nullptr, max);
}

diopiError_t diopiClampMaxScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* max) {
    DiopiTensor max_tensor_tmp;
    makeTensorFromScalar(ctx, max, max_tensor_tmp);
    diopiTensorHandle_t max_tensor = max_tensor_tmp.tensorHandle();
    return clampCommon(ctx, input, out, nullptr, max_tensor);
}

diopiError_t diopiClampMax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t max) {
    return clampCommon(ctx, input, out, nullptr, max);
}

diopiError_t diopiClampMinInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* min) {
    DiopiTensor min_tensor_tmp;
    makeTensorFromScalar(ctx, min, min_tensor_tmp);
    diopiTensorHandle_t min_tensor = min_tensor_tmp.tensorHandle();
    return clampCommon(ctx, input, input, min_tensor, nullptr);
}

diopiError_t diopiClampMinInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t min) {
    return clampCommon(ctx, input, input, min, nullptr);
}

diopiError_t diopiClampMinScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* min) {
    DiopiTensor min_tensor_tmp;
    makeTensorFromScalar(ctx, min, min_tensor_tmp);
    diopiTensorHandle_t min_tensor = min_tensor_tmp.tensorHandle();
    return clampCommon(ctx, input, out, min_tensor, nullptr);
}

diopiError_t diopiClampMin(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t min) {
    return clampCommon(ctx, input, out, min, nullptr);
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
