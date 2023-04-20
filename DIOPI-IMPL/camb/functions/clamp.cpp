#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {
extern "C" {

diopiError_t getClampBoundPtr(diopiContextHandle_t ctx, diopiConstTensorHandle_t bound, diopiDtype_t desire_dtype, void** out) {
    if (nullptr != bound) {
        DiopiTensor bound_tensor(bound);
        DIOPI_CHECK(bound_tensor.numel() == 1, "only supported when min and max are scalar or one element Tensor currently");
        if (desire_dtype != bound_tensor.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, bound_tensor, desire_dtype));
        }
        *out = bound_tensor.data();
        return diopiSuccess;
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

    DiopiTensor output32_tensor = output_tensor;
    if (DiopiDataType::isInteger(input_tensor.dtype())) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_int32));
        DIOPI_CALL(dataTypeCast(ctx, output32_tensor, diopi_dtype_int32));
    } else if (input_tensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, output32_tensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output32Desc(output32_tensor, CNNL_LAYOUT_ARRAY);

    void* min_ptr = nullptr;
    void* max_ptr = nullptr;
    DIOPI_CALL(getClampBoundPtr(ctx, min, input_tensor.dtype(), &min_ptr));
    DIOPI_CALL(getClampBoundPtr(ctx, max, input_tensor.dtype(), &max_ptr));

    DIOPI_CALLCNNL(
        cnnlClip_v2(handle, CNNL_POINTER_MODE_DEVICE, inputDesc.get(), input_tensor.data(), min_ptr, max_ptr, output32Desc.get(), output32_tensor.data()));
    if (output_tensor.dtype() != output32_tensor.dtype()) {
        if (output_tensor.dtype() != diopi_dtype_uint8) {
            DIOPI_CALL(dataTypeCast(ctx, output_tensor, output32_tensor));
        } else {
            DIOPI_CALL(dataTypeCast(ctx, output32_tensor, diopi_dtype_float32));
            DIOPI_CALL(dataTypeCast(ctx, output_tensor, output32_tensor));
        }
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
