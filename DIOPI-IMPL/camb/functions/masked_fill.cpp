#include <string.h>
#include <iostream>
#include <memory>
#include <numeric>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                       diopiConstTensorHandle_t value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto input_tensor = DiopiTensor(input);
    auto mask_tensor = DiopiTensor(mask);
    auto value_tensor = DiopiTensor(value);
    auto out_tensor = DiopiTensor(out);

    std::vector<DiopiTensor*> pTensors{&input_tensor, &value_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_bool};

    std::vector<DiopiTensor*> MTensors{&mask_tensor};
    std::set<diopiDtype_t> supportedDtypes_mask{diopi_dtype_int8, diopi_dtype_uint8, diopi_dtype_bool};

    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DIOPI_CALL(autoCastTensorType(ctx, MTensors, supportedDtypes_mask));

    DiopiTensor input_tensor_tmp = *pTensors[0];
    DiopiTensor value_tensor_tmp = *pTensors[1];
    DiopiTensor mask_tensor_tmp = *MTensors[0];
    DiopiTensor out_tensor_tmp = out_tensor;
    DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, input_tensor_tmp.dtype()));

    CnnlTensorDesc input_desc(input_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mask_desc(mask_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

    CnnlTensorDesc value_desc;
    if (value_tensor_tmp.shape().size() > 0) {
        DIOPI_CALL(value_desc.set(value_tensor_tmp, CNNL_LAYOUT_ARRAY));
    } else {
        std::vector<int> value_dims = {1};
        DIOPI_CALL(value_desc.set(value_tensor_tmp, CNNL_LAYOUT_ARRAY, value_dims));
    }

    DiopiTensor value_cast_tensor;
    CnnlTensorDesc value_cast_desc;

    bool value_cast = false;
    if (input_tensor_tmp.dtype() != value_tensor_tmp.dtype()) {
        value_cast = true;
        value_cast_tensor = value_tensor_tmp;
        DIOPI_CALL(dataTypeCast(ctx, value_tensor, input_tensor_tmp.dtype()));
        value_cast_desc.set(value_cast_tensor, CNNL_LAYOUT_ARRAY);
    }

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(
        handle, CNNL_MASKED_FILL, input_desc.get(), mask_desc.get(), value_cast ? value_cast_desc.get() : value_desc.get(), out_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlMasked_v3(handle,
                                 CNNL_MASKED_FILL,
                                 input_desc.get(),
                                 input_tensor_tmp.data(),
                                 mask_desc.get(),
                                 mask_tensor_tmp.data(),
                                 value_cast ? value_cast_desc.get() : value_desc.get(),
                                 value_cast ? value_cast_tensor.data() : value_tensor_tmp.data(),
                                 workspace,
                                 workspace_size,
                                 out_desc.get(),
                                 out_tensor_tmp.data(),
                                 nullptr));

    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    DIOPI_CALL(diopiMaskedFill(ctx, input, input, mask, value));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                             const diopiScalar_t* value) {
    DiopiTensor value_tensor;
    makeTensorFromScalar(ctx, value, value_tensor);
    DIOPI_CALL(diopiMaskedFill(ctx, out, input, mask, static_cast<diopiTensorHandle_t>(value_tensor)));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask,
                                                const diopiScalar_t* value) {
    DiopiTensor value_tensor;
    makeTensorFromScalar(ctx, value, value_tensor);
    DIOPI_CALL(diopiMaskedFill(ctx, input, input, mask, static_cast<diopiTensorHandle_t>(value_tensor)));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
