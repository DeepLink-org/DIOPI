#include <diopi/functions.h>
#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor exponent_tensor(exponent);
    DiopiTensor out_tensor(out);

    std::vector<DiopiTensor*> pTensors_in{&input_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors_in, supportedDtypes));
    DiopiTensor input_tensor_tmp = *pTensors_in[0];
    DiopiTensor out_tensor_tmp = out_tensor;
    DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, input_tensor_tmp.dtype()));

    CnnlTensorDesc input_desc(input_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

    std::vector<DiopiTensor*> pTensors_exp{&exponent_tensor};
    if (input_tensor.dtype() == diopi_dtype_float16) {
        DIOPI_CALL(autoCastTensorType(ctx, pTensors_exp, {diopi_dtype_float16, diopi_dtype_int16}));
    } else if (input_tensor.dtype() == diopi_dtype_float32) {
        DIOPI_CALL(autoCastTensorType(ctx, pTensors_exp, {diopi_dtype_float32, diopi_dtype_int16}));
    } else {
        DIOPI_CHECK(false, "input datatype not supported, only float16, float32 supported");
    }

    DiopiTensor exponent_tensor_tmp = *pTensors_exp[0];
    CnnlTensorDesc exponent_desc(exponent_tensor_tmp, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetPowWorkspaceSize(handle, input_desc.get(), exponent_desc.get(), out_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlPow(handle,
                            CNNL_COMPUTATION_HIGH_PRECISION,
                            input_desc.get(),
                            input_tensor_tmp.data(),
                            exponent_desc.get(),
                            exponent_tensor_tmp.data(),
                            workspace,
                            workspace_size,
                            out_desc.get(),
                            out_tensor_tmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    DIOPI_CALL(diopiPowTensor(ctx, input, input, exponent));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    DiopiTensor exponent_tensor;
    makeTensorFromScalar(ctx, exponent, exponent_tensor);
    DIOPI_CALL(diopiPowTensor(ctx, out, input, static_cast<diopiTensorHandle_t>(exponent_tensor)));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {
    DIOPI_CALL(diopiPow(ctx, input, input, exponent));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    DiopiTensor input_tensor;
    makeTensorFromScalar(ctx, input, input_tensor);
    DIOPI_CALL(diopiPowTensor(ctx, out, static_cast<diopiTensorHandle_t>(input_tensor), exponent));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
