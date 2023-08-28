

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiPowTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor exponentTensor(exponent);
    DiopiTensor outTensor(out);

    std::vector<DiopiTensor*> pTensorsIn{&inputTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensorsIn, supportedDtypes));
    DiopiTensor inputTensorTmp = *pTensorsIn[0];
    DiopiTensor outTensorTmp = outTensor;
    DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, inputTensorTmp.dtype()));

    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);

    std::vector<DiopiTensor*> pTensorsExp{&exponentTensor};
    if (inputTensor.dtype() == diopi_dtype_float16) {
        DIOPI_CALL(autoCastTensorType(ctx, pTensorsExp, {diopi_dtype_float16, diopi_dtype_int16}));
    } else if (inputTensor.dtype() == diopi_dtype_float32) {
        DIOPI_CALL(autoCastTensorType(ctx, pTensorsExp, {diopi_dtype_float32, diopi_dtype_int16}));
    } else {
        DIOPI_CHECK(false, "input datatype not supported, only float16, float32 supported");
    }

    DiopiTensor exponentTensorTmp = *pTensorsExp[0];
    CnnlTensorDesc exponentDesc(exponentTensorTmp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetPowWorkspaceSize(handle, inputDesc.get(), exponentDesc.get(), outDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlPow(handle,
                           CNNL_COMPUTATION_HIGH_PRECISION,
                           inputDesc.get(),
                           inputTensorTmp.data(),
                           exponentDesc.get(),
                           exponentTensorTmp.data(),
                           workspace,
                           workspaceSize,
                           outDesc.get(),
                           outTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    return diopiSuccess;
}

diopiError_t diopiPowInpTensor(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t exponent) {
    DIOPI_CALL(diopiPowTensor(ctx, input, input, exponent));
    return diopiSuccess;
}

diopiError_t diopiPow(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* exponent) {
    DiopiTensor exponentTensor;
    makeTensorFromScalar(ctx, exponent, exponentTensor);
    DIOPI_CALL(diopiPowTensor(ctx, out, input, static_cast<diopiTensorHandle_t>(exponentTensor)));
    return diopiSuccess;
}

diopiError_t diopiPowInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, const diopiScalar_t* exponent) {
    DIOPI_CALL(diopiPow(ctx, input, input, exponent));
    return diopiSuccess;
}

diopiError_t diopiPowScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t* input, diopiConstTensorHandle_t exponent) {
    DiopiTensor inputTensor;
    makeTensorFromScalar(ctx, input, inputTensor);
    DIOPI_CALL(diopiPowTensor(ctx, out, static_cast<diopiTensorHandle_t>(inputTensor), exponent));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
