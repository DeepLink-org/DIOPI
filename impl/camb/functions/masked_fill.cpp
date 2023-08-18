#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiMaskedFill(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                             diopiConstTensorHandle_t value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor maskTensor(mask);
    DiopiTensor valueTensor(value);
    DiopiTensor outTensor(out);

    std::vector<DiopiTensor*> pTensors{&inputTensor, &valueTensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_bool};

    std::vector<DiopiTensor*> mTensors{&maskTensor};
    std::set<diopiDtype_t> supportedDtypesMask{diopi_dtype_int8, diopi_dtype_uint8, diopi_dtype_bool};

    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DIOPI_CALL(autoCastTensorType(ctx, mTensors, supportedDtypesMask));

    DiopiTensor inputTensorTmp = *pTensors[0];
    DiopiTensor valueTensorTmp = *pTensors[1];
    DiopiTensor maskTensorTmp = *mTensors[0];
    DiopiTensor outTensorTmp = outTensor;
    DIOPI_CALL(dataTypeCast(ctx, outTensorTmp, inputTensorTmp.dtype()));

    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc maskDesc(maskTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);

    CnnlTensorDesc valueDesc;
    if (!valueTensorTmp.shape().empty()) {
        DIOPI_CALL(valueDesc.set(valueTensorTmp, CNNL_LAYOUT_ARRAY));
    } else {
        std::vector<int> valueDims = {1};
        DIOPI_CALL(valueDesc.set(valueTensorTmp, CNNL_LAYOUT_ARRAY, valueDims));
    }

    DiopiTensor valueCastTensor;
    CnnlTensorDesc valueCastDesc;

    bool valueCast = false;
    if (inputTensorTmp.dtype() != valueTensorTmp.dtype()) {
        valueCast = true;
        valueCastTensor = valueTensorTmp;
        DIOPI_CALL(dataTypeCast(ctx, valueTensor, inputTensorTmp.dtype()));
        valueCastDesc.set(valueCastTensor, CNNL_LAYOUT_ARRAY);
    }

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(
        handle, CNNL_MASKED_FILL, inputDesc.get(), maskDesc.get(), valueCast ? valueCastDesc.get() : valueDesc.get(), outDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlMasked_v3(handle,
                                 CNNL_MASKED_FILL,
                                 inputDesc.get(),
                                 inputTensorTmp.data(),
                                 maskDesc.get(),
                                 maskTensorTmp.data(),
                                 valueCast ? valueCastDesc.get() : valueDesc.get(),
                                 valueCast ? valueCastTensor.data() : valueTensorTmp.data(),
                                 workspace,
                                 workspaceSize,
                                 outDesc.get(),
                                 outTensorTmp.data(),
                                 nullptr));

    DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    return diopiSuccess;
}

diopiError_t diopiMaskedFillInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, diopiConstTensorHandle_t value) {
    DIOPI_CALL(diopiMaskedFill(ctx, input, input, mask, value));
    return diopiSuccess;
}

diopiError_t diopiMaskedFillScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                   const diopiScalar_t* value) {
    DiopiTensor valueTensor;
    makeTensorFromScalar(ctx, value, valueTensor);
    DIOPI_CALL(diopiMaskedFill(ctx, out, input, mask, static_cast<diopiTensorHandle_t>(valueTensor)));
    return diopiSuccess;
}

diopiError_t diopiMaskedFillInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t mask, const diopiScalar_t* value) {
    DiopiTensor valueTensor;
    makeTensorFromScalar(ctx, value, valueTensor);
    DIOPI_CALL(diopiMaskedFill(ctx, input, input, mask, static_cast<diopiTensorHandle_t>(valueTensor)));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
