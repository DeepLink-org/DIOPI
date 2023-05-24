#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                    diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor1(tensor1);
    DiopiTensor otherTensor2(tensor2);
    DiopiTensor outTensor(out);

    std::vector<DiopiTensor*> pTensors{&inputTensor, &otherTensor1, &otherTensor2};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_float16};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor outTensorTemp = outTensor;
    if (outTensor.dtype() != inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensorTemp, inputTensor.dtype()));
    }

    CnnlTensorDesc inputTensorDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherTensor1Desc(otherTensor1, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherTensor2Desc(otherTensor2, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTensorDesc(outTensorTemp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAddcdivWorkspaceSize(handle, inputTensorDesc.get(), otherTensor1Desc.get(), otherTensor2Desc.get(), &workspaceSize));
    void* workspace = nullptr;
    float scalarValue;
    if (DiopiDataType::isInteger(value->stype)) {
        scalarValue = value->ival;
    } else {
        scalarValue = value->fval;
    }

    workspace = requiresBuffer(ctx, workspaceSize).data();
    DIOPI_CALLCNNL(cnnlAddcdiv(handle,
                               inputTensorDesc.get(),
                               inputTensor.data(),
                               &(scalarValue),
                               otherTensor1Desc.get(),
                               otherTensor1.data(),
                               otherTensor2Desc.get(),
                               otherTensor2.data(),
                               workspace,
                               workspaceSize,
                               outTensorDesc.get(),
                               outTensorTemp.data()))
    if (outTensorTemp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));
    }
    return diopiSuccess;
}
diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                       const diopiScalar_t* value) {
    diopiAddcdiv(ctx, input, input, tensor1, tensor2, value);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
