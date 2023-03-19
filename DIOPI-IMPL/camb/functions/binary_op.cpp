/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#include <cnrt.h>
#include <diopi/functions.h>

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" DIOPI_API diopiError_t
diopiAdd(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiTensorHandle_t input_ = diopiTensorHandle_t(input);
    diopiTensorHandle_t other_ = diopiTensorHandle_t(other);
    auto trInput = makeTensor(input_);
    auto trOther = makeTensor(other_);
    auto trOut = makeTensor(out);
    std::vector<DiopiTensorT*> pTensors{&trInput, &trOther};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32};

    autoCastTensorType(ctx, pTensors, supportedDtypes);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput, layout);
    CnnlTensorDesc descOther(trOther, layout);
    CnnlTensorDesc descOut(trOut, layout);
    DiopiTensorT trOutTmp;
    CnnlTensorDesc descOutTmp;
    if (trInput.dtype() == trOut.dtype()) {
        trOutTmp = trOut;
        descOutTmp = descOut;
    } else {
        trOutTmp = requiresTensor(ctx, vec2diopiSize_t(trOut.shape()), trInput.dtype());
        descOutTmp.set(trOutTmp, CNNL_LAYOUT_ARRAY);
    }

    std::unique_ptr<void, void (*)(void*)> pAlphaIn(malloc(4), free);
    std::unique_ptr<void, void (*)(void*)> pBetaIn(malloc(4), free);
    if (CnnlDataType::isInteger(dtype)) {
        *reinterpret_cast<int32_t*>(pBetaIn.get()) = 0;
        if (alpha->stype <= 7) {
            *reinterpret_cast<int32_t*>(pAlphaIn.get()) = static_cast<int32_t>(alpha->ival);
        } else {
            *reinterpret_cast<int32_t*>(pAlphaIn.get()) = static_cast<int32_t>(static_cast<float>(alpha->fval));
        }
    } else {
        *reinterpret_cast<float*>(pBetaIn.get()) = 0.0f;
        if (alpha->stype <= 7) {
            *reinterpret_cast<float*>(pAlphaIn.get()) = static_cast<float>(static_cast<int32_t>(alpha->ival));
        } else {
            *reinterpret_cast<float*>(pAlphaIn.get()) = static_cast<float>(alpha->fval);
        }
    }
    DIOPI_CALLCNNL(
        cnnlTransform_v2(handle, CNNL_POINTER_MODE_HOST, pAlphaIn.get(), descOther.get(), trOther.data(), pBetaIn.get(), descOther.get(), trOther.data()));
    const cnnlTensorDescriptor_t inputDescs[2] = {descInput.get(), descOther.get()};
    const void* inputs[2] = {trInput.data(), trOther.data()};
    uint32_t inputNum = 2;
    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetAddNWorkspaceSize(handle, inputDescs, inputNum, descOutTmp.get(), &workspaceSize));
    auto buff = requiresBuffer(ctx, workspaceSize);
    void* pWorkspace = buff.data();

    DIOPI_CALLCNNL(cnnlAddN_v2(handle, inputDescs, inputs, inputNum, descOutTmp.get(), trOutTmp.data(), pWorkspace, workspaceSize));
    if (trOutTmp.dtype() != trOut.dtype()) {
        dataTypeCast(ctx, trOut, trOutTmp);
    }
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiAddInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t other, const diopiScalar_t* alpha) {
    diopiAdd(ctx, input, input, other, alpha);
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t
diopiAddScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* other, const diopiScalar_t* alpha) {
    DiopiTensorT trOther = makeTensorFromScalar(ctx, other);
    DIOPI_CALL(diopiAdd(ctx, out, input, trOther, alpha));
    return diopiSuccess;
}

extern "C" DIOPI_API diopiError_t diopiAddInpScalar(diopiContextHandle_t ctx,
                                                    diopiTensorHandle_t input,
                                                    const diopiScalar_t* other,
                                                    const diopiScalar_t* alpha) {
    diopiAddScalar(ctx, input, input, other, alpha);
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
