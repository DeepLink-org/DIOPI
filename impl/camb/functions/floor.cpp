/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
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

extern "C" diopiError_t diopiFloor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    DiopiTensor trInput(input);
    DiopiTensor trOut(out);
    std::vector<DiopiTensor*> pTensors{&trInput};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};

    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput, layout);
    CnnlTensorDesc descOut;
    DiopiTensor trOutTmp;
    if (trInput.dtype() == trOut.dtype()) {
        trOutTmp = trOut;
        descOut.set(trOut, layout);
    } else {
        trOutTmp = requiresTensor(ctx, vec2diopiSizeT(trOut.shape()), trInput.dtype());
        descOut.set(trOutTmp, CNNL_LAYOUT_ARRAY);
    }

    DIOPI_CALLCNNL(cnnlFloor(handle,
                             descInput.get(),

                             trInput.data(),
                             descOut.get(),
                             trOutTmp.data()));
    if (trOutTmp.dtype() != trOut.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, trOut, trOutTmp));
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiFloorInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    diopiFloor(ctx, input, input);
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
