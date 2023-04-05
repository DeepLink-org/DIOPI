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

extern "C" DIOPI_API diopiError_t diopiExpand(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    auto trInput = DiopiTensor(input);
    auto trOut = DiopiTensor(out);

    diopiSize_t size;
    diopiGetTensorShape(out, &size);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput, layout);
    CnnlTensorDesc descOut(trOut, layout);
    DiopiTensor trOutTmp;
    if (trInput.dtype() == trOut.dtype()) {
        trOutTmp = trOut;
        descOut.set(trOut, layout);
    } else {
        trOutTmp = requiresTensor(ctx, vec2diopiSize_t(trOut.shape()), trInput.dtype());
        descOut.set(trOutTmp, CNNL_LAYOUT_ARRAY);
    }

    DIOPI_CALLCNNL(cnnlExpand(handle, descInput.get(), trInput.data(), descOut.get(), trOutTmp.data()));
    if (trOutTmp.dtype() != trOut.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, trOut, trOutTmp));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
