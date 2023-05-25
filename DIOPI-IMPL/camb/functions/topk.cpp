/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

diopiError_t diopiTopk(diopiContextHandle_t ctx,
                                 diopiTensorHandle_t values,
                                 diopiTensorHandle_t indices,
                                 diopiConstTensorHandle_t input,
                                 int64_t k,
                                 int64_t dim,
                                 bool largest,
                                 bool sorted) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor indicesTensor(indices);
    DiopiTensor valuesTensor(values);

    DiopiTensor valuesTensorTemp = valuesTensor;
    DiopiTensor inputTensorTemp = inputTensor;
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensorTemp, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, valuesTensorTemp, diopi_dtype_float32));
    } else if (inputTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensorTemp, diopi_dtype_int32));
        DIOPI_CALL(dataTypeCast(ctx, valuesTensorTemp, diopi_dtype_int32));
    } else {
        inputTensorTemp = DiopiTensor(input);
        valuesTensorTemp = DiopiTensor(values);
    }

    DiopiTensor indicesTensorTemp = indicesTensor;
    DIOPI_CALL(dataTypeCast(ctx, indicesTensorTemp, diopi_dtype_int32));
    CnnlTensorDesc inputDesc(inputTensorTemp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc valuesDesc(valuesTensorTemp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indicesDesc(indicesTensorTemp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetTopKTensorWorkspaceSize(handle, inputDesc.get(), k, dim, largest, valuesDesc.get(), indicesDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    const bool lowerIndexFirst = true;
    DIOPI_CALLCNNL(cnnlTopKTensor_v3(handle,
                                     inputDesc.get(),
                                     inputTensorTemp.data(),
                                     k,
                                     dim,
                                     largest,
                                     sorted,
                                     lowerIndexFirst,
                                     workspace,
                                     workspaceSize,
                                     valuesDesc.get(),
                                     valuesTensorTemp.data(),
                                     indicesDesc.get(),
                                     indicesTensorTemp.data()))
    if (valuesTensorTemp.dtype() != valuesTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, valuesTensor, valuesTensorTemp));
    }

    if (indicesTensorTemp.dtype() != indicesTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, indicesTensor, indicesTensorTemp));
    }

    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
