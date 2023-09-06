/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim,
                       bool descending, const bool* stable) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto inputTensor = DiopiTensor(input);
    auto indicesTensor = DiopiTensor(indices);
    auto valuesTensor = DiopiTensor(values);
    DiopiTensor valuesTensorTemp = valuesTensor;

    // since input can be changed by cnnlTopKTensor_v3 when input_shape is (24180),
    // need to requires a temp Tensor to bypass this bug
    DiopiTensor inputTensorTemp = requiresTensor(ctx, inputTensor.shape(), inputTensor.dtype());
    DIOPI_CALL(diopiCopyInp(ctx, input, diopiTensorHandle_t(inputTensorTemp)));

    if (inputTensorTemp.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensorTemp, diopi_dtype_float32));
        DIOPI_CALL(dataTypeCast(ctx, valuesTensorTemp, diopi_dtype_float32));
    } else if (inputTensorTemp.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensorTemp, diopi_dtype_int32));
        DIOPI_CALL(dataTypeCast(ctx, valuesTensorTemp, diopi_dtype_int32));
    }

    DiopiTensor indicesTensorTemp = indicesTensor;
    DIOPI_CALL(dataTypeCast(ctx, indicesTensorTemp, diopi_dtype_int32));
    CnnlTensorDesc inputDesc(inputTensorTemp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc valuesDesc(valuesTensorTemp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indicesDesc(indicesTensorTemp, CNNL_LAYOUT_ARRAY);

    uint64_t k;
    std::vector<int64_t> inputShape = inputTensorTemp.shape();

    if (dim < 0) {
        dim += inputShape.size();
    }
    k = inputShape[dim];

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetTopKTensorWorkspaceSize(handle, inputDesc.get(), k, dim, descending, valuesDesc.get(), indicesDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(cnnlTopKTensor_v3(handle,
                                     inputDesc.get(),
                                     inputTensorTemp.data(),
                                     k,
                                     dim,
                                     descending,
                                     true,
                                     stable,
                                     workspace,
                                     workspaceSize,
                                     valuesDesc.get(),
                                     valuesTensorTemp.data(),
                                     indicesDesc.get(),
                                     indicesTensorTemp.data()));
    if (valuesTensorTemp.dtype() != valuesTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, valuesTensor, valuesTensorTemp));
    }

    if (indicesTensorTemp.dtype() != indicesTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, indicesTensor, indicesTensorTemp));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
