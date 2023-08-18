

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    if (diopi_dtype_float64 == inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    } else if (diopi_dtype_int64 == inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_int32));
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);

    DiopiTensor outTensor(out);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    std::vector<int32_t> dimension(dims.len);
    for (int i = 0; i < dims.len; i++) {
        dimension[i] = dims.data[i];
    }

    if (outTensor.dtype() == inputTensor.dtype()) {
        DIOPI_CALLCNNL(cnnlFlip(handle, dimension.data(), dims.len, inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data()));
    } else {
        DiopiTensor outTemp = requiresTensor(ctx, outTensor.shape(), inputTensor.dtype());
        CnnlTensorDesc outTempDesc(outTemp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlFlip(handle, dimension.data(), dims.len, inputDesc.get(), inputTensor.data(), outTempDesc.get(), outTemp.data()));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTemp));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
