#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {
diopiError_t diopiFlip(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dims) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    if (diopi_dtype_float64 == input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_float32));
    } else if (diopi_dtype_int64 == input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_int32));
    }
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);

    DiopiTensor out_tensor(out);
    CnnlTensorDesc outDesc(out_tensor, CNNL_LAYOUT_ARRAY);

    std::vector<int32_t> dimension(dims.len);
    for (int i = 0; i < dims.len; i++) {
        dimension[i] = dims.data[i];
    }

    if (out_tensor.dtype() == input_tensor.dtype()) {
        DIOPI_CALLCNNL(cnnlFlip(handle, dimension.data(), dims.len, inputDesc.get(), input_tensor.data(), outDesc.get(), out_tensor.data()));
    } else {
        DiopiTensor out_temp = requiresTensor(ctx, out_tensor.shape(), input_tensor.dtype());
        CnnlTensorDesc out_tempDesc(out_temp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlFlip(handle, dimension.data(), dims.len, inputDesc.get(), input_tensor.data(), out_tempDesc.get(), out_temp.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_temp));
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
