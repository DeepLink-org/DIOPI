#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
namespace {

int getDim(DiopiTensor tensor, int64_t dim) {
    int shape_size = tensor.shape().size();
    int dim_ = static_cast<int>(dim);
    if (dim_ < 0) {
        dim_ = dim_ + shape_size;
    }
    return dim_;
}

}  // namespace

extern "C" {

DIOPI_API diopiError_t diopiCumsum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);
    DIOPI_CALL(autoCastTensorType(ctx, {&input_tensor}, {diopi_dtype_int32, diopi_dtype_float32, diopi_dtype_float16}));

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor, CNNL_LAYOUT_ARRAY);
    int axis = getDim(input_tensor, dim);

    if (input_tensor.dtype() == out_tensor.dtype()) {
        DIOPI_CALLCNNL(cnnlCumsum(handle, input_desc.get(), input_tensor.data(), axis, false, false, CNNL_PROPAGATE_NAN, out_desc.get(), out_tensor.data()));
    } else {
        DiopiTensor out_temp = requiresTensor(ctx, out_tensor.shape(), input_tensor.dtype());
        CnnlTensorDesc out_temp_desc(out_temp, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlCumsum(handle, input_desc.get(), input_tensor.data(), axis, false, false, CNNL_PROPAGATE_NAN, out_temp_desc.get(), out_temp.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_temp));
    }

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
