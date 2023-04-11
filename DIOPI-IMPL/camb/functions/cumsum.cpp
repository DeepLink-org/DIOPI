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

    std::vector<DiopiTensor*> pTensors{&input_tensor};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));
    DiopiTensor input_tensor_tmp = *pTensors[0];
    DiopiTensor out_tensor_tmp = out_tensor;
    DIOPI_CALL(dataTypeCast(ctx, out_tensor_tmp, input_tensor_tmp.dtype()));

    CnnlTensorDesc input_desc(input_tensor_tmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_tmp, CNNL_LAYOUT_ARRAY);

    int axis = getDim(input_tensor, dim);

    DIOPI_CALLCNNL(
        cnnlCumsum(handle, input_desc.get(), input_tensor_tmp.data(), axis, false, false, CNNL_PROPAGATE_NAN, out_desc.get(), out_tensor_tmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_tmp));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
