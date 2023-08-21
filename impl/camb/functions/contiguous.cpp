

#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiContiguous(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat) {
    DiopiTensor inputTensor(input);
    DIOPI_CALL(contiguous(ctx, inputTensor, memoryFormat));
    *out = inputTensor.tensorHandle();
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
