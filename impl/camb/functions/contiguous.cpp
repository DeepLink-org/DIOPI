
#include <diopi/functions.h>

#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiContiguous(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input, diopiMemoryFormat_t memoryFormat) {
    DiopiTensor inputTensor(input);
    // DIOPI_CALL(dataTypeCast(ctx, inputTensor, dtype));
    //  std::cout << "11 :" << inputTensor.tensorHandle() << std::endl;
    DIOPI_CALL(contiguous(ctx, inputTensor, memoryFormat));
    // std::cout << "22 :" << inputTensor.tensorHandle() << std::endl;
    *out = inputTensor.tensorHandle();
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
