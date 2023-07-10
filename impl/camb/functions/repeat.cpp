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

diopiError_t diopiRepeat(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t repeatsSize) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlTile(handle, inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data()));
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
