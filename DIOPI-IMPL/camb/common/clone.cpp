/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "common.hpp"

namespace impl {
namespace camb {

diopiError_t clone(diopiContextHandle_t ctx, const DiopiTensor& inTensor, DiopiTensor& outTensor) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    outTensor = requiresTensor(ctx, inTensor.shape(), inTensor.dtype());
    CnnlTensorDesc inTensorDesc(inTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTensorDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlCopy(handle, inTensorDesc.get(), inTensor.data(), outTensorDesc.get(), outTensor.data()));
}

}  // namespace camb
}  // namespace impl
