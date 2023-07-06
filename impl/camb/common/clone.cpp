/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "common.hpp"

namespace impl {
namespace camb {

diopiError_t clone(diopiContextHandle_t ctx, const DiopiTensor& inTensor, DiopiTensor& outTensor, MemoryFormat memoryFormat) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    if (memoryFormat == MemoryFormat::Preserve) {
        outTensor = requiresTensor(ctx, inTensor.shape(), inTensor.stride(), inTensor.dtype());
    } else {
        outTensor = requiresTensor(ctx, inTensor.shape(), inTensor.dtype(), memoryFormat);
    }
    CnnlTensorDesc inTensorDesc(inTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outTensorDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlCopy(handle, inTensorDesc.get(), inTensor.data(), outTensorDesc.get(), outTensor.data()));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
