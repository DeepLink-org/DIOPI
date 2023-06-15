/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor tensor(inout);
    if (!tensor.defined()) {
        return diopiSuccess;
    }
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, tensor.dtype()));
    cnnlRandGenerator_t generator;
    DIOPI_CALLCNNL(cnnlRandCreateGenerator(&generator, CNNL_RAND_RNG_FAST));

    DIOPI_CALLCNNL(cnnlRandGenerateUniform(handle, generator, dtype, nullptr, tensor.numel(), from, to, tensor.data()));

    DIOPI_CALLCNNL(cnnlRandDestroyGenerator(generator));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
