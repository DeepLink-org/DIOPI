/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cfloat>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor tensor(inout);
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, tensor.dtype()));
    cnnlRandGenerator_t generator;
    DIOPI_CALLCNNL(cnnlRandCreateGenerator(&generator, CNNL_RAND_RNG_FAST));

    if (dtype == CNNL_DTYPE_FLOAT || dtype == CNNL_DTYPE_HALF) {
        float min = from;
        float max;
        if (to != nullptr) {
            max = *to - 1;
        } else {
            max = FLT_MAX;
        }
        DIOPI_CALLCNNL(cnnlRandGenerateUniform(handle, generator, dtype, nullptr, tensor.numel(), min, max, tensor.data()));
    } else {
        setLastErrorString("%s%d", "cnnl random not support datatype: ", dtype);
        return diopiDtypeNotSupported;
    }
    DIOPI_CALLCNNL(cnnlRandDestroyGenerator(generator));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
