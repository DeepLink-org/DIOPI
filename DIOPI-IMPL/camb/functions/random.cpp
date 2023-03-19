/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#include <diopi/functions.h>
#include <float.h>

#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" DIOPI_API diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, int64_t idx) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    auto tensor = makeTensor(inout);
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
        set_last_error_string("%s%d", "cnnl random not support datatype: ", dtype);
        return diopiDtypeNotSupported;
    }
    DIOPI_CALLCNNL(cnnlRandDestroyGenerator(generator));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
