/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cfloat>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to,
                                       diopiConstGeneratorHandle_t generator) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor tensor(inout);
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, tensor.dtype()));
    cnnlRandGenerator_t cnnl_generator;
    DIOPI_CALLCNNL(cnnlRandCreateGenerator(&cnnl_generator, CNNL_RAND_RNG_MTGP32));

    if (dtype == CNNL_DTYPE_FLOAT || dtype == CNNL_DTYPE_HALF) {
        float min = from;
        float max;
        if (to != nullptr) {
            max = *to - 1;
        } else {
            max = FLT_MAX;
        }
        diopiTensorHandle_t state_handle = nullptr;
        DIOPI_CALL(diopiGeneratorGetState(ctx, generator, &state_handle));
        void* state_ptr = nullptr;
        DIOPI_CALL(diopiGetTensorData(state_handle, &state_ptr));
        DIOPI_CALLCNNL(cnnlRandGenerateUniform(handle, cnnl_generator, dtype, state_ptr, tensor.numel(), min, max, tensor.data()));
        DIOPI_CALL(diopiGeneratorSetState(generator, state_handle));
    } else {
        setLastErrorString("%s%d", "cnnl random not support datatype: ", dtype);
        return diopiDtypeNotSupported;
    }
    DIOPI_CALLCNNL(cnnlRandDestroyGenerator(cnnl_generator));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
