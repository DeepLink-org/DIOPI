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

extern "C" diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to,
                                       diopiConstGeneratorHandle_t generator) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor tensor(inout);
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, tensor.dtype()));
    cnnlRandGenerator_t cnnlGenerator;
    DIOPI_CALLCNNL(cnnlRandCreateGenerator(&cnnlGenerator, CNNL_RAND_RNG_MTGP32));

    if (dtype == CNNL_DTYPE_FLOAT || dtype == CNNL_DTYPE_HALF) {
        float min = from;
        float max;
        if (to != nullptr) {
            max = *to - 1;
        } else {
            max = FLT_MAX;
        }
        diopiTensorHandle_t stateHandle = nullptr;
        DIOPI_CALL(diopiGeneratorGetState(ctx, generator, &stateHandle));
        void* statePtr = nullptr;
        DIOPI_CALL(diopiGetTensorData(stateHandle, &statePtr));
        DIOPI_CALLCNNL(cnnlRandGenerateUniform(handle, cnnlGenerator, dtype, statePtr, tensor.numel(), min, max, tensor.data()));
        DIOPI_CALL(diopiGeneratorSetState(generator, stateHandle));
    } else {
        setLastErrorString("%s%d", "cnnl random not support datatype: ", dtype);
        return diopiDtypeNotSupported;
    }
    DIOPI_CALLCNNL(cnnlRandDestroyGenerator(cnnlGenerator));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
