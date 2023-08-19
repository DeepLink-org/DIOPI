/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <random>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiConstGeneratorHandle_t generator) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor tensor(out);
    if (!(tensor.defined() && tensor.numel())) {
        return diopiSuccess;
    }

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, tensor.dtype()));
    cnnlRandGenerator_t cnnl_generator;
    // cnnlRandRngType_t rng_type is recommended to be set as CNNL_RAND_RNG_MTGP32 on MLU300 series and CNNL_RAND_RNG_FAST on MLU200 series.
    DIOPI_CALLCNNL(cnnlRandCreateGenerator(&cnnl_generator, CNNL_RAND_RNG_MTGP32));

    diopiTensorHandle_t state_handle = nullptr;
    DIOPI_CALL(diopiGeneratorGetState(ctx, generator, &state_handle));
    void* state_ptr = nullptr;
    DIOPI_CALL(diopiGetTensorData(state_handle, &state_ptr));
    DIOPI_CALLCNNL(cnnlRandGenerateNormal(handle, cnnl_generator, dtype, state_ptr, tensor.numel(), mean, std, tensor.data()));
    DIOPI_CALL(diopiGeneratorSetState(generator, state_handle));
    DIOPI_CALLCNNL(cnnlRandDestroyGenerator(cnnl_generator));
    return diopiSuccess;
}

diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiConstGeneratorHandle_t generator) {
    DIOPI_CALL(diopiNormal(ctx, inout, mean, std, generator));
    return diopiSuccess;
}
}

}  // namespace camb
}  // namespace impl
