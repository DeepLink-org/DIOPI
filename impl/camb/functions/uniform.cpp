/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <random>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, diopiConstGeneratorHandle_t generator) {
    DIOPI_CHECK_NULLPTR_ABORT(generator);
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor tensor(inout);
    if (!(tensor.defined() && tensor.numel())) {
        return diopiSuccess;
    }

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, tensor.dtype()));
    cnnlRandGenerator_t cnnl_generator = nullptr;
    // cnnlRandRngType_t rng_type is recommended to be set as CNNL_RAND_RNG_MTGP32 on MLU300 series and CNNL_RAND_RNG_FAST on MLU200 series.
    DIOPI_CALLCNNL(cnnlRandCreateGenerator(&cnnl_generator, CNNL_RAND_RNG_MTGP32));

    diopiTensorHandle_t state_handle = nullptr;
    DIOPI_CALL(diopiGeneratorGetState(ctx, generator, &state_handle));
    void* state_ptr = nullptr;
    DIOPI_CALL(diopiGetTensorData(state_handle, &state_ptr));
    DIOPI_CALLCNNL(cnnlRandGenerateUniform(handle, cnnl_generator, dtype, state_ptr, tensor.numel(), from, to, tensor.data()));
    DIOPI_CALL(diopiGeneratorSetState(generator, state_handle));
    DIOPI_CALLCNNL(cnnlRandDestroyGenerator(cnnl_generator));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
