/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <random>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiGeneratorHandle_t generator) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor tensor(out);
    if (!(tensor.defined() && tensor.numel())) {
        return diopiSuccess;
    }

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, tensor.dtype()));
    cnnlRandGenerator_t cnnlGenerator;
    // cnnlRandRngType_t rng_type is recommended to be set as CNNL_RAND_RNG_MTGP32 on MLU300 series and CNNL_RAND_RNG_FAST on MLU200 series.
    DIOPI_CALL_CNNL(cnnlRandCreateGenerator(&cnnlGenerator, CNNL_RAND_RNG_MTGP32));

    diopiTensorHandle_t stateHandle = nullptr;
    DIOPI_CALL(diopiGeneratorGetState(ctx, generator, &stateHandle));
    void* statePtr = nullptr;
    DIOPI_CALL(diopiGetTensorData(stateHandle, &statePtr));
    DIOPI_CALL_CNNL(cnnlRandGenerateNormal(handle, cnnlGenerator, dtype, statePtr, tensor.numel(), mean, std, tensor.data()));
    DIOPI_CALL(diopiGeneratorSetState(generator, stateHandle));
    DIOPI_CALL_CNNL(cnnlRandDestroyGenerator(cnnlGenerator));
    return diopiSuccess;
}

diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
    DIOPI_CALL(diopiNormal(ctx, inout, mean, std, generator));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
