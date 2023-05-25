#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {
DIOPI_API diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor outTensor(out);
    auto type = CNNL_DTYPE_FLOAT;

    // create and set the rand_generator
    cnnlRandGenerator_t generator;
    // MTGP32 algorithm performs better on MLU300 series than MLU200 series
    DIOPI_CALLCNNL(cnnlRandCreateGenerator(&generator, CNNL_RAND_RNG_MTGP32));
    // set the period to the generator
    DIOPI_CALLCNNL(cnnlRandSetMTGP32Period(generator, CNNL_RAND_MTGP32_P11213));
    // create and set the state
    size_t sizeState = 0;
    DIOPI_CALLCNNL(cnnlRandGetMTGP32StateSize(generator, &sizeState));
    void* state = nullptr;
    state = requiresBuffer(ctx, sizeState).data();
    cnnlMTGP32FastParams_t params;
    DIOPI_CALLCNNL(cnnlRandGetMTGP32HostParam(generator, &params));
    size_t sizeKernel = 0;
    DIOPI_CALLCNNL(cnnlRandGetMTGP32KernelParamSize(generator, &sizeKernel));
    void* kernelParams = nullptr;
    kernelParams = requiresBuffer(ctx, sizeKernel).data();
    DIOPI_CALLCNNL(cnnlRandMakeMTGP32Constants(handle, params, kernelParams));
    int randSeed = time(nullptr);
    DIOPI_CALLCNNL(cnnlRandMakeMTGP32KernelState(handle, state, params, kernelParams, randSeed));

    DIOPI_CALLCNNL(cnnlRandGenerateNormal(handle, generator, type, state, outTensor.numel(), mean, std, outTensor.data()));
    return diopiSuccess;
}
diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std) {
    DIOPI_CALL(diopiNormal(ctx, inout, mean, std));
    return diopiSuccess;
}

// DIOPI_API diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std);
// DIOPI_API diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std);
// DIOPI_API diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std);
}
}  // namespace camb
}  // namespace impl