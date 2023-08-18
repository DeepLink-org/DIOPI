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

static uint32_t getSeed() {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<uint32_t> distribution(0, std::numeric_limits<uint32_t>::max());
    uint32_t seed = distribution(generator);
    return seed;
}

diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, int64_t idx) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor tensor(inout);
    if (!tensor.defined()) {
        return diopiSuccess;
    }

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, tensor.dtype()));
    cnnlRandGenerator_t generator;
    // cnnlRandRngType_t rng_type is recommended to be set as CNNL_RAND_RNG_MTGP32 on MLU300 series and CNNL_RAND_RNG_FAST on MLU200 series.
    DIOPI_CALLCNNL(cnnlRandCreateGenerator(&generator, CNNL_RAND_RNG_MTGP32));

    size_t stateSize = 0;
    DIOPI_CALLCNNL(cnnlRandGetMTGP32StateSize(generator, &stateSize));
    DiopiTensor state = requiresBuffer(ctx, stateSize);

    uint32_t seed = getSeed();
    DIOPI_CALLCNNL(cnnlRandMakeMTGP32KernelState(handle, state.data(), nullptr, nullptr, seed));
    DIOPI_CALLCNNL(cnnlRandGenerateUniform(handle, generator, dtype, state.data(), tensor.numel(), from, to, tensor.data()));
    DIOPI_CALLCNNL(cnnlRandDestroyGenerator(generator));
    return diopiSuccess;
}
}

}  // namespace camb
}  // namespace impl
