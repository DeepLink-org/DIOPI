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

// static uint32_t getSeed() {
//     std::random_device rd;
//     std::mt19937 generator(rd());
//     std::uniform_int_distribution<uint32_t> distribution(0, std::numeric_limits<uint32_t>::max());
//     uint32_t seed = distribution(generator);
//     return seed;
// }

diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, diopiConstGeneratorHandle_t generator) {
    DIOPI_CHECK_NULLPTR_ABORT(generator);
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor tensor(inout);
    if (!tensor.defined()) {
        return diopiSuccess;
    }

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, tensor.dtype()));
    cnnlRandGenerator_t cnnl_generator = nullptr;
    // cnnlRandRngType_t rng_type is recommended to be set as CNNL_RAND_RNG_MTGP32 on MLU300 series and CNNL_RAND_RNG_FAST on MLU200 series.
    DIOPI_CALLCNNL(cnnlRandCreateGenerator(&cnnl_generator, CNNL_RAND_RNG_MTGP32));

    // DIOPI_CALL(diopiGeneratorInitState(generator));
    // DIOPI_CALL(diopiGeneratorUpdateState(generator));
    void* state_ptr = nullptr;
    DIOPI_CALL(diopiGeneratorGetState(generator, &state_ptr));

    DIOPI_CALLCNNL(cnnlRandGenerateUniform(handle, cnnl_generator, dtype, state_ptr, tensor.numel(), from, to, tensor.data()));
    DIOPI_CALLCNNL(cnnlRandDestroyGenerator(cnnl_generator));
    return diopiSuccess;
}
}

}  // namespace camb
}  // namespace impl
