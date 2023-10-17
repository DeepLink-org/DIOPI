/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/float16.hpp"

namespace impl {
namespace camb {

namespace {

// clamp x into range [-largest, largest]
int clamp(int64_t x, int largest) { return static_cast<int>(std::min<int64_t>(std::max<int64_t>(x, -largest), largest)); }

}  // namespace

extern "C" diopiError_t diopiRandomInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, const int64_t* to, diopiGeneratorHandle_t generator) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor tensor(inout);
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, tensor.dtype()));
    cnnlRandGenerator_t cnnlGenerator;
    DIOPI_CALLCNNL(cnnlRandCreateGenerator(&cnnlGenerator, CNNL_RAND_RNG_MTGP32));

    if (dtype == CNNL_DTYPE_FLOAT || dtype == CNNL_DTYPE_HALF) {
        auto mantissa = dtype == CNNL_DTYPE_FLOAT ? std::numeric_limits<float>::digits : std::numeric_limits<half_float::half>::digits;
        auto largestIntAsFp = 1 << mantissa;
        int min = clamp(from, largestIntAsFp);
        int max = to ? clamp(*to - 1, largestIntAsFp) : largestIntAsFp;
        diopiTensorHandle_t stateHandle = nullptr;
        DIOPI_CALL(diopiGeneratorGetState(ctx, generator, &stateHandle));
        void* statePtr = nullptr;
        DIOPI_CALL(diopiGetTensorData(stateHandle, &statePtr));
        DIOPI_CALLCNNL(cnnlRandGenerateDescreteUniform(handle, cnnlGenerator, dtype, statePtr, tensor.numel(), min, max, tensor.data()));
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
