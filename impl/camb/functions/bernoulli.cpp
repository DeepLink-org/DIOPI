/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

DIOPI_API diopiError_t diopiBernoulli(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiGeneratorHandle_t generator) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, inputTensor.dtype()));

    cnnlRandGenerator_t cnnlGenerator = nullptr;
    DIOPI_CALL_CNNL(cnnlRandCreateGenerator(&cnnlGenerator, CNNL_RAND_RNG_MTGP32));

    diopiTensorHandle_t stateHandle = nullptr;
    DIOPI_CALL(diopiGeneratorGetState(ctx, generator, &stateHandle));
    void* statePtr = nullptr;
    DIOPI_CALL(diopiGetTensorData(stateHandle, &statePtr));

    if (outputTensor.dtype() != inputTensor.dtype()) {
        DiopiTensor outTemp = requiresTensor(ctx, inputTensor.shape(), inputTensor.dtype());
        DIOPI_CALL_CNNL(cnnlRandGenerateUniform(handle, cnnlGenerator, dtype, statePtr, inputTensor.numel(), 0, 1, outTemp.data()));
        DIOPI_CALL(diopiGeneratorSetState(generator, stateHandle));
        DIOPI_CALL(diopiLtInp(ctx, diopiTensorHandle_t(outTemp), input));
        DIOPI_CALL(dataTypeCast(ctx, outputTensor, outTemp));
    } else {
        DIOPI_CALL_CNNL(cnnlRandGenerateUniform(handle, cnnlGenerator, dtype, statePtr, inputTensor.numel(), 0, 1, outputTensor.data()));
        DIOPI_CALL(diopiGeneratorSetState(generator, stateHandle));
        DIOPI_CALL(diopiLtInp(ctx, out, input));
    }

    DIOPI_CALL_CNNL(cnnlRandDestroyGenerator(cnnlGenerator));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBernoulliInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, diopiGeneratorHandle_t generator) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(inout);

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, inputTensor.dtype()));
    DiopiTensor outTemp = requiresTensor(ctx, inputTensor.shape(), inputTensor.dtype());

    cnnlRandGenerator_t cnnlGenerator = nullptr;
    DIOPI_CALL_CNNL(cnnlRandCreateGenerator(&cnnlGenerator, CNNL_RAND_RNG_MTGP32));

    diopiTensorHandle_t stateHandle = nullptr;
    DIOPI_CALL(diopiGeneratorGetState(ctx, generator, &stateHandle));
    void* statePtr = nullptr;
    DIOPI_CALL(diopiGetTensorData(stateHandle, &statePtr));
    DIOPI_CALL_CNNL(cnnlRandGenerateUniform(handle, cnnlGenerator, dtype, statePtr, inputTensor.numel(), 0, 1, outTemp.data()));
    DIOPI_CALL(diopiGeneratorSetState(generator, stateHandle));
    DIOPI_CALL(diopiLt(ctx, inout, diopiTensorHandle_t(outTemp), inout));

    DIOPI_CALL_CNNL(cnnlRandDestroyGenerator(cnnlGenerator));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBernoulliScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, double p, diopiGeneratorHandle_t generator) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor outTensor(out);

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, outTensor.dtype()));

    cnnlRandGenerator_t cnnlGenerator = nullptr;
    DIOPI_CALL_CNNL(cnnlRandCreateGenerator(&cnnlGenerator, CNNL_RAND_RNG_MTGP32));

    diopiTensorHandle_t stateHandle = nullptr;
    DIOPI_CALL(diopiGeneratorGetState(ctx, generator, &stateHandle));
    void* statePtr = nullptr;
    DIOPI_CALL(diopiGetTensorData(stateHandle, &statePtr));
    DIOPI_CALL_CNNL(cnnlRandGenerateUniform(handle, cnnlGenerator, dtype, statePtr, outTensor.numel(), 0, 1, outTensor.data()));
    DIOPI_CALL(diopiGeneratorSetState(generator, stateHandle));

    diopiScalar_t scalar = constructDiopiScalarT(outTensor.dtype(), p);
    DIOPI_CALL(diopiLtInpScalar(ctx, out, &scalar));

    DIOPI_CALL_CNNL(cnnlRandDestroyGenerator(cnnlGenerator));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
