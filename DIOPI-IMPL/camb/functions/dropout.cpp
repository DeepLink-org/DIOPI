/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train) {
    if (train) {
        cnnlHandle_t handle = cnnlHandlePool.get(ctx);
        DiopiTensor inputTensor(input);
        DiopiTensor outputTensor(out);
        DiopiTensor maskTensor(mask);

        // Do this Check to use DIOPI-TEST because non-float data not supported in PyTorch unless p==0
        DIOPI_CHECK(((DiopiDataType::isFloatPoint(inputTensor.dtype()) || p == 0)), "result type Float can't be cast to the desired type");
        std::vector<DiopiTensor*> pTensors{&inputTensor};
        std::set<diopiDtype_t> supportedDtypes{
            diopi_dtype_int8, diopi_dtype_uint8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
        DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

        DiopiTensor outputTensorTemp = outputTensor;
        if ((outputTensor.dtype() != inputTensor.dtype())) {
            DIOPI_CALL(dataTypeCast(ctx, outputTensorTemp, inputTensor.dtype()));
        }

        CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc outputDesc(outputTensorTemp, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc maskDesc(maskTensor, CNNL_LAYOUT_ARRAY);

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

        // cases for dropout2d when input_shape != mask_shape
        if (inputTensor.shape() != maskTensor.shape()) {
            DiopiTensor tempTensor = ones(ctx, maskTensor.shape(), diopi_dtype_float32);
            CnnlTensorDesc tempDesc(tempTensor, CNNL_LAYOUT_ARRAY);

            DIOPI_CALLCNNL(cnnlFusedDropout_v2(
                handle, generator, tempDesc.get(), tempTensor.data(), p, state, maskDesc.get(), maskTensor.data(), tempDesc.get(), tempTensor.data()));

            DiopiTensor bcastTempTensor;
            DIOPI_CALL(dataTypeCast(ctx, tempTensor, outputTensorTemp.dtype()));
            broadcastHelper(ctx, tempTensor, outputTensorTemp, &bcastTempTensor);
            CnnlTensorDesc bcastTempDesc(bcastTempTensor, CNNL_LAYOUT_ARRAY);

            cnnlTensorDescriptor_t inputDescs[] = {inputDesc.get(), bcastTempDesc.get()};
            const void* inputs[] = {inputTensor.data(), bcastTempTensor.data()};
            DIOPI_CALLCNNL(cnnlMulN(handle, inputDescs, inputs, 2, outputDesc.get(), outputTensorTemp.data()))
        } else {
            // cases for dropout
            DIOPI_CALLCNNL(cnnlFusedDropout_v2(handle,
                                               generator,
                                               inputDesc.get(),
                                               inputTensor.data(),
                                               p,
                                               state,
                                               maskDesc.get(),
                                               maskTensor.data(),
                                               outputDesc.get(),
                                               outputTensorTemp.data()));
        }
        if (outputTensorTemp.dtype() != outputTensor.dtype()) {
            DIOPI_CALL(dataTypeCast(ctx, outputTensor, outputTensorTemp));
        }
        DIOPI_CALLCNNL(cnnlRandDestroyGenerator(generator));

    } else {  // if in test_mode
        diopiCopyInp(ctx, input, out);
    }
    return diopiSuccess;
}

diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train) {
    diopiDropout(ctx, input, mask, input, p, train);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
