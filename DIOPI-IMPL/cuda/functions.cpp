/**************************************************************************************************
 * Copyright (c) 2022, SenseTime Inc.
 * License
 * Author
 *
 *************************************************************************************************/
#include <cstdio>
#include <vector>

#include <diopi/functions.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <mutex>

#include "helper.hpp"


#define DIOPI_CALLCUDNN(Expr) {                                                         \
        ::cudnnStatus_t ret = Expr;                                                     \
        if (CUDNN_STATUS_SUCCESS != ret) {                                              \
            char strLastError[2048] = {0};                                              \
            sprintf(strLastError, "cudnn error %d : %s at %s:%s",                       \
                    ret, cudnnGetErrorString(ret), __FILE__, __LINE__);                 \
            set_error_string(strLastError);                                             \
            return diopiErrorOccurred;                                                  \
        }}                                                                              \


static diopiError_t convertType(cudnnDataType_t *cudnnType, diopiDtype_t type){
    switch (type){
    case diopi_dtype_int8:
        *cudnnType = CUDNN_DATA_INT8;
        break;
    case diopi_dtype_uint8:
        *cudnnType = CUDNN_DATA_UINT8;
        break;
    case diopi_dtype_int32:
        *cudnnType = CUDNN_DATA_INT32;
        break;
    case diopi_dtype_float16:
        *cudnnType = CUDNN_DATA_HALF;
        break;
    case diopi_dtype_float32:
        *cudnnType = CUDNN_DATA_FLOAT;
        break;
    case diopi_dtype_float64:
        *cudnnType = CUDNN_DATA_DOUBLE;
        break;
#if CUDNN_VESION > 1100
    case diopi_dtype_bool:
        *cudnnType = CUDNN_DATA_BOOLEAN;
        break;
    case diopi_dtype_bfloat16:
        *cudnnType = CUDNN_DATA_BFLOAT16;
        break;
    case diopi_dtype_int64:
        *cudnnType = CUDNN_DATA_INT64;
        break;
#endif // CUDNN_VESION > 1100
    default:
        char strLastError[2048] = {0};
        sprintf(strLastError, "unkown diopitype error %d at %s:%s", type, __FILE__, __LINE__);
        set_error_string(strLastError);
        return diopiDtypeNotSupported;
    }
    return diopiSuccess;
}

static diopiError_t prepareTensorDesc(diopiDtype_t type, const diopiSize_t shape,
                                      const diopiSize_t stride, cudnnTensorDescriptor_t desc)
{
    cudnnDataType_t cudnnType;
    DIOPI_CALL(convertType(&cudnnType, type));

    int len = shape.len;
    int size = len < 4 ? 4 : len;
    std::vector<int> shapeArray(size);
    std::vector<int> strideArray(size);

    for(int i = 0; i < len; i++){
        shapeArray[i] = shape.data[i];
        strideArray[i] = stride.data[i];
    }
    for(int i = len; i < 4; i++){
        shapeArray[i] = 1;
        strideArray[i] = 1;
    }

    DIOPI_CALLCUDNN(cudnnSetTensorNdDescriptor(desc,
        cudnnType, size, shapeArray.data(), strideArray.data()));

    return diopiSuccess;

}

extern "C" diopiError_t diopiSoftmax(diopiContextHandle_t ctx, diopiTensorHandle_t out,
                                     const diopiTensorHandle_t input, int64_t dim, ScalarType dtype)
{
    cudnnHandle_t handle(nullptr);
    cudnnTensorDescriptor_t desc(nullptr);
    DIOPI_CALLCUDNN(cudnnCreate(&handle));
    DIOPI_CALLCUDNN(cudnnCreateTensorDescriptor(&desc));

    auto stream  = impl::cuda::getStream(ctx);
    auto trIn = impl::cuda::makeTensor(input);
    auto trOut   = impl::cuda::makeTensor(out);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    DIOPI_CALLCUDNN(cudnnSetStream(handle, stream));
    DIOPI_CALL(prepareTensorDesc(trIn.dtype(), trIn.shape(), trIn.stride(), desc));

    DIOPI_CALLCUDNN(cudnnSoftmaxForward(handle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, trIn.data(),
        &beta, desc, trOut.data()));

    DIOPI_CALLCUDNN(cudnnDestroyTensorDescriptor(desc));
    DIOPI_CALLCUDNN(cudnnDestroy(handle));
    return diopiSuccess;
}

extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input)
{
    cudnnHandle_t handle(nullptr);
    cudnnTensorDescriptor_t desc(nullptr);
    cudnnActivationDescriptor_t descAct(nullptr);
    DIOPI_CALLCUDNN(cudnnCreate(&handle));
    DIOPI_CALLCUDNN(cudnnCreateTensorDescriptor(&desc));
    DIOPI_CALLCUDNN(cudnnCreateActivationDescriptor(&descAct));

    auto stream  = impl::cuda::getStream(ctx);
    auto trIn = impl::cuda::makeTensor(input);
    auto trOut   = impl::cuda::makeTensor(out);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    DIOPI_CALLCUDNN(cudnnSetStream(handle, stream));
    DIOPI_CALL(prepareTensorDesc(trIn.dtype(), trIn.shape(), trIn.stride(), desc));

    DIOPI_CALLCUDNN(cudnnSetActivationDescriptor(descAct,
        CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 1.));

    DIOPI_CALLCUDNN(cudnnActivationForward(
        handle, descAct, &alpha, desc,
        trIn.data(), &beta, desc, trOut.data()));

    DIOPI_CALLCUDNN(cudnnDestroyTensorDescriptor(desc));
    DIOPI_CALLCUDNN(cudnnDestroy(handle));
    return diopiSuccess;
}
