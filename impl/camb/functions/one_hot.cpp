/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {

diopiError_t maxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(max);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outputDesc(outputTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CHECK(inputTensor.dtype() == outputTensor.dtype(), "input->dtype should equal to output->dtype");

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, inputTensor.dtype()));
    std::vector<int64_t> dims(inputTensor.dim());
    for (int i = 0; i < inputTensor.dim(); i++) {
        dims[i] = i;
    }
    CnnlReduceDescriptor reduceDesc;
    reduceDesc.set(inputTensor, dims, CNNL_REDUCE_MAX, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES, dtype);

    size_t workspaceSize(0);
    DIOPI_CALL_CNNL(cnnlGetReduceOpWorkspaceSize(handle, inputDesc.get(), outputDesc.get(), reduceDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    size_t indicesSizeInbytes(0);
    void* indices = nullptr;
    void* alpha = nullptr;
    void* beta = nullptr;
    DIOPI_CALL_CNNL(cnnlReduce(handle,
                               reduceDesc.get(),
                               workspace,
                               workspaceSize,
                               alpha,
                               inputDesc.get(),
                               inputTensor.data(),
                               indicesSizeInbytes,
                               indices,
                               beta,
                               outputDesc.get(),
                               outputTensor.data()));
    return diopiSuccess;
}

diopiError_t getClassNum(diopiContextHandle_t ctx, DiopiTensor inputTensor, int32_t* clsNum) {
    std::vector<int64_t> dims(1, 1);
    DiopiTensor max = requiresTensor(ctx, dims, inputTensor.dtype());
    DIOPI_CALL(maxAll(ctx, (diopiTensorHandle_t)max, (diopiTensorHandle_t)inputTensor));

    syncStreamInCtx(ctx);
    int32_t* ptr = reinterpret_cast<int32_t*>(malloc(max.numel() * sizeof(int32_t)));
    cnrtMemcpy(ptr, max.data(), max.numel() * sizeof(int32_t), cnrtMemcpyDevToHost);
    *clsNum = *ptr + 1;
    free(ptr);

    return diopiSuccess;
}

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numClasses) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(out);
    // input must be int32
    if (diopi_dtype_int32 != inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_int32));
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);

    int32_t clsNum = 0;
    if (-1 == numClasses) {
        DIOPI_CALL(getClassNum(ctx, inputTensor, &clsNum));
    } else {
        clsNum = numClasses;
    }

    diopiTensorHandle_t onValue, offValue;
    std::vector<int64_t> dims(1, 1);
    diopiSize_t shape{dims.data(), 1};
    DIOPI_CALL(diopiRequireTensor(ctx, &onValue, &shape, nullptr, diopi_dtype_int32, diopi_device));
    DIOPI_CALL(diopiRequireTensor(ctx, &offValue, &shape, nullptr, diopi_dtype_int32, diopi_device));
    DiopiTensor onValueTensor(onValue);
    DiopiTensor offValueTensor(offValue);
    CnnlTensorDesc onTensorDesc(onValueTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc offTensorDesc(offValueTensor, CNNL_LAYOUT_ARRAY);
    int32_t one = 1;
    int32_t zero = 0;
    DIOPI_CALL_CNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &one, onTensorDesc.get(), onValueTensor.data()));
    DIOPI_CALL_CNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &zero, offTensorDesc.get(), offValueTensor.data()));
    int axis = -1;

    // output must be int32, float16, float32
    if (diopi_dtype_int32 != outTensor.dtype()) {
        DiopiTensor out32Tensor = requiresTensor(ctx, outTensor.shape(), diopi_dtype_int32);
        DIOPI_CALL_CNNL(cnnlOneHot(
            handle, inputDesc.get(), inputTensor.data(), clsNum, onValueTensor.data(), offValueTensor.data(), axis, CNNL_DTYPE_INT32, out32Tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, out32Tensor));
    } else {
        DIOPI_CALL_CNNL(cnnlOneHot(
            handle, inputDesc.get(), inputTensor.data(), clsNum, onValueTensor.data(), offValueTensor.data(), axis, CNNL_DTYPE_INT32, outTensor.data()));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
