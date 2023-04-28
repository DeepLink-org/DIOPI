/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {

extern "C" {

diopiError_t maxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor output_tensor(max);
    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CHECK(input_tensor.dtype() == output_tensor.dtype(), "input->dtype should equal to output->dtype");

    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, input_tensor.dtype()));
    std::vector<int64_t> dims(input_tensor.dim());
    for (int i = 0; i < input_tensor.dim(); i++) {
        dims[i] = i;
    }
    diopiSize_t dim = {dims.data(), input_tensor.dim()};
    CnnlReduceDescriptor reduce_desc;
    reduce_desc.set(input_tensor, dims, CNNL_REDUCE_MAX, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES, dtype);

    size_t workspace_size(0);
    DIOPI_CALLCNNL(cnnlGetReduceOpWorkspaceSize(handle, input_desc.get(), output_desc.get(), reduce_desc.get(), &workspace_size));
    void* workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    size_t indices_size_inbytes(0);
    void* indices = nullptr;
    void* alpha = nullptr;
    void* beta = nullptr;
    DIOPI_CALLCNNL(cnnlReduce(handle,
                              reduce_desc.get(),
                              workspace,
                              workspace_size,
                              alpha,
                              input_desc.get(),
                              input_tensor.data(),
                              indices_size_inbytes,
                              indices,
                              beta,
                              output_desc.get(),
                              output_tensor.data()));
    return diopiSuccess;
}

diopiError_t getClassNum(diopiContextHandle_t ctx, DiopiTensor input_tensor, int32_t* cls_num) {
    std::vector<int64_t> dims(1, 1);
    DiopiTensor max = requiresTensor(ctx, dims, input_tensor.dtype());
    DIOPI_CALL(maxAll(ctx, (diopiTensorHandle_t)max, (diopiTensorHandle_t)input_tensor));

    syncStreamInCtx(ctx);
    int32_t* ptr = reinterpret_cast<int32_t*>(malloc(max.numel() * sizeof(int32_t)));
    cnrtMemcpy(ptr, max.data(), max.numel() * sizeof(int32_t), cnrtMemcpyDevToHost);
    *cls_num = *ptr + 1;
    free(ptr);

    return diopiSuccess;
}

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numClasses) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor input_tensor(input);
    DiopiTensor out_tensor(out);
    // input must be int32
    if (diopi_dtype_int32 != input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, diopi_dtype_int32));
    }
    CnnlTensorDesc inputDesc(input_tensor, CNNL_LAYOUT_ARRAY);

    int32_t cls_num = 0;
    if (-1 == numClasses) {
        DIOPI_CALL(getClassNum(ctx, input_tensor, &cls_num));
    } else {
        cls_num = numClasses;
    }

    diopiTensorHandle_t on_value, off_value;
    std::vector<int64_t> dims(1, 1);
    diopiSize_t shape(dims.data(), 1);
    DIOPI_CALL(diopiRequireTensor(ctx, &on_value, &shape, nullptr, diopi_dtype_int32, diopi_device));
    DIOPI_CALL(diopiRequireTensor(ctx, &off_value, &shape, nullptr, diopi_dtype_int32, diopi_device));
    DiopiTensor on_value_tensor(on_value);
    DiopiTensor off_value_tensor(off_value);
    CnnlTensorDesc on_tensor_desc(on_value_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc off_tensor_desc(off_value_tensor, CNNL_LAYOUT_ARRAY);
    int32_t one = 1;
    int32_t zero = 0;
    DIOPI_CALLCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &one, on_tensor_desc.get(), on_value_tensor.data()));
    DIOPI_CALLCNNL(cnnlFill_v3(handle, CNNL_POINTER_MODE_HOST, &zero, off_tensor_desc.get(), off_value_tensor.data()));
    int axis = -1;

    // output must be int32, float16, float32
    if (diopi_dtype_int32 != out_tensor.dtype()) {
        DiopiTensor out32_tensor = requiresTensor(ctx, out_tensor.shape(), diopi_dtype_int32);
        DIOPI_CALLCNNL(cnnlOneHot(handle,
                                  inputDesc.get(),
                                  input_tensor.data(),
                                  cls_num,
                                  on_value_tensor.data(),
                                  off_value_tensor.data(),
                                  axis,
                                  CNNL_DTYPE_INT32,
                                  out32_tensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out32_tensor));
    } else {
        DIOPI_CALLCNNL(cnnlOneHot(handle,
                                  inputDesc.get(),
                                  input_tensor.data(),
                                  cls_num,
                                  on_value_tensor.data(),
                                  off_value_tensor.data(),
                                  axis,
                                  CNNL_DTYPE_INT32,
                                  out_tensor.data()));
    }

    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
