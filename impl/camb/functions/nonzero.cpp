

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t nonzeroCount(diopiContextHandle_t ctx, DiopiTensor inputTensor, DiopiTensor* numTrue) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    std::vector<int64_t> shape = {1};
    *numTrue = requiresTensor(ctx, shape, diopi_dtype_int32);
    CnnlTensorDesc numTrueDesc(*numTrue, CNNL_LAYOUT_ARRAY);

    DIOPI_CALL_CNNL(cnnlNumTrue_v2(handle, inputDesc.get(), inputTensor.data(), numTrueDesc.get(), numTrue->data()));
    return diopiSuccess;
}

diopiError_t diopiNonzero(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiConstTensorHandle_t input) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    if (DiopiDataType::isInteger(inputTensor.dtype())) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_int32));
    } else if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);

    DiopiTensor numTrue;
    nonzeroCount(ctx, inputTensor, &numTrue);
    CnnlTensorDesc numTrueDesc(numTrue, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize(0);
    DIOPI_CALL_CNNL(cnnlGetWhereWorkspaceSize(handle, numTrueDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    syncStreamInCtx(ctx);
    int32_t count = 0;
    cnrtMemcpy(&count, numTrue.data(), sizeof(int32_t), CNRT_MEM_TRANS_DIR_DEV2HOST);

    std::vector<int64_t> shape(2);
    shape[0] = count;
    shape[1] = inputTensor.dim();
    auto outTensor = requiresTensor(ctx, shape, diopi_dtype_int32);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALL_CNNL(cnnlWhere_v2(
        handle, inputDesc.get(), inputTensor.data(), numTrueDesc.get(), numTrue.data(), false, workspace, workspaceSize, outDesc.get(), outTensor.data()));
    DIOPI_CALL(dataTypeCast(ctx, outTensor, diopi_dtype_int64));
    *out = diopiTensorHandle_t(outTensor);
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
