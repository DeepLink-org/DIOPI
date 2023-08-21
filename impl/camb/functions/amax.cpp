/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

diopiError_t diopiAmax(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim, bool keepdim) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor = DiopiTensor(input);
    DiopiTensor outTensor = DiopiTensor(out);
    DiopiTensor inputTensorTmp = inputTensor;
    DiopiTensor outTensorTmp = outTensor;

    DIOPI_CALL(autoCastTensorType(ctx, {&inputTensorTmp}, {diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32}));
    if (inputTensorTmp.dtype() != outTensor.dtype()) {
        outTensorTmp = requiresTensor(ctx, outTensor.shape(), inputTensorTmp.dtype());
    }

    const int64_t* dimPtr = dim.data;
    int64_t len = dim.len;
    std::vector<int32_t> axis;
    if (dimPtr != nullptr && len != 0) {
        for (int32_t i = 0; i < len; ++i) {
            int32_t dim = *(dimPtr + i);
            axis.emplace_back(dim < 0 ? dim + inputTensorTmp.dim() : dim);
        }
        std::set<int32_t> set(axis.begin(), axis.end());
        axis.assign(set.begin(), set.end());
    } else {
        for (int32_t i = 0; i < inputTensorTmp.dim(); ++i) {
            axis.emplace_back(i);
        }
    }

    cnnlDataType_t tensorType;
    CnnlDataType::convertToCnnlType(&tensorType, inputTensorTmp.dtype());
    CnnlResourceGuard<cnnlReduceDescriptor_t, cnnlCreateReduceDescriptor, cnnlDestroyReduceDescriptor> reduceDesc;
    DIOPI_CALLCNNL(cnnlSetReduceDescriptor_v2(
        reduceDesc.get(), axis.data(), axis.size(), CNNL_REDUCE_MAX, tensorType, CNNL_NOT_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES, CNNL_32BIT_INDICES, 0.0));

    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetReduceOpWorkspaceSize(handle, inputDesc.get(), outDesc.get(), reduceDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLCNNL(cnnlReduce(handle,
                              reduceDesc.get(),
                              workspace,
                              workspaceSize,
                              nullptr,
                              inputDesc.get(),
                              inputTensorTmp.data(),
                              0,
                              nullptr,
                              nullptr,
                              outDesc.get(),
                              outTensorTmp.data()));
    if (outTensorTmp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
