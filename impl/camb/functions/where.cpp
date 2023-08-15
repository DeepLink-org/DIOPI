

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                        diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor otherTensor(other);
    DiopiTensor condTensor(condition);
    DiopiTensor outTensor(out);

    std::vector<DiopiTensor*> inputs{&inputTensor, &otherTensor, &condTensor};
    std::set<diopiDtype_t> inputsSupportDtype{
        diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_int64, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, inputs, inputsSupportDtype));
    std::vector<DiopiTensor*> cond{&condTensor};
    std::set<diopiDtype_t> condSupportDtype{diopi_dtype_uint8, diopi_dtype_bool};
    DIOPI_CALL(autoCastTensorType(ctx, cond, condSupportDtype));

    DiopiTensor outTensorTemp = outTensor;
    if (outTensorTemp.dtype() != inputTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensorTemp, inputTensor.dtype()));
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc otherDesc(otherTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc condDesc(condTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTemp, CNNL_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetSelectV2WorkspaceSize(handle, condDesc.get(), inputDesc.get(), otherDesc.get(), &workspaceSize));
    void* workspace = nullptr;
    workspace = requiresBuffer(ctx, workspaceSize).data();

    DIOPI_CALLCNNL(cnnlSelectV2(handle,
                                condDesc.get(),
                                condTensor.data(),
                                inputDesc.get(),
                                inputTensor.data(),
                                otherDesc.get(),
                                otherTensor.data(),
                                workspace,
                                workspaceSize,
                                outDesc.get(),
                                outTensorTemp.data()));

    if (outTensorTemp.dtype() != outTensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTemp));
    }
    return diopiSuccess;
}
}  // namespace camb
}  // namespace impl
