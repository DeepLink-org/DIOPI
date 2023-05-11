#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

diopiError_t diopiAddcmul(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                    diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor other_tensor1(tensor1);
    DiopiTensor other_tensor2(tensor2);
    DiopiTensor out_tensor(out);

    std::vector<DiopiTensor*> pTensors{&input_tensor, &other_tensor1, &other_tensor2};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float32, diopi_dtype_float16};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    DiopiTensor out_tensor_temp = out_tensor;
    if (out_tensor.dtype() != input_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor_temp, input_tensor.dtype()));
    }

    CnnlTensorDesc input_tensor_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_tensor1_desc(other_tensor1, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_tensor2_desc(other_tensor2, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_tensor_desc(out_tensor_temp, CNNL_LAYOUT_ARRAY);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetAddcmulWorkspaceSize(handle, input_tensor_desc.get(), other_tensor1_desc.get(), other_tensor2_desc.get(), &workspace_size));
    void* workspace = nullptr;
    float scalar_value;
    if (DiopiDataType::isInteger(value->stype)) {
        scalar_value = value->ival;
    } else {
        scalar_value = value->fval;
    }

    workspace = requiresBuffer(ctx, workspace_size).data();
    DIOPI_CALLCNNL(cnnlAddcmul(handle,
                               input_tensor_desc.get(),
                               input_tensor.data(),
                               &(scalar_value),
                               other_tensor1_desc.get(),
                               other_tensor1.data(),
                               other_tensor2_desc.get(),
                               other_tensor2.data(),
                               workspace,
                               workspace_size,
                               out_tensor_desc.get(),
                               out_tensor_temp.data()))
    if (out_tensor_temp.dtype() != out_tensor.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_temp));
    }
    return diopiSuccess;
}
diopiError_t diopiAddcmulInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                       const diopiScalar_t* value) {
    diopiAddcmul(ctx, input, input, tensor1, tensor2, value);
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
