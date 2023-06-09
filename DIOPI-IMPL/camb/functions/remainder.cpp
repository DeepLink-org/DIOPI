/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/debug.hpp"

namespace impl {
namespace camb {
extern "C" {

DIOPI_API diopiError_t diopiRemainderTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t other) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor other_tensor(other);
    DiopiTensor out_tensor(out);
    std::cout << "before dtype cast" << std::endl;
    std::cout << "input.dtype = " << input_tensor.dtype() << std::endl;
    std::cout << "other.dtype = " << other_tensor.dtype() << std::endl;
    std::cout << "out.dtype = " << out_tensor.dtype() << std::endl;
    std::cout << "input_tensor = ";
    printDevData(ctx, input_tensor);
    std::cout << "other_tensor = ";
    printDevData(ctx, other_tensor);
    std::cout << "out_tensor = ";
    printDevData(ctx, out_tensor);

    DiopiTensor out_tensor_temp = out_tensor;
    std::vector<DiopiTensor *> pTensors{&input_tensor, &other_tensor, &out_tensor_temp};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int32};
    DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedDtypes));

    // std::set<diopiDtype_t> supportedFloatDtypes{diopi_dtype_float16, diopi_dtype_float32};
    // std::set<diopiDtype_t> supportedIntDtypes{diopi_dtype_int32};

    // if (DiopiDataType::isInteger(input_tensor.dtype())) {
    //     DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedIntDtypes));
    // } else {
    //     DIOPI_CALL(autoCastTensorType(ctx, pTensors, supportedFloatDtypes));
    // }

    std::cout << "after dtypecast" << std::endl;
    std::cout << "input.dtype = " << input_tensor.dtype() << std::endl;
    std::cout << "other.dtype = " << other_tensor.dtype() << std::endl;
    std::cout << "out.dtype = " << out_tensor_temp.dtype() << std::endl;

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc other_desc(other_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(out_tensor_temp, CNNL_LAYOUT_ARRAY);
    std::cout << "input_tensor = ";
    printDevData(ctx, input_tensor);
    std::cout << "other_tensor = ";
    printDevData(ctx, other_tensor);

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetFloorModWorkspaceSize(handle, input_desc.get(), other_desc.get(), out_desc.get(), &workspace_size));
    void *workspace = nullptr;
    if (workspace_size != 0) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    DIOPI_CALLCNNL(cnnlFloorMod(handle,
                                input_desc.get(),
                                input_tensor.data(),
                                other_desc.get(),
                                other_tensor.data(),
                                out_desc.get(),
                                out_tensor_temp.data(),
                                workspace,
                                workspace_size));

    if (out_tensor.dtype() != out_tensor_temp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, out_tensor, out_tensor_temp));
    }
    std::cout << "out_tensor = ";
    printDevData(ctx, out_tensor);

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRemainderScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t *other) {
    DiopiTensor other_tensor;
    makeTensorFromScalar(ctx, other, other_tensor);
    diopiTensorHandle_t other_tensor_ptr = other_tensor.tensorHandle();
    DIOPI_CALL(diopiRemainderTensor(ctx, out, input, other_tensor_ptr));
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiRemainder(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiScalar_t *input, diopiConstTensorHandle_t other) {
    DiopiTensor input_tensor;
    makeTensorFromScalar(ctx, input, input_tensor);
    diopiTensorHandle_t input_tensor_ptr = input_tensor.tensorHandle();
    DIOPI_CALL(diopiRemainderTensor(ctx, out, input_tensor_ptr, other));
    return diopiSuccess;
}
}  // extern "C"
}  // namespace camb
}  // namespace impl