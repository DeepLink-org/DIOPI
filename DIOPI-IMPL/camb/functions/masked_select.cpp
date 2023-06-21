#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
#include "../common/debug.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t nonzeroCount(diopiContextHandle_t ctx, DiopiTensor input_tensor, DiopiTensor *num_true);

diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor input_tensor(input);
    DiopiTensor mask_tensor(mask);

    std::vector<DiopiTensor *> pmask{&mask_tensor};
    std::set<diopiDtype_t> mask_dtypes{diopi_dtype_bool, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pmask, mask_dtypes));
    // When the data type of masked tensor is not bool, the data type of input
    // tensor must be same with the data type of the masked tensor.
    diopiDtype_t origin_dtype = input_tensor.dtype();

    if (mask_tensor.dtype() != diopi_dtype_bool) {
        DIOPI_CALL(dataTypeCast(ctx, input_tensor, mask_tensor.dtype()))
    } else {
        std::vector<DiopiTensor *> pinput{&input_tensor};
        std::set<diopiDtype_t> input_dtypes{
            diopi_dtype_bool, diopi_dtype_int8, diopi_dtype_uint8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
        DIOPI_CALL(autoCastTensorType(ctx, pinput, input_dtypes));
    }

    std::vector<int64_t> input_numel(1, int64_t(input_tensor.numel()));
    auto temp_output_tensor = requiresTensor(ctx, input_numel, input_tensor.dtype());

    CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc mask_desc(mask_tensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc out_desc(temp_output_tensor, CNNL_LAYOUT_ARRAY);
    cnnlMaskedOp_t masked_mode = CNNL_MASKED_SELECT;

    size_t workspace_size = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(handle, masked_mode, input_desc.get(), mask_desc.get(), nullptr, out_desc.get(), &workspace_size));
    void *workspace = nullptr;
    if (0 != workspace_size) {
        workspace = requiresBuffer(ctx, workspace_size).data();
    }

    std::vector<int64_t> num_true_shape(1, 1);
    auto num_true = requiresTensor(ctx, num_true_shape, diopi_dtype_uint32);
    DIOPI_CALLCNNL(cnnlMasked_v3(handle,
                                 masked_mode,
                                 input_desc.get(),
                                 input_tensor.data(),
                                 mask_desc.get(),
                                 mask_tensor.data(),
                                 nullptr,
                                 nullptr,
                                 workspace,
                                 workspace_size,
                                 out_desc.get(),
                                 temp_output_tensor.data(),
                                 static_cast<uint32_t *>(num_true.data())));

    syncStreamInCtx(ctx);
    int64_t num_true_host = 0;
    cnrtMemcpy(&num_true_host, num_true.data(), sizeof(num_true.dtype()), CNRT_MEM_TRANS_DIR_DEV2HOST);
    std::vector<int64_t> output_shape(1, num_true_host);
    auto output_tensor = requiresTensor(ctx, output_shape, temp_output_tensor.dtype());

    DIOPI_CALL(diopiSlice(ctx, diopiTensorHandle_t(output_tensor), diopiTensorHandle_t(temp_output_tensor), 0, 0, num_true_host, 1));
    *out = diopiTensorHandle_t(output_tensor);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor Gradinput_tensor(grad_input);
    DiopiTensor Gradoutput_tensor(grad_output);
    DiopiTensor mask_tensor(mask);

    if (not Gradinput_tensor.defined()) {
        std::cout << "grad_input not defined !!!" << std::endl;
        return diopiSuccess;
    }

    if (not Gradoutput_tensor.defined()) {
        std::cout << "grad_output not defined !!!" << std::endl;
        return diopiSuccess;
    }

    if (not mask_tensor.defined()) {
        std::cout << "mask not defined !!!" << std::endl;
        return diopiSuccess;
    }

    DIOPI_CALL(diopiMul(ctx, grad_input, mask, grad_output); return diopiSuccess;)
}
}  // extern "C"

}  // namespace camb
}  // namespace impl