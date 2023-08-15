#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t nonzeroCount(diopiContextHandle_t ctx, DiopiTensor inputTensor, DiopiTensor *numTrue);

diopiError_t diopiMaskedSelect(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor maskTensor(mask);

    std::vector<DiopiTensor *> pmask{&maskTensor};
    std::set<diopiDtype_t> maskDtypes{diopi_dtype_bool};
    DIOPI_CALL(autoCastTensorType(ctx, pmask, maskDtypes));
    // When the data type of masked tensor is not bool, the data type of input
    // tensor must be same with the data type of the masked tensor.

    std::vector<DiopiTensor *> pinput{&inputTensor};
    std::set<diopiDtype_t> inputDtypes{
        diopi_dtype_bool, diopi_dtype_int8, diopi_dtype_uint8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pinput, inputDtypes));

    std::vector<int64_t> inputNumel(1, int64_t(inputTensor.numel()));
    auto tempOutputTensor = requiresTensor(ctx, inputNumel, inputTensor.dtype());

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc maskDesc(maskTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(tempOutputTensor, CNNL_LAYOUT_ARRAY);
    cnnlMaskedOp_t maskMode = CNNL_MASKED_SELECT;

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(handle, maskMode, inputDesc.get(), maskDesc.get(), nullptr, outDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

    std::vector<int64_t> numTrueShape(1, 1);
    auto numTrue = requiresTensor(ctx, numTrueShape, diopi_dtype_uint32);

// version should be greater than 1.15.2
#if (CNNL_MAJOR * 10000 + CNNL_MINOR * 100 + CNNL_PATCHLEVEL >= 11502)
    DIOPI_CALLCNNL(cnnlMasked_v4(handle,
                                 maskMode,
                                 inputDesc.get(),
                                 inputTensor.data(),
                                 maskDesc.get(),
                                 maskTensor.data(),
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 workspace,
                                 workspaceSize,
                                 outDesc.get(),
                                 tempOutputTensor.data(),
                                 reinterpret_cast<uint32_t *>(numTrue.data())));
#else
    DIOPI_CALLCNNL(cnnlMasked_v3(handle,
                                 maskMode,
                                 inputDesc.get(),
                                 inputTensor.data(),
                                 maskDesc.get(),
                                 maskTensor.data(),
                                 nullptr,
                                 nullptr,
                                 workspace,
                                 workspaceSize,
                                 outDesc.get(),
                                 tempOutputTensor.data(),
                                 reinterpret_cast<uint32_t *>(numTrue.data())));
#endif
    syncStreamInCtx(ctx);
    uint32_t numTrueHost = 0;
    cnrtMemcpy(&numTrueHost, numTrue.data(), sizeof(numTrue.dtype()), CNRT_MEM_TRANS_DIR_DEV2HOST);
    std::vector<int64_t> outputShape(1, static_cast<int64_t>(numTrueHost));
    auto outputTensor = requiresTensor(ctx, outputShape, tempOutputTensor.dtype());

    DIOPI_CALL(diopiSlice(ctx, diopiTensorHandle_t(outputTensor), diopiTensorHandle_t(tempOutputTensor), 0, 0, numTrueHost, 1));
    *out = diopiTensorHandle_t(outputTensor);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMaskedSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                 diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor gradInputTensor(gradInput);    // output
    DiopiTensor gradOutputTensor(gradOutput);  // src
    DiopiTensor maskTensor(mask);              // mask
    DiopiTensor tempGradInputTensor = ones(ctx, gradInputTensor.shape(), gradInputTensor.dtype());

    if (!gradOutputTensor.defined()) {  // if mask is full-zero, output is empty, gradInput is full-zero
        auto scalar = diopiScalar_t();
        scalar.stype = gradInputTensor.dtype();
        scalar.ival = 0;
        scalar.fval = 0.0;
        diopiFill(ctx, gradInput, &scalar);
    }

    std::vector<DiopiTensor *> pmask{&maskTensor};
    std::set<diopiDtype_t> maskDtypes{diopi_dtype_bool};
    DIOPI_CALL(autoCastTensorType(ctx, pmask, maskDtypes));

    std::vector<DiopiTensor *> pGradInput{&tempGradInputTensor, &gradOutputTensor};
    std::set<diopiDtype_t> gradInputDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pGradInput, gradInputDtypes));

    CnnlTensorDesc gradInputDesc(tempGradInputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutDesc(gradOutputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc maskDesc(maskTensor, CNNL_LAYOUT_ARRAY);
    cnnlMaskedOp_t maskMode = CNNL_MASKED_SCATTER;

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(handle, maskMode, gradInputDesc.get(), maskDesc.get(), gradOutDesc.get(), gradInputDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

// version should be greater than 1.15.2
#if (CNNL_MAJOR * 10000 + CNNL_MINOR * 100 + CNNL_PATCHLEVEL >= 11502)
    DIOPI_CALLCNNL(cnnlMasked_v4(handle,
                                 maskMode,
                                 gradInputDesc.get(),
                                 tempGradInputTensor.data(),
                                 maskDesc.get(),
                                 maskTensor.data(),
                                 gradOutDesc.get(),
                                 gradOutputTensor.data(),
                                 nullptr,
                                 workspace,
                                 workspaceSize,
                                 gradInputDesc.get(),
                                 gradInputTensor.data(),
                                 nullptr));
#else
    DIOPI_CALLCNNL(cnnlMasked_v3(handle,
                                 maskMode,
                                 gradInputDesc.get(),
                                 tempGradInputTensor.data(),
                                 maskDesc.get(),
                                 maskTensor.data(),
                                 gradOutDesc.get(),
                                 gradOutputTensor.data(),
                                 workspace,
                                 workspaceSize,
                                 gradInputDesc.get(),
                                 gradInputTensor.data(),
                                 nullptr));
#endif
    DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, tempGradInputTensor));
    DIOPI_CALL(diopiMul(ctx, gradInput, mask, gradInput));
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
