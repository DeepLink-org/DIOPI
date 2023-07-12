/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

DIOPI_API diopiError_t diopiMaskedScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask,
                                          diopiConstTensorHandle_t source) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor outTensor(out);      // output
    DiopiTensor inputTensor(input);  // input
    DiopiTensor maskTensor(mask);    // mask
    DiopiTensor srcTensor(source);   // source
    DiopiTensor tempOutTensor = requiresTensor(ctx, outTensor.shape(), outTensor.dtype());

    std::vector<DiopiTensor *> pmask{&maskTensor};
    std::set<diopiDtype_t> maskDtypes{diopi_dtype_bool};
    DIOPI_CALL(autoCastTensorType(ctx, pmask, maskDtypes));

    std::vector<DiopiTensor *> pInput{&tempOutTensor, &inputTensor, &srcTensor};
    std::set<diopiDtype_t> inputDtypes{diopi_dtype_float16, diopi_dtype_float32};
    DIOPI_CALL(autoCastTensorType(ctx, pInput, inputDtypes));

    CnnlTensorDesc outDesc(tempOutTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc maskDesc(maskTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc srcDesc(srcTensor, CNNL_LAYOUT_ARRAY);
    cnnlMaskedOp_t maskMode = CNNL_MASKED_SCATTER;

    size_t workspaceSize = 0;
    DIOPI_CALLCNNL(cnnlGetMaskedWorkspaceSize(handle, maskMode, inputDesc.get(), maskDesc.get(), srcDesc.get(), outDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }

#if (CNNL_MAJOR >= 1 && CNNL_MINOR >= 15 && CNNL_PATCHLEVEL >= 2)
    DIOPI_CALLCNNL(cnnlMasked_v4(handle,
                                 maskMode,
                                 inputDesc.get(),
                                 inputTensor.data(),
                                 maskDesc.get(),
                                 maskTensor.data(),
                                 srcDesc.get(),
                                 srcTensor.data(),
                                 nullptr,
                                 workspace,
                                 workspaceSize,
                                 outDesc.get(),
                                 tempOutTensor.data(),
                                 nullptr));
#else
    DIOPI_CALLCNNL(cnnlMasked_v3(handle,
                                 maskMode,
                                 inputDesc.get(),
                                 inputTensor.data(),
                                 maskDesc.get(),
                                 maskTensor.data(),
                                 srcDesc.get(),
                                 srcTensor.data(),
                                 workspace,
                                 workspaceSize,
                                 outDesc.get(),
                                 tempOutTensor.data(),
                                 nullptr));
#endif
    DIOPI_CALL(dataTypeCast(ctx, outTensor, tempOutTensor));
    return diopiSuccess;
}
}  // extern "C"

}  // namespace camb
}  // namespace impl
