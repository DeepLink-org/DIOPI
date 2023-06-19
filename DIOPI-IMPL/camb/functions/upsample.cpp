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

extern "C" diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    if (inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_CHECK(inputTensor.dim() == 4 || inputTensor.dim() == 3, "Camb kernel only supports UpsampleNearest 2d now.")
    DIOPI_CHECK(inputTensor.isContiguous(MemoryFormat::ChannelsLast), "inputTensor's memory format should be channelsLast");
    DIOPI_CHECK(outputTensor.isContiguous(MemoryFormat::ChannelsLast), "outputTensor's memory format should be channelsLast");

    cnnlTensorLayout_t layout = inputTensor.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;

    CnnlTensorDesc inputDesc(inputTensor, layout);
    CnnlTensorDesc outputDesc(outputTensor, layout);

    CnnlInterpDescriptor interpDesc;

    DIOPI_CALL(interpDesc.set(inputDesc.get(), CNNL_INTERP_NEAREST, CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO3, nullptr));

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DIOPI_CALLCNNL(cnnlInterp_v3(handle, interpDesc.get(), inputDesc.get(), inputTensor.data(), outputDesc.get(), outputTensor.data()));

    return diopiSuccess;
}

extern "C" diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                     diopiSize_t outSize, diopiSize_t inSize) {
    DiopiTensor inputTensor(gradOutput);
    DiopiTensor outputTensor(gradInput);

    if (inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_CHECK(inputTensor.dim() == 4 || inputTensor.dim() == 3, "Camb kernel only supports UpsampleNearest 2d now.")
    DIOPI_CHECK(inputTensor.isContiguous(MemoryFormat::ChannelsLast), "inputTensor's memory format should be channelsLast");
    DIOPI_CHECK(outputTensor.isContiguous(MemoryFormat::ChannelsLast), "outputTensor's memory format should be channelsLast");

    cnnlTensorLayout_t layout = inputTensor.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;

    CnnlTensorDesc inputDesc(inputTensor, layout);
    CnnlTensorDesc outputDesc(outputTensor, layout);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DIOPI_CALLCNNL(cnnlInterpBackward_v2(
        handle, false, false, CNNL_INTERP_BACKWARD_NEAREST, nullptr, true, inputDesc.get(), inputTensor.data(), outputDesc.get(), outputTensor.data()));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
