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

namespace {
struct DescData {
    int dim;
    uint64_t total_num;
    uint64_t total_size;
    int dims[8];
};

}  // namespace

extern "C" diopiError_t diopiUpsampleNearest(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t size) {
    DiopiTensor inputTensor(input);
    DiopiTensor outputTensor(out);

    if (inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_CHECK(inputTensor.dim() == 4 || inputTensor.dim() == 3, "Camb kernel only supports UpsampleNearest 2d now.")
    DIOPI_CHECK(inputTensor.isContiguous(), "inputTensor should be contiguous");
    DIOPI_CHECK(outputTensor.isContiguous(), "outputTensor should be contiguous");

    MemoryFormat format = inputTensor.dim() > 4 ? MemoryFormat::ChannelsLast3d : MemoryFormat::ChannelsLast;
    cnnlTensorLayout_t layout = inputTensor.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
    DIOPI_CALL(contiguous(ctx, inputTensor, format));
    DiopiTensor outputTmpTensor = requiresTensor(ctx, outputTensor.shape(), outputTensor.dtype(), format);

    CnnlTensorDesc inputDesc(inputTensor, layout);
    CnnlTensorDesc outputDesc(outputTmpTensor, layout);

    CnnlInterpDescriptor interpDesc;

    DIOPI_CALL(interpDesc.set(inputDesc.get(), CNNL_INTERP_NEAREST, CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO3, nullptr));

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DIOPI_CALLCNNL(cnnlInterp_v3(handle, interpDesc.get(), inputDesc.get(), inputTensor.data(), outputDesc.get(), outputTmpTensor.data()));

    // channels last -> contiguous
    DIOPI_CALL(contiguous(ctx, outputTmpTensor, MemoryFormat::Contiguous));
    // Copy back to origin
    DIOPI_CALL(diopiCopyInp(ctx, outputTmpTensor.tensorHandle(), outputTensor.tensorHandle()));

    return diopiSuccess;
}

extern "C" diopiError_t diopiUpsampleNearestBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                                     diopiSize_t out_size, diopiSize_t in_size) {
    DiopiTensor inputTensor(grad_output);
    DiopiTensor outputTensor(grad_input);

    if (inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    DIOPI_CHECK(inputTensor.dim() == 4 || inputTensor.dim() == 3, "Camb kernel only supports UpsampleNearest 2d now.")
    DIOPI_CHECK(inputTensor.isContiguous(), "inputTensor should be contiguous");
    DIOPI_CHECK(outputTensor.isContiguous(), "outputTensor should be contiguous");

    MemoryFormat format = inputTensor.dim() > 4 ? MemoryFormat::ChannelsLast3d : MemoryFormat::ChannelsLast;
    cnnlTensorLayout_t layout = inputTensor.dim() > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
    DIOPI_CALL(contiguous(ctx, inputTensor, format));
    DiopiTensor outputTmpTensor = requiresTensor(ctx, outputTensor.shape(), outputTensor.dtype(), format);

    CnnlTensorDesc inputDesc(inputTensor, layout);
    CnnlTensorDesc outputDesc(outputTmpTensor, layout);

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DIOPI_CALLCNNL(cnnlInterpBackward_v2(
        handle, false, false, CNNL_INTERP_BACKWARD_NEAREST, nullptr, true, inputDesc.get(), inputTensor.data(), outputDesc.get(), outputTmpTensor.data()));

    // channels last -> contiguous
    DIOPI_CALL(contiguous(ctx, outputTmpTensor, MemoryFormat::Contiguous));
    // Copy back to origin
    DIOPI_CALL(diopiCopyInp(ctx, outputTmpTensor.tensorHandle(), outputTensor.tensorHandle()));

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
