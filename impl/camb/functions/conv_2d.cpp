/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>
#include <vector>

#include "../common/common.hpp"
#include "../common/debug.hpp"

namespace impl {
namespace camb {

#define REQUIRES_TENSOR_BY_DTYPE_OR_NOT(tensor1, tensor2, targetDtype)   \
    DiopiTensor tensor1 = tensor2;                                       \
    if (tensor2.defined() && tensor1.dtype() != targetDtype) {           \
        tensor1 = requiresTensor(ctx, tensor1.shape(), tensor2.dtype()); \
    }

// namespace {
// diopiError_t diopiTensorPermote(diopiContextHandle_t ctx, DiopiTensor &dstTensor, DiopiTensor srcTensor, std::vector<int64_t> permAxis) {
//     if (!dstTensor.defined()) {
//         std::vector<int64_t> srcShapeT64(srcTensor.shape().size());
//         for (int i = 0; i < srcTensor.shape().size(); ++i) {
//             srcShapeT64[i] = srcTensor.shape()[permAxis[i]];
//         }
//         diopiSize_t srcTShape(srcShapeT64.data(), srcShapeT64.size());
//         auto dstHandle = dstTensor.tensorHandle();
//         DIOPI_CALL(diopiRequireTensor(ctx, &dstHandle, &srcTShape, nullptr, srcTensor.dtype(), diopi_device));
//         dstTensor = DiopiTensor(dstHandle);
//     }
//     diopiSize_t axisSize(permAxis.data(), 4);
//     DIOPI_CALL(diopiPermute(ctx, dstTensor.tensorHandle(), srcTensor.tensorHandle(), axisSize));
//     return diopiSuccess;
// }

// }  // namespace

extern "C" diopiError_t diopiConvolution2d(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                                           diopiConstTensorHandle_t bias, diopiSize_t stride, diopiSize_t padding, diopiSize_t dilation, int64_t groups) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor outputTensor(out);
    DiopiTensor biasTensor(bias);

    DIOPI_CHECK(inputTensor.isContiguous(MemoryFormat::ChannelsLast), "inputTensor should be ChannelsLast");
    DIOPI_CHECK(weightTensor.isContiguous(MemoryFormat::ChannelsLast), "weightTensor should be ChannelsLast");
    DIOPI_CHECK(outputTensor.isContiguous(MemoryFormat::ChannelsLast), "outputTensor should be ChannelsLast");

    std::vector<DiopiTensor *> tensors{&inputTensor, &weightTensor, &biasTensor};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(outputTensorTmp, outputTensor, inputTensor.dtype());

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc outputDesc(outputTensorTmp, CNNL_LAYOUT_NHWC);

    CnnlTensorDesc biasDesc;
    if (biasTensor.defined()) {
        DIOPI_CALL(biasDesc.set(biasTensor, CNNL_LAYOUT_ARRAY));
    }

    std::vector<int> strideVec{stride.data, stride.data + stride.len};
    std::vector<int> paddingVec{padding.data, padding.data + padding.len};
    std::vector<int> dilationVec{dilation.data, dilation.data + dilation.len};

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> convDesc;

    int paddingTmp[4] = {paddingVec[0], paddingVec[0], paddingVec[1], paddingVec[1]};
    int strideTmp[2] = {strideVec[0], strideVec[1]};
    int dilationTmp[2] = {dilationVec[0], dilationVec[1]};

    cnnlDataType_t computeType;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&computeType, inputTensor.dtype()));
    DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(convDesc.get(), 4, paddingTmp, strideTmp, dilationTmp, groups, computeType));

    size_t workspaceSize;
    // CNNL_CONVOLUTION_FWD_ALGO_GEMM CNNL_CONVOLUTION_FWD_ALGO_DIRECT
    DIOPI_CALLCNNL(cnnlGetConvolutionForwardWorkspaceSize(
        handle, inputDesc.get(), weightDesc.get(), outputDesc.get(), biasDesc.get(), convDesc.get(), CNNL_CONVOLUTION_FWD_ALGO_DIRECT, &workspaceSize));

    void *workspace = nullptr;
    if (0 != workspaceSize) {
        workspace = requiresBuffer(ctx, workspaceSize).data();
    }
    DIOPI_CALLCNNL(cnnlConvolutionForward(handle,
                                          convDesc.get(),
                                          CNNL_CONVOLUTION_FWD_ALGO_DIRECT,
                                          nullptr,
                                          inputDesc.get(),
                                          inputTensor.data(),
                                          weightDesc.get(),
                                          weightTensor.data(),
                                          biasTensor.defined() ? biasDesc.get() : nullptr,
                                          biasTensor.defined() ? biasTensor.data() : nullptr,
                                          workspace,
                                          workspaceSize,
                                          nullptr,
                                          outputDesc.get(),
                                          outputTensorTmp.data()));
    DIOPI_CALL(dataTypeCast(ctx, outputTensor, outputTensorTmp));
    return diopiSuccess;
}

extern "C" diopiError_t diopiConvolution2dBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight,
                                                   diopiTensorHandle_t grad3, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                                   diopiConstTensorHandle_t weight, diopiSize_t *biasSizes, diopiSize_t stride, diopiSize_t padding,
                                                   diopiSize_t dilation, bool transposed, diopiSize_t outputPadding, int64_t groups) {
    if (!gradInput && !gradWeight && !grad3) {
        // do nothing
        return diopiSuccess;
    }
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    DiopiTensor weightTensor(weight);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradWeightTensor(gradWeight);
    DiopiTensor gradBiasTensor(grad3);
    DIOPI_CHECK(inputTensor.isContiguous(MemoryFormat::ChannelsLast), "inputTensor should be ChannelsLast");
    if (gradInputTensor.defined()) {
        DIOPI_CHECK(gradInputTensor.isContiguous(MemoryFormat::ChannelsLast), "gradInputTensor should be ChannelsLast");
    }
    std::vector<DiopiTensor *> tensors{&inputTensor, &weightTensor, &gradOutputTensor};

    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float16, diopi_dtype_float32}));

    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(gradWeightTensorTmp, gradWeightTensor, inputTensor.dtype());
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(gradInputTensorTmp, gradInputTensor, inputTensor.dtype());
    REQUIRES_TENSOR_BY_DTYPE_OR_NOT(gradBiasTensorTmp, gradBiasTensor), inputTensor.dtype();

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc weightDesc(weightTensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc gradOutputDesc(gradOutputTensor, CNNL_LAYOUT_NHWC);
    CnnlTensorDesc weightGradDesc(gradWeightTensorTmp, CNNL_LAYOUT_NHWC);

    CnnlResourceGuard<cnnlConvolutionDescriptor_t, cnnlCreateConvolutionDescriptor, cnnlDestroyConvolutionDescriptor> convDesc;

    std::vector<int> strideVec{stride.data, stride.data + stride.len};
    std::vector<int> paddingVec{padding.data, padding.data + padding.len};
    std::vector<int> dilationVec{dilation.data, dilation.data + dilation.len};

    int paddingTmp[4] = {paddingVec[0], paddingVec[1], paddingVec[0], paddingVec[1]};
    int strideTmp[2] = {strideVec[0], strideVec[1]};
    int dilationTmp[2] = {dilationVec[0], dilationVec[1]};
    if (gradWeightTensor.defined()) {
        cnnlDataType_t computeType;
        DIOPI_CALL(CnnlDataType::convertToCnnlType(&computeType, inputTensor.dtype()));
        DIOPI_CALLCNNL(cnnlSetConvolutionDescriptor(convDesc.get(), 4, paddingTmp, strideTmp, dilationTmp, groups, computeType));

        size_t workspaceSizeFilter = 0;
        DIOPI_CALLCNNL(cnnlGetConvolutionBackwardFilterWorkspaceSize(
            handle, inputDesc.get(), gradOutputDesc.get(), weightDesc.get(), convDesc.get(), CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT, &workspaceSizeFilter));

        void *workspaceFilter = nullptr;
        if (workspaceSizeFilter != 0) {
            workspaceFilter = requiresBuffer(ctx, workspaceSizeFilter).data();
        }

        DIOPI_CALLCNNL(cnnlConvolutionBackwardFilter(handle,
                                                     nullptr,
                                                     inputDesc.get(),
                                                     inputTensor.data(),
                                                     gradOutputDesc.get(),
                                                     gradOutputTensor.data(),
                                                     convDesc.get(),
                                                     CNNL_CONVOLUTION_BWD_FILTER_ALGO_DIRECT,
                                                     workspaceFilter,
                                                     workspaceSizeFilter,
                                                     nullptr,
                                                     weightGradDesc.get(),
                                                     gradWeightTensorTmp.data()));
        DIOPI_CALL(dataTypeCast(ctx, gradWeightTensor, gradWeightTensorTmp));
    }

    if (gradInputTensor.defined()) {
        CnnlTensorDesc inputGradDesc(gradInputTensorTmp, CNNL_LAYOUT_NHWC);
        size_t workspaceSizeInput;
        DIOPI_CALLCNNL(cnnlGetConvolutionBackwardDataWorkspaceSize(
            handle, weightDesc.get(), gradOutputDesc.get(), convDesc.get(), inputGradDesc.get(), CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT, &workspaceSizeInput));

        void *workspaceInput;
        if (workspaceSizeInput != 0) {
            workspaceInput = requiresBuffer(ctx, workspaceSizeInput).data();
        }

        DIOPI_CALLCNNL(cnnlConvolutionBackwardData(handle,
                                                   nullptr,
                                                   weightDesc.get(),
                                                   weightTensor.data(),
                                                   gradOutputDesc.get(),
                                                   gradOutputTensor.data(),
                                                   convDesc.get(),
                                                   CNNL_CONVOLUTION_BWD_DATA_ALGO_DIRECT,
                                                   workspaceInput,
                                                   workspaceSizeInput,
                                                   nullptr,
                                                   inputGradDesc.get(),
                                                   gradInputTensorTmp.data()));

        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTensorTmp));
    }

    if (grad3 != nullptr) {
        DiopiTensor biasGradTensor(grad3);
        CnnlTensorDesc gradBiasDesc(gradBiasTensorTmp, CNNL_LAYOUT_ARRAY);
        std::vector<int64_t> biasShape{gradBiasTensorTmp.shape().begin(), gradBiasTensorTmp.shape().end()};
        biasSizes->data = biasShape.data();
        biasSizes->len = biasShape.size();
        size_t workspaceSizeBias;
        int channelAxis = 3;
        DIOPI_CALLCNNL(cnnlGetBiasAddBackwardWorkspaceSize(handle, gradOutputDesc.get(), gradBiasDesc.get(), channelAxis, &workspaceSizeBias))
        void *workspaceBias = nullptr;
        if (0 != workspaceSizeBias) {
            workspaceBias = requiresBuffer(ctx, workspaceSizeBias).data();
        }
        DIOPI_CALLCNNL(cnnlBiasAddBackward_v2(handle,
                                              gradOutputDesc.get(),
                                              gradOutputTensor.data(),
                                              channelAxis,
                                              gradBiasDesc.get(),
                                              gradBiasTensorTmp.data(),
                                              workspaceBias,
                                              workspaceSizeBias));
        DIOPI_CALL(dataTypeCast(ctx, gradBiasTensor, gradBiasTensorTmp))
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
