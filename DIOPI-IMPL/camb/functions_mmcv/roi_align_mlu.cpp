/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include <iostream>

#include "../cnnl_helper.hpp"
#include "../diopi_helper.hpp"
#include "../mlu_helper.hpp"

namespace impl {

namespace camb {

void kernelRoiAlign(cnrtDim3_t kDim, cnrtFunctionType_t kType, cnrtQueue_t queue, const cnrtDataType_t dType, const void *input, const void *rois,
                    const int channels, const bool aligned, const int pooledHeight, const int pooledWidth, const int inputHeight, const int inputWidth,
                    const int samplingRatio, const float spatialScale, const int numRois, void *output);

void kernelRoiAlignBackward(cnrtDim3_t kDim, cnrtFunctionType_t kType, cnrtQueue_t queue, const cnrtDataType_t dtype, const void *grads, const void *boxes,
                            void *gradsImage, const int boxesNum, const int hi, const int wi, const int c, const int no, const int ho, const int wo,
                            const float spatialScale, const int samplingRatio, const bool aligned);

}  // namespace camb

}  // namespace impl

extern "C" DIOPI_API diopiError_t diopiRoiAlignMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t output, diopiTensorHandle_t argmaxY,
                                                    diopiTensorHandle_t argmaxX, diopiConstTensorHandle_t input, diopiConstTensorHandle_t rois,
                                                    int64_t alignedHeight, int64_t alignedWidth, int64_t samplingRatio, int64_t poolMode, float spatialScale,
                                                    bool aligned) {
    auto inputTr = impl::camb::DiopiTensor(input);
    auto roisTr = impl::camb::DiopiTensor(rois);
    auto outputTr = impl::camb::DiopiTensor(output);
    auto argmaxYTr = impl::camb::DiopiTensor(argmaxY);
    auto argmaxXTr = impl::camb::DiopiTensor(argmaxX);

    auto memoryFormat = impl::camb::MemoryFormat::ChannelsLast;
    auto inputTensor = inputTr.contiguous(ctx, memoryFormat);
    cnnlHandle_t handle = impl::camb::cnnlHandlePool.get(ctx);
    cnnlTranspose(ctx, handle, inputTr, inputTensor, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

    auto numRois = roisTr.size(0);
    auto channels = inputTr.size(1);
    int height = inputTr.size(2);
    int width = inputTr.size(3);

    if (outputTr.numel() == 0) {
        auto dtype = inputTr.dtype();
        outputTr = impl::camb::requiresTensor(ctx, {numRois, channels, alignedHeight, alignedWidth}, dtype);
        diopiScalar_t scalar = {dtype, 0.0};
        if (impl::camb::DiopiDataType().isInteger(dtype)) scalar = {dtype, 0};
        diopiFill(ctx, diopiTensorHandle_t(outputTr), &scalar);
        return diopiSuccess;
    }

    auto outputTrTemp = impl::camb::requiresTensor(ctx, {numRois, channels, alignedHeight, alignedWidth}, inputTr.dtype());
    auto outputTrTmp = outputTrTemp.contiguous(ctx, memoryFormat);

    // get compute queue
    auto queue = impl::camb::getStream(ctx);

    cnrtJobType_t kType = CNRT_FUNC_TYPE_UNION1;
    cnrtDim3_t kDim;
    kDim.x = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    kDim.y = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
    kDim.z = 1;
    cnrtDataType_t dataType = impl::camb::dtype2CnrtDtype(inputTr.dtype());

    impl::camb::kernelRoiAlign(kDim,
                               kType,
                               queue,
                               dataType,
                               inputTensor.data(),
                               roisTr.data(),
                               channels,
                               aligned,
                               alignedHeight,
                               alignedWidth,
                               height,
                               width,
                               samplingRatio,
                               spatialScale,
                               numRois,
                               outputTrTmp.data());
    cnnlTranspose(ctx, handle, outputTrTmp, outputTr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);
    return diopiSuccess;
}

static int nearestPower2(int x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    return x;
}

extern "C" diopiError_t diopiRoiAlignBackwardMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput,
                                                  diopiConstTensorHandle_t rois, diopiConstTensorHandle_t argmaxY, diopiConstTensorHandle_t argmaxX,
                                                  int64_t alignedHeight, int64_t alignedWidth, int64_t samplingRatio, int64_t poolMode, float spatialScale,
                                                  bool aligned) {
    return diopiErrorOccurred;
    auto gradTr = impl::camb::DiopiTensor(gradOutput);
    auto roisTr = impl::camb::DiopiTensor(rois);
    auto argmaxYTr = impl::camb::DiopiTensor(argmaxY);
    auto argmaxXTr = impl::camb::DiopiTensor(argmaxX);
    auto gradInputTr = impl::camb::DiopiTensor(gradInput);

    int batchSize = gradInputTr.size(0);
    int channels = gradInputTr.size(1);
    int height = gradInputTr.size(2);
    int width = gradInputTr.size(3);

    auto memoryFormat = impl::camb::MemoryFormat::ChannelsLast;
    auto gradTrTmp = gradTr.contiguous(ctx, memoryFormat);
    cnnlHandle_t handle = impl::camb::cnnlHandlePool.get(ctx);
    cnnlTranspose(ctx, handle, gradTr, gradTrTmp, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

    auto dtype = gradTr.dtype();
    auto gradInputTrTmp = impl::camb::requiresTensor(ctx, {batchSize, channels, height, width}, dtype);
    diopiScalar_t scalar = {dtype, 0.0};
    if (impl::camb::DiopiDataType().isInteger(dtype)) {
        scalar = {dtype, 0};
    }
    diopiFill(ctx, diopiTensorHandle_t(gradInputTrTmp), &scalar);

    auto gradInputTr1 = gradInputTrTmp.contiguous(ctx, memoryFormat);
    cnnlTranspose(ctx, handle, gradInputTrTmp, gradInputTr1, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);

    int boxesNum = roisTr.size(0);
    int hi = gradTr.size(2);
    int wi = gradTr.size(3);
    int c = gradTr.size(1);

    int no = gradInputTr.size(0);
    int ho = gradInputTr.size(2);
    int wo = gradInputTr.size(3);

    // get compute queue
    auto queue = impl::camb::getStream(ctx);

    cnrtJobType_t kType = CNRT_FUNC_TYPE_UNION1;
    int needCore = nearestPower2(boxesNum);
    int unionNumber = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
    uint32_t dimX = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    uint32_t dimY = (needCore - 1) / dimX + 1;
    dimY = (dimY > unionNumber) ? unionNumber : dimY;
    cnrtDim3_t kDim = {dimX, dimY, 1};
    cnrtDataType_t kDtype = impl::camb::dtype2CnrtDtype(gradTr.dtype());

    impl::camb::kernelRoiAlignBackward(kDim,
                                       kType,
                                       queue,
                                       kDtype,
                                       gradTrTmp.data(),
                                       roisTr.data(),
                                       gradInputTr1.data(),
                                       boxesNum,
                                       hi,
                                       wi,
                                       c,
                                       no,
                                       ho,
                                       wo,
                                       spatialScale,
                                       samplingRatio,
                                       aligned);

    cnnlTranspose(ctx, handle, gradInputTr1, gradInputTr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);
    return diopiSuccess;
}
