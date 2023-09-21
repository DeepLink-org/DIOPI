/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions_mmcv.h>

#include <memory>

#include "../common/common.hpp"
#include "../diopi_helper.hpp"
#include "../mlu_ops_helper.hpp"

extern "C" DIOPI_API diopiError_t diopiNmsRotatedMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t dets,
                                                      diopiConstTensorHandle_t scores, diopiConstTensorHandle_t order, diopiConstTensorHandle_t detsSorted,
                                                      diopiConstTensorHandle_t labels, float iouThreshold, bool multiLabel) {
    mluOpHandle_t handle = impl::camb::mluOpHandlePool.get(ctx);

    impl::camb::DiopiTensor boxesTensor(dets);
    impl::camb::DiopiTensor scoresTensor(scores);
    impl::camb::DiopiTensor labelsTensor(labels);

    DIOPI_CHECK(boxesTensor.dtype() == diopi_dtype_float32 && scoresTensor.dtype() == diopi_dtype_float32, "mlu ops only support float32");
    DIOPI_CHECK(!multiLabel, "mlu ops currently does not support multi_label nms_rotated");
    // DIOPI_CHECK(labelsTensor.dim() == 2 || scoresTensor.dim() == 2, "labels' dim or scores' dim must equal to 2");

    if (boxesTensor.numel() == 0) {
        std::vector<int64_t> emptyShape{0};
        auto outputTensor = impl::camb::requiresTensor(ctx, emptyShape, diopi_dtype_int64);
        *out = diopiTensorHandle_t(outputTensor);
        return diopiSuccess;
    }

    int boxesNum = boxesTensor.size(0);
    auto outputTensor = impl::camb::requiresTensor(ctx, {boxesNum}, diopi_dtype_int32);
    auto outputSize = impl::camb::requiresTensor(ctx, {1}, diopi_dtype_int32);

    impl::camb::MluOpTensorDesc boxesDesc(boxesTensor, MLUOP_LAYOUT_ARRAY);
    impl::camb::MluOpTensorDesc scoresDesc(scoresTensor, MLUOP_LAYOUT_ARRAY);
    impl::camb::MluOpTensorDesc outputDesc(outputTensor, MLUOP_LAYOUT_ARRAY);

    size_t workspaceSize = 0;
    DIOPI_CALLMLUOP(mluOpGetNmsRotatedWorkspaceSize(handle, boxesDesc.get(), &workspaceSize));
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        workspace = impl::camb::requiresBuffer(ctx, workspaceSize).data();
    }

    DIOPI_CALLMLUOP(mluOpNmsRotated(handle,
                                    iouThreshold,
                                    boxesDesc.get(),
                                    boxesTensor.data(),
                                    scoresDesc.get(),
                                    scoresTensor.data(),
                                    workspace,
                                    workspaceSize,
                                    outputDesc.get(),
                                    outputTensor.data(),
                                    reinterpret_cast<int *>(outputSize.data())));

    int32_t outputNum;
    cnrtMemcpyAsync(&outputNum, outputSize.data(), sizeof(int32_t) * outputSize.numel(), impl::camb::getStream(ctx), cnrtMemcpyDevToHost);
    impl::camb::syncStreamInCtx(ctx);

    auto finalOutputTensor = impl::camb::requiresTensor(ctx, {outputNum}, outputTensor.dtype());
    DIOPI_CALL(impl::camb::diopiSlice(ctx, diopiTensorHandle_t(finalOutputTensor), diopiTensorHandle_t(outputTensor), 0, 0, outputNum, 1));
    DIOPI_CALL(impl::camb::dataTypeCast(ctx, finalOutputTensor, diopi_dtype_int64));
    *out = diopiTensorHandle_t(finalOutputTensor);
    return diopiSuccess;
}
