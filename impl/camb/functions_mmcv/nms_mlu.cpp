/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include <memory>

#include "../common/common.hpp"
#include "../diopi_helper.hpp"
#include "../mlu_helper.hpp"

namespace impl {

namespace camb {

void kernelNms(cnrtDim3_t kDim, cnrtFunctionType_t kType, cnrtQueue_t queue, const cnrtDataType_t dataTypeInput, const void *boxesPtr, const void *scoresPtr,
               const int inputNumBoxes, const int maxOutputBoxes, const float iouThreshold, const float offset, void *workspacePtr, void *outputSizePtr,
               void *outputPtr);
}  // namespace camb

}  // namespace impl

int selectUnionType(uint32_t useJob, int boxNumPerCore) {
    // the boxNumPerCore should be at least 256, otherwise the real IO
    // bandwidth would be very low
    while (boxNumPerCore < 256 && useJob >= 4) {
        boxNumPerCore *= 2;
        useJob /= 2;
    }
    return useJob;
}

static cnnlStatus_t policyFunc(cnrtDim3_t *kDim, cnrtFunctionType_t *kType, int &coreNumPerClass, const int inputBoxNum) {
    uint32_t coreDim = impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    uint32_t clusterNumber = impl::camb::getDeviceAttr(cnrtAttrClusterCount);
    uint32_t jobLimit = impl::camb::getJobLimitCapability();
    uint32_t coreNumber = jobLimit;

    int boxNumPerCore = (inputBoxNum + coreNumber - 1) / coreNumber;
    int useJob = selectUnionType(jobLimit, boxNumPerCore);
    // initiate kType as Union1
    kDim->x = coreDim;
    kDim->y = 1;
    kDim->z = 1;
    *kType = CNRT_FUNC_TYPE_UNION1;
    switch (jobLimit) {
        case CN_KERNEL_CLASS_BLOCK:
        case CN_KERNEL_CLASS_UNION:
        case CN_KERNEL_CLASS_UNION2:
        case CN_KERNEL_CLASS_UNION4:
        case CN_KERNEL_CLASS_UNION8:
        case CN_KERNEL_CLASS_UNION16: {
            if (useJob < 4) {
                kDim->x = 1;
                *kType = CNRT_FUNC_TYPE_BLOCK;
            } else if (useJob == 4) {
                kDim->x = coreDim;
                *kType = CNRT_FUNC_TYPE_UNION1;
            } else {
                kDim->x = useJob;
                *kType = (cnrtFunctionType_t)useJob;
            }
        }; break;
        default:
            DIOPI_CHECK_ABORT(false,
                              "%s",
                              "[cnnlNms_v2]: got unsupported job limit number. Use "
                              "default CN_KERNEL_CLASS_UNION1 with UNION1 task.");
    }
    return CNNL_STATUS_SUCCESS;
}

extern "C" DIOPI_API diopiError_t diopiNmsMmcv(diopiContextHandle_t ctx, diopiTensorHandle_t *out, diopiConstTensorHandle_t boxesTr,
                                               diopiConstTensorHandle_t scoresTr, double iouThreshold, int64_t offset) {
    auto boxes = impl::camb::DiopiTensor(boxesTr);
    auto scores = impl::camb::DiopiTensor(scoresTr);

    if (boxes.numel() == 0) {
        diopiScalar_t scalar = impl::camb::constructDiopiScalarT(diopi_dtype_int64, 1);
        auto tempOut = impl::camb::requiresTensor(ctx, {1}, diopi_dtype_int64);
        DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(tempOut), &scalar));
        *out = diopiTensorHandle_t(tempOut);
        return diopiSuccess;
    }

    int inputNumBoxes = boxes.size(0);
    int maxOutputBoxes = boxes.size(0);

    cnrtDataType_t dataTypeInput = impl::camb::dtype2CnrtDtype(boxes.dtype());
    cnrtDim3_t kDim;
    cnrtJobType_t kType;

    int coreNumPerClass;
    policyFunc(&kDim, &kType, coreNumPerClass, inputNumBoxes);

    // transpose boxes (n, 4) to (4, n) for better performance
    auto boxesT = impl::camb::requiresTensor(ctx, {boxes.size(1), boxes.size(0)}, boxes.dtype());

    DIOPI_CALL(diopiTranspose(ctx, diopiTensorHandle_t(boxesT), diopiTensorHandle_t(boxes), 0, 1));
    auto output = impl::camb::requiresTensor(ctx, {maxOutputBoxes}, diopi_dtype_int32);
    auto outputSize = impl::camb::requiresTensor(ctx, {1}, diopi_dtype_int32);

    // workspace
    const int infoNum = 5;  // x1, x2, y1, y2 and score
    size_t spaceSize = 0;
    if (boxes.dtype() == diopi_dtype_float16) {
        spaceSize = inputNumBoxes * sizeof(int16_t) * infoNum + sizeof(float);
    } else {
        spaceSize = inputNumBoxes * sizeof(float) * infoNum + sizeof(float);
    }
#if __BANG_ARCH__ > 370
    int cluster_num = getCoreNumOfJobLimitCapability() / impl::camb::getDeviceAttr(cnrtAttrMcorePerCluster);
    spaceSize += clusterNumber * sizeof(float) * 7;
#endif
    auto workspace = impl::camb::requiresTensor(ctx, {static_cast<int64_t>(spaceSize)}, diopi_dtype_uint8);

    // get compute queue
    auto queue = impl::camb::getStream(ctx);

    impl::camb::kernelNms(kDim,
                          kType,
                          queue,
                          dataTypeInput,
                          boxesT.data(),
                          scores.data(),
                          inputNumBoxes,
                          maxOutputBoxes,
                          iouThreshold,
                          offset,
                          workspace.data(),
                          outputSize.data(),
                          output.data());

    int bytes = sizeof(int) * outputSize.numel();
    std::unique_ptr<char> outputSizeCpu(new char[bytes]);
    cnrtMemcpyAsync(outputSizeCpu.get(), outputSize.data(), bytes, impl::camb::getStream(ctx), cnrtMemcpyDevToHost);
    impl::camb::syncStreamInCtx(ctx);
    int outputNum = reinterpret_cast<int *>(outputSizeCpu.get())[0];

    auto tempOut = impl::camb::requiresTensor(ctx, {outputNum}, output.dtype());
    DIOPI_CALL(diopiSlice(ctx, diopiTensorHandle_t(tempOut), diopiTensorHandle_t(output), 0, 0, outputNum, 1));
    DIOPI_CALL(impl::camb::dataTypeCast(ctx, tempOut, diopi_dtype_int64));
    *out = diopiTensorHandle_t(tempOut);
    return diopiSuccess;
}
