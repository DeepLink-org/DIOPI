/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cmath>
#include <vector>

#include "../common/common.hpp"

namespace impl {
namespace camb {

void bangAdamInternal(void* grad, void* m, void* v, void* vMax, void* variable, size_t sizes, int tensorNum, float beta1, float beta2, float epsilonCorrection,
                      float learningRateCorrection, int adamMode, float decay, float decayCorrection, cnrtDim3_t kDim, cnrtFunctionType_t kType,
                      cnrtQueue_t queue, cnrtDataType_t cnrtType, bool amsgrad);

diopiError_t bangAdam(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t grad, diopiTensorHandle_t expAvg,
                      diopiTensorHandle_t expAvgSq, diopiTensorHandle_t maxExpAvgSq, float lr, float beta1, float beta2, float eps, float weightDecay,
                      int64_t step, bool amsgrad, int adamMode = 0) {
    cnrtQueue_t queue = getStream(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor gradTensor(grad);
    DiopiTensor expAvgTensor(expAvg);
    DiopiTensor expAvgSqTensor(expAvgSq);
    DiopiTensor maxExpAvgSqTensor(maxExpAvgSq);

    if (amsgrad) {
        DIOPI_CALL(autoCastTensorType(
            ctx, {&inputTensor, &gradTensor, &expAvgTensor, &expAvgSqTensor, &maxExpAvgSqTensor}, {diopi_dtype_float16, diopi_dtype_float32}));
    } else {
        DIOPI_CALL(autoCastTensorType(ctx, {&inputTensor, &gradTensor, &expAvgTensor, &expAvgSqTensor}, {diopi_dtype_float16, diopi_dtype_float32}));
        // To temporarily bypass the bug in the camb kernel, apply for a fake tensor
        maxExpAvgSqTensor = requiresTensor(ctx, expAvgSqTensor.shape(), expAvgSqTensor.dtype());
    }

    float beta1CorrectionRecip = 1;
    float beta2CorrectionRecip = 1;
    beta1CorrectionRecip = 1 / (1 - std::pow(beta1, step));
    beta2CorrectionRecip = 1 / (1 - std::pow(beta2, step));
    float epsilonCorrection = eps / std::sqrt(beta2CorrectionRecip);
    float learningrateCorrection = lr * beta1CorrectionRecip / std::sqrt(beta2CorrectionRecip);

    float decayCorrection = 1 - lr * weightDecay;
    cnrtDim3_t kDim;
    int clusterCount = 0;
    int corePerCluster = 0;
    cnrtRet_t ret = cnrtDeviceGetAttribute(&clusterCount, cnrtAttrClusterCount, 0);
    if (ret != cnrtSuccess) {
        return diopiErrorOccurred;
    }
    ret = cnrtDeviceGetAttribute(&corePerCluster, cnrtAttrMcorePerCluster, 0);
    if (ret != cnrtSuccess) {
        return diopiErrorOccurred;
    }
    kDim.x = corePerCluster;
    kDim.y = clusterCount;
    kDim.z = 1;
    cnrtFunctionType_t kType = CNRT_FUNC_TYPE_UNION1;
    cnrtDataType_t cnrtType;
    if (inputTensor.dtype() == diopi_dtype_float32) {
        cnrtType = cnrtFloat32;
    } else {
        cnrtType = cnrtFloat16;
    }

    bangAdamInternal(gradTensor.data(),
                     expAvgTensor.data(),
                     expAvgSqTensor.data(),
                     maxExpAvgSqTensor.data(),
                     inputTensor.data(),
                     inputTensor.numel(),
                     1,
                     beta1,
                     beta2,
                     epsilonCorrection,
                     learningrateCorrection,
                     adamMode,
                     weightDecay,
                     decayCorrection,
                     kDim,
                     kType,
                     queue,
                     cnrtType,
                     amsgrad);

    return diopiSuccess;
}

diopiError_t diopiAdam(diopiContextHandle_t ctx, diopiTensorHandle_t param, diopiConstTensorHandle_t grad, diopiTensorHandle_t expAvg,
                       diopiTensorHandle_t expAvgSq, diopiTensorHandle_t maxExpAvgSq, float lr, float beta1, float beta2, float eps, float weightDecay,
                       int64_t step, bool amsgrad) {
    DIOPI_CALL(bangAdam(ctx, param, grad, expAvg, expAvgSq, maxExpAvgSq, lr, beta1, beta2, eps, weightDecay, step, amsgrad, 0));
    return diopiSuccess;
}

diopiError_t diopiAdamW(diopiContextHandle_t ctx, diopiTensorHandle_t param, diopiConstTensorHandle_t grad, diopiTensorHandle_t expAvg,
                        diopiTensorHandle_t expAvgSq,

                        diopiTensorHandle_t maxExpAvgSq, float lr, float beta1, float beta2, float eps, float weightDecay, int64_t step, bool amsgrad) {
    DIOPI_CALL(bangAdam(ctx, param, grad, expAvg, expAvgSq, maxExpAvgSq, lr, beta1, beta2, eps, weightDecay, step, amsgrad, 1));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
