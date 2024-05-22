/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiOneHot(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t numClasses) {
    if (-1 == numClasses) {
        diopiTensorHandle_t maxTensor;
        int64_t sizeTmp[1] = {1};
        diopiSize_t sSize = arrayToDiopiSize(sizeTmp, 1);
        diopiRequireTensor(ctx, &maxTensor, &sSize, nullptr, diopi_dtype_int64, diopi_device);
        DIOPI_ASCEND_CALL_ACLNN(aclnnMax, ctx, input, maxTensor);

        void* dataPtr = nullptr;
        diopiGetTensorData(maxTensor, &dataPtr);

        int64_t maxValue;
        diopiStreamHandle_t stream;
        diopiGetStream(ctx, &stream);
        aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream));
        aclrtMemcpy(&maxValue, sizeof(int64_t), dataPtr, sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
        numClasses = maxValue + 1;
    }

    diopiTensorHandle_t onValue, offValue;
    int64_t sizeTmp[1] = {1};
    diopiSize_t sSize = arrayToDiopiSize(sizeTmp, 1);
    diopiRequireTensor(ctx, &onValue, &sSize, nullptr, diopi_dtype_int64, diopi_device);
    diopiRequireTensor(ctx, &offValue, &sSize, nullptr, diopi_dtype_int64, diopi_device);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceOne, ctx, onValue);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, offValue);

    diopiSize_t inputShape;
    diopiGetTensorShape(input, &inputShape);

    DIOPI_ASCEND_CALL_ACLNN(aclnnOneHot, ctx, input, numClasses, onValue, offValue, inputShape.len, out);
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
