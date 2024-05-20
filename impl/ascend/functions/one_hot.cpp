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
        diopiScalar_t maxScalar = constructDiopiScalarT(diopi_dtype_int64, -1);
        diopiTensorHandle_t maxTensor;
        makeTensorFromScalar(ctx, &maxScalar, &maxTensor, diopi_device);
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
    diopiScalar_t onValueScalar = constructDiopiScalarT(diopi_dtype_int64, 1);
    diopiScalar_t offValueScalar = constructDiopiScalarT(diopi_dtype_int64, 0);
    makeTensorFromScalar(ctx, &onValueScalar, &onValue, diopi_device);
    makeTensorFromScalar(ctx, &offValueScalar, &offValue, diopi_device);

    diopiSize_t inputShape;
    diopiGetTensorShape(input, &inputShape);

    DIOPI_ASCEND_CALL_ACLNN(aclnnOneHot, ctx, input, numClasses, onValue, offValue, inputShape.len, out);
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
