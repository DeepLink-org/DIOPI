/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                         diopiConstTensorHandle_t bias) {
    diopiTensorHandle_t weightT;
    diopiSize_t weightShape;
    diopiGetTensorShape(weight, &weightShape);
    diopiDtype_t weightDtype;
    diopiGetTensorDtype(weight, &weightDtype);
    std::vector<int64_t> weightSize(weightShape.data, weightShape.data + weightShape.len);
    weightSize[weightShape.len - 1] = weightShape.data[weightShape.len - 2];
    weightSize[weightShape.len - 2] = weightShape.data[weightShape.len - 1];
    diopiSize_t weightTShape = {weightSize.data(), weightSize.size()};
    diopiRequireTensor(ctx, &weightT, &weightTShape, nullptr, weightDtype, diopi_device);
    std::vector<int64_t> dims = {weightShape.len - 1, weightShape.len - 2};
    DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, weight, dims, weightT);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMatmul, ctx, input, weightT, out, 0);

    if (nullptr != bias) {
        diopiScalar_t alpha = constructDiopiScalarT(diopi_dtype_int64, 1);
        DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceAdd, ctx, out, bias, &alpha);
    }

    return diopiSuccess;
}

diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                 diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    if (nullptr != gradInput) {
        DIOPI_ASCEND_CALL_ACLNN(aclnnMatmul, ctx, gradOutput, weight, gradInput, 0);
    }

    if (nullptr != gradWeight) {
        diopiTensorHandle_t gradOutputT;
        diopiSize_t gradOutputShape;
        diopiGetTensorShape(gradOutput, &gradOutputShape);
        diopiDtype_t gradOutputDtype;
        diopiGetTensorDtype(gradOutput, &gradOutputDtype);
        std::vector<int64_t> gradOutputSize(gradOutputShape.data, gradOutputShape.data + gradOutputShape.len);
        gradOutputSize[gradOutputShape.len - 1] = gradOutputShape.data[gradOutputShape.len - 2];
        gradOutputSize[gradOutputShape.len - 2] = gradOutputShape.data[gradOutputShape.len - 1];
        diopiSize_t gradOutputTShape = {gradOutputSize.data(), gradOutputSize.size()};
        diopiRequireTensor(ctx, &gradOutputT, &gradOutputTShape, nullptr, gradOutputDtype, diopi_device);
        std::vector<int64_t> dims = {gradOutputShape.len - 1, gradOutputShape.len - 2};
        DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, gradOutput, dims, gradOutputT);
        DIOPI_ASCEND_CALL_ACLNN(aclnnMatmul, ctx, gradOutputT, input, gradWeight, 0);
    }

    if (nullptr != gradBias) {
        diopiSize_t gradOutputSize;
        diopiGetTensorShape(gradOutput, &gradOutputSize);
        std::vector<int64_t> dims(gradOutputSize.len - 1);
        std::iota(std::begin(dims), std::end(dims), 0);

        diopiDtype_t biasDtype;
        diopiGetTensorDtype(gradBias, &biasDtype);
        aclDataType dtype = getAclDataType(biasDtype);
        DIOPI_ASCEND_CALL_ACLNN(aclnnReduceSum, ctx, gradOutput, dims, false, dtype, gradBias);
    }

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
