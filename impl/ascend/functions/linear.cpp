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
    diopiSize_t weightSize;
    diopiGetTensorShape(weight, &weightSize);
    diopiDtype_t weightDtype;
    diopiGetTensorDtype(weight, &weightDtype);
    std::vector<int64_t> weightTShape(weightSize.data, weightSize.data + weightSize.len);
    weightTShape[weightSize.len - 1] = weightSize.data[weightSize.len - 2];
    weightTShape[weightSize.len - 2] = weightSize.data[weightSize.len - 1];
    diopiSize_t weightTSize = {weightTShape.data(), weightTShape.size()};
    diopiRequireTensor(ctx, &weightT, &weightTSize, nullptr, weightDtype, diopi_device);
    std::vector<int64_t> dims = {1, 0};
    DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, weight, dims, weightT);
    DIOPI_ASCEND_CALL_ACLNN(aclnnMatmul, ctx, input, weightT, out, 0);

    if (nullptr != bias) {
        diopiDtype_t outDtype;
        diopiGetTensorDtype(out, &outDtype);
        diopiScalar_t alpha = constructDiopiScalarT(outDtype, 1);
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
        AscendTensor input2D(input);
        if (input2D.dim() > 2) transTensorTo2D(ctx, input2D);
        AscendTensor gradOutput2D(gradOutput);
        if (gradOutput2D.dim() > 2) transTensorTo2D(ctx, gradOutput2D);

        diopiTensorHandle_t gradOutput2DT;
        std::vector<int64_t> gradOutput2DTShape = {gradOutput2D.shape()[1], gradOutput2D.shape()[0]};
        diopiSize_t gradOutput2DTSize = {gradOutput2DTShape.data(), gradOutput2DTShape.size()};
        diopiRequireTensor(ctx, &gradOutput2DT, &gradOutput2DTSize, nullptr, gradOutput2D.dtype(), diopi_device);

        std::vector<int64_t> dims = {1, 0};
        DIOPI_ASCEND_CALL_ACLNN(aclnnPermute, ctx, gradOutput2D, dims, gradOutput2DT);
        DIOPI_ASCEND_CALL_ACLNN(aclnnMatmul, ctx, gradOutput2DT, input2D, gradWeight, 0);
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
