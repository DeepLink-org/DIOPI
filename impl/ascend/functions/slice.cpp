/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/adaptor.hpp"
#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t nullOut, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end,
                        int64_t step) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnSlice, ctx, input, dim, start, end, step, nullOut);
    return diopiSuccess;
}

diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t inputSizes,
                                int64_t dim, int64_t start, int64_t end, int64_t step) {
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceZero, ctx, gradInput);

    if (start < 0) start += inputSizes.data[dim];
    if (end < 0) end += inputSizes.data[dim];

    diopiTensorHandle_t index;
    std::array<int64_t, 1> indexSize = {(end - start) / step};
    diopiSize_t indexShape = {indexSize.data(), 1};
    diopiRequireTensor(ctx, &index, &indexShape, nullptr, diopi_dtype_int64, diopi_device);

    diopiScalar_t startScalar = constructDiopiScalarT(diopi_dtype_int64, start);
    diopiScalar_t endScalar = constructDiopiScalarT(diopi_dtype_int64, end);
    diopiScalar_t stepScalar = constructDiopiScalarT(diopi_dtype_int64, step);
    DIOPI_ASCEND_CALL_ACLNN(aclnnArange, ctx, &startScalar, &endScalar, &stepScalar, index);

    diopiScalar_t one = constructDiopiScalarT(diopi_dtype_int64, 1);
    DIOPI_ASCEND_CALL_ACLNN(aclnnIndexAdd, ctx, gradInput, dim, index, gradOutput, &one, gradInput);

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
