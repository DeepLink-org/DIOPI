/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../aclnn/acl_scalar.hpp"
#include "../aclnn/adaptor.hpp"

namespace impl {
namespace ascend {

static int64_t getReduce(const char* reduce) {
    if (strcmp(reduce, "add") == 0) {
        return 1;
    } else if (strcmp(reduce, "multiply") == 0) {
        return 2;
    } else {  // replace
        return 0;
    }
}

diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src,
                          diopiConstTensorHandle_t index, const char* reduce) {
    AscendTensor inputTensor(input);
    if (!inputTensor.defined() || inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    int64_t reduction = getReduce(reduce);
    DIOPI_ASCEND_CALL_ACLNN(aclnnScatter, ctx, input, dim, index, src, reduction, out);
    return diopiSuccess;
}

diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index,
                             const char* reduce) {
    AscendTensor inputTensor(input);
    if (!inputTensor.defined() || inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    int64_t reduction = getReduce(reduce);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceScatter, ctx, input, dim, index, src, reduction);
    return diopiSuccess;
}

diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                diopiConstTensorHandle_t index, const char* reduce) {
    AscendTensor inputTensor(input);
    if (!inputTensor.defined() || inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    int64_t reduction = getReduce(reduce);
    DIOPI_ASCEND_CALL_ACLNN(aclnnScatterValue, ctx, input, dim, index, value, reduction, out);
    return diopiSuccess;
}

diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index,
                                   const char* reduce) {
    AscendTensor inputTensor(input);
    if (!inputTensor.defined() || inputTensor.numel() == 0) {
        return diopiSuccess;
    }

    int64_t reduction = getReduce(reduce);
    DIOPI_ASCEND_CALL_ACLNN(aclnnInplaceScatterValue, ctx, input, dim, index, value, reduction);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
