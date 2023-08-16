/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

namespace {
aclDataType dtypeConvertor(diopiDtype_t type) {
    auto dtype = getAclDataType(type);
    if (dtype == ACL_BOOL) {
        return ACL_UINT8;
    }
    return dtype;
}

}  // namespace

DIOPI_API diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p,
                                    bool train) {
    if (train) {
        diopiTensorHandle_t maskTempTensor;

        int64_t numels;
        diopiGetTensorNumel(input, &numels);
        uint32_t length = (numels + 128 - 1) / 128 * 128;
        int64_t maskTempShape[1] = {length / 8};
        diopiSize_t maskTempSize{maskTempShape, 1};
        diopiRequireTensor(ctx, &maskTempTensor, &maskTempSize, nullptr, diopi_dtype_bool, diopi_device);
        diopiSize_t inputSize;
        diopiGetTensorShape(input, &inputSize);

        int64_t offsetList[2] = {0, 0};
        diopiSize_t offset{offsetList, 2};

        float prob = 1. - p;
        AclOpRunner<5, 1, dtypeConvertor>("StatelessDropOutGenMask", ctx)
            .addConstInput(inputSize)
            .addConstInput<float>(prob)
            .addConstInput<int>(0)
            .addConstInput<int>(0)
            .addConstInput(offset)
            .addOutput(maskTempTensor)
            .run();
        AclOpRunner<3, 1, dtypeConvertor>("DropOutDoMask", ctx).addInput(input, maskTempTensor).addConstInput<float>(prob).addOutput(out).run();

    } else {
        diopiCopyInp(ctx, input, out);
    }
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
