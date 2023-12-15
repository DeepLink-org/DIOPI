/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

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

void dropoutTrainCore(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p,
                      diopiGeneratorHandle_t generator) {
    AscendTensor inputAt(input);
    diopiTensorHandle_t maskTempTensor;
    uint32_t length = (inputAt.numel() + 128 - 1) / 128 * 128;
    int64_t maskTempShape[1] = {length / 8};
    diopiSize_t maskTempSize = arrayToDiopiSize(maskTempShape, 1);
    diopiRequireTensor(ctx, &maskTempTensor, &maskTempSize, nullptr, diopi_dtype_bool, diopi_device);

    diopiSize_t inputSize;
    diopiGetTensorShape(input, &inputSize);

    auto pair = getSeedAndOffset(ctx, generator, 10);
    int64_t offsetList[2] = {0, pair.second};
    diopiSize_t offset = arrayToDiopiSize(offsetList, 2);

    float prob = 1. - p;
    AclOpRunner<5, 1, dtypeConvertor>("StatelessDropOutGenMask", ctx)
        .addConstInput(inputSize)
        .addConstInput(prob, diopi_dtype_float32)
        .addConstInput(pair.first, diopi_dtype_int32)
        .addConstInput(0, diopi_dtype_int32)
        .addConstInput(offset)
        .addOutput(maskTempTensor)
        .run();

    diopiScalar_t oneScalar = constructDiopiScalarT(diopi_dtype_float64, 1);
    diopiTensorHandle_t oneTh;
    makeTensorFromScalar(ctx, &oneScalar, &oneTh, inputAt.dtype(), diopi_device);
    AclOpRunner<3, 1, dtypeConvertor>("DropOutDoMask", ctx).addInput(input).addInput(maskTempTensor).addInput(oneTh).addOutput(out).run();

    diopiEq(ctx, mask, input, out);
    diopiScalar_t probReciprocalScalar = constructDiopiScalarT(diopi_dtype_float64, 1. / prob);
    diopiMulInpScalar(ctx, out, &probReciprocalScalar);
}

diopiError_t diopiDropout(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t mask, diopiConstTensorHandle_t input, double p, bool train,
                          diopiGeneratorHandle_t generator) {
    if (train) {
        AscendTensor inputAt(input);
        if (inputAt.shape() != AscendTensor(mask).shape()) {
            diopiTensorHandle_t inputFor2d;
            makeOnesLike(ctx, &inputFor2d, mask, inputAt.dtype());
            diopiTensorHandle_t outFor2d;
            makeTensorLike(ctx, &outFor2d, mask, inputAt.dtype());
            dropoutTrainCore(ctx, outFor2d, mask, inputFor2d, p, generator);
            diopiMul(ctx, out, input, outFor2d);
        } else {
            dropoutTrainCore(ctx, out, mask, input, p, generator);
        }
    } else {
        diopiCopyInp(ctx, input, out);
    }
    return diopiSuccess;
}

diopiError_t diopiDropoutInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiTensorHandle_t mask, double p, bool train,
                             diopiGeneratorHandle_t generator) {
    if (train) {
        diopiTensorHandle_t inputCopy;
        makeTensorLike(ctx, &inputCopy, input);
        diopiCopyInp(ctx, input, inputCopy);
        diopiDropout(ctx, input, mask, inputCopy, p, train, generator);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
