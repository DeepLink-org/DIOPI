/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiLinear(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight,
                         diopiConstTensorHandle_t bias) {
    AscendTensor inputCopy(input);
    AscendTensor outputCopy(out);
    AscendTensor biasCopy(bias);
    AscendTensor weightCopy(weight);
    diopiDtype_t inputDtype;
    inputDtype = inputCopy.dtype();
    diopiDtype_t execType = inputDtype;
    if (inputDtype == diopi_dtype_float64) execType = diopi_dtype_float32;
    AclOpRunner<3, 1> runner("MatMulV2", ctx);
    if (inputCopy.shape().size() > 2) transTensorTo2D(ctx, inputCopy);
    if (outputCopy.shape().size() > 2) transTensorTo2D(ctx, outputCopy);
    runner.addInput(inputCopy, execType).addInput(weightCopy, execType).setAttr<uint8_t>("transpose_x1", false).setAttr<uint8_t>("transpose_x2", true);
    if (bias) runner.addInput(biasCopy, execType);
    diopiCastDtype(ctx, out, const_cast<diopiTensorHandle_t>(outputCopy.tensorHandle()));
    runner.run();
    return diopiSuccess;
}

diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                 diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    AscendTensor gradInputCopy(gradInput);
    AscendTensor gradWeightCopy(gradWeight);
    AscendTensor gradBiasCopy(gradBias);
    AscendTensor gradOutputCopy(gradOutput);
    AscendTensor inputCopy(input);
    AscendTensor weightCopy(weight);
    diopiDtype_t execType = gradOutputCopy.dtype();

    if (execType == diopi_dtype_float64) execType = diopi_dtype_float32;
    transTensorTo2D(ctx, gradOutputCopy);

    if (weightCopy.shape().size() > 2) transTensorTo2D(ctx, weightCopy);
    if (gradOutputCopy.shape().size() > 2) transTensorTo2D(ctx, gradOutputCopy);
    if (gradInputCopy.shape().size() > 2) transTensorTo2D(ctx, gradInputCopy);
    AclOpRunner<2, 1>("MatMul", ctx)
        .addInput(gradOutputCopy)
        .addInput(weightCopy)
        .setAttr<uint8_t>("transpose_x1", false)
        .setAttr<uint8_t>("transpose_x2", false)
        .addOutput(gradInputCopy)
        .run();

    if (inputCopy.shape().size() > 2) transTensorTo2D(ctx, inputCopy);
    if (gradWeightCopy.shape().size() > 2) transTensorTo2D(ctx, gradWeightCopy);
    AclOpRunner<2, 1>("MatMul", ctx)
        .addInput(gradOutputCopy)
        .addInput(inputCopy)
        .setAttr<uint8_t>("transpose_x1", true)
        .setAttr<uint8_t>("transpose_x2", false)
        .addOutput(gradWeightCopy)
        .run();

    diopiTensorHandle_t gradBiasCopyDiopi = const_cast<diopiTensorHandle_t>(gradBiasCopy.tensorHandle());
    diopiTensorHandle_t gradOutputCopyDiopi = const_cast<diopiTensorHandle_t>(gradOutputCopy.tensorHandle());
    diopiTensorHandle_t gradInputCopyDiopi = const_cast<diopiTensorHandle_t>(gradInputCopy.tensorHandle());
    diopiTensorHandle_t gradWeightCopyDiopi = const_cast<diopiTensorHandle_t>(gradWeightCopy.tensorHandle());
    if (gradBias) {
        std::vector<int64_t> dimVec(gradOutputCopy.shape().size() - 1);
        std::iota(std::begin(dimVec), std::end(dimVec), 0);
        diopiSize_t dim = vectorToDiopiSize(dimVec);
        diopiSum(ctx, gradBiasCopyDiopi, gradOutputCopyDiopi, dim);
    }

    // return gradInputCopy、gradWeightCopy、gradBiasCopy
    diopiCastDtype(ctx, gradInput, gradInputCopyDiopi);
    diopiCastDtype(ctx, gradBias, gradBiasCopyDiopi);
    diopiCastDtype(ctx, gradWeight, gradWeightCopyDiopi);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
