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
    // convert inputs to AscendTensor class
    contiguous(ctx, out);
    contiguous(ctx, input);
    AscendTensor inputCopy(input);
    AscendTensor outputCopy(out);
    AscendTensor weightCopy(weight);
    const std::vector<int64_t> outputPrimaryShape = outputCopy.shape();

    if (inputCopy.numel() == 0 || weightCopy.numel() == 0) {
        diopiScalar_t zero = constructDiopiScalarT(outputCopy.dtype(), 0.0);
        diopiFill(ctx, out, &zero);
        return diopiSuccess;
    }

    // mm's input matrix must be 2D, it needs to be converted if it isn't
    bool transTensorTo2DFalg = false;
    if (inputCopy.shape().size() > 2) {
        transTensorTo2D(ctx, inputCopy);
    }
    if (outputCopy.shape().size() > 2) {
        transTensorTo2DFalg = true;
        transTensorTo2D(ctx, outputCopy);
    }

    AclOpRunner<3, 1> runner("MatMulV2", ctx);
    runner.addInput(inputCopy).addInput(weightCopy).setAttr<uint8_t>("transpose_x1", false).setAttr<uint8_t>("transpose_x2", true).addOutput(outputCopy);

    // if bias is not nullptr, also add bias to input
    if (bias) {
        runner.addInput(bias);
    }
    runner.run();

    if (transTensorTo2DFalg) outputCopy.view(outputPrimaryShape);
    diopiTensorHandle_t diopiOutputCopy = const_cast<diopiTensorHandle_t>(outputCopy.tensorHandle());
    diopiCastDtype(ctx, out, diopiOutputCopy);

    return diopiSuccess;
}

diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                 diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    AscendTensor gradInputCopy(gradInput);
    AscendTensor gradWeightCopy(gradWeight);
    AscendTensor gradOutputCopy(gradOutput);
    AscendTensor inputCopy(input);
    AscendTensor weightCopy(weight);

    const std::vector<int64_t> gradInputPrimaryShape = gradInputCopy.shape();
    bool transTensorTo2DFalg = false;

    if (weightCopy.shape().size() > 2) transTensorTo2D(ctx, weightCopy);
    if (gradOutputCopy.shape().size() > 2) transTensorTo2D(ctx, gradOutputCopy);
    if (gradInputCopy.shape().size() > 2) {
        transTensorTo2DFalg = true;
        transTensorTo2D(ctx, gradInputCopy);
    }

    std::cout << "gradOutput shape: ";
    std::for_each(gradOutputCopy.shape().begin(), gradOutputCopy.shape().end(), [](const int& i) { std::cout << i << " "; });
    std::cout<<std::endl;
    std::cout << "weight shape: ";
    std::for_each(weightCopy.shape().begin(), weightCopy.shape().end(), [](const int& i) { std::cout << i << " "; });
    std::cout<<std::endl;
    std::cout<< "gradInputCopy shape";
    std::for_each(gradInputCopy.shape().begin(), gradInputCopy.shape().end(), [](const int& i) { std::cout << i << " "; });
    std::cout<<std::endl;

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

    if (transTensorTo2DFalg) {
        gradInputCopy.view(gradInputPrimaryShape);
    }
    diopiTensorHandle_t diopiGradOutputCopy = const_cast<diopiTensorHandle_t>(gradOutputCopy.tensorHandle());
    diopiTensorHandle_t diopiGradInputCopy = const_cast<diopiTensorHandle_t>(gradInputCopy.tensorHandle());
    diopiTensorHandle_t diopiGradWeightCopy = const_cast<diopiTensorHandle_t>(gradWeightCopy.tensorHandle());
    if (gradBias) {
        std::vector<int64_t> dimVec(gradOutputCopy.shape().size() - 1);
        std::iota(std::begin(dimVec), std::end(dimVec), 0);
        diopiSize_t dim = vectorToDiopiSize(dimVec);
        diopiSum(ctx, gradBias, diopiGradOutputCopy, dim);
    }

    // return gradInputCopy、gradWeightCopy、gradBiasCopy
    diopiCastDtype(ctx, gradInput, diopiGradInputCopy);
    diopiCastDtype(ctx, gradWeight, diopiGradWeightCopy);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl