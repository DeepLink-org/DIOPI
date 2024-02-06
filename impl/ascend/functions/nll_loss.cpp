/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    // test;
    std::cout << "come into diopiNLLLossBackward" << std::endl;
    AclTensor inputAcl(input), targetAcl(target);
    if (inputAcl.numel() == 0) {
        return diopiSuccess;
    }
    int64_t reductionVal = static_cast<int64_t>(reduction);
    diopiTensorHandle_t out, totalWeight, weightCopy;
    makeTensorLike(ctx, &out, gradInput, inputAcl.getAclTensor().dtype());
    auto totalWeightSizeVec = std::vector<int64_t>({1});
    auto totalWeightSize = vectorToDiopiSize(totalWeightSizeVec);
    diopiRequireTensor(ctx, &totalWeight, &totalWeightSize, nullptr, inputAcl.getAclTensor().dtype(), diopi_device);
    diopiScalar_t scalar = constructDiopiScalarT(inputAcl.getAclTensor().dtype(), targetAcl.numel());
    diopiFill(ctx, totalWeight, &scalar);
    if (weight) {
        weightCopy = const_cast<diopiTensorHandle_t>(weight);
    } else {
        // AscendTensor inputAt(input);
        diopiSize_t inputShape;
        diopiGetTensorShape(input, &inputShape);

        // weight shape is (C). C is number of classes
        int64_t weightDim[1];
        if (inputShape.len == 1)
            weightDim[0] = inputShape.data[0];
        else
            weightDim[0] = inputShape.data[1];
        diopiSize_t weightShape = arrayToDiopiSize(weightDim, 1);
        diopiRequireTensor(ctx, &weightCopy, &weightShape, nullptr, inputAcl.getAclTensor().dtype(), diopi_device);
        fillTensor(ctx, weightCopy, static_cast<float>(1.0));
    }
    AclTensor gradInputAcl(gradInput), gradOutputAcl(gradOutput), weightAcl(weightCopy), outAcl(out),
        totalWeightAcl(totalWeight);
    auto inputSize = inputAcl.getAclTensor().shape();
    int c = inputAcl.getAclTensor().dim() == 1 ? inputSize[0] : inputSize[1];
    int n = std::accumulate(inputSize.begin(), inputSize.end(), 1, std::multiplies<>()) / c;
    if (inputAcl.getAclTensor().dim() == 1) {
        inputAcl.reshape({1, inputSize[0]});
    } else if (inputAcl.getAclTensor().dim() == 2) {
        // it's OK
    } else {
        inputAcl.reshape({1, inputAcl.getAclTensor().shape(0)});
    }
    targetAcl.reshape({targetAcl.numel()});
    gradInputAcl.reshape(inputAcl.getAclTensor().shape());

    // if (!inputAcl.defined()) {
    //     std::cout << "!inputAcl.defined()" << std::endl;
    // } else {
    //     printContiguousTensor(ctx, inputAcl.getAclTensor(), "inputAcl");
    // }
    // if (!targetAcl.defined()) {
    //     std::cout << "!targetAcl.defined()" << std::endl;
    // } else {
    //     printContiguousTensor(ctx, targetAcl.getAclTensor(), "targetAcl");
    // }
    // if (!weightAcl.defined()) {
    //     std::cout << "!weightAcl.defined()" << std::endl;
    // } else {
    //     printContiguousTensor(ctx, weightAcl.getAclTensor(), "weightAcl");
    // }
    // if (!outAcl.defined()) {
    //     std::cout << "!outAcl.defined()" << std::endl;
    // } else {
    //     printContiguousTensor(ctx, outAcl.getAclTensor(), "outAcl");
    // }
    // if (!totalWeightAcl.defined()) {
    //     std::cout << "!totalWeightAcl.defined()" << std::endl;
    // } else {
    //     printContiguousTensor(ctx, totalWeightAcl.getAclTensor(), "totalWeightAcl");
    // }
    std::cout << "aclnnNLLLossBackward begin." << std::endl;


    ACLNN_ADAPTOR(aclnnNLLLossBackward, ctx, gradOutputAcl, inputAcl, targetAcl, weightAcl, reductionVal, ignoreIndex, totalWeightAcl, gradInputAcl);
    std::cout << "aclnnNLLLossBackward finished." << std::endl;
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
