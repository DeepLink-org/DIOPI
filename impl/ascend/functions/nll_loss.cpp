/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

int64_t getReductionVal(diopiReduction_t reduction) {
    if (reduction == diopiReduction_t::ReductionNone) {
        return 0;
    } else if (reduction == diopiReduction_t::ReductionMean) {
        return 1;
    } else if (reduction == diopiReduction_t::ReductionSum) {
        return 2;
    } else {
        std::cout << "unsupport type:" << reduction << std::endl;
        std::abort();
    }
}

diopiError_t diopiNLLLoss0(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t target,
                          diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    AclTensor inputAcl(input), targetAcl(target);
    if (inputAcl.numel() == 0) {
        diopiScalar_t scalar = constructDiopiScalarT(inputAcl.getAscendTensor().dtype(), reduction == 1 ? std::numeric_limits<float>::quiet_NaN() : 0);
        DIOPI_CALL(diopiFill(ctx, out, &scalar));
        return diopiSuccess;
    }
    auto inputSize = inputAcl.getAscendTensor().shape();
    int64_t c = inputAcl.getAscendTensor().dim() == 1 ? inputSize[0] : inputSize[1];
    int64_t n = std::accumulate(inputSize.begin(), inputSize.end(), 1, std::multiplies<>()) / c;
    diopiTensorHandle_t weightCopy, totalWeight;
    if (weight) {
        weightCopy = const_cast<diopiTensorHandle_t>(weight);
    } else {
        diopiSize_t weightShape = arrayToDiopiSize(&c, 1);
        diopiRequireTensor(ctx, &weightCopy, &weightShape, nullptr, inputAcl.getAscendTensor().dtype(), diopi_device);
        fillTensor(ctx, weightCopy, static_cast<float>(1.0));
    }
    if (inputAcl.getAscendTensor().dim() == 1) {
        inputAcl.reshape({1, inputSize[0]});
    } else if (inputAcl.getAscendTensor().dim() == 2) {
        // it's OK
    } else {
        inputAcl.reshape({n, c});
    }
    targetAcl.reshape({targetAcl.numel()});
    auto totalWeightSizeVec = std::vector<int64_t>({1});
    auto totalWeightSize = vectorToDiopiSize(totalWeightSizeVec);
    diopiRequireTensor(ctx, &totalWeight, &totalWeightSize, nullptr, inputAcl.getAscendTensor().dtype(), diopi_device);
    AclTensor weightAcl(weightCopy), outAcl(out), totalWeightAcl(totalWeight);
    // if (outAcl.getAscendTensor().shape(0) != n) {
    outAcl.reshape({n});
    if (reduction == diopiReduction_t::ReductionNone && inputAcl.getAscendTensor().dim() != 1) {
        outAcl.reshape({inputAcl.getAscendTensor().shape(0)});
    }
    // }

    if (!inputAcl.defined()) {
        std::cout << "!inputAcl.defined()" << std::endl;
    } else {
        printContiguousTensor(ctx, inputAcl.getAscendTensor(), "inputAcl");
    }
    if (!targetAcl.defined()) {
        std::cout << "!targetAcl.defined()" << std::endl;
    } else {
        printContiguousTensor(ctx, targetAcl.getAscendTensor(), "targetAcl");
    }
    if (!weightAcl.defined()) {
        std::cout << "!weightAcl.defined()" << std::endl;
    } else {
        printContiguousTensor(ctx, weightAcl.getAscendTensor(), "weightAcl");
    }
    if (!outAcl.defined()) {
        std::cout << "!outAcl.defined()" << std::endl;
    } else {
        printContiguousTensor(ctx, outAcl.getAscendTensor(), "outAcl");
    }
    if (!totalWeightAcl.defined()) {
        std::cout << "!totalWeightAcl.defined()" << std::endl;
    } else {
        printContiguousTensor(ctx, totalWeightAcl.getAscendTensor(), "totalWeightAcl");
    }
    int64_t reductionVal = static_cast<int64_t>(reduction);
    reductionVal = getReductionVal(reduction);
    // reductionVal = 0;
    ACLNN_ADAPTOR(aclnnNLLLoss, ctx, inputAcl, targetAcl, weightAcl, reductionVal, ignoreIndex, outAcl, totalWeightAcl);

    std::cout << "~~~aclnnNLLLoss finished." << std::endl;
    return diopiSuccess;
}

diopiError_t diopiNLLLossBackward0(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target, diopiConstTensorHandle_t weight, diopiReduction_t reduction, int64_t ignoreIndex) {
    // test;
    std::cout << "come into diopiNLLLossBackward" << std::endl;
    AclTensor inputAcl(input), targetAcl(target);
    if (inputAcl.numel() == 0) {
        return diopiSuccess;
    }
    int64_t reductionVal = static_cast<int64_t>(reduction);
    reductionVal = getReductionVal(reduction);
    diopiTensorHandle_t out, totalWeight, weightCopy;
    makeTensorLike(ctx, &out, gradInput, inputAcl.getAscendTensor().dtype());
    auto totalWeightSizeVec = std::vector<int64_t>({1});
    auto totalWeightSize = vectorToDiopiSize(totalWeightSizeVec);
    diopiRequireTensor(ctx, &totalWeight, &totalWeightSize, nullptr, inputAcl.getAscendTensor().dtype(), diopi_device);
    diopiScalar_t scalar = constructDiopiScalarT(inputAcl.getAscendTensor().dtype(), targetAcl.numel());
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
        diopiRequireTensor(ctx, &weightCopy, &weightShape, nullptr, inputAcl.getAscendTensor().dtype(), diopi_device);
        fillTensor(ctx, weightCopy, static_cast<float>(1.0));
    }
    AclTensor gradInputAcl(gradInput), gradOutputAcl(gradOutput), weightAcl(weightCopy), outAcl(out), totalWeightAcl(totalWeight);
    auto inputSize = inputAcl.getAscendTensor().shape();
    int c = inputAcl.getAscendTensor().dim() == 1 ? inputSize[0] : inputSize[1];
    int n = std::accumulate(inputSize.begin(), inputSize.end(), 1, std::multiplies<>()) / c;
    if (inputAcl.getAscendTensor().dim() == 1) {
        inputAcl.reshape({1, inputSize[0]});
    } else if (inputAcl.getAscendTensor().dim() == 2) {
        // it's OK
    } else {
        inputAcl.reshape({n, c});
    }
    targetAcl.reshape({targetAcl.numel()});
    gradInputAcl.reshape(inputAcl.getAscendTensor().shape());
    if (reduction == diopiReduction_t::ReductionNone && inputAcl.getAscendTensor().dim() != 1) {
        outAcl.reshape({inputAcl.getAscendTensor().shape(0)});
    }

    // if (!inputAcl.defined()) {
    //     std::cout << "!inputAcl.defined()" << std::endl;
    // } else {
    //     printContiguousTensor(ctx, inputAcl.getAscendTensor(), "inputAcl");
    // }
    // if (!targetAcl.defined()) {
    //     std::cout << "!targetAcl.defined()" << std::endl;
    // } else {
    //     printContiguousTensor(ctx, targetAcl.getAscendTensor(), "targetAcl");
    // }
    // if (!weightAcl.defined()) {
    //     std::cout << "!weightAcl.defined()" << std::endl;
    // } else {
    //     printContiguousTensor(ctx, weightAcl.getAscendTensor(), "weightAcl");
    // }
    // if (!outAcl.defined()) {
    //     std::cout << "!outAcl.defined()" << std::endl;
    // } else {
    //     printContiguousTensor(ctx, outAcl.getAscendTensor(), "outAcl");
    // }
    // if (!totalWeightAcl.defined()) {
    //     std::cout << "!totalWeightAcl.defined()" << std::endl;
    // } else {
    //     printContiguousTensor(ctx, totalWeightAcl.getAscendTensor(), "totalWeightAcl");
    // }
    std::cout << "aclnnNLLLossBackward begin." << std::endl;

    ACLNN_ADAPTOR(aclnnNLLLossBackward, ctx, gradOutputAcl, inputAcl, targetAcl, weightAcl, reductionVal, ignoreIndex, totalWeightAcl, gradInputAcl);
    std::cout << "~~~aclnnNLLLossBackward finished." << std::endl;
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
