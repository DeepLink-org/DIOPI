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
    diopiSize_t inputSize, outputSize;
    diopiGetTensorShape(input, &inputSize);
    diopiGetTensorShape(out, &outputSize);
    int64_t numel, numelOut, elemsize;
    diopiDtype_t dtype;
    diopiGetTensorNumel(input, &numel);
    diopiGetTensorNumel(out, &numelOut);
    diopiGetTensorElemSize(input, &elemsize);
    diopiGetTensorDtype(input, &dtype);

    AclOpRunner<3, 1> runner("MatMulV2", ctx);

    if (inputSize.len > 2) {
        const void* data;
        diopiGetTensorDataConst(input, &data);
        std::vector<int64_t> dims({1, inputSize.data[inputSize.len - 1]});
        for (int i = 0; i < inputSize.len - 1; i++) {
            dims[0] = dims[0] * inputSize.data[i];
        }
        runner.addInput(data, numel * elemsize, dims, ACL_FORMAT_ND, dtype);
    } else {
        runner.addInput(input);
    }
    runner.addInput(weight).setAttr<uint8_t>("transpose_x1", false).setAttr<uint8_t>("transpose_x2", true);

    if (outputSize.len > 2) {
        void* data;
        diopiGetTensorData(out, &data);
        std::vector<int64_t> dims({1, outputSize.data[outputSize.len - 1]});
        for (int i = 0; i < outputSize.len - 1; i++) {
            dims[0] = dims[0] * outputSize.data[i];
        }
        runner.addOutput(data, numelOut * elemsize, dims, ACL_FORMAT_ND, dtype);
    } else {
        runner.addOutput(out);
    }
    if (bias) runner.addInput(bias);
    runner.run();
    return diopiSuccess;
}

diopiError_t diopiLinearBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiTensorHandle_t gradWeight, diopiTensorHandle_t gradBias,
                                 diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input, diopiConstTensorHandle_t weight) {
    diopiSize_t inputSize, gradOutSize;
    diopiGetTensorShape(input, &inputSize);
    diopiGetTensorShape(gradOutput, &gradOutSize);
    int64_t numel, numelGradOut, elemsize, elemSizeGrad;
    diopiDtype_t dtype, dtypeGrad;
    diopiGetTensorNumel(input, &numel);
    diopiGetTensorNumel(gradOutput, &numelGradOut);
    diopiGetTensorElemSize(input, &elemsize);
    diopiGetTensorElemSize(gradInput, &elemSizeGrad);
    diopiGetTensorDtype(input, &dtype);
    diopiGetTensorDtype(gradInput, &dtypeGrad);

    std::vector<int64_t> dimsGradOut({gradOutSize.data[gradOutSize.len - 2], gradOutSize.data[gradOutSize.len - 1]});
    for (int i = 0; i < gradOutSize.len - 2; i++) {
        dimsGradOut[0] = dimsGradOut[0] * gradOutSize.data[i];
    }
    const void* dataGrad;
    diopiGetTensorDataConst(gradOutput, &dataGrad);

    std::vector<int64_t> dims({inputSize.data[inputSize.len - 2], inputSize.data[inputSize.len - 1]});

    AclOpRunner<2, 1> runner("MatMulV2", ctx);
    runner.addInput(dataGrad, numelGradOut * elemSizeGrad, dimsGradOut, ACL_FORMAT_ND, dtypeGrad)
        .addInput(weight)
        .setAttr<uint8_t>("transpose_x1", false)
        .setAttr<uint8_t>("transpose_x2", false);
    if (inputSize.len > 2) {
        void* data;
        diopiGetTensorData(gradInput, &data);
        for (int i = 0; i < inputSize.len - 2; i++) {
            dims[0] = dims[0] * inputSize.data[i];
        }
        runner.addOutput(data, numel * elemSizeGrad, dims, ACL_FORMAT_ND, dtypeGrad);
    } else {
        runner.addOutput(gradInput);
    }
    runner.run();

    AclOpRunner<2, 1> runner2("MatMulV2", ctx);
    runner2.addInput(dataGrad, numelGradOut * elemSizeGrad, dimsGradOut, ACL_FORMAT_ND, dtypeGrad)
        .setAttr<uint8_t>("transpose_x1", true)
        .setAttr<uint8_t>("transpose_x2", false)
        .addOutput(gradWeight);
    if (inputSize.len > 2) {
        const void* data;
        diopiGetTensorDataConst(input, &data);
        runner2.addInput(data, numel * elemsize, dims, ACL_FORMAT_ND, dtype);
    } else {
        runner2.addInput(input);
    }
    runner2.run();

    if (gradBias) {
        std::vector<int64_t> dimVec(gradOutSize.len - 1);
        std::iota(std::begin(dimVec), std::end(dimVec), 0);
        diopiSize_t dim = vectorToDiopiSize(dimVec);
        diopiSum(ctx, gradBias, gradOutput, dim);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
