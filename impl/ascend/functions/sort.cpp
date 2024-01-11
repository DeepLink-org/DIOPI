/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiSort(diopiContextHandle_t ctx, diopiTensorHandle_t values, diopiTensorHandle_t indices, diopiConstTensorHandle_t input, int64_t dim,
                       bool descending, const bool* pStable) {
    diopiSize_t inputShape;
    diopiGetTensorShape(input, &inputShape);

    int64_t inputSize = inputShape.len;
    int64_t lastdim = inputSize - 1;

    if (dim < 0) {
        dim = dim + inputSize;
    }

    if (dim != lastdim) {
        AscendTensor inputA(input);
        std::vector<int64_t> perms = inputA.shape();
        std::swap(perms[dim], perms[lastdim]);
        const diopiSize_t tranShape = vectorToDiopiSize(perms);

        diopiTensorHandle_t inputT, indicesT, valuesT;
        diopiRequireTensor(ctx, &inputT, &tranShape, nullptr, inputA.dtype(), inputA.device());
        diopiRequireTensor(ctx, &indicesT, &tranShape, nullptr, inputA.dtype(), inputA.device());
        diopiRequireTensor(ctx, &valuesT, &tranShape, nullptr, inputA.dtype(), inputA.device());

        diopiTranspose(ctx, inputT, input, dim, lastdim);
        AclOpRunner<1, 2>("Sort", ctx)
            .addInput(inputT)
            .setAttr("axis", lastdim)
            .setAttr("descending", descending)
            .setAttr("stable", nullptr == pStable ? false : *pStable)
            .addOutput(valuesT)
            .addOutput(indicesT)
            .run();

        diopiTranspose(ctx, indices, indicesT, dim, lastdim);
        diopiTranspose(ctx, values, valuesT, dim, lastdim);

    } else {
        AclOpRunner<1, 2>("Sort", ctx)
            .addInput(input)
            .setAttr("axis", dim)
            .setAttr("descending", descending)
            .setAttr("stable", nullptr == pStable ? false : *pStable)
            .addOutput(values)
            .addOutput(indices)
            .run();
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
