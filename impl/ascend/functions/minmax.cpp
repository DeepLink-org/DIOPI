/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiMax(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiTensorHandle_t maxIndices, diopiConstTensorHandle_t input, int64_t dim) {
    AclOpRunner<1, 2>("ArgMaxWithValue", ctx).setAttr<int>("dimension", static_cast<int>(dim)).addInput(input).addOutput(maxIndices).addOutput(max).run();
    return diopiSuccess;
}

diopiError_t diopiMaxAll(diopiContextHandle_t ctx, diopiTensorHandle_t max, diopiConstTensorHandle_t input) {
    diopiSize_t inS;
    diopiGetTensorShape(input, &inS);
    std::vector<int64_t> dimAllVector(inS.len);
    std::iota(std::begin(dimAllVector), std::end(dimAllVector), 0);
    diopiSize_t dimAll = vectorToDiopiSize(dimAllVector);
    AclOpRunner<2, 1>("ReduceMax", ctx).addInput(input).addConstInput(dimAll).addOutput(max).run();
    return diopiSuccess;
}

diopiError_t diopiMin(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiTensorHandle_t minIndices, diopiConstTensorHandle_t input, int64_t dim) {
    AclOpRunner<1, 2>("ArgMinWithValue", ctx).setAttr<int>("dimension", static_cast<int>(dim)).addInput(input).addOutput(minIndices).addOutput(min).run();
    return diopiSuccess;
}

diopiError_t diopiMinAll(diopiContextHandle_t ctx, diopiTensorHandle_t min, diopiConstTensorHandle_t input) {
    diopiSize_t inS;
    diopiGetTensorShape(input, &inS);
    std::vector<int64_t> dimAllVector(inS.len);
    std::iota(std::begin(dimAllVector), std::end(dimAllVector), 0);
    diopiSize_t dimAll = vectorToDiopiSize(dimAllVector);
    AclOpRunner<2, 1>("ReduceMin", ctx).addInput(input).addConstInput(dimAll).addOutput(min).run();
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
