/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <numeric>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    bool keepdim = true;
    diopiSize_t inS, outS;
    diopiGetTensorShape(input, &inS);
    diopiGetTensorShape(out, &outS);
    AclOpRunner<2, 1> runner("ReduceSum", ctx);
    runner.addInput(input);

    if (dim.len > 0) {
        runner.addConstInput(dim);
    } else {
        std::vector<int64_t> dimAllVector(inS.len);
        std::iota(std::begin(dimAllVector), std::end(dimAllVector), 0);
        diopiSize_t dimAll = vectorToDiopiSize(dimAllVector);
        runner.addConstInput(dimAll);
    }
    if (inS.len != outS.len) {
        keepdim = false;
    }
    runner.setAttr<uint8_t>("keep_dims", keepdim).addOutput(out).run();
    return diopiSuccess;
}

extern "C" diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    bool keepdim = true;
    diopiSize_t inS, outS;
    diopiGetTensorShape(input, &inS);
    diopiGetTensorShape(out, &outS);
    AclOpRunner<2, 1> runner("ReduceMean", ctx);
    runner.addInput(input);

    if (dim.len > 0) {
        runner.addConstInput(dim);
    } else {
        std::vector<int64_t> dimAllVector(inS.len);
        std::iota(std::begin(dimAllVector), std::end(dimAllVector), 0);
        diopiSize_t dimAll = vectorToDiopiSize(dimAllVector);
        runner.addConstInput(dimAll);
    }
    if (inS.len != outS.len) {
        keepdim = false;
    }
    runner.setAttr<uint8_t>("keep_dims", keepdim).addOutput(out).run();
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
