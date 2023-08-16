/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    bool keepdim = true;
    diopiSize_t inS, outS;
    diopiGetTensorShape(input, &inS);
    diopiGetTensorShape(out, &outS);
    AclOpRunner<2, 1> runner("ReduceSum", ctx);
    runner.addInput(input);

    if (dim.getLen() > 0) {
        runner.addConstInput(dim);
    } else {
        std::vector<int64_t> dimAllVector(inS.getLen());
        std::iota(std::begin(dimAllVector), std::end(dimAllVector), 0);
        diopiSize_t dimAll{dimAllVector.data(), static_cast<int64_t>(dimAllVector.size())};
        runner.addConstInput(dimAll);
    }
    if (inS.getLen() != outS.getLen()) {
        keepdim = false;
    } else {
        for (int i = 0; i < inS.getLen(); i++) {
            if (inS.data[i] != outS.data[i]) {
                keepdim = false;
                break;
            }
        }
    }
    runner.setAttr<uint8_t>("keep_dims", keepdim).addOutput(out).run();
    return diopiSuccess;
}

diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    bool keepdim = true;
    diopiSize_t inS, outS;
    diopiGetTensorShape(input, &inS);
    diopiGetTensorShape(out, &outS);
    AclOpRunner<2, 1> runner("ReduceMean", ctx);
    runner.addInput(input);

    if (dim.getLen() > 0) {
        runner.addConstInput(dim);
    } else {
        std::vector<int64_t> dimAllVector(inS.getLen());
        std::iota(std::begin(dimAllVector), std::end(dimAllVector), 0);
        diopiSize_t dimAll{dimAllVector.data(), static_cast<int64_t>(dimAllVector.size())};
        runner.addConstInput(dimAll);
    }
    if (inS.getLen() != outS.getLen()) {
        keepdim = false;
    } else {
        for (int i = 0; i < inS.getLen(); i++) {
            if (inS.data[i] != outS.data[i]) {
                keepdim = false;
                break;
            }
        }
    }
    runner.setAttr<uint8_t>("keep_dims", keepdim).addOutput(out).run();
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
