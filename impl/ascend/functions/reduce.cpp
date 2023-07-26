#include <diopi/functions.h>

#include <numeric>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiTensorHandle_t dimT = nullptr;
    bool keepdim = true;
    diopiSize_t inS, outS;
    diopiGetTensorShape(input, &inS);
    diopiGetTensorShape(out, &outS);
    if (dim.getLen() > 0) {
        makeTensorFromSize(ctx, &dim, &dimT);
    } else {
        std::vector<int64_t> v(inS.getLen());
        std::iota(std::begin(v), std::end(v), 0);
        diopiSize_t dim_(v.data(), v.size());
        makeTensorFromSize(ctx, &dim_, &dimT);
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
    AclOpRunner<2, 1>("ReduceSum").addInput(input).addConstInput(dimT, ACL_FORMAT_ND).setAttr<uint8_t>("keep_dims", keepdim).addOutput(out).run(ctx);
    return diopiSuccess;
}

extern "C" diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    diopiTensorHandle_t dimT = nullptr;
    bool keepdim = true;
    diopiSize_t inS, outS;
    diopiGetTensorShape(input, &inS);
    diopiGetTensorShape(out, &outS);
    if (dim.getLen() > 0) {
        makeTensorFromSize(ctx, &dim, &dimT);
    } else {
        std::vector<int64_t> v(inS.getLen());
        std::iota(std::begin(v), std::end(v), 0);
        diopiSize_t dim_(v.data(), v.size());
        makeTensorFromSize(ctx, &dim_, &dimT);
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
    AclOpRunner<2, 1>("ReduceMean").addInput(input).addConstInput(dimT, ACL_FORMAT_ND).setAttr<uint8_t>("keep_dims", keepdim).addOutput(out).run(ctx);
    return diopiSuccess;
}
}  // namespace ascend
}  // namespace impl
