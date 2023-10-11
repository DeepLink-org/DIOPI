/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <cmath>
#include <numeric>
#include <set>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern diopiError_t negativeInputRtnFillNan(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input);

diopiError_t diopiSum(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (0 == numel) {
        AclOpRunner<1, 1>("Fills", ctx).addInput(out).setAttr<float>("value", 0).addOutput(out).run();
        return diopiSuccess;
    }

    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    std::set<diopiDtype_t> typeSet{diopi_dtype_float16, diopi_dtype_float32};
    diopiTensorHandle_t inputTemp;
    diopiTensorHandle_t outTemp;
    if (typeSet.find(dtype) == typeSet.end()) {
        makeTensorLike(ctx, &inputTemp, input, diopi_dtype_float32);
        makeTensorLike(ctx, &outTemp, out, diopi_dtype_float32);
        diopiCastDtype(ctx, inputTemp, input);
    } else {
        inputTemp = const_cast<diopiTensorHandle_t>(input);
        outTemp = out;
    }

    bool keepdim = true;
    diopiSize_t inS, outS;
    diopiGetTensorShape(input, &inS);
    diopiGetTensorShape(out, &outS);
    AclOpRunner<2, 1> runner("ReduceSum", ctx);
    runner.addInput(inputTemp);

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
    runner.setAttr<uint8_t>("keep_dims", keepdim).addOutput(outTemp).run();
    if (typeSet.find(dtype) == typeSet.end()) {
        diopiCastDtype(ctx, out, outTemp);
    }
    return diopiSuccess;
}

diopiError_t diopiMean(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t dim) {
    int64_t numel = 0;
    diopiGetTensorNumel(input, &numel);
    if (0 == numel) {
        diopiScalar_t nanScalar = {diopi_dtype_float64, NAN};
        diopiFill(ctx, out, &nanScalar);
        return diopiSuccess;
    }

    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    std::set<diopiDtype_t> typeSet{diopi_dtype_float16, diopi_dtype_float32, diopi_dtype_int8, diopi_dtype_uint8};

    diopiTensorHandle_t inputTemp;
    diopiTensorHandle_t outTemp;
    if (typeSet.find(dtype) == typeSet.end()) {
        makeTensorLike(ctx, &inputTemp, input, diopi_dtype_float32);
        makeTensorLike(ctx, &outTemp, out, diopi_dtype_float32);
        diopiCastDtype(ctx, inputTemp, input);
    } else {
        inputTemp = const_cast<diopiTensorHandle_t>(input);
        outTemp = out;
    }

    bool keepdim = true;
    diopiSize_t inS, outS;
    diopiGetTensorShape(input, &inS);
    diopiGetTensorShape(out, &outS);
    AclOpRunner<2, 1> runner("ReduceMean", ctx);
    runner.addInput(inputTemp);

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
    runner.setAttr<uint8_t>("keep_dims", keepdim).addOutput(outTemp).run();
    if (typeSet.find(dtype) == typeSet.end()) {
        diopiCastDtype(ctx, out, outTemp);
    }
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
