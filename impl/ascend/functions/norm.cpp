/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <vector>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiNorm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, const diopiScalar_t* p, diopiSize_t dim) {
    diopiSize_t size;
    diopiGetTensorShape(out, &size);
    diopiDtype_t dtype;
    diopiGetTensorDtype(input, &dtype);
    diopiTensorHandle_t resultTmp;
    diopiRequireTensor(ctx, &resultTmp, &size, nullptr, dtype, diopi_device);
    AclOpRunner<1, 1> runner("LpNormReduce", ctx);
    if (diopi_dtype_float32 != dtype) {
        diopiTensorHandle_t fp32;
        makeTensorLike(ctx, &fp32, input, diopi_dtype_float32);
        diopiCastDtype(ctx, fp32, input);
        runner.addInput(fp32);
    } else {
        runner.addInput(input);
    }
    std::vector<int> dimVec;
    for (size_t i = 0; i < dim.len; ++i) {
        dimVec.emplace_back(dim.data[i]);
    }
    // default value is 2
    int64_t pValue = (p ? getValue<int64_t>(p) : 2);
    runner.addOutput(resultTmp).setAttr("p", pValue).setAttr("axes", dimVec).setAttr("keepdim", false).setAttr<float>("epsilon", 0.0).run();

    AclOpRunner<1, 1>("LpNormUpdate", ctx).addInput(resultTmp).addOutput(out).setAttr("p", pValue).setAttr<float>("epsilon", 0.0).run();

    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
