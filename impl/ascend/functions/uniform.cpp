/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <cfloat>
#include <cmath>
#include <limits>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

extern "C" {
DIOPI_API diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, diopiGeneratorHandle_t generator) {
    auto pair = getSeedAndOffset(ctx, generator, 10);
    uint64_t seedList[1] = {pair.first};
    uint64_t offsetList[2] = {0, static_cast<uint64_t>(pair.second)};
    int32_t alg = 1;
    diopiSize_t inputSize;
    diopiGetTensorShape(inout, &inputSize);
    diopiDtype_t dtype;
    diopiGetTensorDtype(inout, &dtype);
    std::vector<int64_t> seedDim{1}, offsetDim{2};
    AclOpRunner<4, 1>("StatelessRandomUniformV2", ctx)
        .addConstInput(inputSize)
        .addConstInput(seedList, sizeof(uint64_t) * 1, seedDim, ACL_FORMAT_ND, diopi_dtype_uint64)
        .addConstInput(offsetList, sizeof(uint64_t) * 2, offsetDim, ACL_FORMAT_ND, diopi_dtype_uint64)
        .addConstInput(alg, diopi_dtype_int32)
        .addOutput(inout)
        .setAttr("dtype", getAclDataType(dtype))
        .run();

    // output: U(0~1) --> U(from~to)
    diopiTensorHandle_t tmp;
    diopiScalar_t fromScalar{diopi_dtype_float64, {from}};
    diopiScalar_t toScalar{diopi_dtype_float64, {to}};
    diopiScalar_t zeroScalar{diopi_dtype_float64, {0}};
    makeTensorLike(ctx, &tmp, inout);
    diopiMulInpScalar(ctx, tmp, &fromScalar);
    diopiSubInpScalar(ctx, tmp, &fromScalar, &zeroScalar);
    diopiMulInpScalar(ctx, inout, &toScalar);
    diopiSubInp(ctx, inout, tmp, &zeroScalar);
    return diopiSuccess;
}
}  // extern "C"

}  // namespace ascend
}  // namespace impl
