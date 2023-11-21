/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiUniformInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double from, double to, diopiGeneratorHandle_t generator) {
    auto pair = getSeedAndOffset(ctx, generator, 10);
    diopiScalar_t seedScalar = constructDiopiScalarT(diopi_dtype_int64, pair.first);
    diopiTensorHandle_t seedTh;
    makeTensorFromScalar(ctx, &seedScalar, &seedTh);
    diopiScalar_t offsetScalar = constructDiopiScalarT(diopi_dtype_int64, pair.second);
    diopiTensorHandle_t offsetTh;
    makeTensorFromScalar(ctx, &offsetScalar, &offsetTh);
    diopiScalar_t alg = constructDiopiScalarT(diopi_dtype_int64, 1);
    AclOpRunner<4, 1>("StatelessRandomUniformV2", ctx)
        .addConstInput(AscendTensor(inout).shape())
        .addConstInput(seedTh, false, ACL_UINT64)
        .addConstInput(offsetTh, false, ACL_UINT64)
        .addConstInput(alg, diopi_dtype_int32)
        .setAttr("dtype", getAclDataType(inout))
        .addOutput(inout)
        .run();

    // StatelessRandomUniformV2 output: U(0~1) --> U(from~to)
    diopiScalar_t diffScalar = constructDiopiScalarT(diopi_dtype_float64, to - from);
    diopiMulInpScalar(ctx, inout, &diffScalar);
    diopiScalar_t fromScalar = constructDiopiScalarT(diopi_dtype_float64, from);
    diopiScalar_t alphaScalar = constructDiopiScalarT(diopi_dtype_float64, 1);
    diopiAddInpScalar(ctx, inout, &fromScalar, &alphaScalar);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
