/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

void stdNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiGeneratorHandle_t generator) {
    AscendTensor outputAt(out);
    auto pair = getSeedAndOffset(ctx, generator, 10);
    diopiScalar_t seedScalar = constructDiopiScalarT(diopi_dtype_int64, pair.first);
    diopiTensorHandle_t seedTh;
    makeTensorFromScalar(ctx, &seedScalar, &seedTh);
    diopiScalar_t offsetScalar = constructDiopiScalarT(diopi_dtype_int64, pair.second);
    diopiTensorHandle_t offsetTh;
    makeTensorFromScalar(ctx, &offsetScalar, &offsetTh);
    diopiScalar_t alg = constructDiopiScalarT(diopi_dtype_int64, 1);
    AclOpRunner<4, 1>("StatelessRandomNormalV2", ctx)
        .addConstInput(outputAt.dim() == 0 ? std::vector<int64_t>{1} : outputAt.shape())
        .addConstInput(seedTh, false, ACL_UINT64)
        .addConstInput(offsetTh, false, ACL_UINT64)
        .addConstInput(alg, diopi_dtype_int32)
        .setAttr("dtype", getAclDataType(out))
        .addOutput(out)
        .run();
}

diopiError_t diopiNormal(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, double std, diopiGeneratorHandle_t generator) {
    stdNormal(ctx, out, generator);

    // N(0,1) --> N(mean,std)
    diopiScalar_t stdScalar = constructDiopiScalarT(diopi_dtype_float64, std);
    diopiMulInpScalar(ctx, out, &stdScalar);
    diopiScalar_t meanScalar = constructDiopiScalarT(diopi_dtype_float64, mean);
    diopiScalar_t alphaScalar = constructDiopiScalarT(diopi_dtype_float64, 1);
    diopiAddInpScalar(ctx, out, &meanScalar, &alphaScalar);
    return diopiSuccess;
}

diopiError_t diopiNormalInp(diopiContextHandle_t ctx, diopiTensorHandle_t inout, double mean, double std, diopiGeneratorHandle_t generator) {
    return diopiNormal(ctx, inout, mean, std, generator);
}

diopiError_t diopiNormalTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, diopiConstTensorHandle_t std,
                               diopiGeneratorHandle_t generator) {
    stdNormal(ctx, out, generator);

    // N(0,1) --> N(mean,std)
    diopiMulInp(ctx, out, std);
    diopiScalar_t alphaScalar = constructDiopiScalarT(diopi_dtype_float64, 1);
    diopiAddInp(ctx, out, mean, &alphaScalar);
    return diopiSuccess;
}

diopiError_t diopiNormalScalarTensor(diopiContextHandle_t ctx, diopiTensorHandle_t out, double mean, diopiConstTensorHandle_t std,
                                     diopiGeneratorHandle_t generator) {
    stdNormal(ctx, out, generator);

    // N(0,1) --> N(mean,std)
    diopiMulInp(ctx, out, std);
    diopiScalar_t meanScalar = constructDiopiScalarT(diopi_dtype_float64, mean);
    diopiScalar_t alphaScalar = constructDiopiScalarT(diopi_dtype_float64, 1);
    diopiAddInpScalar(ctx, out, &meanScalar, &alphaScalar);
    return diopiSuccess;
}

diopiError_t diopiNormalTensorScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t mean, double std,
                                     diopiGeneratorHandle_t generator) {
    stdNormal(ctx, out, generator);

    // N(0,1) --> N(mean,std)
    diopiScalar_t stdScalar = constructDiopiScalarT(diopi_dtype_float64, std);
    diopiMulInpScalar(ctx, out, &stdScalar);
    diopiScalar_t alphaScalar = constructDiopiScalarT(diopi_dtype_float64, 1);
    diopiAddInp(ctx, out, mean, &alphaScalar);
    return diopiSuccess;
}

}  // namespace ascend
}  // namespace impl
