/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {

diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1,
                          diopiConstTensorHandle_t batch2, double beta, double alpha) {
    diopiDtype_t outDtype;
    diopiGetTensorDtype(out, &outDtype);
    diopiDtype_t execType;

    // adjust the input's and output's data type
    if (outDtype == diopi_dtype_float64) {
        execType = diopi_dtype_float32;
    } else {
        execType = outDtype;
    }

    AscendTensor inputCopy(input);
    AscendTensor outputCopy(out);
    AscendTensor batch1Copy(batch1);
    AscendTensor batch2Copy(batch2);
    castTensor(ctx, outputCopy, execType);
    castTensor(ctx, batch1Copy, execType);
    castTensor(ctx, inputCopy, execType);
    castTensor(ctx, batch2Copy, execType);

    // get the size of batch1 * batch2
    AscendTensor asBatch1 = AscendTensor(batch1Copy);
    AscendTensor asBatch2 = AscendTensor(batch2Copy);
    std::vector<int64_t> batch1Shape = asBatch1.shape();
    std::vector<int64_t> batch2Shape = asBatch2.shape();
    std::vector<int64_t> vectorSizeBatchMatMulTensor = {batch1Shape[0], batch1Shape[1], batch2Shape[2]};

    // init a tensor according to the size of batch1 * batch2 ;
    diopiSize_t diopiSizeBatchMatMulTensor = vectorToDiopiSize(vectorSizeBatchMatMulTensor);
    AscendTensor asBatchMatMulTensor;
    makeTensor(ctx, asBatchMatMulTensor, &diopiSizeBatchMatMulTensor, execType, diopiDevice_t::diopi_device);

    // does batch1/batch2 need to transpose?
    bool isSelfT = false;
    bool isMat2T = false;

    // do batch1 times batch2 -> BatchMatMulTensor
    AclOpRunner<2, 1>("BatchMatMul", ctx)
        .addInput(batch1Copy)
        .addInput(batch2Copy)
        .addOutput(asBatchMatMulTensor)
        .setAttr("adj_x1", isSelfT)
        .setAttr("adj_x2", isMat2T)
        .run();

    // init memory based on the size of alphaMulTensor and betaMulTensor
    AscendTensor alphaMulTensor;
    AscendTensor betaMulTensor;
    makeTensorLike(ctx, alphaMulTensor, asBatchMatMulTensor, execType);
    makeTensorLike(ctx, betaMulTensor, inputCopy, execType);

    diopiScalar_t alphaScalar;
    alphaScalar.stype = execType;
    alphaScalar.fval = alpha;
    diopiScalar_t betaScalar;
    betaScalar.stype = execType;
    betaScalar.fval = beta;

    // transform ascendTensor to diopiTensorHandle_t
    diopiTensorHandle_t diopiAlphaMulTensor = const_cast<diopiTensorHandle_t>(alphaMulTensor.tensorHandle());
    diopiTensorHandle_t diopiBateMulTensor = const_cast<diopiTensorHandle_t>(betaMulTensor.tensorHandle());
    diopiTensorHandle_t diopiAsBatchMatMulTensor = const_cast<diopiTensorHandle_t>(asBatchMatMulTensor.tensorHandle());
    diopiTensorHandle_t diopiInputCopy = const_cast<diopiTensorHandle_t>(inputCopy.tensorHandle());

    // alpha times BatchMatMulTensor -> alphaMulTensor and beta times input -> betaMulTensor
    diopiMulScalar(ctx, diopiAlphaMulTensor, diopiAsBatchMatMulTensor, &alphaScalar);
    diopiMulScalar(ctx, diopiBateMulTensor, diopiInputCopy, &betaScalar);

    diopiScalar_t other;
    other.fval = 1;
    other.stype = outDtype;
    diopiTensorHandle_t diopiOutputCopy = const_cast<diopiTensorHandle_t>(outputCopy.tensorHandle());
    diopiAdd(ctx, diopiOutputCopy, diopiAlphaMulTensor, diopiBateMulTensor, &other);
    diopiCastDtype(ctx, out, diopiOutputCopy);

    return diopiSuccess;
}

diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta,
                             double alpha) {
    return diopiBaddbmm(ctx, input, input, batch1, batch2, beta, alpha);
}

}  // namespace ascend
}  // namespace impl
