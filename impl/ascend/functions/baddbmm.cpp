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

    AscendTensor inputAt(input);
    AscendTensor outputAt(out);
    AscendTensor batch1At(batch1);
    AscendTensor batch2At(batch2);

    // get the size of batch1 * batch2
    std::vector<int64_t> batch1Shape = batch1At.shape();
    std::vector<int64_t> batch2Shape = batch2At.shape();
    std::vector<int64_t> vectorSizeBatchMatMulTensor = {batch1Shape[0], batch1Shape[1], batch2Shape[2]};

    // init a tensor according to the size of batch1 * batch2 ;
    diopiSize_t diopiSizeBatchMatMulTensor = vectorToDiopiSize(vectorSizeBatchMatMulTensor);
    AscendTensor batchMatMulTensorAt;
    makeTensor(ctx, batchMatMulTensorAt, &diopiSizeBatchMatMulTensor, outDtype, diopiDevice_t::diopi_device);

    // does batch1/batch2 need to transpose?
    bool isSelfT = false;
    bool isMat2T = false;

    // do batch1 times batch2 -> BatchMatMulTensor
    AclOpRunner<2, 1>("BatchMatMul", ctx)
        .addInput(batch1At)
        .addInput(batch2At)
        .addOutput(batchMatMulTensorAt)
        .setAttr("adj_x1", isSelfT)
        .setAttr("adj_x2", isMat2T)
        .run();

    // init memory based on the size of alphaMulTensor and betaMulTensor
    AscendTensor alphaMulTensor;
    AscendTensor betaMulTensor;
    makeTensorLike(ctx, alphaMulTensor, batchMatMulTensorAt, outDtype);
    makeTensorLike(ctx, betaMulTensor, inputAt, outDtype);

    diopiScalar_t alphaScalar = constructDiopiScalarT(outDtype, alpha);
    diopiScalar_t betaScalar = constructDiopiScalarT(outDtype, beta);

    // transform ascendTensor to diopiTensorHandle_t
    diopiTensorHandle_t diopiAlphaMulTensor = const_cast<diopiTensorHandle_t>(alphaMulTensor.tensorHandle());
    diopiTensorHandle_t diopiBateMulTensor = const_cast<diopiTensorHandle_t>(betaMulTensor.tensorHandle());
    diopiTensorHandle_t diopiAsBatchMatMulTensor = const_cast<diopiTensorHandle_t>(batchMatMulTensorAt.tensorHandle());
    diopiTensorHandle_t diopiInput = const_cast<diopiTensorHandle_t>(inputAt.tensorHandle());

    // alpha times BatchMatMulTensor -> alphaMulTensor and beta times input -> betaMulTensor
    diopiMulScalar(ctx, diopiAlphaMulTensor, diopiAsBatchMatMulTensor, &alphaScalar);
    diopiMulScalar(ctx, diopiBateMulTensor, diopiInput, &betaScalar);

    diopiScalar_t otherScalar = constructDiopiScalarT(outDtype, 1);
    diopiTensorHandle_t diopiOutput = const_cast<diopiTensorHandle_t>(outputAt.tensorHandle());
    diopiAdd(ctx, diopiOutput, diopiAlphaMulTensor, diopiBateMulTensor, &otherScalar);
    return diopiSuccess;
}

diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2, double beta,
                             double alpha) {
    return diopiBaddbmm(ctx, input, input, batch1, batch2, beta, alpha);
}

}  // namespace ascend
}  // namespace impl
