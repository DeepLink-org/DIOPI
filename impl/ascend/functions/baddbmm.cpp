/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../ascend_tensor.hpp"
#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" {

DIOPI_API diopiError_t diopiBaddbmm(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t batch1,
                                    diopiConstTensorHandle_t batch2, double beta, double alpha) {
    diopiDtype_t outDtype;
    diopiGetTensorDtype(out, &outDtype);

    // adjust the input's and output's data type
    diopiDtype_t execType = diopi_dtype_float32;

    diopiTensorHandle_t inCopy;
    makeTensorLike(ctx, &inCopy, input, execType);
    diopiCastDtype(ctx, inCopy, input);
    AscendTensor as_inCopy(inCopy);

    diopiTensorHandle_t outCopy;
    makeTensorLike(ctx, &outCopy, out, execType);
    diopiCastDtype(ctx, outCopy, out);

    diopiTensorHandle_t batch1Copy;
    makeTensorLike(ctx, &batch1Copy, batch1, execType);
    diopiCastDtype(ctx, batch1Copy, batch1);

    diopiTensorHandle_t batch2Copy;
    makeTensorLike(ctx, &batch2Copy, batch2, execType);
    diopiCastDtype(ctx, batch2Copy, batch2);

    // get the size of batch1 * batch2
    AscendTensor asTensor1 = AscendTensor(batch1);
    AscendTensor asTensor2 = AscendTensor(batch2);
    std::vector<int64_t> batch1_shape = asTensor1.shape();
    std::vector<int64_t> batch2_shape = asTensor2.shape();
    std::vector<int64_t> vectorSize_BatchMatMulTensor = {batch1_shape[0], batch1_shape[1], batch2_shape[2]};

    // init a tensor according to the size of batch1 * batch2 ;
    diopiSize_t diopiSize_BatchMatMulTensor = vectorToDiopiSize(vectorSize_BatchMatMulTensor);
    AscendTensor as_BatchMatMulTensor;
    makeTensor(ctx, as_BatchMatMulTensor, &diopiSize_BatchMatMulTensor, diopi_dtype_float32, diopiDevice_t::diopi_device);

    // does batch1/batch2 need to transpose?
    bool isSelfT = false;
    bool isMat2T = false;

    // do batch1 times batch2 -> BatchMatMulTensor
    AclOpRunner<5, 1>("BatchMatMul", ctx)
        .addInput(batch1Copy)
        .addInput(batch2Copy)
        .addOutput(as_BatchMatMulTensor)
        .setAttr("adj_x1", isSelfT)
        .setAttr("adj_x2", isMat2T)
        .run();

    // init memory based on the size of alphaMulTensor and betaMulTensor
    AscendTensor alphaMulTensor;
    AscendTensor betaMulTensor;
    makeTensorLike(ctx, alphaMulTensor, as_BatchMatMulTensor, diopi_dtype_float32);
    makeTensorLike(ctx, betaMulTensor, as_inCopy, diopi_dtype_float32);

    diopiScalar_t alpha_scalar = {diopi_dtype_float32, alpha};
    diopiScalar_t beta_scalar = {diopi_dtype_float32, beta};

    // transform ascendTensor to diopiTensorHandle_t
    diopiTensorHandle_t diopiAlphaMulTensor = const_cast<diopiTensorHandle_t>(alphaMulTensor.tensorHandle());
    diopiTensorHandle_t diopiBateMulTensor = const_cast<diopiTensorHandle_t>(betaMulTensor.tensorHandle());
    diopiTensorHandle_t diopi_as_BatchMatMulTensor = const_cast<diopiTensorHandle_t>(as_BatchMatMulTensor.tensorHandle());

    //\alpha times BatchMatMulTensor -> alphaMulTensor and \beta times input -> betaMulTensor
    diopiMulScalar(ctx, diopiAlphaMulTensor, diopi_as_BatchMatMulTensor, &alpha_scalar);
    diopiMulScalar(ctx, diopiBateMulTensor, inCopy, &beta_scalar);

    diopiScalar_t other;
    other.fval = 1;
    other.stype = outDtype;

    diopiAdd(ctx, outCopy, diopiAlphaMulTensor, diopiBateMulTensor, &other);
    diopiCastDtype(ctx, out, outCopy);

    return diopiSuccess;
}

DIOPI_API diopiError_t diopiBaddbmmInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t batch1, diopiConstTensorHandle_t batch2,
                                       double beta, double alpha) {
    diopiBaddbmm(ctx, input, input, batch1, batch2, beta, alpha);
    return diopiSuccess;
}
}
}  // namespace ascend
}  // namespace impl