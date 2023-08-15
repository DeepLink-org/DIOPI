/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <numeric>
#include <vector>

#include "../common/common.hpp"

namespace impl {
namespace camb {

std::vector<int64_t> inferSize(std::vector<int64_t> a, std::vector<int64_t> b) {
    int32_t dimsA = a.size();
    int32_t dimsB = b.size();
    int32_t ndim = dimsA > dimsB ? dimsA : dimsB;
    std::vector<int64_t> expandedSize(ndim);
    for (auto i = ndim - 1; i >= 0; --i) {
        auto offset = ndim - 1 - i;
        auto dimA = dimsA - 1 - offset;
        auto dimB = dimsB - 1 - offset;
        auto sizeA = (dimA >= 0) ? a[dimA] : 1;
        auto sizeB = (dimB >= 0) ? b[dimB] : 1;
        assert((sizeA == sizeB || sizeA == 1 || sizeB == 1) && "The size of tensor a must match the size of tensor b at a non-singleton dimension");
        expandedSize[i] = sizeA == 1 ? sizeB : sizeA;
    }
    return expandedSize;
}

diopiError_t expand(diopiContextHandle_t ctx, DiopiTensor inputTensor, DiopiTensor outTensor) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensorTmp = outTensor;
    if (inputTensor.dtype() != outTensor.dtype()) {
        outTensorTmp = requiresTensor(ctx, outTensor.shape(), inputTensor.dtype());
    }

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlExpand(handle, inputDesc.get(), inputTensor.data(), outDesc.get(), outTensorTmp.data()));
    if (outTensor.dtype() != outTensorTmp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    }
    return diopiSuccess;
}

diopiError_t diopiCdist(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p,
                        const int64_t *computeMode) {
    DIOPI_CHECK(p == 1.0, "Currently only 1-norm is supported by cnnl");
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensor(out);
    DiopiTensor input1Tensor(input1);
    DiopiTensor input2Tensor(input2);

    std::vector<DiopiTensor *> tensors{&input1Tensor, &input2Tensor};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float32}));

    DiopiTensor outTensorTmp = outTensor;
    if (outTensor.dtype() != input1Tensor.dtype()) {
        outTensorTmp = requiresTensor(ctx, outTensor.shape(), input1Tensor.dtype());
    }

    int64_t dim1 = input1Tensor.dim();
    int64_t dim2 = input2Tensor.dim();
    int64_t c1 = input1Tensor.shape()[dim1 - 1];
    int64_t c2 = input2Tensor.shape()[dim2 - 1];
    int64_t r1 = input1Tensor.shape()[dim1 - 2];
    int64_t r2 = input2Tensor.shape()[dim2 - 2];

    if (r1 == 0 || r2 == 0) {
        return diopiSuccess;
    } else if (c1 == 0) {
        diopiScalar_t value = constructDiopiScalarT(outTensor.dtype(), 0);
        DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(outTensor), &value));
        return diopiSuccess;
    }

    std::vector<int64_t> batchInput1Tensor(input1Tensor.shape().begin(), input1Tensor.shape().begin() + dim1 - 2);
    std::vector<int64_t> batchInput2Tensor(input2Tensor.shape().begin(), input2Tensor.shape().begin() + dim2 - 2);
    std::vector<int64_t> expandBatchPortion = inferSize(batchInput1Tensor, batchInput2Tensor);
    std::vector<int64_t> input1TensorExpandSize(expandBatchPortion);
    input1TensorExpandSize.insert(input1TensorExpandSize.end(), {r1, c1});
    std::vector<int64_t> input2TensorExpandSize(expandBatchPortion);
    input2TensorExpandSize.insert(input2TensorExpandSize.end(), {r2, c2});

    int64_t expandBatchProduct = std::accumulate(expandBatchPortion.begin(), expandBatchPortion.end(), 1LL, std::multiplies<>());

    std::vector<int64_t> input1ShapeCnnl{expandBatchProduct, r1, c1};
    DiopiTensor input1TensorExpand = requiresTensor(ctx, input1TensorExpandSize, input1Tensor.dtype());
    DIOPI_CALL(expand(ctx, input1Tensor, input1TensorExpand));
    input1TensorExpand.reshape(input1ShapeCnnl);

    std::vector<int64_t> input2ShapeCnnl{expandBatchProduct, r2, c2};
    DiopiTensor input2TensorExpand = requiresTensor(ctx, input2TensorExpandSize, input2Tensor.dtype());
    DIOPI_CALL(expand(ctx, input2Tensor, input2TensorExpand));
    input2TensorExpand.reshape(input2ShapeCnnl);

    std::vector<int64_t> outputShape = outTensor.shape();
    std::vector<int64_t> outputShapeCnnl{expandBatchProduct, r1, r2};
    outTensorTmp.reshape(outputShapeCnnl);

    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc input1Desc(input1TensorExpand, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc input2Desc(input2TensorExpand, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlCdistForward(
        handle, input1Desc.get(), input1TensorExpand.data(), input2Desc.get(), input2TensorExpand.data(), p, outDesc.get(), outTensorTmp.data()));
    outTensorTmp.reshape(outputShape);
    if (outTensor.dtype() != outTensorTmp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    }
    return diopiSuccess;
}

diopiError_t diopiCdistBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiConstTensorHandle_t input1,
                                diopiConstTensorHandle_t input2, double p, diopiConstTensorHandle_t cdist) {
    DIOPI_CHECK(p == 1.0, "Currently only 1-norm is supported by cnnl");

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradOutputTensor(gradOutput);
    DiopiTensor input1Tensor(input1);
    DiopiTensor input2Tensor(input2);
    DiopiTensor cdistTensor(cdist);

    std::vector<DiopiTensor *> tensors{&gradOutputTensor, &input1Tensor, &input2Tensor, &cdistTensor};
    DIOPI_CALL(autoCastTensorType(ctx, tensors, {diopi_dtype_float32}));

    DiopiTensor gradInputTensorTmp = gradInputTensor;
    if (gradInputTensor.dtype() != input1Tensor.dtype()) {
        gradInputTensorTmp = requiresTensor(ctx, gradInputTensor.shape(), input1Tensor.dtype());
    }

    int64_t dim1 = input1Tensor.dim();
    int64_t dim2 = input2Tensor.dim();
    int64_t c1 = input1Tensor.shape()[dim1 - 1];
    int64_t c2 = input2Tensor.shape()[dim2 - 1];
    int64_t r1 = input1Tensor.shape()[dim1 - 2];
    int64_t r2 = input2Tensor.shape()[dim2 - 2];

    std::vector<int64_t> batchInput1Tensor(input1Tensor.shape().begin(), input1Tensor.shape().begin() + dim1 - 2);
    std::vector<int64_t> batchInput2Tensor(input2Tensor.shape().begin(), input2Tensor.shape().begin() + dim2 - 2);
    std::vector<int64_t> expandBatchPortion = inferSize(batchInput1Tensor, batchInput2Tensor);
    std::vector<int64_t> input1TensorExpandSize(expandBatchPortion);
    input1TensorExpandSize.insert(input1TensorExpandSize.end(), {r1, c1});
    std::vector<int64_t> input2TensorExpandSize(expandBatchPortion);
    input2TensorExpandSize.insert(input2TensorExpandSize.end(), {r2, c2});

    int64_t expandBatchProduct = std::accumulate(expandBatchPortion.begin(), expandBatchPortion.end(), 1LL, std::multiplies<>());

    // Gracefully handle empty Tensors
    if (r1 == 0 || r2 == 0 || c1 == 0 || expandBatchProduct == 0) {
        diopiScalar_t value = constructDiopiScalarT(gradInputTensor.dtype(), 0);
        DIOPI_CALL(diopiFill(ctx, diopiTensorHandle_t(gradInputTensor), &value));
        return diopiSuccess;
    }

    std::vector<int64_t> input1ShapeCnnl{expandBatchProduct, r1, c1};
    DiopiTensor input1TensorExpand = requiresTensor(ctx, input1TensorExpandSize, input1Tensor.dtype());
    DIOPI_CALL(expand(ctx, input1Tensor, input1TensorExpand));
    input1TensorExpand.reshape(input1ShapeCnnl);

    std::vector<int64_t> input2ShapeCnnl{expandBatchProduct, r2, c2};
    DiopiTensor input2TensorExpand = requiresTensor(ctx, input2TensorExpandSize, input2Tensor.dtype());
    DIOPI_CALL(expand(ctx, input2Tensor, input2TensorExpand));
    input2TensorExpand.reshape(input2ShapeCnnl);

    std::vector<int64_t> gradInputShapeCnnl{expandBatchProduct, r1, c1};
    gradInputTensorTmp.reshape(gradInputShapeCnnl);

    std::vector<int64_t> gradOutputShapeCnnl{expandBatchProduct, r1, r2};
    gradOutputTensor.reshape(gradOutputShapeCnnl);

    std::vector<int64_t> cdistShapeCnnl{expandBatchProduct, r1, r2};
    cdistTensor.reshape(cdistShapeCnnl);

    CnnlTensorDesc gradInputDesc(gradInputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradOutputDesc(gradOutputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc input1Desc(input1TensorExpand, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc input2Desc(input2TensorExpand, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc cdistDesc(cdistTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlCdistBackward(handle,
                                     input1Desc.get(),
                                     input1TensorExpand.data(),
                                     input2Desc.get(),
                                     input2TensorExpand.data(),
                                     cdistDesc.get(),
                                     cdistTensor.data(),
                                     gradOutputDesc.get(),
                                     gradOutputTensor.data(),
                                     p,
                                     gradInputDesc.get(),
                                     gradInputTensorTmp.data()));
    if (gradInputTensor.dtype() != gradInputTensorTmp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, gradInputTensorTmp));
    }

    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
