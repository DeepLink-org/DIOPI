/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, DeepLink.
 */

#include <pybind11/pybind11.h>
#include "litert.hpp"
#ifdef TEST_USE_ADAPTOR
#include <diopi_adaptors.hpp>
#endif
#include <diopi/diopirt.h>
namespace py = pybind11;

PYBIND11_MODULE(diopi_functions, m) {
    m.doc() = "pybind11 example-1 plugin"; // optional module docstring
    m.def("diopiGetVendorName", &diopiGetVendorName);
    m.def("diopiGetImplVersion", &diopiGetImplVersion);
    m.def("diopiGetVersion", &diopiGetVersion);
    m.def("diopiGetLastErrorString", &diopiGetLastErrorString);
    m.def("diopiConvolution2d", diopiConvolution2d);
    m.def("diopiConvolution2dBackward", diopiConvolution2dBackward);
    m.def("diopiBatchNorm", diopiBatchNorm);
    m.def("diopiBatchNormBackward", diopiBatchNormBackward);
    m.def("diopiRelu", diopiRelu);
    m.def("diopiReluInp", diopiReluInp);
    m.def("diopiHardtanh", diopiHardtanh);
    m.def("diopiHardtanhInp", diopiHardtanhInp);
    m.def("diopiHardtanhBackward", diopiHardtanhBackward);
    m.def("diopiHardswish", diopiHardswish);
    m.def("diopiHardswishInp", diopiHardswishInp);
    m.def("diopiHardswishBackward", diopiHardswishBackward);
    m.def("diopiThreshold", diopiThreshold);
    m.def("diopiThresholdInp", diopiThresholdInp);
    m.def("diopiThresholdBackward", diopiThresholdBackward);
    m.def("diopiGelu", diopiGelu);
    m.def("diopiGeluBackward", diopiGeluBackward);
    m.def("diopiLeakyRelu", diopiLeakyRelu);
    m.def("diopiLeakyReluInp", diopiLeakyReluInp);
    m.def("diopiLeakyReluBackward", diopiLeakyReluBackward);
    m.def("diopiAvgPool2d", diopiAvgPool2d);
    m.def("diopiAvgPool2d", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad) {
    
        diopiError_t ret = diopiAvgPool2d(ctx, out, input, kernel_size, stride, padding, ceil_mode, count_include_pad, nullptr);
    
        return ret;
    });
    m.def("diopiAvgPool2dBackward", diopiAvgPool2dBackward);
    m.def("diopiAvgPool2dBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output, diopiConstTensorHandle_t input, diopiSize_t kernel_size, diopiSize_t stride, diopiSize_t padding, bool ceil_mode, bool count_include_pad) {
    
        diopiError_t ret = diopiAvgPool2dBackward(ctx, grad_input, grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad, nullptr);
    
        return ret;
    });
    m.def("diopiMaxPool2d", diopiMaxPool2d);
    m.def("diopiMaxPool2dWithIndices", diopiMaxPool2dWithIndices);
    m.def("diopiMaxPool2dBackward", diopiMaxPool2dBackward);
    m.def("diopiAdaptiveAvgPool2d", diopiAdaptiveAvgPool2d);
    m.def("diopiAdaptiveAvgPool2dBackward", diopiAdaptiveAvgPool2dBackward);
    m.def("diopiAdaptiveMaxPool2d", diopiAdaptiveMaxPool2d);
    m.def("diopiAdaptiveMaxPool2dWithIndices", diopiAdaptiveMaxPool2dWithIndices);
    m.def("diopiAdaptiveMaxPool2dBackward", diopiAdaptiveMaxPool2dBackward);
    m.def("diopiDropout", diopiDropout);
    m.def("diopiDropoutInp", diopiDropoutInp);
    m.def("diopiMSELoss", diopiMSELoss);
    m.def("diopiMSELossBackward", diopiMSELossBackward);
    m.def("diopiSigmoidFocalLoss", diopiSigmoidFocalLoss);
    m.def("diopiSigmoidFocalLossBackward", diopiSigmoidFocalLossBackward);
    m.def("diopiCrossEntropyLoss", diopiCrossEntropyLoss);
    m.def("diopiCrossEntropyLossBackward", diopiCrossEntropyLossBackward);
    m.def("diopiNLLLoss", diopiNLLLoss);
    m.def("diopiNLLLossBackward", diopiNLLLossBackward);
    m.def("diopiBCEWithLogits", diopiBCEWithLogits);
    m.def("diopiBCEWithLogitsBackward", diopiBCEWithLogitsBackward);
    m.def("diopiBCELoss", diopiBCELoss);
    m.def("diopiBCELossBackward", diopiBCELossBackward);
    m.def("diopiSign", diopiSign);
    m.def("diopiAbsInp", diopiAbsInp);
    m.def("diopiAbs", diopiAbs);
    m.def("diopiNegInp", diopiNegInp);
    m.def("diopiNeg", diopiNeg);
    m.def("diopiFloorInp", diopiFloorInp);
    m.def("diopiFloor", diopiFloor);
    m.def("diopiSqrtInp", diopiSqrtInp);
    m.def("diopiSqrt", diopiSqrt);
    m.def("diopiRsqrtInp", diopiRsqrtInp);
    m.def("diopiRsqrt", diopiRsqrt);
    m.def("diopiSinInp", diopiSinInp);
    m.def("diopiSin", diopiSin);
    m.def("diopiCosInp", diopiCosInp);
    m.def("diopiCos", diopiCos);
    m.def("diopiTanhInp", diopiTanhInp);
    m.def("diopiTanh", diopiTanh);
    m.def("diopiTanhBackward", diopiTanhBackward);
    m.def("diopiSigmoidInp", diopiSigmoidInp);
    m.def("diopiSigmoid", diopiSigmoid);
    m.def("diopiSigmoidBackward", diopiSigmoidBackward);
    m.def("diopiSiluInp", diopiSiluInp);
    m.def("diopiSilu", diopiSilu);
    m.def("diopiSiluBackward", diopiSiluBackward);
    m.def("diopiExpInp", diopiExpInp);
    m.def("diopiExp", diopiExp);
    m.def("diopiLogInp", diopiLogInp);
    m.def("diopiLog", diopiLog);
    m.def("diopiLog2Inp", diopiLog2Inp);
    m.def("diopiLog2", diopiLog2);
    m.def("diopiLog10Inp", diopiLog10Inp);
    m.def("diopiLog10", diopiLog10);
    m.def("diopiErfInp", diopiErfInp);
    m.def("diopiErf", diopiErf);
    m.def("diopiPowScalar", diopiPowScalar);
    m.def("diopiPow", diopiPow);
    m.def("diopiPowInp", diopiPowInp);
    m.def("diopiPowTensor", diopiPowTensor);
    m.def("diopiPowInpTensor", diopiPowInpTensor);
    m.def("diopiAdd", diopiAdd);
    m.def("diopiAddInp", diopiAddInp);
    m.def("diopiAddScalar", diopiAddScalar);
    m.def("diopiAddInpScalar", diopiAddInpScalar);
    m.def("diopiSub", diopiSub);
    m.def("diopiSubInp", diopiSubInp);
    m.def("diopiSubScalar", diopiSubScalar);
    m.def("diopiSubInpScalar", diopiSubInpScalar);
    m.def("diopiMul", diopiMul);
    m.def("diopiMulInp", diopiMulInp);
    m.def("diopiMulScalar", diopiMulScalar);
    m.def("diopiMulInpScalar", diopiMulInpScalar);
    m.def("diopiDiv", diopiDiv);
    m.def("diopiDivInp", diopiDivInp);
    m.def("diopiDivScalar", diopiDivScalar);
    m.def("diopiDivInpScalar", diopiDivInpScalar);
    m.def("diopiBmm", diopiBmm);
    m.def("diopiBaddbmm", diopiBaddbmm);
    m.def("diopiBaddbmmInp", diopiBaddbmmInp);
    m.def("diopiAddcmul", diopiAddcmul);
    m.def("diopiAddcmulInp", diopiAddcmulInp);
    m.def("diopiMatmul", diopiMatmul);
    m.def("diopiAddcdiv", diopiAddcdiv);
    m.def("diopiAddcdivInp", diopiAddcdivInp);
    m.def("diopiAddmm", diopiAddmm);
    m.def("diopiCholesky", diopiCholesky);
    m.def("diopiCholeskyBackward", diopiCholeskyBackward);
    m.def("diopiTriangularSolve", diopiTriangularSolve);
    m.def("diopiTriangularSolveBackward", diopiTriangularSolveBackward);
    m.def("diopiClampInpScalar", diopiClampInpScalar);
    m.def("diopiClampInp", diopiClampInp);
    m.def("diopiClampScalar", diopiClampScalar);
    m.def("diopiClamp", diopiClamp);
    m.def("diopiClampMaxInpScalar", diopiClampMaxInpScalar);
    m.def("diopiClampMaxInp", diopiClampMaxInp);
    m.def("diopiClampMaxScalar", diopiClampMaxScalar);
    m.def("diopiClampMax", diopiClampMax);
    m.def("diopiClampMinInpScalar", diopiClampMinInpScalar);
    m.def("diopiClampMinInp", diopiClampMinInp);
    m.def("diopiClampMinScalar", diopiClampMinScalar);
    m.def("diopiClampMin", diopiClampMin);
    m.def("diopiFill", diopiFill);
    m.def("diopiLogicalAnd", diopiLogicalAnd);
    m.def("diopiLogicalAndInp", diopiLogicalAndInp);
    m.def("diopiLogicalOr", diopiLogicalOr);
    m.def("diopiLogicalOrInp", diopiLogicalOrInp);
    m.def("diopiLogicalNot", diopiLogicalNot);
    m.def("diopiLogicalNotInp", diopiLogicalNotInp);
    m.def("diopiBitwiseAnd", diopiBitwiseAnd);
    m.def("diopiBitwiseAndInp", diopiBitwiseAndInp);
    m.def("diopiBitwiseAndScalar", diopiBitwiseAndScalar);
    m.def("diopiBitwiseAndInpScalar", diopiBitwiseAndInpScalar);
    m.def("diopiBitwiseOr", diopiBitwiseOr);
    m.def("diopiBitwiseOrInp", diopiBitwiseOrInp);
    m.def("diopiBitwiseOrScalar", diopiBitwiseOrScalar);
    m.def("diopiBitwiseOrInpScalar", diopiBitwiseOrInpScalar);
    m.def("diopiBitwiseNot", diopiBitwiseNot);
    m.def("diopiBitwiseNotInp", diopiBitwiseNotInp);
    m.def("diopiEqScalar", diopiEqScalar);
    m.def("diopiEqInpScalar", diopiEqInpScalar);
    m.def("diopiEq", diopiEq);
    m.def("diopiEqInp", diopiEqInp);
    m.def("diopiNeScalar", diopiNeScalar);
    m.def("diopiNeInpScalar", diopiNeInpScalar);
    m.def("diopiNe", diopiNe);
    m.def("diopiNeInp", diopiNeInp);
    m.def("diopiGeScalar", diopiGeScalar);
    m.def("diopiGeInpScalar", diopiGeInpScalar);
    m.def("diopiGe", diopiGe);
    m.def("diopiGeInp", diopiGeInp);
    m.def("diopiGtScalar", diopiGtScalar);
    m.def("diopiGtInpScalar", diopiGtInpScalar);
    m.def("diopiGt", diopiGt);
    m.def("diopiGtInp", diopiGtInp);
    m.def("diopiLeScalar", diopiLeScalar);
    m.def("diopiLeInpScalar", diopiLeInpScalar);
    m.def("diopiLe", diopiLe);
    m.def("diopiLeInp", diopiLeInp);
    m.def("diopiLtScalar", diopiLtScalar);
    m.def("diopiLtInpScalar", diopiLtInpScalar);
    m.def("diopiLt", diopiLt);
    m.def("diopiLtInp", diopiLtInp);
    m.def("diopiMean", diopiMean);
    m.def("diopiSum", diopiSum);
    m.def("diopiStd", diopiStd);
    m.def("diopiMin", diopiMin);
    m.def("diopiMinAll", diopiMinAll);
    m.def("diopiMax", diopiMax);
    m.def("diopiMaxAll", diopiMaxAll);
    m.def("diopiAny", diopiAny);
    m.def("diopiAny", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    
        diopiError_t ret = diopiAny(ctx, out, input, nullptr);
    
        return ret;
    });
    m.def("diopiAll", diopiAll);
    m.def("diopiAll", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    
        diopiError_t ret = diopiAll(ctx, out, input, nullptr);
    
        return ret;
    });
    m.def("diopiSoftmax", diopiSoftmax);
    m.def("diopiSoftmaxBackward", diopiSoftmaxBackward);
    m.def("diopiLogSoftmax", diopiLogSoftmax);
    m.def("diopiLogSoftmaxBackward", diopiLogSoftmaxBackward);
    m.def("diopiIndex", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, py::list& indices, int64_t nums) {
        std::vector<diopiConstTensorHandle_t> indicesV(nums);
        for (int i = 0; i < nums; ++i)
            indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
        auto indicesDIOPI = indicesV.data();
        diopiTensorHandle_t outHandle = nullptr;
        diopiError_t ret = diopiIndex(ctx, &outHandle, input, indicesDIOPI, nums);
        if (out.get() != nullptr)
             *out = *outHandle;
        return ret;
    });
    m.def("diopiIndexBackward", [](diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiTensorHandle_t zeros_like_input, py::list& indices, int64_t nums, diopiConstTensorHandle_t grad) {
        std::vector<diopiConstTensorHandle_t> indicesV(nums);
        for (int i = 0; i < nums; ++i)
            indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
        auto indicesDIOPI = indicesV.data();
        diopiError_t ret = diopiIndexBackward(ctx, grad_input, zeros_like_input, indicesDIOPI, nums, grad);
    
        return ret;
    });
    m.def("diopiIndexSelect", diopiIndexSelect);
    m.def("diopiIndexSelectBackward", diopiIndexSelectBackward);
    m.def("diopiSelect", diopiSelect);
    m.def("diopiSelectBackward", diopiSelectBackward);
    m.def("diopiSelectScatter", diopiSelectScatter);
    m.def("diopiSliceScatter", diopiSliceScatter);
    m.def("diopiSlice", diopiSlice);
    m.def("diopiSliceBackward", diopiSliceBackward);
    m.def("diopiMaskedScatter", diopiMaskedScatter);
    m.def("diopiNms", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t dets, diopiConstTensorHandle_t scores, double iou_threshold) {
        diopiTensorHandle_t outHandle = nullptr;
        diopiError_t ret = diopiNms(ctx, &outHandle, dets, scores, iou_threshold);
        if (out.get() != nullptr)
             *out = *outHandle;
        return ret;
    });
    m.def("diopiNonzero", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input) {
        diopiTensorHandle_t outHandle = nullptr;
        diopiError_t ret = diopiNonzero(ctx, &outHandle, input);
        if (out.get() != nullptr)
             *out = *outHandle;
        return ret;
    });
    m.def("diopiLinear", diopiLinear);
    m.def("diopiLinearBackward", diopiLinearBackward);
    m.def("diopiRoiAlign", diopiRoiAlign);
    m.def("diopiRoiAlignBackward", diopiRoiAlignBackward);
    m.def("diopiSgd", diopiSgd);
    m.def("diopiClipGradNorm", [](diopiContextHandle_t ctx, void* out, py::list& grads, int64_t num_grads, double max_norm, double norm_type, bool error_if_nonfinite) {
        std::vector<diopiTensorHandle_t> gradsV(num_grads);
        for (int i = 0; i < num_grads; ++i)
            gradsV[i] = grads[i].cast<PtrWrapper<diopiTensor>>().get();
        auto gradsDIOPI = gradsV.data();
        diopiError_t ret = diopiClipGradNorm(ctx, reinterpret_cast<double*>(out), gradsDIOPI, num_grads, max_norm, norm_type, error_if_nonfinite);
    
        return ret;
    });
    m.def("diopiEmbeddingRenorm_", diopiEmbeddingRenorm_);
    m.def("diopiEmbedding", diopiEmbedding);
    m.def("diopiEmbeddingBackward", diopiEmbeddingBackward);
    m.def("diopiTril", diopiTril);
    m.def("diopiCat", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, py::list& tensors, int64_t num_inputs, int64_t dim) {
        std::vector<diopiConstTensorHandle_t> tensorsV(num_inputs);
        for (int i = 0; i < num_inputs; ++i)
            tensorsV[i] = tensors[i].cast<PtrWrapper<diopiTensor>>().get();
        auto tensorsDIOPI = tensorsV.data();
        diopiError_t ret = diopiCat(ctx, out, tensorsDIOPI, num_inputs, dim);
    
        return ret;
    });
    m.def("diopiSplitWithSizes", [](diopiContextHandle_t ctx, py::list& outs, int64_t num_outs, diopiConstTensorHandle_t input, const diopiSize_t splitSizes, int64_t dim) {
        std::vector<diopiTensorHandle_t> outsV(num_outs);
        for (int i = 0; i < num_outs; ++i)
            outsV[i] = outs[i].cast<PtrWrapper<diopiTensor>>().get();
        auto outsDIOPI = outsV.data();
        diopiError_t ret = diopiSplitWithSizes(ctx, outsDIOPI, num_outs, input, splitSizes, dim);
    
        return ret;
    });
    m.def("diopiStack", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, py::list& tensors, int64_t numTensors, int64_t dim) {
        std::vector<diopiConstTensorHandle_t> tensorsV(numTensors);
        for (int i = 0; i < numTensors; ++i)
            tensorsV[i] = tensors[i].cast<PtrWrapper<diopiTensor>>().get();
        auto tensorsDIOPI = tensorsV.data();
        diopiError_t ret = diopiStack(ctx, out, tensorsDIOPI, numTensors, dim);
    
        return ret;
    });
    m.def("diopiSort", diopiSort);
    m.def("diopiTopk", diopiTopk);
    m.def("diopiTranspose", diopiTranspose);
    m.def("diopiOneHot", diopiOneHot);
    m.def("diopiWhere", diopiWhere);
    m.def("diopiMaskedFill", diopiMaskedFill);
    m.def("diopiMaskedFillInp", diopiMaskedFillInp);
    m.def("diopiMaskedFillScalar", diopiMaskedFillScalar);
    m.def("diopiMaskedFillInpScalar", diopiMaskedFillInpScalar);
    m.def("diopiReciprocal", diopiReciprocal);
    m.def("diopiReciprocalInp", diopiReciprocalInp);
    m.def("diopiAdamW", diopiAdamW);
    m.def("diopiConvTranspose2d", diopiConvTranspose2d);
    m.def("diopiUnfold", diopiUnfold);
    m.def("diopiUnfoldBackward", diopiUnfoldBackward);
    m.def("diopiCumsum", diopiCumsum);
    m.def("diopiCdist", diopiCdist);
    m.def("diopiCdist", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input1, diopiConstTensorHandle_t input2, double p) {
    
        diopiError_t ret = diopiCdist(ctx, out, input1, input2, p, nullptr);
    
        return ret;
    });
    m.def("diopiCdistBackward", diopiCdistBackward);
    m.def("diopiArgmax", diopiArgmax);
    m.def("diopiArgmax", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, bool keepdim) {
    
        diopiError_t ret = diopiArgmax(ctx, out, input, nullptr, keepdim);
    
        return ret;
    });
    m.def("diopiAdadelta", diopiAdadelta);
    m.def("diopiAdam", diopiAdam);
    m.def("diopiRmsprop", diopiRmsprop);
    m.def("diopiSmoothL1Loss", diopiSmoothL1Loss);
    m.def("diopiSmoothL1LossBackward", diopiSmoothL1LossBackward);
    m.def("diopiConvolution3d", diopiConvolution3d);
    m.def("diopiConvolution3dBackward", diopiConvolution3dBackward);
    m.def("diopiMaxPool3d", diopiMaxPool3d);
    m.def("diopiMaxPool3dWithIndices", diopiMaxPool3dWithIndices);
    m.def("diopiMaxPool3dBackward", diopiMaxPool3dBackward);
    m.def("diopiAdaptiveAvgPool3d", diopiAdaptiveAvgPool3d);
    m.def("diopiAdaptiveAvgPool3dBackward", diopiAdaptiveAvgPool3dBackward);
    m.def("diopiAdaptiveMaxPool3d", diopiAdaptiveMaxPool3d);
    m.def("diopiAdaptiveMaxPool3dWithIndices", diopiAdaptiveMaxPool3dWithIndices);
    m.def("diopiAdaptiveMaxPool3dBackward", diopiAdaptiveMaxPool3dBackward);
    m.def("diopiMaskedSelect", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t mask) {
        diopiTensorHandle_t outHandle = nullptr;
        diopiError_t ret = diopiMaskedSelect(ctx, &outHandle, input, mask);
        if (out.get() != nullptr)
             *out = *outHandle;
        return ret;
    });
    m.def("diopiMaskedSelectBackward", diopiMaskedSelectBackward);
    m.def("diopiMaximum", diopiMaximum);
    m.def("diopiMinimum", diopiMinimum);
    m.def("diopiMm", diopiMm);
    m.def("diopiIndexFillScalar", diopiIndexFillScalar);
    m.def("diopiIndexFill", diopiIndexFill);
    m.def("diopiIndexFillInpScalar", diopiIndexFillInpScalar);
    m.def("diopiIndexFillInp", diopiIndexFillInp);
    m.def("diopiExpand", diopiExpand);
    m.def("diopiLinspace", diopiLinspace);
    m.def("diopiPermute", diopiPermute);
    m.def("diopiPad", diopiPad);
    m.def("diopiPad", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiSize_t pad, const char* mode) {
    
        diopiError_t ret = diopiPad(ctx, out, input, pad, mode, nullptr);
    
        return ret;
    });
    m.def("diopiRoll", diopiRoll);
    m.def("diopiFlip", diopiFlip);
    m.def("diopiNorm", diopiNorm);
    m.def("diopiGroupNorm", diopiGroupNorm);
    m.def("diopiGroupNormBackward", diopiGroupNormBackward);
    m.def("diopiUnique", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, const int64_t* dim, bool sorted, bool return_counts, diopiTensorHandle_t indices, PtrWrapper<diopiTensor> counts) {
        diopiTensorHandle_t outHandle = nullptr;
        diopiTensorHandle_t countsHandle = nullptr;
        diopiError_t ret = diopiUnique(ctx, &outHandle, input, dim, sorted, return_counts, indices, &countsHandle);
        if (out.get() != nullptr)
             *out = *outHandle;
        if (counts.get() != nullptr)
             *counts = *countsHandle;
        return ret;
    });
    m.def("diopiUnique", [](diopiContextHandle_t ctx, PtrWrapper<diopiTensor> out, diopiConstTensorHandle_t input, bool sorted, bool return_counts, diopiTensorHandle_t indices, PtrWrapper<diopiTensor> counts) {
        diopiTensorHandle_t outHandle = nullptr;
        diopiTensorHandle_t countsHandle = nullptr;
        diopiError_t ret = diopiUnique(ctx, &outHandle, input, nullptr, sorted, return_counts, indices, &countsHandle);
        if (out.get() != nullptr)
             *out = *outHandle;
        if (counts.get() != nullptr)
             *counts = *countsHandle;
        return ret;
    });
    m.def("diopiProd", diopiProd);
    m.def("diopiProd", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    
        diopiError_t ret = diopiProd(ctx, out, input, nullptr);
    
        return ret;
    });
    m.def("diopiCTCLoss", diopiCTCLoss);
    m.def("diopiCTCLossBackward", diopiCTCLossBackward);
    m.def("diopiRemainderTensor", diopiRemainderTensor);
    m.def("diopiRemainderScalar", diopiRemainderScalar);
    m.def("diopiRemainder", diopiRemainder);
    m.def("diopiGather", diopiGather);
    m.def("diopiGatherBackward", diopiGatherBackward);
    m.def("diopiScatterInp", diopiScatterInp);
    m.def("diopiScatterInpScalar", diopiScatterInpScalar);
    m.def("diopiScatter", diopiScatter);
    m.def("diopiScatterScalar", diopiScatterScalar);
    m.def("diopiIndexPutInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t values, py::list& indices, int64_t indices_counts, bool accumulate) {
        std::vector<diopiConstTensorHandle_t> indicesV(indices_counts);
        for (int i = 0; i < indices_counts; ++i)
            indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
        auto indicesDIOPI = indicesV.data();
        diopiError_t ret = diopiIndexPutInp(ctx, input, values, indicesDIOPI, indices_counts, accumulate);
    
        return ret;
    });
    m.def("diopiIndexPut", [](diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t values, py::list& indices, int64_t indices_counts, bool accumulate) {
        std::vector<diopiConstTensorHandle_t> indicesV(indices_counts);
        for (int i = 0; i < indices_counts; ++i)
            indicesV[i] = indices[i].cast<PtrWrapper<diopiTensor>>().get();
        auto indicesDIOPI = indicesV.data();
        diopiError_t ret = diopiIndexPut(ctx, out, input, values, indicesDIOPI, indices_counts, accumulate);
    
        return ret;
    });
    m.def("diopiRandomInp", diopiRandomInp);
    m.def("diopiRandomInp", [](diopiContextHandle_t ctx, diopiTensorHandle_t inout, int64_t from, int64_t idx) {
    
        diopiError_t ret = diopiRandomInp(ctx, inout, from, nullptr, idx);
    
        return ret;
    });
    m.def("diopiUniformInp", diopiUniformInp);
    m.def("diopiBernoulli", diopiBernoulli);
    m.def("diopiBernoulliInp", diopiBernoulliInp);
    m.def("diopiBernoulliScalar", diopiBernoulliScalar);
    m.def("diopiArange", diopiArange);
    m.def("diopiRandperm", diopiRandperm);
    m.def("diopiNormal", diopiNormal);
    m.def("diopiNormalTensorScalar", diopiNormalTensorScalar);
    m.def("diopiNormalScalarTensor", diopiNormalScalarTensor);
    m.def("diopiNormalTensor", diopiNormalTensor);
    m.def("diopiNormalInp", diopiNormalInp);
    m.def("diopiMeshGrid", [](diopiContextHandle_t ctx, py::list& outs, py::list& inputs, int64_t inputsNum) {
        std::vector<diopiConstTensorHandle_t> inputsV(inputsNum);
        for (int i = 0; i < inputsNum; ++i)
            inputsV[i] = inputs[i].cast<PtrWrapper<diopiTensor>>().get();
        auto inputsDIOPI = inputsV.data();
        std::vector<diopiTensorHandle_t> outsV(inputsNum);
        for (int i = 0; i < inputsNum; ++i)
            outsV[i] = outs[i].cast<PtrWrapper<diopiTensor>>().get();
        auto outsDIOPI = outsV.data();
        diopiError_t ret = diopiMeshGrid(ctx, outsDIOPI, inputsDIOPI, inputsNum);
    
        return ret;
    });
    m.def("diopiMultinomial", diopiMultinomial);
    m.def("diopiLayerNorm", diopiLayerNorm);
    m.def("diopiLayerNormBackward", diopiLayerNormBackward);
    m.def("diopiCopyInp", diopiCopyInp);
    m.def("diopiUpsampleNearest", diopiUpsampleNearest);
    m.def("diopiUpsampleNearestBackward", diopiUpsampleNearestBackward);
    m.def("diopiUpsampleLinear", diopiUpsampleLinear);
    m.def("diopiUpsampleLinearBackward", diopiUpsampleLinearBackward);
    m.def("diopiErfinv", diopiErfinv);
    m.def("diopiErfinvInp", diopiErfinvInp);
    m.def("diopiIm2Col", diopiIm2Col);
    m.def("diopiCol2Im", diopiCol2Im);
    m.def("diopiRepeat", diopiRepeat);
    m.def("diopiCastDtype", diopiCastDtype);
}

