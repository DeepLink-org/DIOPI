#include <diopi/functions.h>

#include <numeric>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"
namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiNLLLoss(diopiContextHandle_t ctx,
                          diopiTensorHandle_t out,
                          diopiConstTensorHandle_t input,
                          diopiConstTensorHandle_t target,
                          diopiConstTensorHandle_t weight,
                          diopiReduction_t reduction,
                          int64_t ignore_index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tr = DiopiTensor(input);
    auto output_tr = DiopiTensor(out);
    auto target_tr = DiopiTensor(target);
    auto weight_tr = DiopiTensor(weight);

    DIOPI_CHECK(input_tr.dtype() != diopi_dtype_float16, "Half is not supported currently")
    DIOPI_CHECK(input_tr.numel() != 0, "input tensor is empty")

    if (target_tr.dtype() != diopi_dtype_int32) {
        target_tr = dataTypeCast(ctx, target_tr, diopi_dtype_int32);
    }
    if (!weight_tr.defined()) {
        weight_tr = ones(ctx, {input_tr.shape()[1]}, input_tr.dtype());
    }
    DIOPI_CHECK(input_tr.is_contiguous(), "input tensor should be contiguous");
    DIOPI_CHECK(weight_tr.is_contiguous(), "weight tensor should be contiguous");
    DIOPI_CHECK(target_tr.is_contiguous(), "input tensor should be contiguous");

    auto input_contiguous = input_tr;

    auto dim = input_tr.dim();
    if (dim == 2 || dim == 1) {
        DIOPI_CHECK(target_tr.dim() == 1, "1D target_tr tensor expected, multi-target_tr not supported");
        DIOPI_CHECK(input_tr.shape()[0] == target_tr.shape()[0], "size mismatch ");
        DIOPI_CHECK(!weight_tr.defined() || weight_tr.numel() == input_tr.shape()[1],
                    "weight_tr tensor should be defined either for all classes or no classes");
    } else if (dim == 4) {
        input_contiguous = input_tr.contiguous(ctx, MemoryFormat::ChannelsLast);
        cnnl_transpose(ctx, handle, input_tr, input_contiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    } else if (dim == 3) {
        int64_t input_last_size = 1;
        for (int i = 2; i < input_tr.dim(); ++i) {
            input_last_size *= input_tr.shape()[i];
        }
        input_tr.reshape({input_tr.shape()[0], input_tr.shape()[1], 1, input_last_size});

        input_contiguous = input_tr.contiguous(ctx, MemoryFormat::ChannelsLast);
        cnnl_transpose(ctx, handle, input_tr, input_contiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    } else {
        DIOPI_CHECK(false, "unexpected input tensor dim")
    }

    auto input_size = input_contiguous.shape();
    int C = input_size[1];
    int N = std::accumulate(input_size.begin(), input_size.end(), 1, std::multiplies<int64_t>()) / C;
    DIOPI_CHECK(N == target_tr.numel(), "Target size need be equal as input N*H*W.");
    DIOPI_CHECK(C == weight_tr.numel(), "Weight size need be equal as input C.");
    std::vector<int> output_size(input_size.begin(), input_size.end());

    cnnlNlllossAlgorithm_t reduction_mode;
    switch (reduction) {
        case 0: {
            reduction_mode = CNNL_REDUCTION_NONE;
            output_size.erase(output_size.begin() + 1);
            break;
        }
        case 1: {
            reduction_mode = CNNL_REDUCTION_MEAN;
            output_size = {1};
            break;
        }
        case 2: {
            reduction_mode = CNNL_REDUCTION_SUM;
            output_size = {1};
            break;
        }
        default:
            DIOPI_CHECK(false, "unexpected nll_loss reduciton mode");
    }
    auto total_weight_tr = requiresTensor(ctx, {1}, weight_tr.dtype());
    diopiScalar_t scalar({weight_tr.dtype(), static_cast<double>(target_tr.numel())});
    diopiFill(ctx, total_weight_tr.tensor_handle(), &scalar);

    CnnlTensorDesc input_desc;
    CnnlTensorDesc target_desc;
    CnnlTensorDesc weight_desc(weight_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc tw_desc(total_weight_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc output_desc;
    input_desc.set(input_contiguous, CNNL_LAYOUT_ARRAY, {N, C});
    target_desc.set(target_tr, CNNL_LAYOUT_ARRAY, {N});
    output_desc.set(output_tr, CNNL_LAYOUT_ARRAY, output_size);

    size_t workspace_size = 0;
    DIOPI_CHECKCNNL(cnnlGetNlllossWorkspaceSize(handle, input_desc.get(), &workspace_size));
    void* workspace_ptr = workspace_size == 0 ? nullptr : requiresBuffer(ctx, workspace_size).data();

    DIOPI_CALLCNNL(cnnlNlllossForward(handle,
                                      reduction_mode,
                                      workspace_ptr,
                                      workspace_size,
                                      input_desc.get(),
                                      input_contiguous.data(),
                                      target_desc.get(),
                                      target_tr.data(),
                                      static_cast<int>(ignore_index),
                                      weight_desc.get(),
                                      weight_tr.data(),
                                      tw_desc.get(),
                                      total_weight_tr.data(),
                                      output_desc.get(),
                                      output_tr.data()));

    return diopiSuccess;
}

diopiError_t diopiNLLLossBackward(diopiContextHandle_t ctx,
                                  diopiTensorHandle_t grad_input,
                                  diopiConstTensorHandle_t grad_output,
                                  diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t target,
                                  diopiConstTensorHandle_t weight,
                                  diopiReduction_t reduction,
                                  int64_t ignore_index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    auto input_tr = DiopiTensor(input);
    auto grad_input_tr = DiopiTensor(grad_input);
    auto grad_output_tr = DiopiTensor(grad_output);
    auto target_tr = DiopiTensor(target);
    auto weight_tr = DiopiTensor(weight);

    DIOPI_CHECK(input_tr.dtype() != diopi_dtype_float16, "Half is not supported currently")
    DIOPI_CHECK(input_tr.numel() != 0, "input tensor is empty")

    if (target_tr.dtype() != diopi_dtype_int32) {
        target_tr = dataTypeCast(ctx, target_tr, diopi_dtype_int32);
    }
    if (!weight_tr.defined()) {
        weight_tr = ones(ctx, {input_tr.shape()[1]}, input_tr.dtype());
    }
    DIOPI_CHECK(input_tr.is_contiguous(), "input tensor should be contiguous");
    DIOPI_CHECK(weight_tr.is_contiguous(), "weight tensor should be contiguous");
    DIOPI_CHECK(target_tr.is_contiguous(), "input tensor should be contiguous");

    auto input_contiguous = input_tr;

    auto dim = input_tr.dim();
    if (dim == 2 || dim == 1) {
        DIOPI_CHECK(target_tr.dim() == 1, "1D target_tr tensor expected, multi-target_tr not supported");
        DIOPI_CHECK(input_tr.shape()[0] == target_tr.shape()[0], "size mismatch ");
        DIOPI_CHECK(!weight_tr.defined() || weight_tr.numel() == input_tr.shape()[1],
                    "weight_tr tensor should be defined either for all classes or no classes");
    } else if (dim == 4) {
        input_contiguous = input_tr.contiguous(ctx, MemoryFormat::ChannelsLast);
        cnnl_transpose(ctx, handle, input_tr, input_contiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    } else if (dim == 3) {
        int64_t input_last_size = 1;
        for (int i = 2; i < input_tr.dim(); ++i) {
            input_last_size *= input_tr.shape()[i];
        }
        input_tr.reshape({input_tr.shape()[0], input_tr.shape()[1], 1, input_last_size});

        input_contiguous = input_tr.contiguous(ctx, MemoryFormat::ChannelsLast);
        cnnl_transpose(ctx, handle, input_tr, input_contiguous, CNNL_LAYOUT_NCHW, CNNL_LAYOUT_NHWC);
    } else {
        DIOPI_CHECK(false, "unexpected input tensor dim")
    }

    auto input_size = input_contiguous.shape();
    int C = input_size[1];
    int N = std::accumulate(input_size.begin(), input_size.end(), 1, std::multiplies<int64_t>()) / C;
    DIOPI_CHECK(N == target_tr.numel(), "Target size need be equal as input N*H*W.");
    DIOPI_CHECK(C == weight_tr.numel(), "Weight size need be equal as input C.");

    cnnlNlllossAlgorithm_t reduction_mode;
    switch (reduction) {
        case 0:
            reduction_mode = CNNL_REDUCTION_NONE;
            break;
        case 1:
            reduction_mode = CNNL_REDUCTION_MEAN;
            break;
        case 2:
            reduction_mode = CNNL_REDUCTION_SUM;
            break;
        default:
            DIOPI_CHECK(false, "unexpected nll_loss reduciton mode");
    }

    auto grad_input_real_tr = requiresTensor(ctx, {N, C}, input_contiguous.dtype());

    auto total_weight_tr = requiresTensor(ctx, {1}, weight_tr.dtype());
    diopiScalar_t scalar({weight_tr.dtype(), static_cast<double>(target_tr.numel())});
    diopiFill(ctx, total_weight_tr.tensor_handle(), &scalar);

    CnnlTensorDesc grad_output_desc(grad_output_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc target_desc;
    CnnlTensorDesc weight_desc(weight_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc tw_desc(total_weight_tr, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc grad_input_desc(grad_input_real_tr, CNNL_LAYOUT_ARRAY);
    target_desc.set(target_tr, CNNL_LAYOUT_ARRAY, {N});

    DIOPI_CALLCNNL(cnnlNlllossBackward(handle,
                                       reduction_mode,
                                       grad_output_desc.get(),
                                       grad_output_tr.data(),
                                       target_desc.get(),
                                       target_tr.data(),
                                       static_cast<int>(ignore_index),
                                       weight_desc.get(),
                                       weight_tr.data(),
                                       tw_desc.get(),
                                       total_weight_tr.data(),
                                       grad_input_desc.get(),
                                       grad_input_real_tr.data()));
    if (dim > 2) {
        // NHWC -> NCHW
        grad_input_real_tr.reshape(input_contiguous.shape());
        grad_input_tr.reshape(input_contiguous.shape());
        cnnl_transpose(ctx, handle, grad_input_real_tr, grad_input_tr, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NCHW);
    } else {
        diopiCopyInp(ctx, grad_input_real_tr.tensor_handle(), grad_input);
    }

    return diopiSuccess;
}

diopiError_t diopiCrossEntropyLoss(diopiContextHandle_t ctx,
                                   diopiTensorHandle_t out,
                                   diopiConstTensorHandle_t input,
                                   diopiConstTensorHandle_t target,
                                   diopiConstTensorHandle_t weight,
                                   diopiReduction_t reduction,
                                   int64_t ignore_index,
                                   double label_smoothing) {
    auto input_tr = DiopiTensor(input);
    auto target_tr = DiopiTensor(target);

    DIOPI_CHECK(label_smoothing == 0, "param label_smoothing is not supported")
    DIOPI_CHECK(target_tr.dim() == input_tr.dim() - 1, "Probabilities for each class are not supported");

    auto log_tr = requiresTensor(ctx, input_tr.shape(), input_tr.dtype());
    diopiLogSoftmax(ctx, log_tr.tensor_handle(), input, 1);
    diopiNLLLoss(ctx, out, log_tr.tensor_handle(), target, weight, reduction, ignore_index);
    return diopiSuccess;
}
diopiError_t diopiCrossEntropyLossBackward(diopiContextHandle_t ctx,
                                           diopiTensorHandle_t grad_input,
                                           diopiConstTensorHandle_t grad_output,
                                           diopiConstTensorHandle_t input,
                                           diopiConstTensorHandle_t target,
                                           diopiConstTensorHandle_t weight,
                                           diopiReduction_t reduction,
                                           int64_t ignore_index,
                                           double label_smoothing) {
    auto input_tr = DiopiTensor(input);
    auto target_tr = DiopiTensor(target);
    auto grad_input_tr = DiopiTensor(grad_input);

    DIOPI_CHECK(label_smoothing == 0, "param label_smoothing is not supported")
    DIOPI_CHECK(target_tr.dim() == input_tr.dim() - 1, "Probabilities for each class are not supported");

    auto log_tr = requiresTensor(ctx, input_tr.shape(), input_tr.dtype());
    auto grad_tmp_tr = requiresTensor(ctx, grad_input_tr.shape(), grad_input_tr.dtype());

    diopiLogSoftmax(ctx, log_tr.tensor_handle(), input, 1);
    // for nll loss backward, `input` should be logsoftmax out.
    diopiNLLLossBackward(ctx, grad_tmp_tr.tensor_handle(), grad_output, log_tr.tensor_handle(), target, weight, reduction, ignore_index);
    // for softmax backward, `output` should be logsoftmax out
    diopiLogSoftmaxBackward(ctx, grad_input, grad_tmp_tr.tensor_handle(), log_tr.tensor_handle(), 1);
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMSELoss(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input,
                                    diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    auto trInput = DiopiTensor(input);
    auto trTarget = DiopiTensor(target);
    auto trOut = DiopiTensor(out);
    std::vector<DiopiTensor*> pTensors{&trInput, &trTarget};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    autoCastTensorType(ctx, pTensors, supportedDtypes);

    cnnlMSELossReduction_t cnnl_reduction;
    if (reduction == ReductionMean) {
        cnnl_reduction = CNNL_MSE_LOSS_MEAN;
        DIOPI_CHECK(trOut.dim() == 0, "Output dim must be 0.");
    } else if (reduction == ReductionSum) {
        cnnl_reduction = CNNL_MSE_LOSS_SUM;
        DIOPI_CHECK(trOut.dim() == 0, "Output dim must be 0.");
    } else {
        cnnl_reduction = CNNL_MSE_LOSS_NONE;
        DIOPI_CHECK(trOut.dim() == trInput.dim(), "Output dim must be the same as input.");
    }

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput, layout);
    CnnlTensorDesc descTarget(trTarget, layout);
    CnnlTensorDesc descOut(trOut, layout);
    DiopiTensor trOutTmp;
    CnnlTensorDesc descOutTmp;
    if (trInput.dtype() == trOut.dtype()) {
        trOutTmp = trOut;
        descOutTmp = descOut;
    } else {
        trOutTmp = requiresTensor(ctx, vec2diopiSize_t(trOut.shape()), trInput.dtype());
        descOutTmp.set(trOutTmp, CNNL_LAYOUT_ARRAY);
    }

    DIOPI_CALLCNNL(cnnlMSELoss(handle, cnnl_reduction, descInput.get(), trInput.data(), descTarget.get(), trTarget.data(), descOutTmp.get(), trOutTmp.data()));
    if (trOutTmp.dtype() != trOut.dtype()) {
        dataTypeCast(ctx, trOut, trOutTmp);
    }
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiMSELossBackward(diopiContextHandle_t ctx, diopiTensorHandle_t grad_input, diopiConstTensorHandle_t grad_output,
                                            diopiConstTensorHandle_t input, diopiConstTensorHandle_t target, diopiReduction_t reduction) {
    auto trInput = DiopiTensor(input);
    auto trGradOutput = DiopiTensor(grad_output);
    auto trTarget = DiopiTensor(target);
    auto trGradInput = DiopiTensor(grad_input);

    std::vector<DiopiTensor*> pTensors{&trInput, &trGradOutput, &trTarget};
    std::set<diopiDtype_t> supportedDtypes{diopi_dtype_float16, diopi_dtype_float32};
    autoCastTensorType(ctx, pTensors, supportedDtypes);

    cnnlMSELossReduction_t cnnl_reduction;
    if (reduction == ReductionMean) {
        cnnl_reduction = CNNL_MSE_LOSS_MEAN;
        DIOPI_CHECK(trGradOutput.dim() == 0, "Grad output dim must be 0.");
    } else if (reduction == ReductionSum) {
        cnnl_reduction = CNNL_MSE_LOSS_SUM;
        DIOPI_CHECK(trGradOutput.dim() == 0, "Grad output dim must be 0.");
    } else {
        cnnl_reduction = CNNL_MSE_LOSS_NONE;
        DIOPI_CHECK(trGradOutput.dim() == trInput.dim(), "Output dim must be the same as input.");
    }

    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
    cnnlDataType_t dtype;
    DIOPI_CALL(CnnlDataType::convertToCnnlType(&dtype, trInput.dtype()));

    CnnlTensorDesc descInput(trInput, layout);
    CnnlTensorDesc descTarget(trTarget, layout);
    CnnlTensorDesc descGradOutput(trGradOutput, layout);
    CnnlTensorDesc descGradInput(trGradInput, layout);
    DiopiTensor trGradInputTmp;
    CnnlTensorDesc descGradInputTmp;
    if (trInput.dtype() == trGradInput.dtype()) {
        trGradInputTmp = trGradInput;
        descGradInputTmp = descGradInput;
    } else {
        trGradInputTmp = requiresTensor(ctx, vec2diopiSize_t(trGradInput.shape()), trInput.dtype());
        descGradInputTmp.set(trGradInputTmp, CNNL_LAYOUT_ARRAY);
    }

    DIOPI_CALLCNNL(cnnlMSELossBackward(handle, cnnl_reduction, descInput.get(), trInput.data(), descTarget.get(), \
    trTarget.data(), descGradOutput.get(), trGradOutput.data(), descGradInputTmp.get(), trGradInputTmp.data()));
    if (trGradInputTmp.dtype() != trGradInput.dtype()) {
        dataTypeCast(ctx, trGradInput, trGradInputTmp);
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
