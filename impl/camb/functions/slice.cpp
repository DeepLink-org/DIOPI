/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" {

diopiError_t diopiIndexSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    }
    DiopiTensor indexTensor(index);
    DiopiTensor outTensor(out);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(indexTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    if (dim < 0) {
        dim = dim + inputTensor.dim();
    }
    if (outTensor.dtype() == inputTensor.dtype()) {
        DIOPI_CALLCNNL(cnnlIndexSelect(handle, dim, inputDesc.get(), inputTensor.data(), indexDesc.get(), indexTensor.data(), outDesc.get(), outTensor.data()));
    } else {
        DiopiTensor outTempTensor = requiresTensor(ctx, outTensor.shape(), inputTensor.dtype());
        CnnlTensorDesc outTempDesc(outTempTensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(
            cnnlIndexSelect(handle, dim, inputDesc.get(), inputTensor.data(), indexDesc.get(), indexTensor.data(), outTempDesc.get(), outTempTensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTempTensor));
    }
    return diopiSuccess;
}

diopiError_t diopiIndexSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t grad, diopiSize_t inputSizes,
                                      int64_t dim, diopiConstTensorHandle_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    diopiScalar_t zero = {diopi_dtype_int64, 0};
    DIOPI_CALL(diopiFill(ctx, gradInput, &zero));
    DiopiTensor gradInputTensor(gradInput);
    DiopiTensor gradTensor(grad);
    DiopiTensor outTensor(gradInput);
    diopiDtype_t outDtype = gradInputTensor.dtype();
    if (gradInputTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, diopi_dtype_int32));
    } else if (gradInputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, diopi_dtype_float32));
    }
    if (gradTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, gradTensor, diopi_dtype_int32));
    } else if (gradTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, gradTensor, diopi_dtype_float32));
    }
    CnnlTensorDesc gradInputDesc(gradInputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradDesc(gradTensor, CNNL_LAYOUT_ARRAY);

    DiopiTensor indexTensor(index);
    if (indexTensor.dtype() != diopi_dtype_int32) {
        DIOPI_CALL(dataTypeCast(ctx, indexTensor, diopi_dtype_int32));
    }
    CnnlTensorDesc indexDesc(indexTensor, CNNL_LAYOUT_ARRAY);

    if (dim < 0) {
        dim = dim + inputSizes.len;
    }

    if (gradInputTensor.dtype() == outDtype) {
        DIOPI_CALLCNNL(cnnlIndexAdd(handle,
                                    dim,
                                    gradInputDesc.get(),
                                    gradInputTensor.data(),
                                    indexDesc.get(),
                                    indexTensor.data(),
                                    gradDesc.get(),
                                    gradTensor.data(),
                                    gradInputDesc.get(),
                                    gradInputTensor.data()));
    } else {
        DIOPI_CALLCNNL(cnnlIndexAdd(handle,
                                    dim,
                                    gradInputDesc.get(),
                                    gradInputTensor.data(),
                                    indexDesc.get(),
                                    indexTensor.data(),
                                    gradDesc.get(),
                                    gradTensor.data(),
                                    gradInputDesc.get(),
                                    gradInputTensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, gradInputTensor));
    }
    return diopiSuccess;
}

diopiError_t diopiSelect(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, int64_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor inputTensor(input);
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    }

    diopiScalar_t indexScalar;
    indexScalar.stype = diopi_dtype_int64;
    indexScalar.ival = index;
    DiopiTensor indexTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, &indexScalar, indexTensor));
    DiopiTensor outTensor(out);

    if (dim < 0) {
        dim = dim + inputTensor.dim();
    }
    std::vector<int64_t> shape(outTensor.shape());
    shape.insert(shape.begin() + dim, 1);
    outTensor.view(shape);

    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(indexTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    if (outTensor.dtype() == inputTensor.dtype()) {
        DIOPI_CALLCNNL(cnnlIndexSelect(handle, dim, inputDesc.get(), inputTensor.data(), indexDesc.get(), indexTensor.data(), outDesc.get(), outTensor.data()));
    } else {
        DiopiTensor outTempTensor = requiresTensor(ctx, outTensor.shape(), inputTensor.dtype());
        CnnlTensorDesc outTempDesc(outTempTensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(
            cnnlIndexSelect(handle, dim, inputDesc.get(), inputTensor.data(), indexDesc.get(), indexTensor.data(), outTempDesc.get(), outTempTensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTempTensor));
    }
    return diopiSuccess;
}

diopiError_t diopiSelectBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t inputSizes,
                                 int64_t dim, int64_t index) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    diopiScalar_t zero = {diopi_dtype_int64, 0};
    DIOPI_CALL(diopiFill(ctx, gradInput, &zero));
    DiopiTensor gradInputTensor(gradInput);
    diopiDtype_t outDtype = gradInputTensor.dtype();
    if (dim < 0) {
        dim = dim + inputSizes.len;
    }
    DiopiTensor gradTensor(gradOutput);
    std::vector<int64_t> shape(gradTensor.shape());
    shape.insert(shape.begin() + dim, 1);
    gradTensor.view(shape);

    if (gradInputTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, diopi_dtype_int32));
    } else if (gradInputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, gradInputTensor, diopi_dtype_float32));
    }
    if (gradTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, gradTensor, diopi_dtype_int32));
    } else if (gradTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, gradTensor, diopi_dtype_float32));
    }
    CnnlTensorDesc gradInputDesc(gradInputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc gradDesc(gradTensor, CNNL_LAYOUT_ARRAY);

    diopiScalar_t indexScalar;
    indexScalar.stype = diopi_dtype_int64;
    indexScalar.ival = index;
    DiopiTensor indexTensor;
    DIOPI_CALL(makeTensorFromScalar(ctx, &indexScalar, indexTensor));
    if (indexTensor.dtype() == diopi_dtype_int64) {
        DIOPI_CALL(dataTypeCast(ctx, indexTensor, diopi_dtype_int32));
    }
    CnnlTensorDesc indexDesc(indexTensor, CNNL_LAYOUT_ARRAY);

    if (gradInputTensor.dtype() == outDtype) {
        DIOPI_CALLCNNL(cnnlIndexAdd(handle,
                                    dim,
                                    gradInputDesc.get(),
                                    gradInputTensor.data(),
                                    indexDesc.get(),
                                    indexTensor.data(),
                                    gradDesc.get(),
                                    gradTensor.data(),
                                    gradInputDesc.get(),
                                    gradInputTensor.data()));
    } else {
        DIOPI_CALLCNNL(cnnlIndexAdd(handle,
                                    dim,
                                    gradInputDesc.get(),
                                    gradInputTensor.data(),
                                    indexDesc.get(),
                                    indexTensor.data(),
                                    gradDesc.get(),
                                    gradTensor.data(),
                                    gradInputDesc.get(),
                                    gradInputTensor.data()));
        DiopiTensor outTensor(gradInput);
        DIOPI_CALL(dataTypeCast(ctx, outTensor, gradInputTensor));
    }

    return diopiSuccess;
}

diopiError_t diopiSlice(diopiContextHandle_t ctx, diopiTensorHandle_t nullOut, diopiConstTensorHandle_t input, int64_t dim, int64_t start, int64_t end,
                        int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(input);
    DiopiTensor outTensor(nullOut);
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    std::vector<int32_t> start32(inputTensor.dim(), 0);
    std::vector<int32_t> step32(inputTensor.dim(), 1);
    std::vector<int32_t> end32(inputTensor.shape().begin(), inputTensor.shape().end());
    start32[dim] = start;
    step32[dim] = step;
    end32[dim] = end;

    DIOPI_CALLCNNL(cnnlStridedSlice(handle, inputDesc.get(), inputTensor.data(), start32.data(), end32.data(), step32.data(), outDesc.get(), outTensor.data()));
    return diopiSuccess;
}

diopiError_t diopiSliceBackward(diopiContextHandle_t ctx, diopiTensorHandle_t gradInput, diopiConstTensorHandle_t gradOutput, diopiSize_t inputSizes,
                                int64_t dim, int64_t start, int64_t end, int64_t step) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);
    DiopiTensor inputTensor(gradOutput);
    DiopiTensor outTensor(gradInput);
    if (inputTensor.dtype() == diopi_dtype_float64) {
        DIOPI_CALL(dataTypeCast(ctx, inputTensor, diopi_dtype_float32));
    }
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);

    std::vector<int32_t> start32(inputTensor.dim(), 0);
    std::vector<int32_t> step32(inputTensor.dim(), 1);
    std::vector<int32_t> end32(inputTensor.shape().begin(), inputTensor.shape().end());
    start32[dim] = start;
    step32[dim] = step;
    end32[dim] = end;

    if (outTensor.dtype() == inputTensor.dtype()) {
        DIOPI_CALLCNNL(cnnlStridedSliceBackward(
            handle, start32.data(), end32.data(), step32.data(), inputDesc.get(), inputTensor.data(), outDesc.get(), outTensor.data()));
    } else {
        DiopiTensor outTempTensor = requiresTensor(ctx, outTensor.shape(), inputTensor.dtype());
        CnnlTensorDesc outTempDesc(outTempTensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlStridedSliceBackward(
            handle, start32.data(), end32.data(), step32.data(), inputDesc.get(), inputTensor.data(), outTempDesc.get(), outTempTensor.data()));
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTempTensor));
    }
    return diopiSuccess;
}

}  // extern "C"

}  // namespace camb
}  // namespace impl
