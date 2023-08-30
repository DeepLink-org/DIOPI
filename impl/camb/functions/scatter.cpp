/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

static diopiError_t slice(cnnlHandle_t handle, DiopiTensor outTensor, DiopiTensor inputTensor, std::vector<int32_t> start, std::vector<int32_t> end,
                          std::vector<int32_t> step) {
    CnnlTensorDesc inputDesc(inputTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc outDesc(outTensor, CNNL_LAYOUT_ARRAY);
    DIOPI_CALLCNNL(cnnlStridedSlice(handle, inputDesc.get(), inputTensor.data(), start.data(), end.data(), step.data(), outDesc.get(), outTensor.data()));
    return diopiSuccess;
}

static diopiError_t scatter(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor inputTensor, int64_t dim, DiopiTensor srcTensor,
                            DiopiTensor indexTensor, const char* reduce) {
    cnnlHandle_t handle = cnnlHandlePool.get(ctx);

    DiopiTensor outTensorTmp = outTensor;
    DiopiTensor inputTensorTmp = inputTensor;
    DiopiTensor srcTensorTmp = srcTensor;

    std::vector<DiopiTensor*> tensor{&indexTensor};
    DIOPI_CALL(autoCastTensorType(ctx, tensor, {diopi_dtype_int32, diopi_dtype_int64}));

    cnnlScatterMode_t mode = CNNL_SCATTER;
    if (strcmp(reduce, "") == 0) {
        mode = CNNL_SCATTER;
        std::vector<DiopiTensor*> tensors{&outTensorTmp, &inputTensorTmp, &srcTensorTmp};
        std::set<diopiDtype_t> supportedDtypes{
            diopi_dtype_bool, diopi_dtype_int8, diopi_dtype_int16, diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, supportedDtypes));

    } else if (strcmp(reduce, "add") == 0) {
        mode = CNNL_SCATTER_ADD;
        std::vector<DiopiTensor*> tensors{&outTensorTmp, &inputTensorTmp, &srcTensorTmp};
        std::set<diopiDtype_t> supportedDtypes{diopi_dtype_int32, diopi_dtype_float16, diopi_dtype_float32};
        DIOPI_CALL(autoCastTensorType(ctx, tensors, supportedDtypes));
    }
    DiopiTensor actualSrcTensor;
    if (srcTensorTmp.numel() != 1 && indexTensor.shape() != srcTensorTmp.shape()) {
        actualSrcTensor = requiresTensor(ctx, indexTensor.shape(), srcTensorTmp.dtype());
        int32_t ndim = indexTensor.dim();
        std::vector<int32_t> start(indexTensor.dim(), 0);
        std::vector<int32_t> step(indexTensor.dim(), 1);
        std::vector<int32_t> end(indexTensor.dim());
        for (int dim = 0; dim < ndim; ++dim) {
            end[dim] = indexTensor.shape()[dim];
        }
        DIOPI_CALL(slice(handle, actualSrcTensor, srcTensorTmp, start, end, step));
    } else {
        actualSrcTensor = srcTensorTmp;
    }

    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc srcDesc(actualSrcTensor, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(indexTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlScatter(handle,
                               dim,
                               inputDesc.get(),
                               inputTensorTmp.data(),
                               indexDesc.get(),
                               indexTensor.data(),
                               srcDesc.get(),
                               actualSrcTensor.data(),
                               outDesc.get(),
                               outTensorTmp.data(),
                               mode));
    if (outTensor.dtype() != outTensorTmp.dtype()) {
        DIOPI_CALL(dataTypeCast(ctx, outTensor, outTensorTmp));
    }
    return diopiSuccess;
}

diopiError_t diopiScatter(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src,
                          diopiConstTensorHandle_t index, const char* reduce) {
    DIOPI_CHECK(reduce != nullptr, "reduce can not be nullptr");
    DIOPI_CHECK(strcmp(reduce, "") == 0 || strcmp(reduce, "add") == 0, "The reduction operation of multiply is not supported by cnnl");

    DiopiTensor outTensor(out);
    DiopiTensor inputTensor(input);
    DiopiTensor srcTensor(src);
    DiopiTensor indexTensor(index);

    DIOPI_CALL(scatter(ctx, outTensor, inputTensor, dim, srcTensor, indexTensor, reduce));
    return diopiSuccess;
}

diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index,
                             const char* reduce) {
    DIOPI_CHECK(reduce != nullptr, "reduce can not be nullptr");
    DIOPI_CHECK(strcmp(reduce, "") == 0 || strcmp(reduce, "add") == 0, "The reduction operation of multiply is not supported by cnnl");

    DiopiTensor outTensor(input);
    DiopiTensor inputTensor(input);
    DiopiTensor srcTensor(src);
    DiopiTensor indexTensor(index);

    DIOPI_CALL(scatter(ctx, outTensor, inputTensor, dim, srcTensor, indexTensor, reduce));
    return diopiSuccess;
}

diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                diopiConstTensorHandle_t index, const char* reduce) {
    DIOPI_CHECK(reduce != nullptr, "reduce can not be nullptr");
    DIOPI_CHECK(strcmp(reduce, "") == 0 || strcmp(reduce, "add") == 0, "The reduction operation of multiply is not supported by cnnl");

    DiopiTensor outTensor(out);
    DiopiTensor inputTensor(input);
    DiopiTensor indexTensor(index);
    DiopiTensor srcTensor = requiresTensor(ctx, indexTensor.shape(), inputTensor.dtype());

    diopiTensorHandle_t src = srcTensor.tensorHandle();
    DIOPI_CALL(diopiFill(ctx, src, value));
    DIOPI_CALL(scatter(ctx, outTensor, inputTensor, dim, srcTensor, indexTensor, reduce));
    return diopiSuccess;
}

diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index,
                                   const char* reduce) {
    DIOPI_CHECK(reduce != nullptr, "reduce can not be nullptr");
    DIOPI_CHECK(strcmp(reduce, "") == 0 || strcmp(reduce, "add") == 0, "The reduction operation of multiply is not supported by cnnl");

    DiopiTensor outTensor(input);
    DiopiTensor inputTensor(input);
    DiopiTensor indexTensor(index);
    DiopiTensor srcTensor = requiresTensor(ctx, indexTensor.shape(), inputTensor.dtype());

    diopiTensorHandle_t src = srcTensor.tensorHandle();
    DIOPI_CALL(diopiFill(ctx, src, value));
    DIOPI_CALL(scatter(ctx, outTensor, inputTensor, dim, srcTensor, indexTensor, reduce));
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
