/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {
extern "C" {

bool compareVectors(const std::vector<int64_t>& vec1, const std::vector<int64_t>& vec2) {
    if (vec1.size() != vec2.size()) {
        return false;
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        if (vec1[i] != vec2[i]) {
            return false;
        }
    }

    return true;
}

diopiError_t scatter(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor inputTensor, int64_t dim, DiopiTensor srcTensor, DiopiTensor indexTensor,
                     const char* reduce) {
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

    CnnlTensorDesc outDesc(outTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc inputDesc(inputTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc srcDesc(srcTensorTmp, CNNL_LAYOUT_ARRAY);
    CnnlTensorDesc indexDesc(indexTensor, CNNL_LAYOUT_ARRAY);

    DIOPI_CALLCNNL(cnnlScatter(handle,
                               dim,
                               inputDesc.get(),
                               inputTensorTmp.data(),
                               indexDesc.get(),
                               indexTensor.data(),
                               srcDesc.get(),
                               srcTensorTmp.data(),
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
    DiopiTensor outTensor(out);
    DiopiTensor inputTensor(input);
    DiopiTensor srcTensor(src);
    DiopiTensor indexTensor(index);

    DIOPI_CHECK(strcmp(reduce, "") == 0 || strcmp(reduce, "add") == 0, "The reduction operation of multiply is not supported by cnnl");
    DIOPI_CHECK(compareVectors(srcTensor.shape(), indexTensor.shape()), "Currently, the shape of src tensor and index tensor must be the same");
    DIOPI_CALL(scatter(ctx, outTensor, inputTensor, dim, srcTensor, indexTensor, reduce));
    return diopiSuccess;
}

diopiError_t diopiScatterInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, diopiConstTensorHandle_t src, diopiConstTensorHandle_t index,
                             const char* reduce) {
    DiopiTensor outTensor(input);
    DiopiTensor inputTensor(input);
    DiopiTensor srcTensor(src);
    DiopiTensor indexTensor(index);

    DIOPI_CHECK(strcmp(reduce, "") == 0 || strcmp(reduce, "add") == 0, "The reduction operation of multiply is not supported by cnnl");
    DIOPI_CHECK(compareVectors(srcTensor.shape(), indexTensor.shape()), "Currently, the shape of src tensor and index tensor must be the same");
    DIOPI_CALL(scatter(ctx, outTensor, inputTensor, dim, srcTensor, indexTensor, reduce));
    return diopiSuccess;
}

diopiError_t diopiScatterScalar(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, int64_t dim, const diopiScalar_t* value,
                                diopiConstTensorHandle_t index, const char* reduce) {
    DiopiTensor outTensor(out);
    DiopiTensor inputTensor(input);
    DiopiTensor indexTensor(index);
    DiopiTensor srcTensor(index);
    DIOPI_CALL(dataTypeCast(ctx, srcTensor, inputTensor.dtype()));
    diopiTensorHandle_t src = srcTensor.tensorHandle();
    DIOPI_CALL(diopiFill(ctx, src, value));

    DIOPI_CHECK(strcmp(reduce, "") == 0 || strcmp(reduce, "add") == 0, "The reduction operation of multiply is not supported by cnnl");
    DIOPI_CHECK(compareVectors(srcTensor.shape(), indexTensor.shape()), "Currently, the shape of src tensor and index tensor must be the same");
    DIOPI_CALL(scatter(ctx, outTensor, inputTensor, dim, srcTensor, indexTensor, reduce));
    return diopiSuccess;
}

diopiError_t diopiScatterInpScalar(diopiContextHandle_t ctx, diopiTensorHandle_t input, int64_t dim, const diopiScalar_t* value, diopiConstTensorHandle_t index,
                                   const char* reduce) {
    DiopiTensor outTensor(input);
    DiopiTensor inputTensor(input);
    DiopiTensor indexTensor(index);
    DiopiTensor srcTensor(index);
    DIOPI_CALL(dataTypeCast(ctx, srcTensor, inputTensor.dtype()));
    diopiTensorHandle_t src = srcTensor.tensorHandle();
    DIOPI_CALL(diopiFill(ctx, src, value));

    DIOPI_CHECK(strcmp(reduce, "") == 0 || strcmp(reduce, "add") == 0, "The reduction operation of multiply is not supported by cnnl");
    DIOPI_CHECK(compareVectors(srcTensor.shape(), indexTensor.shape()), "Currently, the shape of src tensor and index tensor must be the same");
    DIOPI_CALL(scatter(ctx, outTensor, inputTensor, dim, srcTensor, indexTensor, reduce));
    return diopiSuccess;
}

}  // extern "C"
}  // namespace camb
}  // namespace impl
