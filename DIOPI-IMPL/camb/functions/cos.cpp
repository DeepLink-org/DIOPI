/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#include <diopi/functions.h>

#include <vector>

#include "../cnnl_helper.hpp"
#include "../common/common.hpp"

namespace impl {
namespace camb {

extern "C" diopiError_t diopiCosInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    auto handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        auto input_tensor_f32 = dataTypeCast(ctx, input_tensor, diopi_dtype_float32);
        CnnlTensorDesc f32_desc(input_tensor_f32, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlCos_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, f32_desc.get(), input_tensor_f32.data(), f32_desc.get(), input_tensor_f32.data()));
        dataTypeCast(ctx, input_tensor, input_tensor_f32);
    } else {
        CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlCos_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, input_desc.get(), input_tensor.data(), input_desc.get(), input_tensor.data()));
    }
    return diopiSuccess;
}

extern "C" diopiError_t diopiCos(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input) {
    auto handle = cnnlHandlePool.get(ctx);
    auto input_tensor = DiopiTensor(input);
    auto output_tensor = DiopiTensor(out);
    if (input_tensor.dtype() == diopi_dtype_float64) {
        auto input_tensor_f32 = dataTypeCast(ctx, input_tensor, diopi_dtype_float32);
        CnnlTensorDesc f32_desc(input_tensor_f32, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlCos_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, f32_desc.get(), input_tensor_f32.data(), f32_desc.get(), input_tensor_f32.data()));
        dataTypeCast(ctx, output_tensor, input_tensor_f32);
    } else {
        CnnlTensorDesc input_desc(input_tensor, CNNL_LAYOUT_ARRAY);
        CnnlTensorDesc output_desc(output_tensor, CNNL_LAYOUT_ARRAY);
        DIOPI_CALLCNNL(cnnlCos_v2(handle, CNNL_COMPUTATION_HIGH_PRECISION, input_desc.get(), input_tensor.data(), output_desc.get(), output_tensor.data()));
    }
    return diopiSuccess;
}

}  // namespace camb
}  // namespace impl
