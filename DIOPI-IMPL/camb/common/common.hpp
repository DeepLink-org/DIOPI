/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef DIOPI_IMPL_CAMB_COMMON_COMMON_HPP_
#define DIOPI_IMPL_CAMB_COMMON_COMMON_HPP_

#include <set>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {


diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& src, diopiDtype_t destDtype);

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src);

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, DiopiTensor& out);

diopiError_t autoCastTensorType(diopiContextHandle_t ctx, const std::vector<DiopiTensor*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype);

diopiError_t broadcast(diopiContextHandle_t ctx, DiopiTensor& out, const DiopiTensor& input);

diopiError_t broadcastHelper(diopiContextHandle_t ctx, DiopiTensor input_tensor, DiopiTensor target_tensor, DiopiTensor* out_tensor);

diopiError_t contiguous_(diopiContextHandle_t& ctx, DiopiTensor& src, MemoryFormat memory_format);

diopiError_t contiguous_(diopiContextHandle_t& ctx, DiopiTensor& src, MemoryFormat memory_format, cnnlTensorLayout_t layout_in, cnnlTensorLayout_t layout_out);

template<typename T1 = double, typename T2 = double, typename T3 = double>
diopiError_t cnnlOpTensor(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor other, DiopiTensor out, cnnlOpTensorDesc_t op_type, T1 alpha1 = 1.0,
                            T2 alpha2 = 1.0, T3 beta = 0.0);

diopiError_t clone(diopiContextHandle_t ctx, const DiopiTensor& inTensor, DiopiTensor& outTensor);

}  // namespace camb
}  // namespace impl

#endif  // DIOPI_IMPL_CAMB_COMMON_COMMON_HPP_
