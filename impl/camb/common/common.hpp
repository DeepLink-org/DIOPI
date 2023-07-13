/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_CAMB_COMMON_COMMON_HPP_
#define IMPL_CAMB_COMMON_COMMON_HPP_

#include <set>
#include <vector>

#include "../cnnl_helper.hpp"
#include "debug.hpp"

namespace impl {
namespace camb {

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& src, diopiDtype_t destDtype);

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src);

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, DiopiTensor& out);

diopiError_t autoCastTensorType(diopiContextHandle_t ctx, const std::vector<DiopiTensor*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype);

diopiError_t broadcast(diopiContextHandle_t ctx, DiopiTensor& out, const DiopiTensor& input);

diopiError_t broadcastHelper(diopiContextHandle_t ctx, DiopiTensor inputTensor, DiopiTensor targetTensor, DiopiTensor* outTensor);

diopiError_t contiguous(diopiContextHandle_t ctx, DiopiTensor& src, MemoryFormat memoryFormat = MemoryFormat::Contiguous);

diopiError_t contiguous(diopiContextHandle_t ctx, DiopiTensor& src, MemoryFormat memoryFormat, cnnlTensorLayout_t layoutIn, cnnlTensorLayout_t layoutOut);

template <typename T1 = double, typename T2 = double, typename T3 = double>
diopiError_t cnnlOpTensor(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor other, DiopiTensor out, cnnlOpTensorDesc_t opType, T1 alpha1 = 1.0,
                          T2 alpha2 = 1.0, T3 beta = 0.0);

diopiError_t clone(diopiContextHandle_t ctx, const DiopiTensor& inTensor, DiopiTensor& outTensor, MemoryFormat memoryFormat = MemoryFormat::Preserve);

diopiError_t transpose(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor input, int64_t dim0, int64_t dim1);

diopiError_t transpose(diopiContextHandle_t ctx, const DiopiTensor& inputTensor, DiopiTensor& outTensor, std::vector<int32_t> perms);

}  // namespace camb
}  // namespace impl

#endif  // IMPL_CAMB_COMMON_COMMON_HPP_
