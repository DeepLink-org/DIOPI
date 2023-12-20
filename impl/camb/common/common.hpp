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

namespace impl {
namespace camb {

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& src, diopiDtype_t destDtype);

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src);

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, DiopiTensor& out);

diopiError_t autoCastTensorType(diopiContextHandle_t ctx, const std::vector<DiopiTensor*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype);

diopiError_t broadcastContiguous(diopiContextHandle_t ctx, DiopiTensor& out, const DiopiTensor& input);

diopiError_t broadcastContiguous(diopiContextHandle_t ctx, DiopiTensor inputTensor, const std::vector<int64_t>& targetShape, diopiDtype_t targetDtype,
                                 DiopiTensor* outTensor);

bool checkBroadCast(const DiopiTensor& src, const std::vector<int64_t>& targetShape, std::vector<int64_t>& outStrides);

bool broadcast(DiopiTensor inputTensor, const std::vector<int64_t>& targetShape, DiopiTensor* outTensor);

diopiError_t contiguous(diopiContextHandle_t ctx, DiopiTensor& src, diopiMemoryFormat_t memoryFormat = diopiMemoryFormat_t::Contiguous);

diopiError_t contiguousOut(diopiContextHandle_t ctx, DiopiTensor& src, DiopiTensor& dest, diopiMemoryFormat_t destMemoryFormat);

diopiError_t contiguous(diopiContextHandle_t ctx, DiopiTensor& src, diopiMemoryFormat_t memoryFormat, cnnlTensorLayout_t layoutIn,
                        cnnlTensorLayout_t layoutOut);

template <typename T1 = double, typename T2 = double, typename T3 = double>
diopiError_t cnnlOpTensor(diopiContextHandle_t ctx, DiopiTensor input, DiopiTensor other, DiopiTensor out, cnnlOpTensorDesc_t opType, T1 alpha1 = 1.0,
                          T2 alpha2 = 1.0, T3 beta = 0.0);

template <typename T = double>
diopiError_t cnnlTransformAdaptor(diopiContextHandle_t ctx, DiopiTensor out, DiopiTensor input, T other, T alpha, T beta);

diopiError_t clone(diopiContextHandle_t ctx, const DiopiTensor& inTensor, DiopiTensor& outTensor,
                   diopiMemoryFormat_t memoryFormat = diopiMemoryFormat_t::Preserve);

diopiError_t transpose(diopiContextHandle_t ctx, DiopiTensor outTensor, DiopiTensor input, int64_t dim0, int64_t dim1);

diopiError_t transpose(diopiContextHandle_t ctx, const DiopiTensor& inputTensor, DiopiTensor& outTensor, std::vector<int32_t> perms);

bool denseCheck(const DiopiTensor& src);

}  // namespace camb
}  // namespace impl

#endif  // IMPL_CAMB_COMMON_COMMON_HPP_
