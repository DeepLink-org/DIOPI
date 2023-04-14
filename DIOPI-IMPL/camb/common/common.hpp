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

diopiError_t makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar, DiopiTensor& out);

diopiError_t dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src);

diopiError_t autoCastTensorType(diopiContextHandle_t ctx, const std::vector<DiopiTensor*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype);

diopiError_t broadcast(diopiContextHandle_t ctx, DiopiTensor& out, const DiopiTensor& input);

diopiError_t broadcastHelper(diopiContextHandle_t ctx, DiopiTensor input_tensor, DiopiTensor target_tensor, DiopiTensor* out_tensor);

void print_backtrace();

}  // namespace camb
}  // namespace impl

#endif  // IMPL_CAMB_COMMON_COMMON_HPP_
