#ifndef IMPL_CAMB_COMMON_COMMON_HPP_
#define IMPL_CAMB_COMMON_COMMON_HPP_

#include <set>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {


DiopiTensor dataTypeCast(diopiContextHandle_t& ctx, const DiopiTensor& src, diopiDtype_t destDtype);

DiopiTensor makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar);

void dataTypeCast(diopiContextHandle_t ctx, DiopiTensor& dest, const DiopiTensor& src);

diopiDtype_t choiceDtype(const std::set<diopiDtype_t>& opSupportedDtypes);

void autoCastTensorType(diopiContextHandle_t ctx, std::vector<DiopiTensor*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype);

diopiError_t broadcast(diopiContextHandle_t ctx, DiopiTensor& out, const DiopiTensor& input);

DiopiTensor broadcastHelper(diopiContextHandle_t ctx, DiopiTensor input_tensor, DiopiTensor target_tensor);


}  // namespace camb
}  // namespace impl

#endif  // IMPL_CAMB_COMMON_COMMON_HPP_
