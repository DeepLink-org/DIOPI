/**
 * @file
 * @author pjlab
 * @copyright  (c) 2023, SenseTime Inc.
 */

#ifndef IMPL_CAMB_COMMON_COMMON_HPP_
#define IMPL_CAMB_COMMON_COMMON_HPP_

#include <set>
#include <vector>

#include "../cnnl_helper.hpp"

namespace impl {
namespace camb {

using DiopiTensorT = DiopiTensor<diopiTensorHandle_t>;

DiopiTensorT dataTypeCast(diopiContextHandle_t& ctx, const DiopiTensorT& src, diopiDtype_t destDtype);

DiopiTensorT makeTensorFromScalar(diopiContextHandle_t ctx, const diopiScalar_t* scalar);

void dataTypeCast(diopiContextHandle_t ctx, DiopiTensorT& dest, const DiopiTensorT& src);

diopiDtype_t choiceDtype(const std::set<diopiDtype_t>& opSupportedDtypes);

void autoCastTensorType(diopiContextHandle_t ctx, std::vector<DiopiTensorT*>& pTensors, const std::set<diopiDtype_t>& opSupportedDtype);

}  // namespace camb
}  // namespace impl

#endif  // IMPL_CAMB_COMMON_COMMON_HPP_
