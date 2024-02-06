/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "acl_tensor.hpp"

namespace impl {
namespace ascend {

// 需要做的完善一些
void AclTensor::reshape(const std::vector<int64_t>& shape) {
    at_.view(shape);
    // auto shape = at_.getAclMemShape();
    auto stride = at_.stride();
    acl_ = aclCreateTensor(shape.data(),
                            shape.size(),
                            at_.getAclDataType(),
                            stride.data(),
                            at_.storageOffset(),
                            at_.getAclDataFormat(),
                            shape.data(),
                            shape.size(),
                            const_cast<void*>(at_.data()));
}

}  // namespace ascend
}  // namespace impl
