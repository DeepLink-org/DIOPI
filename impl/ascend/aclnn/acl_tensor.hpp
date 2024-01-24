/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#ifndef IMPL_ASCEND_ACLNN_ACLTENSOR_HPP_
#define IMPL_ASCEND_ACLNN_ACLTENSOR_HPP_

#include "../ascend_tensor.hpp"
#include "aclnn/acl_meta.h"

namespace impl {
namespace ascend {

class AclTensor final {
public:
    explicit AclTensor(const diopiConstTensorHandle_t& tensor) : at_(tensor) {
        void* deviceAddr = nullptr;

        // 调用aclCreateTensor接口创建aclTensor
        acl_ = aclCreateTensor(at_.getAclMemShape().data(),
                               at_.getAclMemShape().size(),
                               at_.getAclDataType(),
                               at_.stride().data(),
                               0,
                               at_.getAclDataFormat(),
                               at_.getAclMemShape().data(),
                               at_.getAclMemShape().size(),
                               const_cast<void*>(at_.data()));
    }

    // explicit operator aclTensor() { return *acl_; }
    explicit operator aclTensor*() { return acl_; }

    bool defined() const { return acl_; }

    int64_t numel() const { return at_.numel(); }

private:
    aclTensor* acl_ = nullptr;
    AscendTensor at_;
};

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_ACLNN_ACLTENSOR_HPP_
