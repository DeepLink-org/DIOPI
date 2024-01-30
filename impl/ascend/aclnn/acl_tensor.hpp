/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#ifndef IMPL_ASCEND_ACLNN_ACL_TENSOR_HPP_
#define IMPL_ASCEND_ACLNN_ACL_TENSOR_HPP_

#include "../ascend_tensor.hpp"
#include "../common/debug.hpp"
#include "aclnn/acl_meta.h"
// #include "adaptor.hpp"

namespace impl {
namespace ascend {

class AclTensor final {
public:
    explicit AclTensor(diopiConstTensorHandle_t tensor) /* : at_(tensor) */ {
        at_ = AscendTensor(tensor);
        auto shape = at_.getAclMemShape();
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

    explicit operator aclTensor*() { return acl_; }

    bool defined() const { return acl_; }

    int64_t numel() const { return at_.numel(); }

    const void* data() { return at_.data(); }

    const aclTensor* ptr() { return acl_; }

    void print() const;

private:
    aclTensor* acl_ = nullptr;
    AscendTensor at_;
};

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_ACLNN_ACL_TENSOR_HPP_
