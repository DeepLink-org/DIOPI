/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#ifndef IMPL_ASCEND_ACLNN_ACL_SCALAR_HPP_
#define IMPL_ASCEND_ACLNN_ACL_SCALAR_HPP_

#include "../common/acloprunner.hpp"
#include "aclnn/acl_meta.h"

namespace impl {
namespace ascend {

class AclScalsr final {
public:
    explicit AclScalsr(const diopiScalar_t* scalar) {
        // create aclScalar
        if (scalar->stype == diopiDtype_t::diopi_dtype_float64) {
            auto v = getValue<double>(scalar);
            acl_ = aclCreateScalar(&v, getAclDataType(scalar->stype));
        } else {
            auto v = getValue<int64_t>(scalar);
            acl_ = aclCreateScalar(&v, getAclDataType(scalar->stype));
        }
    }

    explicit operator aclScalar*() { return acl_; }

    bool defined() const { return acl_; }

private:
    aclScalar* acl_ = nullptr;
};

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ACLNN_ACL_SCALAR_HPP_
