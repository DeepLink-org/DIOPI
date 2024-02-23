/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "adaptor.hpp"

#include <acl/acl_rt.h>

#include <functional>
#include <numeric>
#include <valarray>
#include <vector>

#include "../ascend_tensor.hpp"
#include "../common/acloprunner.hpp"
#include "../common/utils.hpp"

namespace impl {
namespace ascend {

int createAclTensor(diopiConstTensorHandle_t input, aclTensor** tensor) {
    impl::ascend::AscendTensor inAt(input);
    auto shape = inAt.getAclMemShape();
    auto stride = inAt.stride();

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(),
                              shape.size(),
                              inAt.getAclDataType(),
                              stride.data(),
                              inAt.storageOffset(),
                              inAt.getAclDataFormat(),
                              inAt.getAclMemShape().data(),
                              inAt.getAclMemShape().size(),
                              const_cast<void*>(inAt.data()));
    return ACL_SUCCESS;
}

}  // namespace ascend
}  // namespace impl
