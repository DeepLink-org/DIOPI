

/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "adaptor.hpp"

#include <numeric>

namespace impl {
namespace ascend {
int createAclTensor1(diopiConstTensorHandle_t input, aclTensor** tensor) {
    impl::ascend::AscendTensor inAt(input);

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(inAt.getAclMemShape().data(),
                              inAt.getAclMemShape().size(),
                              inAt.getAclDataType(),
                              inAt.stride().data(),
                              0,
                              inAt.getAclDataFormat(),
                              inAt.getAclMemShape().data(),
                              inAt.getAclMemShape().size(),
                              const_cast<void*>(inAt.data()));
    return ACL_SUCCESS;
}

}  // namespace ascend
}  // namespace impl
