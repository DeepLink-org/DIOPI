/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_COMMON_ACL_TOOLS_HPP_
#define IMPL_ASCEND_COMMON_ACL_TOOLS_HPP_
#include <acl/acl.h>

#include <algorithm>
#include <sstream>
#include <string>

#include "../macro.hpp"

namespace impl {
namespace ascend {

std::string getAclVersion() {
    int32_t majorVersion, minorVersion, patchVersion;
    CALL_ACLRT(aclrtGetVersion(&majorVersion, &minorVersion, &patchVersion));
    return std::to_string(majorVersion) + "." + std::to_string(minorVersion) + "." + std::to_string(patchVersion);
}



}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_COMMON_ACL_TOOLS_HPP_
