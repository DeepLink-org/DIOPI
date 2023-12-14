/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_COMMON_FORMAT_HELPER_H_
#define IMPL_ASCEND_COMMON_FORMAT_HELPER_H_

#include <diopi/diopirt.h>

#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "../ascend_tensor.hpp"

namespace impl {
namespace ascend {

using Shape = std::vector<int64_t>;
constexpr int blocksize = 16;
struct FormatInfo {
    diopiMemoryFormat_t diopiFormat_ = diopiMemoryFormat_t::Undefined;
    aclFormat format_ = aclFormat::ACL_FORMAT_UNDEFINED;
    aclFormat baseFormat_ = aclFormat::ACL_FORMAT_UNDEFINED;
    std::function<Shape(Shape)> func_ = nullptr;
    std::string formatName_;
    bool isPadded_ = false;
};

class FormatHelper {
public:
    static aclFormat getAclFormat(diopiMemoryFormat_t memoryFormat);
    static std::string getFormatName(diopiMemoryFormat_t format);
    static std::string getFormatName(aclFormat format);
    static bool isBaseFormat(diopiMemoryFormat_t format);
    static bool isBaseFormat(aclFormat format);
    static diopiMemoryFormat_t getDiopiBaseFormat(diopiMemoryFormat_t format);
    static aclFormat getAclBaseFormat(aclFormat format);
    static Shape getStorageSizes(diopiMemoryFormat_t format, const Shape& dims);

private:
    static std::unordered_map<aclFormat, FormatInfo> info;
};  // class FormatHelper

}  // namespace ascend
}  // namespace impl
#endif  // IMPL_ASCEND_COMMON_FORMAT_HELPER_H_
