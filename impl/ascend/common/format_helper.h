/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_COMMON_FORMAT_HELPER_H_
#define IMPL_ASCEND_COMMON_FORMAT_HELPER_H_

#include <diopi/diopirt.h>

#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../ascend_tensor.hpp"

namespace impl {
namespace ascend {

using Shape = std::vector<int64_t>;
class FormatInfo {
public:
    FormatInfo(diopiMemoryFormat_t diopiFormat, aclFormat format, aclFormat baseFormat, std::string formatName, bool isPadded)
        : diopiFormat_(diopiFormat), format_(format), baseFormat_(baseFormat), formatName_(std::move(formatName)), isPadded_(isPadded) {}
    virtual ~FormatInfo() = default;
    virtual Shape inferShape(const Shape& dims) = 0;
    static Shape inferShapeLessTo4(const Shape& dims);
    diopiMemoryFormat_t diopiFormat() { return diopiFormat_; }
    aclFormat format() { return format_; }
    aclFormat baseFormat() { return baseFormat_; }
    std::string& formatName() { return formatName_; }
    bool isPadded() const { return isPadded_; }

private:
    diopiMemoryFormat_t diopiFormat_ = diopiMemoryFormat_t::Undefined;
    aclFormat format_ = aclFormat::ACL_FORMAT_UNDEFINED;
    aclFormat baseFormat_ = aclFormat::ACL_FORMAT_UNDEFINED;
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
    static std::unordered_map<aclFormat, std::shared_ptr<FormatInfo>> info;
};  // class FormatHelper

class NzFormatInfo : public FormatInfo {
public:
    NzFormatInfo() : FormatInfo(diopiMemoryFormat_t::FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND, "FRACTAL_NZ", true) {}
    Shape inferShape(const Shape& dims) override;
};
class NdFormatInfo : public FormatInfo {
public:
    NdFormatInfo() : FormatInfo(diopiMemoryFormat_t::ND, ACL_FORMAT_ND, ACL_FORMAT_ND, "ND", false) {}
    Shape inferShape(const Shape& dims) override;
};
class NC1HWC0FormatInfo : public FormatInfo {
public:
    NC1HWC0FormatInfo() : FormatInfo(diopiMemoryFormat_t::NC1HWC0, ACL_FORMAT_NC1HWC0, ACL_FORMAT_NCHW, "NC1HWC0", true) {}
    Shape inferShape(const Shape& dims) override;
};

class NCHWFormatInfo : public FormatInfo {
public:
    NCHWFormatInfo() : FormatInfo(diopiMemoryFormat_t::NCHW, ACL_FORMAT_NCHW, ACL_FORMAT_NCHW, "NCHW", false) {}
    Shape inferShape(const Shape& dims) override;
};
class NHWCFormatInfo : public FormatInfo {
public:
    NHWCFormatInfo() : FormatInfo(diopiMemoryFormat_t::ChannelsLast, ACL_FORMAT_NHWC, ACL_FORMAT_NHWC, "NHWC", false) {}
    Shape inferShape(const Shape& dims) override;
};
class ZFormatInfo : public FormatInfo {
public:
    ZFormatInfo() : FormatInfo(diopiMemoryFormat_t::FRACTAL_Z, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NCHW, "FRACTAL_Z", true) {}
    Shape inferShape(const Shape& dims) override;
};
class NDHWCFormatInfo : public FormatInfo {
public:
    NDHWCFormatInfo() : FormatInfo(diopiMemoryFormat_t::NDHWC, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW, "NDHWC", false) {}
    Shape inferShape(const Shape& dims) override;
};
class NCDHWFormatInfo : public FormatInfo {
public:
    NCDHWFormatInfo() : FormatInfo(diopiMemoryFormat_t::NCDHW, ACL_FORMAT_NCDHW, ACL_FORMAT_NCDHW, "NCDHW", false) {}
    Shape inferShape(const Shape& dims) override;
};
class NDC1HWC0FormatInfo : public FormatInfo {
public:
    NDC1HWC0FormatInfo() : FormatInfo(diopiMemoryFormat_t::NDC1HWC0, ACL_FORMAT_NDC1HWC0, ACL_FORMAT_NCDHW, "NDC1HWC0", true) {}
    Shape inferShape(const Shape& dims) override;
};
class Z3DFormatInfo : public FormatInfo {
public:
    Z3DFormatInfo() : FormatInfo(diopiMemoryFormat_t::FRACTAL_Z_3D, ACL_FRACTAL_Z_3D, ACL_FORMAT_NCDHW, "FRACTAL_Z_3D", true) {}
    Shape inferShape(const Shape& dims) override;
};

}  // namespace ascend
}  // namespace impl
#endif  // IMPL_ASCEND_COMMON_FORMAT_HELPER_H_
