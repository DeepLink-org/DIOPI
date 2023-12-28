/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "format_helper.h"

#include <diopi/diopirt.h>

#include <memory>
#include <string>

namespace impl {
namespace ascend {
constexpr int blocksize = 16;
std::unordered_map<aclFormat, std::shared_ptr<FormatInfo>> FormatHelper::info = {
    {ACL_FORMAT_NC1HWC0, std::make_shared<NC1HWC0FormatInfo>()},
    {ACL_FORMAT_ND, std::make_shared<NdFormatInfo>()},
    {ACL_FORMAT_NCHW, std::make_shared<NCHWFormatInfo>()},
    {ACL_FORMAT_NHWC, std::make_shared<NHWCFormatInfo>()},
    {ACL_FORMAT_FRACTAL_NZ, std::make_shared<NzFormatInfo>()},
    {ACL_FORMAT_FRACTAL_Z, std::make_shared<ZFormatInfo>()},
    {ACL_FORMAT_NDHWC, std::make_shared<NDHWCFormatInfo>()},
    {ACL_FORMAT_NCDHW, std::make_shared<NCDHWFormatInfo>()},
    {ACL_FORMAT_NDC1HWC0, std::make_shared<NDC1HWC0FormatInfo>()},
    {ACL_FRACTAL_Z_3D, std::make_shared<Z3DFormatInfo>()},
};

std::string FormatHelper::getFormatName(aclFormat format) {
    if (format == aclFormat::ACL_FORMAT_UNDEFINED) {
        return "Undefined";
    }
    const auto& itr = info.find(format);
    ASCEND_CHECK_ABORT(itr != info.end(), "not ascend format:%d", format);
    return itr->second->formatName();
}

std::string FormatHelper::getFormatName(diopiMemoryFormat_t format) { return getFormatName(getAclFormat(format)); }

bool FormatHelper::isBaseFormat(aclFormat format) { return getAclBaseFormat(format) == format; }

bool FormatHelper::isBaseFormat(diopiMemoryFormat_t format) { return getDiopiBaseFormat(format) == format; }

aclFormat FormatHelper::getAclBaseFormat(aclFormat format) {
    const auto& iter = info.find(format);
    ASCEND_CHECK_ABORT(iter != info.end(), "not ascend format:%s", getFormatName(format).c_str());
    return iter->second->baseFormat();
}

diopiMemoryFormat_t FormatHelper::getDiopiBaseFormat(diopiMemoryFormat_t format) {
    aclFormat base = getAclBaseFormat(getAclFormat(format));
    const auto& iter = info.find(base);
    ASCEND_CHECK_ABORT(iter != info.end(), "not ascend format:%s", getFormatName(format).c_str());
    return iter->second->diopiFormat();
}

aclFormat FormatHelper::getAclFormat(diopiMemoryFormat_t memoryFormat) {
    switch (memoryFormat) {
        case diopiMemoryFormat_t::Undefined:
            return aclFormat::ACL_FORMAT_UNDEFINED;
        case diopiMemoryFormat_t::Contiguous:
        case diopiMemoryFormat_t::ND:
            return aclFormat::ACL_FORMAT_ND;
        case diopiMemoryFormat_t::NCHW:
            return aclFormat::ACL_FORMAT_NCHW;
        case diopiMemoryFormat_t::ChannelsLast:
            return aclFormat::ACL_FORMAT_NHWC;
        case diopiMemoryFormat_t::NC1HWC0:
            return aclFormat::ACL_FORMAT_NC1HWC0;
        case diopiMemoryFormat_t::FRACTAL_Z:
            return aclFormat::ACL_FORMAT_FRACTAL_Z;
        case diopiMemoryFormat_t::NC1HWC0_C04:
            return aclFormat::ACL_FORMAT_NC1HWC0_C04;
        case diopiMemoryFormat_t::HWCN:
            return aclFormat::ACL_FORMAT_HWCN;
        case diopiMemoryFormat_t::NDHWC:
            return aclFormat::ACL_FORMAT_NDHWC;
        case diopiMemoryFormat_t::FRACTAL_NZ:
            return aclFormat::ACL_FORMAT_FRACTAL_NZ;
        case diopiMemoryFormat_t::NCDHW:
            return aclFormat::ACL_FORMAT_NCDHW;
        case diopiMemoryFormat_t::NDC1HWC0:
            return aclFormat::ACL_FORMAT_NDC1HWC0;
        case diopiMemoryFormat_t::FRACTAL_Z_3D:
            return aclFormat::ACL_FRACTAL_Z_3D;
        default:
            ASCEND_CHECK_ABORT(false, "acl not support diopiMemoryFormat_t:%d", memoryFormat);
            return aclFormat::ACL_FORMAT_UNDEFINED;
    }
}

Shape FormatHelper::getStorageSizes(diopiMemoryFormat_t format, const Shape& dims) {
    const auto& itr = info.find(getAclFormat(format));
    if (itr != info.end()) {
        return itr->second->inferShape(dims);
    }
    ASCEND_CHECK_ABORT(false, "acl not support format:%s", getFormatName(format).c_str());
    return {};
}

Shape FormatInfo::inferShapeLessTo4(const Shape& dims) {
    Shape res(4);
    ASCEND_CHECK_ABORT(dims.size() <= 4, "check failed in InferShapeLessTo4, input dim > 4");
    switch (dims.size()) {
        case 0:
            res[0] = 1;
            res[1] = 1;
            res[2] = 1;
            res[3] = 1;
            break;
        case 1:  // RESHAPE_TYPE_C
            res[0] = 1;
            res[1] = dims[0];
            res[2] = 1;
            res[3] = 1;
            break;
        case 2:  // RESHAPE_TYPE_CH
            res[0] = 1;
            res[1] = dims[0];
            res[2] = dims[1];
            res[3] = 1;
            break;
        case 3:  // RESHAPE_TYPE_CHW
            res[0] = 1;
            res[1] = dims[0];
            res[2] = dims[1];
            res[3] = dims[2];
            break;
        case 4:
            res[0] = dims[0];
            res[1] = dims[1];
            res[2] = dims[2];
            res[3] = dims[3];
            break;
        default:
            ASCEND_CHECK_ABORT(false, "dims of NCHW shape should not be greater than 4, which is %ld", dims.size());
    }
    return res;
}

Shape NC1HWC0FormatInfo::inferShape(const Shape& dims) {
    Shape res(5);
    ASCEND_CHECK_ABORT(dims.size() == 4, "NC1HWC0FormatInfo::inferShape but input dim != 4");
    res[0] = dims[0];
    res[1] = (dims[1] + blocksize - 1) / blocksize;
    res[2] = dims[2];
    res[3] = dims[3];
    res[4] = blocksize;
    return res;
}

Shape NHWCFormatInfo::inferShape(const Shape& dims) {
    ASCEND_CHECK_ABORT(dims.size() == 4, "input dim should be equal to 4 when InferShapeofNHWC");
    return dims;
}

Shape NzFormatInfo::inferShape(const Shape& dims) {
    Shape res;
    // sum(keepdim = false) may make tensor dim = 0
    Shape dim;
    for (int64_t i : dims) {
        dim.emplace_back(i);
    }
    // this action will move to GuessStorageSizeWhenConvertFormat
    if (dim.empty()) {
        dim.emplace_back(1);
    }
    if (dim.size() == 1) {
        dim.emplace_back(1);
    }
    int i = 0;
    for (; i < dim.size() - 2; i++) {
        res.emplace_back(dim[i]);
    }
    res.emplace_back((dim[i + 1] + blocksize - 1) / blocksize);
    res.emplace_back((dim[i] + blocksize - 1) / blocksize);
    res.emplace_back(blocksize);
    res.emplace_back(blocksize);
    return res;
}

Shape ZFormatInfo::inferShape(const Shape& dims) {
    if (dims.size() < 4) {
        return inferShape(inferShapeLessTo4(dims));
    }
    Shape res(4);
    res[0] = (dims[1] + blocksize - 1) / blocksize * dims[2] * dims[3];
    res[1] = (dims[0] + blocksize - 1) / blocksize;
    res[2] = blocksize;
    res[3] = blocksize;
    return res;
}

// NCDHW -> NDHWC
Shape NDHWCFormatInfo::inferShape(const Shape& dims) {
    ASCEND_CHECK_ABORT(dims.size() == 5, "cannot convert to NDHWC");
    Shape res(5);
    res[0] = dims[0];
    res[1] = dims[2];
    res[2] = dims[3];
    res[3] = dims[4];
    res[4] = dims[1];
    return res;
}

// NCDHW to NCDHW
Shape NCDHWFormatInfo::inferShape(const Shape& dims) {
    ASCEND_CHECK_ABORT(dims.size() == 5, "cannot convert to NCDHW");
    return dims;
}

// NCDHW to NDC1HWC0
Shape NDC1HWC0FormatInfo::inferShape(const Shape& dims) {
    ASCEND_CHECK_ABORT(dims.size() == 5, "cannot convert to NDC1HWC0");
    Shape res(6);
    res[0] = dims[0];
    res[1] = dims[2];
    res[2] = (dims[1] + blocksize - 1) / blocksize;
    res[3] = dims[3];
    res[4] = dims[4];
    res[5] = blocksize;
    return res;
}

// NCDHW to FZ_3D
Shape Z3DFormatInfo::inferShape(const Shape& dims) {
    ASCEND_CHECK_ABORT(dims.size() == 5, "cannot convert to FZ_3D");
    int64_t d1 = dims[2];
    int64_t d2 = (dims[1] + blocksize - 1) / blocksize;
    int64_t d3 = dims[3];
    int64_t d4 = dims[4];
    int64_t d5 = (dims[0] + blocksize - 1) / blocksize;
    int64_t d6 = blocksize;
    int64_t d7 = blocksize;
    // The shape of FZ3D is 7D, but the CANN only accept 4D
    // so we should merge 1st, 2nd, 3rd, 4th dimension.
    Shape res(4);
    res[0] = d1 * d2 * d3 * d4;
    res[1] = d5;
    res[2] = d6;
    res[3] = d7;
    return res;
}

Shape NCHWFormatInfo::inferShape(const Shape& dims) {
    if (dims.size() < 5) {
        return inferShapeLessTo4(dims);
    }
    return dims;
}

Shape NdFormatInfo::inferShape(const Shape& dims) { return dims; }

}  // namespace ascend
}  // namespace impl
