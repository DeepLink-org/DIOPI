/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include "format_helper.h"

#include <diopi/diopirt.h>

#include <string>

namespace impl {
namespace ascend {
namespace {
Shape inferShapeLessTo4(const Shape& dims);
Shape inferShape4To5(const Shape& dims);
Shape inferShapeofNhwc(const Shape& dims);
Shape inferShape5To4(const Shape& dims);
Shape inferShapeNdToNz(const Shape& dims);
Shape inferShapeNdToZ(const Shape& dims);
Shape inferShapeOfNdhwc(const Shape& dims);
Shape inferShapeOfNcdhw(const Shape& dims);
Shape inferShapeOfNdC1HwC0(const Shape& dims);
Shape inferShapeOfFZ3D(const Shape& dims);
Shape inferShapeofNchw(const Shape& dims);
Shape inferShapeofNd(const Shape& dims);
}  // namespace

std::unordered_map<aclFormat, FormatInfo> FormatHelper::info = {
    {ACL_FORMAT_NC1HWC0, (FormatInfo){diopiMemoryFormat_t::NC1HWC0, ACL_FORMAT_NC1HWC0, ACL_FORMAT_NCHW, inferShape4To5, "NC1HWC0", true}},
    {ACL_FORMAT_ND, (FormatInfo){diopiMemoryFormat_t::ND, ACL_FORMAT_ND, ACL_FORMAT_ND, inferShapeofNd, "ND", false}},
    {ACL_FORMAT_NCHW, (FormatInfo){diopiMemoryFormat_t::NCHW, ACL_FORMAT_NCHW, ACL_FORMAT_NCHW, inferShapeofNchw, "NCHW", false}},
    {ACL_FORMAT_NHWC, (FormatInfo){diopiMemoryFormat_t::ChannelsLast, ACL_FORMAT_NHWC, ACL_FORMAT_NHWC, inferShapeofNhwc, "NHWC", false}},
    {ACL_FORMAT_FRACTAL_NZ, (FormatInfo){diopiMemoryFormat_t::FRACTAL_NZ, ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND, inferShapeNdToNz, "FRACTAL_NZ", true}},
    {ACL_FORMAT_FRACTAL_Z, (FormatInfo){diopiMemoryFormat_t::FRACTAL_Z, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NCHW, inferShapeNdToZ, "FRACTAL_Z", true}},
    {ACL_FORMAT_NDHWC, (FormatInfo){diopiMemoryFormat_t::NDHWC, ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW, inferShapeOfNdhwc, "NDHWC", false}},
    {ACL_FORMAT_NCDHW, (FormatInfo){diopiMemoryFormat_t::NCDHW, ACL_FORMAT_NCDHW, ACL_FORMAT_NCDHW, inferShapeOfNcdhw, "NCDHW", false}},
    {ACL_FORMAT_NDC1HWC0, (FormatInfo){diopiMemoryFormat_t::NDC1HWC0, ACL_FORMAT_NDC1HWC0, ACL_FORMAT_NCDHW, inferShapeOfNdC1HwC0, "NDC1HWC0", true}},
    {ACL_FRACTAL_Z_3D, (FormatInfo){diopiMemoryFormat_t::FRACTAL_Z_3D, ACL_FRACTAL_Z_3D, ACL_FORMAT_NCDHW, inferShapeOfFZ3D, "FRACTAL_Z_3D", true}},
};

std::string FormatHelper::getFormatName(aclFormat format) {
    if (format == aclFormat::ACL_FORMAT_UNDEFINED) {
        return "Undefined";
    }
    const auto& itr = info.find(format);
    ASCEND_CHECK_ABORT(itr != info.end(), "not ascend format:%d", format);
    return itr->second.formatName_;
}

std::string FormatHelper::getFormatName(diopiMemoryFormat_t format) { return getFormatName(getAclFormat(format)); }

bool FormatHelper::isBaseFormat(aclFormat format) { return getAclBaseFormat(format) == format; }

bool FormatHelper::isBaseFormat(diopiMemoryFormat_t format) { return getDiopiBaseFormat(format) == format; }

aclFormat FormatHelper::getAclBaseFormat(aclFormat format) {
    const auto& iter = info.find(format);
    ASCEND_CHECK_ABORT(iter != info.end(), "not ascend format:%s", getFormatName(format).c_str());
    return iter->second.baseFormat_;
}

diopiMemoryFormat_t FormatHelper::getDiopiBaseFormat(diopiMemoryFormat_t format) {
    aclFormat base = getAclBaseFormat(getAclFormat(format));
    const auto& iter = info.find(base);
    ASCEND_CHECK_ABORT(iter != info.end(), "not ascend format:%s", getFormatName(format).c_str());
    return iter->second.diopiFormat_;
}

aclFormat FormatHelper::getAclFormat(diopiMemoryFormat_t memoryFormat) {
    switch (memoryFormat) {
        case diopiMemoryFormat_t::Undefined:
            return aclFormat::ACL_FORMAT_UNDEFINED;
        case diopiMemoryFormat_t::Contiguous:
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
        if (itr->second.func_) {
            return itr->second.func_(dims);
        }
    }
    ASCEND_CHECK_ABORT(false, "acl not support format:%s", getFormatName(format).c_str());
    return {};
}

namespace {
Shape inferShapeLessTo4(const Shape& dims) {
    Shape res;
    res.resize(4);
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

Shape inferShape4To5(const Shape& dims) {
    Shape res;
    res.resize(5);
    ASCEND_CHECK_ABORT(dims.size() == 4, "infershape4to5 but input dim != 4");
    res[0] = dims[0];
    res[1] = (dims[1] + 15) / 16;
    res[2] = dims[2];
    res[3] = dims[3];
    res[4] = blocksize;
    return res;
}

Shape inferShapeofNhwc(const Shape& dims) {
    ASCEND_CHECK_ABORT(dims.size() == 4, "input dim should be equal to 4 when InferShapeofNHWC");
    return dims;
}

Shape inferShape5To4(const Shape& dims) {
    ASCEND_CHECK_ABORT(dims.size() >= 4, "input dim must >= 4 in function InferShape5To4");
    Shape res;
    res.emplace_back(dims[0]);
    res.emplace_back(((dims[1] + 15) / 16) * 16);
    res.emplace_back(dims[2]);
    res.emplace_back(dims[3]);
    return res;
}

Shape inferShapeNdToNz(const Shape& dims) {
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
    res.emplace_back((dim[i + 1] + 15) / blocksize);
    res.emplace_back((dim[i] + 15) / blocksize);
    res.emplace_back(blocksize);
    res.emplace_back(blocksize);
    return res;
}

Shape inferShapeNdToZ(const Shape& dims) {
    if (dims.size() < 4) {
        return inferShapeNdToZ(inferShapeLessTo4(dims));
    }
    Shape res;
    res.emplace_back((dims[1] + 15) / blocksize * dims[2] * dims[3]);
    res.emplace_back((dims[0] + 15) / blocksize);
    res.emplace_back(blocksize);
    res.emplace_back(blocksize);
    return res;
}

// NCDHW -> NDHWC
Shape inferShapeOfNdhwc(const Shape& dims) {
    ASCEND_CHECK_ABORT(dims.size() == 5, "cannot convert to NDHWC");
    Shape res;
    res.resize(5);
    res[0] = dims[0];
    res[1] = dims[2];
    res[2] = dims[3];
    res[3] = dims[4];
    res[4] = dims[1];
    return res;
}

// NCDHW to NCDHW
Shape inferShapeOfNcdhw(const Shape& dims) {
    ASCEND_CHECK_ABORT(dims.size() == 5, "cannot convert to NCDHW");
    return dims;
}

// NCDHW to NDC1HWC0
Shape inferShapeOfNdC1HwC0(const Shape& dims) {
    ASCEND_CHECK_ABORT(dims.size() == 5, "cannot convert to NDC1HWC0");
    Shape res;
    res.resize(6);
    res[0] = dims[0];
    res[1] = dims[2];
    res[2] = (dims[1] + blocksize - 1) / blocksize;
    res[3] = dims[3];
    res[4] = dims[4];
    res[5] = blocksize;
    return res;
}

// NCDHW to FZ_3D
Shape inferShapeOfFZ3D(const Shape& dims) {
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
    Shape res;
    res.resize(4);
    res[0] = d1 * d2 * d3 * d4;
    res[1] = d5;
    res[2] = d6;
    res[3] = d7;
    return res;
}

Shape inferShapeofNchw(const Shape& dims) {
    if (dims.size() < 5) {
        return inferShapeLessTo4(dims);
    }
    return inferShapeofNd(dims);
}

Shape inferShapeofNd(const Shape& dims) { return dims; }
}  // namespace

}  // namespace ascend
}  // namespace impl
