/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace OP_IMPL_NS {
static std::unordered_map<diopiCustomFormat_t, aclFormat> diopiAclFormatMap{{diopiCustomFormat_t::Undefined, aclFormat::ACL_FORMAT_UNDEFINED},
                                                                            {diopiCustomFormat_t::ND, aclFormat::ACL_FORMAT_ND},
                                                                            {diopiCustomFormat_t::NCHW, aclFormat::ACL_FORMAT_NCHW},
                                                                            {diopiCustomFormat_t::NHWC, aclFormat::ACL_FORMAT_NHWC},
                                                                            {diopiCustomFormat_t::NC1HWC0, aclFormat::ACL_FORMAT_NC1HWC0},
                                                                            {diopiCustomFormat_t::FRACTAL_Z, aclFormat::ACL_FORMAT_FRACTAL_Z},
                                                                            {diopiCustomFormat_t::NC1HWC0_C04, aclFormat::ACL_FORMAT_NC1HWC0_C04},
                                                                            {diopiCustomFormat_t::HWCN, aclFormat::ACL_FORMAT_HWCN},
                                                                            {diopiCustomFormat_t::FRACTAL_NZ, aclFormat::ACL_FORMAT_FRACTAL_NZ},
                                                                            {diopiCustomFormat_t::NCDHW, aclFormat::ACL_FORMAT_NCDHW},
                                                                            {diopiCustomFormat_t::NDC1HWC0, aclFormat::ACL_FORMAT_NDC1HWC0},
                                                                            {diopiCustomFormat_t::FRACTAL_Z_3D, aclFormat::ACL_FRACTAL_Z_3D}};
static std::unordered_map<aclFormat, diopiCustomFormat_t> aclDiopiFormatMap{{aclFormat::ACL_FORMAT_UNDEFINED, diopiCustomFormat_t::Undefined},
                                                                            {aclFormat::ACL_FORMAT_ND, diopiCustomFormat_t::ND},
                                                                            {aclFormat::ACL_FORMAT_NCHW, diopiCustomFormat_t::NCHW},
                                                                            {aclFormat::ACL_FORMAT_NHWC, diopiCustomFormat_t::NHWC},
                                                                            {aclFormat::ACL_FORMAT_NC1HWC0, diopiCustomFormat_t::NC1HWC0},
                                                                            {aclFormat::ACL_FORMAT_FRACTAL_Z, diopiCustomFormat_t::FRACTAL_Z},
                                                                            {aclFormat::ACL_FORMAT_NC1HWC0_C04, diopiCustomFormat_t::NC1HWC0_C04},
                                                                            {aclFormat::ACL_FORMAT_HWCN, diopiCustomFormat_t::HWCN},
                                                                            {aclFormat::ACL_FORMAT_FRACTAL_NZ, diopiCustomFormat_t::FRACTAL_NZ},
                                                                            {aclFormat::ACL_FORMAT_NCDHW, diopiCustomFormat_t::NCDHW},
                                                                            {aclFormat::ACL_FORMAT_NDC1HWC0, diopiCustomFormat_t::NDC1HWC0},
                                                                            {aclFormat::ACL_FRACTAL_Z_3D, diopiCustomFormat_t::FRACTAL_Z_3D}};
static std::unordered_map<void*, torch_npu::NPUStorageDesc> dataMap;
diopiError_t diopiCustomFormatCast(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiTensorHandle_t in, diopiCustomFormat_t format) {
    const auto& itr = diopiAclFormatMap.find(format);
    DIOPI_CHECK(itr != diopiAclFormatMap.end(), "acl not support.");
    aclFormat dstAclFormat = itr->second;
    BEGIN_CALL_ACL_OP(in);
    const auto& dataItr = dataMap.find(inAt.data_ptr());
    if (dataItr != dataMap.end()) {
        auto& desc = dataItr->second;
        at_npu::native::StorageDescHelper::CopyDesc(inAt, desc);
        if (!desc.base_strides_.empty()) {
            inAt.unsafeGetTensorImpl()->set_sizes_and_strides(desc.base_sizes_, desc.base_strides_);
        } else {
            inAt.unsafeGetTensorImpl()->set_sizes_contiguous(desc.base_sizes_);
        }
    }
    at::Tensor result = at_npu::native::custom_ops::_npu_format_cast(inAt, dstAclFormat);
    if (result.data_ptr() == inAt.data_ptr()) {
        *out = in;
    }
    auto& desc = torch_npu::NPUBridge::GetNpuStorageImpl(result)->npu_desc_;
    *out = const_cast<diopiTensorHandle_t>(desc.diopi_tensor_);
    dataMap[result.data_ptr()] = desc;
    END_CALL_ACL_OP();
    return diopiSuccess;
}

diopiError_t diopiGetCustomFormat(diopiContextHandle_t ctx, diopiTensorHandle_t in, diopiCustomFormat_t* result) {
    BEGIN_CALL_ACL_OP(in);
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    const auto& dataItr = dataMap.find(inAt.data_ptr());
    if (dataItr != dataMap.end()) {
        format = dataItr->second.npu_format_;
    } else {
        format = at_npu::native::FormatHelper::GetFormat(inAt);
    }
    const auto& itr = aclDiopiFormatMap.find(format);
    DIOPI_CHECK(itr != aclDiopiFormatMap.end(), "acl not support.");
    *result = itr->second;
    END_CALL_ACL_OP();
    return diopiSuccess;
}

diopiError_t diopiTensorDestructionHook(diopiContextHandle_t ctx, void* ptr) {
    const auto& it = dataMap.find(ptr);
    if (it != dataMap.end()) {
        dataMap.erase(it);
    }
    return diopiSuccess;
}

}  // namespace OP_IMPL_NS
