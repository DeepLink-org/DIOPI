/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "helper.hpp"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace OP_IMPL_NS {
static std::unordered_map<void*, torch_npu::NPUStorageDesc> dataMap;
diopiError_t diopiNativeMemoryFormatCast(diopiContextHandle_t ctx, diopiTensorHandle_t* out, diopiTensorHandle_t in, int64_t format) {
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
    at::Tensor result = at_npu::native::custom_ops::_npu_format_cast(inAt, format);
    if (result.data_ptr() == inAt.data_ptr()) {
        *out = in;
    }
    auto& desc = torch_npu::NPUBridge::GetNpuStorageImpl(result)->npu_desc_;
    *out = const_cast<diopiTensorHandle_t>(desc.diopi_tensor_);
    dataMap[result.data_ptr()] = desc;
    END_CALL_ACL_OP();
    return diopiSuccess;
}

diopiError_t diopiGetNativeMemoryFormat(diopiContextHandle_t ctx, diopiConstTensorHandle_t in, int64_t* result) {
    BEGIN_CALL_ACL_OP(in);
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    const auto& dataItr = dataMap.find(inAt.data_ptr());
    if (dataItr != dataMap.end()) {
        format = dataItr->second.npu_format_;
    } else {
        format = at_npu::native::FormatHelper::GetFormat(inAt);
    }
    *result = format;
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
