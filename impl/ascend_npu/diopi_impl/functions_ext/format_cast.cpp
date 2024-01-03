/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#include "../helper.hpp"
#include "torch_npu/csrc/aten/CustomFunctions.h"

namespace OP_IMPL_NS {
static std::unordered_map<diopiMemoryFormat_t, aclFormat> diopiAclFormatMap{{diopiMemoryFormat_t::Undefined, aclFormat::ACL_FORMAT_UNDEFINED},
                                                                            {diopiMemoryFormat_t::ND, aclFormat::ACL_FORMAT_ND},
                                                                            {diopiMemoryFormat_t::NCHW, aclFormat::ACL_FORMAT_NCHW},
                                                                            {diopiMemoryFormat_t::ChannelsLast, aclFormat::ACL_FORMAT_NHWC},
                                                                            {diopiMemoryFormat_t::NC1HWC0, aclFormat::ACL_FORMAT_NC1HWC0},
                                                                            {diopiMemoryFormat_t::FRACTAL_Z, aclFormat::ACL_FORMAT_FRACTAL_Z},
                                                                            {diopiMemoryFormat_t::NC1HWC0_C04, aclFormat::ACL_FORMAT_NC1HWC0_C04},
                                                                            {diopiMemoryFormat_t::HWCN, aclFormat::ACL_FORMAT_HWCN},
                                                                            {diopiMemoryFormat_t::FRACTAL_NZ, aclFormat::ACL_FORMAT_FRACTAL_NZ},
                                                                            {diopiMemoryFormat_t::NCDHW, aclFormat::ACL_FORMAT_NCDHW},
                                                                            {diopiMemoryFormat_t::NDC1HWC0, aclFormat::ACL_FORMAT_NDC1HWC0},
                                                                            {diopiMemoryFormat_t::FRACTAL_Z_3D, aclFormat::ACL_FRACTAL_Z_3D}};
static std::unordered_map<aclFormat, diopiMemoryFormat_t> aclDiopiFormatMap{{aclFormat::ACL_FORMAT_UNDEFINED, diopiMemoryFormat_t::Undefined},
                                                                            {aclFormat::ACL_FORMAT_ND, diopiMemoryFormat_t::ND},
                                                                            {aclFormat::ACL_FORMAT_NCHW, diopiMemoryFormat_t::NCHW},
                                                                            {aclFormat::ACL_FORMAT_NHWC, diopiMemoryFormat_t::ChannelsLast},
                                                                            {aclFormat::ACL_FORMAT_NC1HWC0, diopiMemoryFormat_t::NC1HWC0},
                                                                            {aclFormat::ACL_FORMAT_FRACTAL_Z, diopiMemoryFormat_t::FRACTAL_Z},
                                                                            {aclFormat::ACL_FORMAT_NC1HWC0_C04, diopiMemoryFormat_t::NC1HWC0_C04},
                                                                            {aclFormat::ACL_FORMAT_HWCN, diopiMemoryFormat_t::HWCN},
                                                                            {aclFormat::ACL_FORMAT_FRACTAL_NZ, diopiMemoryFormat_t::FRACTAL_NZ},
                                                                            {aclFormat::ACL_FORMAT_NCDHW, diopiMemoryFormat_t::NCDHW},
                                                                            {aclFormat::ACL_FORMAT_NDC1HWC0, diopiMemoryFormat_t::NDC1HWC0},
                                                                            {aclFormat::ACL_FRACTAL_Z_3D, diopiMemoryFormat_t::FRACTAL_Z_3D}};
diopiError_t diopiFormatCast(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiTensorHandle_t in, diopiMemoryFormat_t format) {
    const auto& itr = diopiAclFormatMap.find(format);
    if (itr == diopiAclFormatMap.end()) {
        DIOPI_CHECK(false, "acl not support.");
    }
    impl::aten::setCurCtx(ctx);
    at::Tensor* inPtr = reinterpret_cast<at::Tensor*>(in);
    at::Tensor* outPtr = reinterpret_cast<at::Tensor*>(out);
    at::Tensor inAt = impl::aten::buildATen(in);
    if (typeid(*(inPtr->storage().unsafeGetStorageImpl())) == typeid(torch_npu::NPUStorageImpl)) {
        at_npu::native::StorageDescHelper::CopyDesc(inAt, *inPtr);
    }
    at::Tensor result = at_npu::native::custom_ops::_npu_format_cast(inAt, itr->second);
    c10::intrusive_ptr<c10::StorageImpl> storage = c10::make_intrusive<torch_npu::NPUStorageImpl>(
        at::StorageImpl::use_byte_size_t(),
        at::detail::computeStorageNbytes(result.sizes(), result.strides(), result.options().dtype().itemsize()),
        c10::InefficientStdFunctionContext::makeDataPtr(result.data_ptr(), c10::detail::deleteNothing, outPtr->options().device()),
        nullptr,
        /*resizable=*/false);
    outPtr->set_(std::move(storage), result.storage_offset(), result.sizes(), result.strides());
    at_npu::native::StorageDescHelper::CopyDesc(*outPtr, result);
    impl::aten::unsetCurCtx();
    return diopiSuccess;
}

diopiError_t diopiGetFormat(diopiContextHandle_t ctx, diopiConstTensorHandle_t in, diopiMemoryFormat_t* result) {
    const at::Tensor* ptr = reinterpret_cast<const at::Tensor*>(in);
    torch_npu::utils::torch_check_npu(*ptr);
    impl::aten::setCurCtx(ctx);
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    if (typeid(*(ptr->storage().unsafeGetStorageImpl())) == typeid(torch_npu::NPUStorageImpl)) {
        format = at_npu::native::FormatHelper::GetFormat(*ptr);
    } else {
        format = at_npu::native::FormatHelper::GetFormat(impl::aten::buildATen(in));
    }
    impl::aten::unsetCurCtx();
    const auto& itr = aclDiopiFormatMap.find(format);
    if (itr == aclDiopiFormatMap.end()) {
        DIOPI_CHECK(false, "acl not support.");
    }
    *result = itr->second;
    return diopiSuccess;
}

}  // namespace OP_IMPL_NS
