#include "torch_npu/csrc/framework/DIOPIAdapter.h"

#include <ATen/EmptyTensor.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/record_function.h>
#include <diopi/diopirt.h>
#include <torch/library.h>

#include "../../../ascend/common/gil_scoped_release.hpp"
#include "../../../ascend/common/stream_lock.hpp"
#include "diopi_impl/helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace {
constexpr float EPSILON = 1e-6;

int current_device() {
    int devId_ = 0;
    ::aclrtGetDevice(&devId_);
    return devId_;
}

inline bool enableDumpArgs() { return std::getenv("DIOPI_DEBUG_OP") != nullptr; }

// check all at::ScalarType is not negative
#define ENUM_PAIR_FUNC(_1, n) static_assert(static_cast<int64_t>(at::ScalarType::n) >= 0, #n " is negative");
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(ENUM_PAIR_FUNC)
#undef ENUM_PAIR_FUNC

#define AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_)  \
    _(at::ScalarType::Byte, ACL_UINT8)               \
    _(at::ScalarType::Char, ACL_INT8)                \
    _(at::ScalarType::Short, ACL_INT16)              \
    _(at::ScalarType::Int, ACL_INT32)                \
    _(at::ScalarType::Long, ACL_INT64)               \
    _(at::ScalarType::Half, ACL_FLOAT16)             \
    _(at::ScalarType::Float, ACL_FLOAT)              \
    _(at::ScalarType::Double, ACL_DOUBLE)            \
    _(at::ScalarType::ComplexHalf, ACL_COMPLEX32)    \
    _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)   \
    _(at::ScalarType::ComplexDouble, ACL_COMPLEX128) \
    _(at::ScalarType::Bool, ACL_BOOL)                \
    _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)       \
    _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)      \
    _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)      \
    _(at::ScalarType::BFloat16, ACL_BF16)            \
    _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)    \
    _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)    \
    _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)   \
    _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

constexpr aclDataType kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
    AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

// check at::ScalarType has been changed or not
#define ENUM_PAIR_FUNC(at_dtype, acl_dtype)                                                         \
    static_assert(kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(at_dtype)] == (acl_dtype), \
                  #at_dtype " and " #acl_dtype                                                      \
                            " is not match any more, please check "                                 \
                            "AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR and modify it");
AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(ENUM_PAIR_FUNC)
#undef DEFINE_ENUM

static std::map<const string, const aclDataType> STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP = {
    {"uint16", ACL_UINT16}, {"uint8", ACL_UINT8}, {"uint64", ACL_UINT64}, {"string", ACL_STRING}};

aclError AclrtMemcpyAsyncParamCheck(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind, aclrtStream stream) {
    auto ret = aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
    return ret;
}

aclError AclrtMemcpyParamCheck(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind) {
    auto ret = aclrtMemcpy(dst, destMax, src, count, kind);
    return ret;
}
}  // namespace

namespace at_npu {
namespace native {

bool FormatCastHelper::IsSameGroupType(const at::Tensor& src, const at::Tensor& dst) {
    auto src_format = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_.npu_format_;
    auto dst_format = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_.npu_format_;
    return FormatHelper::GetBaseFormat(src_format) == FormatHelper::GetBaseFormat(dst_format);
}

void FormatCastHelper::base_format_cast_nocheck(at::Tensor& dst, const at::Tensor& src) {
    dst.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());
    NPUNativeFunctions::copy_memory_(dst, src, true);
}

void FormatCastHelper::format_cast_as_base_format(const at::Tensor& src, aclFormat format) {
    AT_ASSERT(FormatHelper::IsBaseFormatType(format), "dst format must be base format");
    AT_ASSERT(FormatHelper::IsBaseFormatType(src), "src format must be base format");

    auto& src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    // due to CANN principle : if the ori format of a tensor is the
    // same as the npu format, then its base shape must be same as storage shape
    // so we should not change the storage shape when format cast between base format
    src_desc.origin_format_ = format;
    src_desc.npu_format_ = format;
    return;
}

bool FormatCastHelper::format_cast_between_group(at::Tensor& dst, const at::Tensor& src, FormatCastHelper::FormatCastFunc format_cast_inside_group) {
    if (FormatHelper::IsBaseFormatType(src)) {
        if (FormatHelper::IsBaseFormatType(dst)) {
            // src base format (src format) -> dst base format
            base_format_cast_nocheck(dst, src);  // only need to copy memory
            return true;
        } else {
            // src base format (src format) -> dst base format
            // dst base format -> dst format
            auto src_base_format = FormatHelper::GetBaseFormat(src);
            format_cast_as_base_format(src, FormatHelper::GetBaseFormat(dst));  // prepare: covert src to dst base format
            format_cast_inside_group(dst, src);                                 // src base format (src format) -> dst base format
            format_cast_as_base_format(src, src_base_format);                   // recover: dst base format -> dst format
            return true;
        }
    } else {
        if (FormatHelper::IsBaseFormatType(dst)) {
            // src format -> src base format
            // src base format -> dst base format (dst format)
            auto dst_base_format = FormatHelper::GetBaseFormat(dst);
            format_cast_as_base_format(dst, FormatHelper::GetBaseFormat(src));  // prepare: cover dst to src base format
            format_cast_inside_group(dst, src);                                 // src format -> src base format
            format_cast_as_base_format(dst, dst_base_format);                   // recover: src base format -> dst format
            return true;
        }
    }
    return false;
}

at::Tensor FormatCastHelper::ApplyBaseFormatTensorBy(const at::Tensor& src) {
    auto format = FormatHelper::GetBaseFormat(src);
    return custom_ops::npu_format_cast(src, format);
}

at::Tensor& FormatCastHelper::CovertSelfToBaseFormat(at::Tensor& src) {
    auto format = FormatHelper::GetBaseFormat(src);
    return custom_ops::npu_format_cast_(src, format);
}

UnifiedResult OpPreparation::binary_op_check(at::Tensor& out, const at::Tensor& a, const at::Tensor& b, bool check_mem_overlap) {
    UnifiedResult unified_result;
    unified_result.common_type = out.scalar_type();
    unified_result.common_shape = out.sizes();
    return unified_result;
}

UnifiedResult OpPreparation::binary_op_check(at::Tensor& out, const at::Tensor& a, const c10::Scalar b, bool check_mem_overlap) {
    UnifiedResult unified_result;
    unified_result.common_type = out.scalar_type();
    unified_result.common_shape = out.sizes();
    return unified_result;
}

void NpuUtils::format_fresh_view(at::Tensor& x, const at::Tensor& y) {
    // x:NPU before inplace_op, y: NPU computed
    // now we fresh x according to y
    RECORD_FUNCTION("format_fresh_view", vector<c10::IValue>({x, y}));
    x.copy_(y);
}

// NOTE [Check Match for Npu at::Tensor]
// check_match is used to ensure that npu tensor satisfies the
// calculation requirements of npu operators.
// The rules are as follows,
// 1、tensor should be contiguous
// Not contiguous means the operator needs to read and write memory
// at intervals according to strides and sizes. Npu operators has
// no such ability for the time being
// 2、metadata should be match
// Resize_ a contiguous cpu tensor from [1,2,3,4] to [4,3,2,1] no
// need to change the physical memory. However, for a contiguous npu
// tensor whose npu_format_ is 5HD, storage shape should be change
// from [1,1,3,4,16] to [4,1,2,1,16]. So metadata not match often
// results in unexpected physical memory. format_contiguous will be
// called preparing correct memory of operand in these case.
bool NpuUtils::check_match(const at::Tensor* tensor) {
    // case1:uncontiguous tensor
    if (!tensor->is_contiguous()) {
        return false;
    }

    // case2:meta data not match, sizes or strides of presentation
    // layer is different from that of storage layer
    if (!StorageDescHelper::MetaDataAreMatch(tensor)) {
        return false;
    }

    // case3:meta data not match, storage_offset of presentation layer
    // is different from that of storage layer
    bool isPadding = FormatHelper::IsPadded(tensor);
    if (isPadding && (!StorageDescHelper::OffsetAreMatch(tensor))) {
        return false;
    }
    return true;
}

bool NpuUtils::check_5d_5d_match(const at::Tensor& tensor) {
    // (1) NC1HWC0 format in storage, NCHW format in des.
    // (2) 4d format situation, only uncontiguous in Channel size
    // (3) size and start point must be 16*, make sure the memory be contiguous
    if (tensor.is_contiguous()) {
        return false;
    }

    if (torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.npu_format_ != ACL_FORMAT_NC1HWC0) {
        return false;
    }

    if (tensor.sizes().size() != 4) {
        return false;
    }

    bool is_c_channel_slice = true;
    int64_t z = 1;
    for (int64_t d = tensor.dim() - 1; d >= 1; d--) {
        if (tensor.size(d) != 1) {
            if (tensor.stride(d) == z) {
                z *= tensor.size(d);
            } else {
                is_c_channel_slice = false;
                break;
            }
        }
    }
    if (!is_c_channel_slice) {
        return false;
    }

    int64_t contiguous_len = 16;
    int64_t c0_len = 16;
    for (const auto i : c10::irange(2, torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.base_sizes_.size())) {
        contiguous_len *= torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.base_sizes_[i];
    }
    bool is_offset_match = (tensor.storage_offset() % contiguous_len == 0);
    bool is_length_match = (tensor.size(1) % c0_len == 0);

    return is_offset_match && is_length_match;
}

void NpuUtils::RefreshFormat(const at::Tensor& tensor) {
    auto& tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    if (tensor_desc.storage_sizes_.size() == 4 && tensor_desc.npu_format_ == ACL_FORMAT_ND) {
        tensor_desc.npu_format_ = ACL_FORMAT_NCHW;
        tensor_desc.origin_format_ = ACL_FORMAT_NCHW;
    } else if (tensor_desc.storage_sizes_.size() != 4 && tensor_desc.npu_format_ == ACL_FORMAT_NCHW) {
        tensor_desc.npu_format_ = ACL_FORMAT_ND;
        tensor_desc.origin_format_ = ACL_FORMAT_ND;
    }
}

at::Tensor metadata_convert_match(const at::Tensor& src, bool numelEq) {
    // Only when a tensor monopolizes a storage can NpuStorageDesc be
    // refreshed. When the original format is not NCHW, the npu_format_cast to
    // NCHW will generate a temporary tensor, which always monopolizes its own
    // storage.
    if (numelEq && (!FormatHelper::IsBaseFormatType(src))) {
        at::Tensor tempTensor = custom_ops::npu_format_cast(src, FormatHelper::GetBaseFormat(src));
        custom_ops::npu_reshape_out(tempTensor, tempTensor.sizes(), true, tempTensor);
        NpuUtils::RefreshFormat(tempTensor);
        return tempTensor;
    } else {
        at::Tensor contiguous_view = at::empty(src.sizes(), src.options());
        contiguous_view.copy_(src);
        NpuUtils::RefreshFormat(contiguous_view);
        return contiguous_view;
    }
}

at::Tensor metadata_convert_match_without_copy_optimize(const at::Tensor& src) {
    TORCH_CHECK(src.device().type() == at_npu::key::NativeDeviceType,
                "Expected all tensors to be on the same device. "
                "Expected NPU tensor, please check whether the input tensor device is correct.");
    auto& src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    bool numelEq = (src.numel() == c10::multiply_integers(src_desc.base_sizes_));
    return metadata_convert_match(src, numelEq);
}

at::Tensor metadata_convert_match_with_copy_optimize(const at::Tensor& src) {
    TORCH_CHECK(src.device().type() == at_npu::key::NativeDeviceType,
                "Expected all tensors to be on the same device. "
                "Expected NPU tensor, please check whether the input tensor device is correct.");
    auto& src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    bool numelEq = (src.numel() == c10::multiply_integers(src_desc.base_sizes_));

    // For unmatched Tensors with base format, we can:
    OptimizationCases optimizations_reshape{"reshapeV2"};
    if (numelEq && src_desc.npu_format_ == ACL_FORMAT_ND && src_desc.origin_format_ == ACL_FORMAT_ND && (src.dim() != 0) && !src_desc.base_sizes_.empty()) {
        // 1. directly rewrite their storage description to get matched tensors.
        src_desc.base_sizes_ = CalcuOpUtil::ConvertIntArrayRefToSmallVector(src.sizes());
        src_desc.base_strides_ = CalcuOpUtil::ConvertIntArrayRefToSmallVector(src.strides());
        src_desc.storage_sizes_ = CalcuOpUtil::ConvertIntArrayRefToSmallVector(src.sizes());
        NpuUtils::RefreshFormat(src);
        return src;
    } else if (TransContiguous::CanOptimize(src, optimizations_reshape)) {
        // 2. using memory-repoint/DMA for other cases.
        auto reshapeTensor = TransContiguous::ContiguousOptimizeWithAnyFormat(src, optimizations_reshape);
        if (reshapeTensor.has_value()) {
            return reshapeTensor.value();
        }
    }
    // 3. common method using transdata and copy_, just the same as:
    // metadata_convert_match_without_copy_optimize
    return metadata_convert_match(src, numelEq);
}

at::Tensor metadata_with_offset_padding_convert_match(const at::Tensor& src) {
    at::Tensor contiguous_view = at::empty(src.sizes(), src.options());
    contiguous_view.copy_(src);
    NpuUtils::RefreshFormat(contiguous_view);
    return contiguous_view;
}

at::Tensor NpuUtils::format_contiguous(const at::Tensor& src) {
    // case1:tensor src is not contiguous
    if (!src.is_contiguous()) {
        RECORD_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
        return src.contiguous();
    }
    // case2:meta data not match, sizes or strides of presentation
    // layer is different from that of storage layer
    if (!StorageDescHelper::MetaDataAreMatch(&src)) {
        // Fix not match case2, tensor should have matched metadata and
        // NPUStorageDesc.
        RECORD_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
        return metadata_convert_match_without_copy_optimize(src);
    }

    // case3:meta data not match, storage_offset of presentation layer
    // is different from that of storage layer
    if (FormatHelper::IsPadded(&src) && (!StorageDescHelper::OffsetAreMatch(&src))) {
        // Fix not match case3, tensor with padding should not have storage-offset.
        RECORD_FUNCTION("format_contiguous", vector<c10::IValue>({src}));
        return metadata_with_offset_padding_convert_match(src);
    }

    return src;
}

at::Tensor NpuUtils::format_contiguous_add_copy_optimize(const at::Tensor& src) {
    // case1:tensor src is not contiguous
    if (!src.is_contiguous()) {
        RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
        return src.contiguous();
    }
    // case2:meta data not match, sizes or strides of presentation
    // layer is different from that of storage layer
    if (!StorageDescHelper::MetaDataAreMatch(&src)) {
        // Fix not match case2, tensor should have matched metadata and
        // NPUStorageDesc.
        RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
        return metadata_convert_match_with_copy_optimize(src);
    }

    // case3:meta data not match, storage_offset of presentation layer
    // is different from that of storage layer
    if (FormatHelper::IsPadded(&src) && (!StorageDescHelper::OffsetAreMatch(&src))) {
        // Fix not match case3, tensor with padding should not have storage-offset.
        RECORD_FUNCTION("format_contiguousV2", vector<c10::IValue>({src}));
        return metadata_with_offset_padding_convert_match(src);
    }

    return src;
}

bool NpuUtils::IsOomError(aclError ret, int index) {
    if (ret == ACL_ERROR_GE_DEVICE_MEMORY_ALLOCATION_FAILED) {
        int deviceId = 0;
        NPU_CHECK_ERROR(aclrtGetDevice(&deviceId));
        AT_ERROR("NPU out of memory. device id: ", deviceId);
        return true;
    }
    return false;
}

void NpuUtils::check_1d(const at::Tensor& t, const char* arg, const char* fn) {
    TORCH_CHECK(t.dim() == 1, fn, ": Expected 1-D argument ", arg, ", but got ", t.dim(), "-D");
}

void NpuUtils::ProfReportMarkData(const std::string& msg) {
    if (msg.empty()) {
        return;
    }
}

void NpuUtils::ProfReportMarkDataToNpuProfiler(uint32_t category, const std::string& data, uint64_t correlation_id) {
    if (data.empty()) {
        return;
    }
}

void NpuUtils::ProfReportMarkDataToNpuProfiler(uint32_t category, void* data, size_t offset) {}

namespace {

constexpr int BLOCKSIZE = 16;

// base format is ND/NCHW
FormatShape InferShapeLessTo4(c10::IntArrayRef dims);
FormatShape InferShape4To5(c10::IntArrayRef dims);
FormatShape InferShape5To4(c10::IntArrayRef dims);
FormatShape InferShapeNDToNZ(c10::IntArrayRef dims);
FormatShape InferShapeNDToZ(c10::IntArrayRef dims);
FormatShape InferShapeofNCHW(c10::IntArrayRef dims);
FormatShape InferShapeofND(c10::IntArrayRef dims);

// converter between base format
FormatShape InferShapeNCHWToND(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims);
FormatShape InferShapeNCDHWToND(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims);
FormatShape InferShapeNDToNCHW(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims);
FormatShape InferShapeNDToNCDHW(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims);

// base format is NCDHW
FormatShape InferShapeOfNDHWC(c10::IntArrayRef dims);
FormatShape InferShapeOfNCDHW(c10::IntArrayRef dims);
FormatShape InferShapeOfNDC1HWC0(c10::IntArrayRef dims);
FormatShape InferShapeOfFZ3D(c10::IntArrayRef dims);

FormatShape InferShapeofNHWC(c10::IntArrayRef dims);
}  // namespace

std::unordered_map<aclFormat, FormatHelper::FormatInfo> FormatHelper::info = {
    {ACL_FORMAT_NC1HWC0, (FormatInfo){ACL_FORMAT_NC1HWC0, ACL_FORMAT_NCHW, InferShape4To5, "NC1HWC0", true}},
    {ACL_FORMAT_ND, (FormatInfo){ACL_FORMAT_ND, ACL_FORMAT_ND, InferShapeofND, "ND", false}},
    {ACL_FORMAT_NCHW, (FormatInfo){ACL_FORMAT_NCHW, ACL_FORMAT_NCHW, InferShapeofNCHW, "NCHW", false}},
    {ACL_FORMAT_NHWC, (FormatInfo){ACL_FORMAT_NHWC, ACL_FORMAT_NHWC, InferShapeofNHWC, "NHWC", false}},
    {ACL_FORMAT_FRACTAL_NZ, (FormatInfo){ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND, InferShapeNDToNZ, "FRACTAL_NZ", true}},
    {ACL_FORMAT_FRACTAL_Z, (FormatInfo){ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NCHW, InferShapeNDToZ, "FRACTAL_Z", true}},
    {ACL_FORMAT_NDHWC, (FormatInfo){ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW, InferShapeOfNDHWC, "NDHWC", false}},
    {ACL_FORMAT_NCDHW, (FormatInfo){ACL_FORMAT_NCDHW, ACL_FORMAT_NCDHW, InferShapeOfNCDHW, "NCDHW", false}},
    {ACL_FORMAT_NDC1HWC0, (FormatInfo){ACL_FORMAT_NDC1HWC0, ACL_FORMAT_NCDHW, InferShapeOfNDC1HWC0, "NDC1HWC0", true}},
    {ACL_FRACTAL_Z_3D, (FormatInfo){ACL_FRACTAL_Z_3D, ACL_FORMAT_NCDHW, InferShapeOfFZ3D, "FRACTAL_Z_3D", true}},
};

bool FormatHelper::IsPadded(const at::Tensor* tensor) {
    auto format = torch_npu::NPUBridge::GetNpuStorageImplDesc(*tensor).npu_format_;
    return IsPadded(format);
}

bool FormatHelper::IsPadded(aclFormat format) {
    auto itr = info.find(format);
    if (itr != info.end()) {
        return itr->second.isPadded;
    }
    AT_ERROR("unknown format type:", format);
    return true;
}

char* FormatHelper::GetFormatName(aclFormat format) {
    const auto& itr = info.find(format);
    if (itr == info.end()) {
        AT_ERROR("unknown format type:", format);
        return nullptr;
    }
    return itr->second.formatName;
}

char* FormatHelper::GetFormatName(const at::Tensor& tensor) {
    auto format = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor).npu_format_;
    return GetFormatName(format);
}

aclFormat FormatHelper::GetBaseFormat(const at::Tensor& tensor) {
    auto format = GetFormat(tensor);
    return GetBaseFormat(format);
}

aclFormat FormatHelper::GetBaseFormat(aclFormat format) {
    const auto& itr = info.find(format);
    if (itr == info.end()) {
        AT_ERROR("unknown format type:", format);
        return ACL_FORMAT_ND;
    }
    return itr->second.baseFormat;
}

aclFormat FormatHelper::GetFormat(const at::Tensor& tensor) { return torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor).npu_format_; }

bool FormatHelper::IsBaseFormatType(aclFormat format) { return GetBaseFormat(format) == format; }

bool FormatHelper::IsBaseFormatType(const at::Tensor& tensor) {
    auto format = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor).npu_format_;
    return IsBaseFormatType(format);
}

FormatShape FormatHelper::GetStorageSizes(const torch_npu::NPUStorageDesc& desc) {
    auto ori_size = desc.base_sizes_;
    auto format = desc.npu_format_;
    return GetStorageSizes(format, ori_size);
}

bool FormatHelper::IsOpInputBaseFormat(const at::Tensor& tensor) {
    if (!torch_npu::utils::is_npu(tensor)) {
        return true;
    }
    const auto format = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor).npu_format_;
    return (format == ACL_FORMAT_ND) || (format == ACL_FORMAT_NCHW) || (format == ACL_FORMAT_NHWC) || (format == ACL_FORMAT_NCDHW);
}

bool FormatHelper::IsOpInputBaseFormat(const c10::optional<at::Tensor>& tensor) {
    if (!tensor.has_value()) {
        return true;
    }
    return IsOpInputBaseFormat(tensor.value());
}

bool FormatHelper::IsOpInputBaseFormat(const c10::List<c10::optional<at::Tensor>>& tensors) {
    const auto& iter = std::find_if(tensors.begin(), tensors.end(), [](const auto& tensor) { return !IsOpInputBaseFormat(tensor); });
    return iter == tensors.end();
}

bool FormatHelper::IsOpInputBaseFormat(const at::TensorList& tensors) {
    const auto& iter = std::find_if(tensors.begin(), tensors.end(), [](const auto& tensor) { return !IsOpInputBaseFormat(tensor); });
    return iter == tensors.end();
}

bool FormatHelper::IsOpInputBaseFormat(const at::ITensorListRef& tensors) {
    auto materialized = tensors.materialize();
    const auto& iter = std::find_if(materialized.begin(), materialized.end(), [](const auto& tensor) { return !IsOpInputBaseFormat(tensor.get()); });
    return iter == materialized.end();
}

//
namespace {
FormatShape InferShapeLessTo4(c10::IntArrayRef dims) {
    FormatShape res;
    res.resize(4);
    AT_ASSERT(dims.size() <= 4, "input dim > 4 when InferShapeLessTo4");
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
            AT_ERROR("dims of NCHW shape should not be greater than 4, which is ", dims.size());
    }
    return res;
}

FormatShape InferShapeofNHWC(c10::IntArrayRef dims) {
    AT_ASSERT(dims.size() == 4, "input dim should be equal to 4 when InferShapeofNHWC");
    return FormatShape(dims.begin(), dims.end());
}

FormatShape InferShape4To5(c10::IntArrayRef dims) {
    FormatShape res;
    res.resize(5);
    if (dims.size() < 4) {
        NPU_LOGD("infershape4to5 but input dim < 4");
        return InferShape4To5(InferShapeLessTo4(dims));
    } else if (dims.size() > 4) {
        NPU_LOGE("infershape4to5 but input dim > 4");
    }
    res[0] = dims[0];
    res[1] = (dims[1] + 15) / 16;
    res[2] = dims[2];
    res[3] = dims[3];
    res[4] = BLOCKSIZE;
    return res;
}

FormatShape InferShape5To4(c10::IntArrayRef dims) {
    FormatShape res;
    res.emplace_back(dims[0]);
    res.emplace_back(((dims[1] + 15) / 16) * 16);
    res.emplace_back(dims[2]);
    res.emplace_back(dims[3]);
    return res;
}

FormatShape InferShapeNDToNZ(c10::IntArrayRef dims) {
    FormatShape res;
    // sum(keepdim = false) may make tensor dim = 0
    FormatShape dim;
    for (int i = 0; i < dims.size(); i++) {
        dim.emplace_back(dims[i]);
    }

    // this action will move to GuessStorageSizeWhenConvertFormat
    if (dim.size() == 0) {
        dim.emplace_back(1);
    }
    if (dim.size() == 1) {
        dim.emplace_back(1);
    }

    int i = 0;
    for (; i < dim.size() - 2; i++) {
        res.emplace_back(dim[i]);
    }

    res.emplace_back((dim[i + 1] + 15) / BLOCKSIZE);
    res.emplace_back((dim[i] + 15) / BLOCKSIZE);
    res.emplace_back(BLOCKSIZE);
    res.emplace_back(BLOCKSIZE);

    return res;
}

FormatShape InferShapeNDToZ(c10::IntArrayRef dims) {
    FormatShape res;
    if (dims.size() < 4) {
        return InferShapeNDToZ(InferShapeLessTo4(dims));
    }

    res.emplace_back((dims[1] + 15) / BLOCKSIZE * dims[2] * dims[3]);
    res.emplace_back((dims[0] + 15) / BLOCKSIZE);
    res.emplace_back(BLOCKSIZE);
    res.emplace_back(BLOCKSIZE);

    return res;
}

FormatShape InferShapeNCHWToND(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims) {
    FormatShape res;
    res.resize(4);
    auto cur_storage_dims = storage_dims;
    if (storage_dims.size() != 4) {
        cur_storage_dims = InferShapeLessTo4(storage_dims);
    }
    AT_ASSERT(cur_storage_dims.size() == 4, "input dim num not equal 4 when InferShapeNCHWToND");

    if (base_dims.size() == 0) {
        FormatShape temp_dims;
        temp_dims.emplace_back(1);
        return InferShapeLessTo4(temp_dims);
    }
    switch (base_dims.size()) {
        case 1:
            res.resize(1);
            res[0] = cur_storage_dims[1];
            AT_ASSERT(cur_storage_dims[0] == 1, "reshape type RESHAPE_TYPE_C erase dim N must be 1");
            AT_ASSERT(cur_storage_dims[2] == 1, "reshape type RESHAPE_TYPE_C erase dim H must be 1");
            AT_ASSERT(cur_storage_dims[3] == 1, "reshape type RESHAPE_TYPE_C erase dim W must be 1");
            break;
        case 2:
            res.resize(2);
            res[0] = cur_storage_dims[1];
            res[1] = cur_storage_dims[2];
            AT_ASSERT(cur_storage_dims[0] == 1, "reshape type RESHAPE_TYPE_CH erase dim N must be 1");
            AT_ASSERT(cur_storage_dims[3] == 1, "reshape type RESHAPE_TYPE_CH erase dim W must be 1");
            break;
        case 3:
            res.resize(3);
            res[0] = cur_storage_dims[1];
            res[1] = cur_storage_dims[2];
            res[2] = cur_storage_dims[3];
            AT_ASSERT(cur_storage_dims[0] == 1, "reshape type RESHAPE_TYPE_CHW erase dim N must be 1");
            break;
        case 4:
            res = cur_storage_dims;
            return res;
        default:
            AT_ERROR("unknown reshape type:");
    }
    return res;
}

FormatShape InferShapeNDToNCHW(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims) {
    AT_ASSERT(storage_dims.size() <= 4, "input storage dim not less than 4");
    AT_ASSERT(base_dims.size() <= 4, "input storage dim not less than 4");
    return InferShapeLessTo4(base_dims);
}

FormatShape InferShapeNDToNCDHW(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims) {
    AT_ASSERT(storage_dims.size() == 5, "ND [", storage_dims, "] failed to convert to NCDHW");
    FormatShape res;
    res.resize(5);
    res = storage_dims;
    return res;
}

FormatShape InferShapeNCDHWToND(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims) {
    FormatShape res;
    res.resize(5);
    res = storage_dims;
    AT_ASSERT(res.size() == 5, "input dim num not equal 5 when InferShapeNCDHWToND");
    return res;
}

// NCDHW -> NDHWC
FormatShape InferShapeOfNDHWC(c10::IntArrayRef dims) {
    if (dims.size() < 5) {
        AT_ERROR("dim (", dims, ") cannot convert to NDHWC");
    }
    FormatShape res;
    res.resize(5);
    res[0] = dims[0];
    res[1] = dims[2];
    res[2] = dims[3];
    res[3] = dims[4];
    res[4] = dims[1];
    return res;
}

// NCDHW to NCDHW
FormatShape InferShapeOfNCDHW(c10::IntArrayRef dims) {
    if (dims.size() < 5) {
        AT_ERROR("dim (", dims, ") cannot convert to NCDHW");
    }
    FormatShape res;
    res.resize(5);
    res[0] = dims[0];
    res[1] = dims[1];
    res[2] = dims[2];
    res[3] = dims[3];
    res[4] = dims[4];
    return res;
}

// NCDHW to NDC1HWC0
FormatShape InferShapeOfNDC1HWC0(c10::IntArrayRef dims) {
    if (dims.size() < 5) {
        AT_ERROR("dim (", dims, ") cannot convert to NDC1HWC0");
    }
    FormatShape res;
    res.resize(6);
    res[0] = dims[0];
    res[1] = dims[2];
    res[2] = (dims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
    res[3] = dims[3];
    res[4] = dims[4];
    res[5] = BLOCKSIZE;
    return res;
}

// NCDHW to FZ_3D
FormatShape InferShapeOfFZ3D(c10::IntArrayRef dims) {
    if (dims.size() < 5) {
        AT_ERROR("dim (", dims, ") cannot convert to FZ_3D");
    }

    int64_t d1 = dims[2];
    int64_t d2 = (dims[1] + BLOCKSIZE - 1) / BLOCKSIZE;
    int64_t d3 = dims[3];
    int64_t d4 = dims[4];
    int64_t d5 = (dims[0] + BLOCKSIZE - 1) / BLOCKSIZE;
    int64_t d6 = BLOCKSIZE;
    int64_t d7 = BLOCKSIZE;

    // The shape of FZ3D is 7D, but the CANN only accept 4D
    // so we should merge 1st, 2nd, 3rd, 4th dimension.
    FormatShape res;
    res.resize(4);
    res[0] = d1 * d2 * d3 * d4;
    res[1] = d5;
    res[2] = d6;
    res[3] = d7;
    return res;
}

FormatShape InferShapeofNCHW(c10::IntArrayRef dims) {
    if (dims.size() < 5) {
        return InferShapeLessTo4(dims);
    } else {
        return InferShapeofND(dims);
    }
}

FormatShape InferShapeofND(c10::IntArrayRef dims) {
    FormatShape res;
    res.resize(dims.size());
    for (int j = 0; j < dims.size(); j++) {
        res[j] = dims[j];
    }
    return res;
}

}  // namespace

at::Tensor& FormatHelper::unsafe_format_cast(at::Tensor& self, int64_t self_format, int64_t result_format) {
    torch_npu::NPUStorageDesc& self_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    if (self_format == ACL_FORMAT_ND && result_format == ACL_FORMAT_NC1HWC0) {
        self_desc.storage_sizes_ = InferShape4To5(self.sizes());
        self_desc.npu_format_ = ACL_FORMAT_NC1HWC0;
    } else if (self_format == ACL_FORMAT_NC1HWC0 && result_format == ACL_FORMAT_ND) {
        self_desc.storage_sizes_ = self_desc.base_sizes_;
        self_desc.npu_format_ = ACL_FORMAT_ND;
    }
    return self;
}

void copy_d2d_by_memcpy(at::Tensor& dst, const at::Tensor& src, int64_t exceptSize) {
    int64_t size = exceptSize;
    auto dst_mem_size = StorageDescHelper::GetMemorySize(dst);
    if (exceptSize == 0) {
        size = dst_mem_size;
    }

    if (!dst.data_ptr()) {
        TORCH_WARN("copy_d2d_by_memcpy, dst.data_ptr() is null.");
        return;
    }

    if (!src.data_ptr()) {
        TORCH_WARN("copy_d2d_by_memcpy, src.data_ptr() is null.");
        return;
    }

    if (dst.data_ptr() == src.data_ptr() && dst.element_size() == src.element_size()) {
        return;
    }

    // The current logic is only used in single op mode.
    aclError error =
        c10_npu::queue::LaunchAsyncCopyTask(dst.data_ptr(), size * dst.element_size(), src.data_ptr(), size * dst.element_size(), ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (error != ACL_ERROR_NONE) {
        AT_ERROR("async copy device to device error.");
        return;
    }
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(aclrtMemcpyAsync(dst.data_ptr(), dst.nbytes(), src.data_ptr(), src.nbytes(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
}

float CalcuOpUtil::GetScalarFloatValue(const c10::Scalar& scalar) {
    float value;
    if (scalar.isFloatingPoint()) {
        value = scalar.toFloat();
    } else {
        value = static_cast<float>(scalar.toInt());
    }

    return value;
}

int64_t CalcuOpUtil::GetTensorNpuFormat(const at::Tensor& tensor) {
    int64_t ndim = tensor.sizes().size();
    if (ndim == 5) {
        return ACL_FORMAT_NCDHW;
    } else if (ndim == 4) {
        return ACL_FORMAT_NCHW;
    }
    return ACL_FORMAT_ND;
}

aclDataType CalcuOpUtil::ConvertToAclDataType(const at::ScalarType& data_type) {
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported")
    return acl_dtype;
}

aclDataType CalcuOpUtil::ConvertToAclDataType(const at::ScalarType& data_type, const string& realDataType) {
    auto acl_dtype = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
    TORCH_CHECK(acl_dtype != ACL_DT_UNDEFINED, std::string(c10::toString(data_type)) + " has not been supported")
    if (!realDataType.empty()) {
        return STRING_SCALAR_TYPE_TO_ACL_TYPE_MAP[realDataType];
    }
    return acl_dtype;
}

at::Tensor CalcuOpUtil::CopyScalarToDevice(const c10::Scalar& cpu_scalar, at::ScalarType scalar_data_type) {
    return CalcuOpUtil::CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

at::Tensor CalcuOpUtil::CopyTensorHostToDevice(const at::Tensor& cpu_tensor) {
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    NPU_CHECK_ERROR(aclrtGetDevice(&deviceIndex));
    return cpuPinMemTensor.to(c10::Device(at_npu::key::NativeDeviceType, deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
}

c10::SmallVector<int64_t, SHAPE_SIZE> CalcuOpUtil::ConvertIntArrayRefToSmallVector(c10::IntArrayRef intArray) {
    c10::SmallVector<int64_t, SHAPE_SIZE> intVec;
    for (const auto i : c10::irange(intArray.size())) {
        intVec.emplace_back(intArray[i]);
    }

    return intVec;
}

using aclCubeMathType = enum : int8_t {
    KEEP_DTYPE = 0,
    ALLOW_FP32_DOWN_PRECISION = 1,
    USE_FP16 = 2,
    USE_HF32 = 3,
};

static std::unordered_map<uint8_t, aclCubeMathType> ACL_CUBE_MATH_TYPE_MAP = {
    {0b00, KEEP_DTYPE}, {0b01, USE_FP16}, {0b10, USE_HF32}, {0b11, ALLOW_FP32_DOWN_PRECISION}};

int8_t CalcuOpUtil::GetCubeMathType(bool allowHf32) {
    bool allowFp32ToFp16 = native::env::IsAllowFP32ToFP16();
    uint8_t CubeMathTypeCode = ((uint8_t)allowHf32 << 1) + (uint8_t)allowFp32ToFp16;
    auto iter = ACL_CUBE_MATH_TYPE_MAP.find(CubeMathTypeCode);
    if (iter == ACL_CUBE_MATH_TYPE_MAP.end()) {
        return ALLOW_FP32_DOWN_PRECISION;
    }
    return iter->second;
}

void assert_no_internal_overlap(const at::Tensor& tensor) {
    auto t = tensor.unsafeGetTensorImpl();
    AT_ASSERT(t->layout() == at::kStrided);
    AT_ASSERT(tensor.is_contiguous());
    auto strides = t->strides();
    auto sizes = t->sizes();
    for (size_t i = 0; i < strides.size(); ++i) {
        if (strides[i] == 0 && sizes[i] > 1) {
            AT_ASSERT(false);
        }
    }
}

void assert_no_partial_overlap(const at::Tensor& tensora, const at::Tensor& tensorb) {
    auto a = tensora.unsafeGetTensorImpl();
    auto b = tensorb.unsafeGetTensorImpl();
    if (a == b) {
        return;
    }
    if (a->numel() == 0 || b->numel() == 0) {
        return;
    }
    if (!a->is_contiguous() || !b->is_contiguous()) {
        return;
    }
    if (a->storage().data() == b->storage().data()) {
        const auto a_begin = static_cast<char*>(a->data());
        const auto a_end = a_begin + a->numel() * static_cast<int64_t>(a->itemsize());
        const auto b_begin = static_cast<char*>(b->data());
        const auto b_end = b_begin + b->numel() * static_cast<int64_t>(b->itemsize());
        if (a_begin == b_begin && a_end == b_end) {
            return;
        }
        if (a_begin < b_end && b_begin < a_end) {
            AT_ASSERT(false);
        }
    }
}

void CalcuOpUtil::CheckMemoryOverLaps(c10::ArrayRef<at::Tensor> inputs, c10::ArrayRef<at::Tensor> outputs) {
    for (const auto i : c10::irange(outputs.size())) {
        if (!outputs[i].defined()) {
            continue;
        }

        assert_no_internal_overlap(outputs[i]);

        for (const auto j : c10::irange(inputs.size())) {
            assert_no_partial_overlap(outputs[i], inputs[j]);
        }
    }
}

NPUStatus CalcuOpUtil::AclrtMemcpyAsync(const std::pair<at::Tensor, int64_t>& dst, size_t dst_size, const std::pair<at::Tensor, int64_t>& src, size_t src_size,
                                        aclrtMemcpyKind kind) {
    void* dst_ptr = reinterpret_cast<uint8_t*>(dst.first.data_ptr()) + dst.second * dst.first.itemsize();
    void* src_ptr = reinterpret_cast<uint8_t*>(src.first.data_ptr()) + src.second * src.first.itemsize();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(aclrtMemcpyAsync(dst_ptr, dst_size, src_ptr, src_size, kind, stream));

    return "SUCCESS";
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair& dst, size_t dstMax, const StorageAndOffsetMemSizePair& src, size_t count,
                                                aclrtMemcpyKind kind) {
    void* dst_ptr = static_cast<void*>(static_cast<uint8_t*>(dst.first->data()) + dst.second);
    void* src_ptr = static_cast<void*>(static_cast<uint8_t*>(src.first->data()) + src.second);
    return AclrtMemcpyParamCheck(dst_ptr, dstMax, const_cast<void*>(src_ptr), count, kind);
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair& dst, size_t dstMax, const void* src, size_t count, aclrtMemcpyKind kind) {
    void* dst_ptr = static_cast<void*>(static_cast<uint8_t*>(dst.first->data()) + dst.second);
    return AclrtMemcpyParamCheck(dst_ptr, dstMax, src, count, kind);
}

aclError CalcuOpUtil::AclrtMemcpyWithModeSwitch(void* dst, size_t dstMax, const StorageAndOffsetMemSizePair& src, size_t count, aclrtMemcpyKind kind) {
    void* src_ptr = static_cast<void*>(static_cast<uint8_t*>(src.first->data()) + src.second);
    return AclrtMemcpyParamCheck(dst, dstMax, const_cast<void*>(src_ptr), count, kind);
}

aclError CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(const at::Tensor& dst, size_t dstMax, const at::Tensor& src, size_t count, aclrtMemcpyKind kind) {
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    NPU_CHECK_ERROR(aclrtMemcpyAsync(dst.data_ptr(), dst.nbytes(), src.data_ptr(), src.nbytes(), kind, stream));
}

void ContiguousTensorDesc::refresh_contiguous_using_size_and_stride() {
    if (c10::multiply_integers(sizes_) == 0) {
        is_contiguous_ = true;
    }
    int64_t infer_axis_size = 1;
    for (int64_t dim = static_cast<int64_t>(sizes_.size()) - 1; dim >= 0; dim--) {
        if (sizes_[dim] != 1) {
            if (strides_[dim] == infer_axis_size) {
                infer_axis_size *= sizes_[dim];
            } else {
                is_contiguous_ = false;
                return;
            }
        }
    }
    is_contiguous_ = true;
}

void ContiguousTensorDesc::reset_optimization_cases(const OptimizationCases& opt_cases) { opt_cases_ = opt_cases; }

void ContiguousTensorDesc::add_optimization_case(const std::string& opt_case) { opt_cases_.emplace_back(opt_case); }

void ContiguousTensorDesc::find_match_optimization_cases() {
    for (const auto i : c10::irange(sizes_.size())) {
        if (strides_[i] == 0) {
            opt_cases_.emplace_back("broadcast");
            return;
        }
    }

    for (const auto i : c10::irange(strides_.size() - 1)) {
        if (strides_[i] < strides_[i + 1]) {
            opt_cases_.emplace_back("permute");
            return;
        }
    }

    // Considering combined-cases, we cannot split slice cases any further.
    if (c10::multiply_integers(sizes_) < c10::multiply_integers(base_sizes_)) {
        opt_cases_.emplace_back("slice");
        opt_cases_.emplace_back("select");
        opt_cases_.emplace_back("indexing");
        return;
    }
}

OptimizationCases TransContiguous::optCasesDefault = {};
OptimizationCases TransContiguous::optCasesAnyFormat = {"reshape", "slice"};

ContiguousTensorDesc TransContiguous::GetTensorDescInfo(const at::Tensor& src, const OptimizationCases& opt_cases) {
    auto src_base_info = torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc();
    c10::SmallVector<int64_t, MAX_DIM> src_size_inferred;
    c10::SmallVector<int64_t, MAX_DIM> src_stride_inferred;
    c10::SmallVector<int64_t, MAX_DIM> src_storage_size_inferred = src_base_info.storage_sizes_;
    if (src.dim() == 0) {
        src_size_inferred = {1};
        src_stride_inferred = {1};
        if (src_storage_size_inferred.size() == 0) {
            src_storage_size_inferred = {1};
        }
    } else {
        src_size_inferred = CalcuOpUtil::ConvertIntArrayRefToSmallVector(src.sizes());
        src_stride_inferred = CalcuOpUtil::ConvertIntArrayRefToSmallVector(src.strides());
    }
    ContiguousTensorDesc src_desc = {src.is_contiguous(),
                                     src_size_inferred,
                                     src_stride_inferred,
                                     src.storage_offset(),
                                     src_base_info.base_sizes_,
                                     src_base_info.base_strides_,
                                     src_storage_size_inferred,
                                     src_base_info.base_offset_,
                                     src_base_info.npu_format_,
                                     opt_cases};
    if (src_desc.opt_cases_.empty()) {
        src_desc.find_match_optimization_cases();
    }
    return src_desc;
}

bool TransContiguous::CheckClone(const at::Tensor& src, at::Tensor& self) {
    // self tensor may not be temporary constructed empty tensor from src, so:
    // 1. contiguous storage is needed:storage_offset and numels eq
    // 2. full memory copy: size match between src and self
    if (StorageDescHelper::OffsetAreMatch(&self) && self.is_contiguous() && src.sizes().equals(self.sizes()) &&
        self.sizes().equals(torch_npu::NPUBridge::GetNpuStorageImpl(self)->get_npu_desc().base_sizes_)) {
        return true;
    }
    return false;
}

bool TransContiguous::can_optimize_(ContiguousTensorDesc& tensor_desc) {
    for (auto opt_case : tensor_desc.opt_cases_) {
        bool res = register_opt::CopyOptRegister::GetInstance()->CanOptimize(opt_case, tensor_desc);
        if (res) {
            // refresh patterns to only keep optimized pattern
            tensor_desc.opt_cases_.clear();
            tensor_desc.opt_cases_.emplace_back(opt_case);
            return true;
        }
    }
    return false;
}

bool TransContiguous::CanOptimize(ContiguousTensorDesc& tensor_desc) { return can_optimize_(tensor_desc); }

bool TransContiguous::CanOptimize(const at::Tensor& tensor, const OptimizationCases& opt_cases) {
    ContiguousTensorDesc tensor_desc = GetTensorDescInfo(tensor, opt_cases);
    return can_optimize_(tensor_desc);
}

bool TransContiguous::contiguous_optimize_with_anyformat_(at::Tensor& self, const at::Tensor& src, ContiguousTensorDesc& src_desc) {
    if (!CheckClone(src, self)) {
        return false;
    }
    for (auto& opt_case : src_desc.opt_cases_) {
        bool res = register_opt::CopyOptRegister::GetInstance()->Run(opt_case, self, src, src_desc);
        if (res) {
            return true;
        }
    }
    return false;
}

bool TransContiguous::ContiguousOptimizeWithAnyFormat(at::Tensor& self, const at::Tensor& src, const OptimizationCases& opt_cases) {
    ContiguousTensorDesc src_desc = GetTensorDescInfo(src, opt_cases);
    return contiguous_optimize_with_anyformat_(self, src, src_desc);
}

c10::optional<at::Tensor> TransContiguous::ContiguousOptimizeWithAnyFormat(const at::Tensor& src, const OptimizationCases& opt_cases) {
    TORCH_CHECK(src.device().type() == at_npu::key::NativeDeviceType,
                "Expected all tensors to be on the same device. "
                "Expected NPU tensor, please check whether the input tensor device is correct.");
    auto self = OpPreparation::ApplyTensorWithFormat(src.sizes(), src.options(), torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc().npu_format_);
    ContiguousTensorDesc src_desc = GetTensorDescInfo(src, opt_cases);
    if (contiguous_optimize_with_anyformat_(self, src, src_desc)) {
        return self;
    }
    return c10::nullopt;
}

bool TransContiguous::ContiguousOptimizeWithBaseFormat(at::Tensor& self, const at::Tensor& src, const OptimizationCases& opt_cases, bool OpenCombined) {
    TORCH_CHECK(FormatHelper::IsBaseFormatType(src),
                "ContiguousOptimizeWithBaseFormat func requires Input Tensor "
                "with base format!");
    // In non-specific cases, classify the cases and simplify judgement.
    ContiguousTensorDesc src_desc = GetTensorDescInfo(src, opt_cases);
    // if (OpenCombined && c10_npu::option::OptionsManager::CheckCombinedOptimizerEnable()) {
    if (OpenCombined) {
        src_desc.add_optimization_case("combined");
    }
    return contiguous_optimize_with_anyformat_(self, src, src_desc);
}

// OpPreparation part

inline void check_size_nonnegative(c10::IntArrayRef& size) {
    for (auto& x : size) {
        TORCH_CHECK(x >= 0, "Trying to create tensor with negative dimension ", x, ": ", size);
    }
}

const char* markedOutputsErrorInfo =
    "Parameters that allocate memory inside the operator need to be marked as output in advance through markAsOutputForApplyTensor";
thread_local std::deque<at::Tensor> markedOutputs;
void OpPreparation::markAsOutputForApplyTensor(at::Tensor& src) { markedOutputs.push_back(src); }

at::Tensor empty_npu(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt,
                     c10::optional<bool> pin_memory_opt, c10::optional<at::MemoryFormat> memory_format_opt) {
    TORCH_CHECK(dtype_opt.has_value());
    diopiSize_t sizeDiopi{size.data(), size.size()};
    diopiDtype_t dtypeDiopi = impl::aten::getDIOPITensorType(dtype_opt.value());
    diopiDevice_t deviceDiopi = diopi_device;

    diopiTensorHandle_t tensorDiopi = nullptr;
    if (enableDumpArgs()) {
        std::cout << __FUNCTION__ << ": diopiRequireTensor shape: " << size << ", dtype:" << dtype_opt.value() << std::endl;
    }
    auto ret = diopiRequireTensor(context, &tensorDiopi, &sizeDiopi, nullptr, dtypeDiopi, deviceDiopi);
    TORCH_CHECK(diopiSuccess == ret);
    return impl::aten::buildATen(tensorDiopi);
}

at::Tensor empty_npu(at::IntArrayRef size, const at::TensorOptions& options) {
    return empty_npu(size, c10::make_optional(c10::typeMetaToScalarType(options.dtype())));
}

at::Tensor empty_strided_npu(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                             c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    at::TensorOptions options(dtype.value());
    int64_t nbytes = at::detail::computeStorageNbytes(size, stride, options.dtype().itemsize()).as_int_unchecked();
    int64_t numel = nbytes / options.dtype().itemsize();
    auto out = at_npu::native::empty_npu({numel}, options);
    return impl::aten::viewStorage(out, c10::asIntArrayRefUnchecked(size), c10::asIntArrayRefUnchecked(stride));
}

at::Tensor empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout, c10::optional<at::Device> device,
                             c10::optional<bool> pin_memory, int64_t acl_format) {
    torch_npu::utils::torch_check_npu(c10::device_or_default(device));
    TORCH_CHECK(!pinned_memory_or_default(pin_memory), "Only dense CPU tensors can be pinned");
    check_size_nonnegative(size);
    // when the shape and format are not match, fix format here.
    aclFormat format = InferFormat::GuessStorageFormat(size, (aclFormat)acl_format);
    int64_t numel = StorageDescHelper::GetMemorySize(size, format);
    auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
    at::TensorOptions options(dtype);
    auto tensor = at_npu::native::empty_npu({numel}, options);
    // Default NPUTensorImpl has size [0]
    if (size.size() != 1 || size[0] != 0) {
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }
    tensor.unsafeGetTensorImpl()->empty_tensor_restride(c10::MemoryFormat::Contiguous);
    StorageDescHelper::SetDesc(tensor, size, tensor.strides(), format);
    return tensor;
}

at::Tensor clone(const at::Tensor& src, c10::optional<at::MemoryFormat> memory_format) {
    OptimizationCases opt_cases{"reshape", "slice", "reshapeV2"};
    if (TransContiguous::CanOptimize(src, opt_cases)) {
        // clone with any npu formats
        auto formatTempTensor = TransContiguous::ContiguousOptimizeWithAnyFormat(src, opt_cases);
        return formatTempTensor.value();
    } else {
        // clone with base formats
        auto baseSelf = OpPreparation::ApplyTensorWithSizes(src.sizes(), src.options());
        at::Tensor baseSrc = src;
        if (!FormatHelper::IsBaseFormatType(src)) {
            baseSrc = FormatCastHelper::ApplyBaseFormatTensorBy(src);
        }
        copy_d2d_dtype_baseformat(baseSelf, baseSrc, false);
        return baseSelf;
    }
}

c10::SmallVector<int64_t, 5> OpPreparation::get_tensor_desc_base_sizes(const at::Tensor& tensor) {
    return torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc().base_sizes_;
}

at::Tensor OpPreparation::CastBackToOriFormat(const at::Tensor& tensor) {
    auto& tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    auto ret = custom_ops::npu_format_cast(tensor, tensor_desc.origin_format_);
    return ret;
}

at::Tensor& OpPreparation::CastBackToOriFormat(at::Tensor& tensor) {
    auto& tensor_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    NPUNativeFunctions::npu_format_cast_(tensor, tensor_desc.origin_format_);
    return tensor;
}

at::Tensor OpPreparation::apply_tensor(const at::Tensor& src) { return apply_tensor(src, src.sizes()); }

at::Tensor OpPreparation::apply_tensor(const at::Tensor& src, c10::IntArrayRef sizes) {
    return apply_tensor_with_format(sizes, src.options(), CalcuOpUtil::GetTensorNpuFormat(src));
}

at::Tensor OpPreparation::apply_tensor(const at::Tensor& src, const c10::TensorOptions& options) {
    return apply_tensor_with_format(src.sizes(), options, CalcuOpUtil::GetTensorNpuFormat(src));
}

at::Tensor OpPreparation::apply_tensor(c10::IntArrayRef sizes, const c10::TensorOptions& options, const at::Tensor& src) {
    return apply_tensor_with_format(sizes, options, CalcuOpUtil::GetTensorNpuFormat(src));
}

at::Tensor OpPreparation::apply_tensor_with_format(const at::Tensor& src, int64_t format, bool keep_format) {
    return apply_tensor_with_format(src, src.sizes(), format, keep_format);
}

at::Tensor OpPreparation::apply_tensor_with_format(const at::Tensor& src, c10::IntArrayRef sizes, int64_t format, bool keep_format) {
    return apply_tensor_with_format(sizes, src.options(), format, keep_format);
}

at::Tensor OpPreparation::apply_tensor_with_format(c10::IntArrayRef sizes, const c10::TensorOptions& options, int64_t format, bool keep_format) {
    if (markedOutputs.size() > 0) {
        auto out = *markedOutputs.begin();
        markedOutputs.pop_front();
        return out;
    }
    TORCH_CHECK(options.device().type() == at_npu::key::NativeDeviceType,
                "Expected all tensors to be on the same device. "
                "Expected NPU tensor, please check whether the input tensor device is correct.");
    auto fixFormat = InferFormat::GuessStorageFormat(sizes, (aclFormat)format);
    return NPUNativeFunctions::unsafe_empty_with_format(
        sizes, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), fixFormat, keep_format);
}

at::Tensor OpPreparation::apply_tensor_with_sizes(c10::IntArrayRef sizes, const c10::TensorOptions& options) {
    if (markedOutputs.size() > 0) {
        auto out = *markedOutputs.begin();
        markedOutputs.pop_front();
        return out;
    }
    auto format = InferFormat::GuessBaseFormat(sizes);
    return NPUNativeFunctions::empty_with_format(
        sizes, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), options.pinned_memory_opt(), format);
}

void OpPreparation::CheckOut(const std::initializer_list<at::Tensor>& inputs, at::Tensor& output, at::Tensor dst) {
    CheckOut(inputs, output, CalcuOpUtil::GetTensorNpuFormat(dst), dst.scalar_type(), dst.sizes());
}

void OpPreparation::CheckOut(const std::initializer_list<at::Tensor>& inputs, at::Tensor& output, at::Tensor dst, c10::IntArrayRef shape) {
    CheckOut(inputs, output, CalcuOpUtil::GetTensorNpuFormat(dst), dst.scalar_type(), shape);
}

void OpPreparation::CheckOut(const std::initializer_list<at::Tensor>& input, at::Tensor& output, int64_t format, at::ScalarType dtype, c10::IntArrayRef shape) {
    // Check that the outputs have no internal overlap
    // and do not share memory with inputs.
    c10::SmallVector<at::Tensor, N> inputs{input};
    c10::SmallVector<at::Tensor, N> outputs = {output};
    CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);
    TORCH_CHECK(at_npu::key::isDeviceTensor(output), "output with device ", output.device(), " doesn't match the desired device NPU");
    TORCH_CHECK(output.scalar_type() == dtype, "expected dtype ", dtype, " but got dtype ", output.scalar_type());

    bool is_read_write = false;
    // check if output is also an input
    for (const auto& input : inputs) {
        if (output.is_same(input)) {
            is_read_write = true;
            break;
        }
    }

    // Preserve legacy resizing behavior of out=... arguments
    if (!output.sizes().equals(shape)) {
        TORCH_CHECK(!is_read_write, "output with shape ", output.sizes(), " doesn't match the broadcast shape ", shape);
        output.resize_(shape);
    }

    if (CalcuOpUtil::GetTensorNpuFormat(output) != format) {
        TORCH_CHECK(!is_read_write, "can not cast format when output is input");
        NPUNativeFunctions::npu_format_cast_(output, format);
    }
}

aclFormat InferFormat::GuessFormatWhenContiguous(const at::Tensor& tensor) {
    auto desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_;
    // fix: NCDHW -> default format
    if ((desc.origin_format_ == ACL_FORMAT_NCDHW)) {
        if ((tensor.sizes().size() != desc.base_sizes_.size()) && (tensor.sizes().size() <= 4)) {
            return ACL_FORMAT_NCHW;
        }
    }
    return desc.origin_format_;
}

// NOTE: this method should cooperate with shape infer.
std::tuple<aclFormat, aclFormat> InferFormat::GuessFormatUnit(const c10::IntArrayRef& size, aclFormat format) {
    aclFormat baseFormat = FormatHelper::GetBaseFormat(format);
    if ((baseFormat == ACL_FORMAT_NCDHW) && (size.size() > 4)) {
        return std::make_tuple(ACL_FORMAT_NCDHW, format);
    } else if (format == ACL_FORMAT_ND && size.size() == 4) {
        // 4 dim tensor must be NCHW, reflush base format
        return std::make_tuple(ACL_FORMAT_NCHW, ACL_FORMAT_NCHW);
    } else {
        if (baseFormat == ACL_FORMAT_NCDHW) {
            // scence: Dimensionality reduction: NCDHW->NCHW, for example: max/min
            // NOTE(NPU Dimensionality reduction)
            if (size.size() == 4) {
                return std::make_tuple(ACL_FORMAT_NCHW, ACL_FORMAT_NCHW);
            }
        }
    }
    return std::make_tuple(baseFormat, format);
}

aclFormat InferFormat::GuessBaseFormat(const c10::IntArrayRef& size) {
    if (size.size() == 5) {
        return ACL_FORMAT_NCDHW;
    } else if (size.size() == 4) {
        return ACL_FORMAT_NCHW;
    }
    return ACL_FORMAT_ND;
}

aclFormat InferFormat::GuessStorageFormat(const c10::IntArrayRef& size, aclFormat format) {
    if (format == ACL_FORMAT_FRACTAL_NZ && size.size() < 2) {
        // scalar scene and rank=1 scene do not support NZ
        return ACL_FORMAT_ND;
    }

    int64_t dim = static_cast<int64_t>(size.size());
    aclFormat baseFormat = FormatHelper::GetBaseFormat(format);
    bool isBaseFormat = (baseFormat == format);
    // if base format and tensor size is not match, we should reflush them
    if ((isBaseFormat) && (baseFormat == ACL_FORMAT_NCDHW)) {
        // scence1: Dimensionality reduction: NCDHW->NCHW, for example: max/min
        // scence2: view, as_strided
        // NOTE(NPU Dimensionality reduction)
        if (dim == 4) {
            return ACL_FORMAT_NCHW;
        } else if (dim == 5) {
            return ACL_FORMAT_NCDHW;
        } else {
            return ACL_FORMAT_ND;
        }
    } else if (format == ACL_FORMAT_NCHW && dim != 4) {
        return ACL_FORMAT_ND;
    } else if ((dim == 0) || ((dim == 1) && (size[0] == 1) && (baseFormat == ACL_FORMAT_ND))) {
        // operators treat tensor with dimensions of 0 or shape = [1] as scalar,
        // so these tensor will stay ND format except NCHW tensor whose origin shape
        // can be expand into four dimensions.
        return ACL_FORMAT_ND;
    }
    return format;
}

FormatShape InferFormat::GuessStorageSizeWhenConvertFormat(const at::Tensor& tensor) {
    auto format = FormatHelper::GetFormat(tensor);
    auto size = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.base_sizes_;
    // TransData: ND->NZ, ND size < 2, we can expand dimension to 2, the storage have no effect.
    // now, only ND->NZ and NZ->ND will call transdata， so we no need to check other format.
    if ((size.size() < 2) && format == ACL_FORMAT_ND) {
        do {
            size.emplace_back(1);
        } while (size.size() < 2);
    }
    return FormatHelper::GetStorageSizes(format, size);
}

bool InferFormat::IsDefiniteTensorWhenMetaDataChanges(const at::Tensor& tensor, const c10::IntArrayRef& size) {
    auto baseformat = FormatHelper::GetBaseFormat(tensor);
    if (baseformat == ACL_FORMAT_NCHW && size.size() >= 5) {
        return true;
    }
    if (baseformat == ACL_FORMAT_NCDHW && size.size() != 5) {
        return true;
    }
    return false;
}

bool StorageDescHelper::MetaDataAreMatch(const at::Tensor* tensor) {
    auto& desc = torch_npu::NPUBridge::GetNpuStorageImplDesc(*tensor);
    return IsSameSize(desc.base_sizes_, tensor->sizes()) && IsSameSize(desc.base_strides_, tensor->strides());
}

// copy related
bool StorageDescHelper::IsSameDesc(const torch_npu::NPUStorageDesc& a, const torch_npu::NPUStorageDesc& b) {
    if ((a.origin_format_ != b.origin_format_) || (a.npu_format_ != b.npu_format_)) {
        if ((!FormatHelper::IsBaseFormatType(a.npu_format_)) || (!FormatHelper::IsBaseFormatType(b.npu_format_))) {
            return false;
        }
    }
    return (a.base_sizes_ == b.base_sizes_) && (a.base_strides_ == b.base_strides_) && (a.storage_sizes_ == b.storage_sizes_);
}

bool StorageDescHelper::IsSameDesc(const at::Tensor& a, const at::Tensor& b) {
    const auto& descA = torch_npu::NPUBridge::GetNpuStorageImplDesc(a);
    const auto& descB = torch_npu::NPUBridge::GetNpuStorageImplDesc(b);
    return IsSameDesc(descA, descB);
}

bool StorageDescHelper::IsSameSize(const c10::SmallVector<int64_t, 5>& a, const c10::IntArrayRef& b) {
    if (a.size() == b.size()) {
        return std::equal(a.begin(), a.end(), b.begin());
    }
    return false;
}

void StorageDescHelper::UpdateDesc(torch_npu::NPUStorageDesc& npuDesc, const c10::IntArrayRef& new_data_sizes, const c10::IntArrayRef& new_shape_sizes) {
    int64_t new_data_numel = c10::multiply_integers(new_data_sizes);
    int64_t new_shape_numel = c10::multiply_integers(new_shape_sizes);
    const c10::IntArrayRef& new_size = new_data_numel > new_shape_numel ? new_data_sizes : new_shape_sizes;

    npuDesc.base_sizes_ = new_size;

    // 计算连续场景下size对应的stride值
    int64_t dim_ = static_cast<int64_t>(new_size.size());
    c10::SmallVector<int64_t, 5> new_stride(dim_);
    if (dim_ > 0) {
        int64_t last_idx = dim_ - 1;
        new_stride[last_idx] = 1;
        for (auto i = last_idx - 1; i >= 0; --i) {
            new_stride[i] = new_stride[i + 1] * std::max<int64_t>(new_size[i + 1], 1);
        }
    }
    npuDesc.base_strides_ = new_stride;

    // 更新物理内存信息
    npuDesc.storage_sizes_ = FormatHelper::GetStorageSizes(npuDesc);
    if (new_data_numel > new_shape_numel) {
        // Refresh format to base format only when flattening storage data
        npuDesc.storage_sizes_ = new_size;
        npuDesc.npu_format_ = InferFormat::GuessStorageFormat(npuDesc.storage_sizes_, ACL_FORMAT_ND);
    }
}

FormatShape StorageDescHelper::ComputeStrideFromShape(const FormatShape& shape) {
    FormatShape compute_stride = shape;
    compute_stride[shape.size() - 1] = 1;
    for (auto i = shape.size() - 1; i > 0; i--) {
        compute_stride[i - 1] = shape[i] * compute_stride[i];
    }
    return compute_stride;
}

int64_t StorageDescHelper::GetMemorySize(const torch_npu::NPUStorageDesc& desc) {
    const auto& physical_size = FormatHelper::GetStorageSizes(desc);
    return c10::multiply_integers(physical_size);
}

int64_t StorageDescHelper::GetMemorySize(const at::Tensor& dst) {
    auto desc = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_;
    return GetMemorySize(desc);
}

int64_t StorageDescHelper::GetMemorySize(const c10::IntArrayRef& size, aclFormat format) {
    const auto& physical_size = FormatHelper::GetStorageSizes(format, size);
    return c10::multiply_integers(physical_size);
}

int64_t StorageDescHelper::GetValidMemorySize(const at::Tensor& tensor) {
    int64_t real_bytes = 0;
    for (int64_t i = tensor.dim() - 1; i >= 0; i--) {
        real_bytes += (tensor.size(i) - 1) * tensor.stride(i);
    }
    return real_bytes + 1;
}

void StorageDescHelper::SetDesc(at::Tensor& dst) { torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_ = SetDesc(dst.dtype()); }

void StorageDescHelper::SetDesc(at::Tensor& dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides) {
    torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_ = SetDesc(dst.dtype(), size, strides);
}

void StorageDescHelper::SetDesc(at::Tensor& dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides, aclFormat format) {
    torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_ = SetDesc(dst.dtype(), size, strides, format);
}

void StorageDescHelper::CopyDesc(at::Tensor& dst, const at::Tensor& src) { CopyDesc(dst, src.storage()); }

void StorageDescHelper::CopyDesc(at::Tensor& dst, const c10::Storage& src) {
    CopyDesc(dst, torch_npu::NPUBridge::GetNpuStorageImpl(src.unsafeGetStorageImpl())->npu_desc_);
}

void StorageDescHelper::CopyDesc(const at::Tensor& dst, const torch_npu::NPUStorageDesc& src_desc) {
    auto& dstDesc = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_;
    dstDesc = src_desc;
}

void StorageDescHelper::ReflushDescBySelf(const at::Tensor& src) {
    auto& desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    desc.base_sizes_ = src.sizes();
    desc.storage_sizes_ = src.sizes();
    desc.base_strides_ = src.strides();
}

torch_npu::NPUStorageDesc StorageDescHelper::SetDesc(const caffe2::TypeMeta& dtype) { return SetDesc(dtype, {0}, {}); }

torch_npu::NPUStorageDesc StorageDescHelper::SetDesc(const caffe2::TypeMeta& dtype, const c10::IntArrayRef& size, const c10::IntArrayRef& strides) {
    return SetDesc(dtype, size, strides, InferFormat::GuessBaseFormat(size));
}

torch_npu::NPUStorageDesc StorageDescHelper::SetDesc(const caffe2::TypeMeta& dtype, const c10::IntArrayRef& size, const c10::IntArrayRef& strides,
                                                     aclFormat format) {
    struct torch_npu::NPUStorageDesc npu_desc;
    npu_desc.data_type_ = dtype;
    npu_desc.base_sizes_ = size;
    npu_desc.base_strides_ = strides;
    // guess ori format and npu format unit by size and dst format
    // eg: size: [2,3,4,5] format: nd
    // we will return [NCHW, NCHW] because 4 dim tensor must be nchw,
    // then the tensor used to be the input of conv2d will not make mistake
    aclFormat baseFormat;
    aclFormat npuFormat;
    std::tie(baseFormat, npuFormat) = InferFormat::GuessFormatUnit(size, format);
    npu_desc.storage_sizes_ = FormatHelper::GetStorageSizes(npuFormat, size);
    npu_desc.origin_format_ = baseFormat;
    npu_desc.npu_format_ = npuFormat;
    return npu_desc;
}

class OpAttrMaker {
public:
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, bool value);
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, int64_t value);
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, float value);
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, string value);
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, const char* value);
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, c10::IntArrayRef value);
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, at::ArrayRef<float> value);
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, at::ArrayRef<uint8_t> value);
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, c10::Scalar value);
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, at::ScalarType value);
    TORCH_NPU_API static void Set(aclopAttr* attr, const string& name, at::ArrayRef<c10::IntArrayRef> value);
};  // class OpAttrMaker

void OpAttrMaker::Set(aclopAttr* attr, const string& name, bool value) { aclopSetAttrBool(attr, name.c_str(), value); }

void OpAttrMaker::Set(aclopAttr* attr, const string& name, int64_t value) { aclopSetAttrInt(attr, name.c_str(), value); }

void OpAttrMaker::Set(aclopAttr* attr, const string& name, float value) { aclopSetAttrFloat(attr, name.c_str(), value); }

void OpAttrMaker::Set(aclopAttr* attr, const string& name, string value) { aclopSetAttrString(attr, name.c_str(), value.c_str()); }

void OpAttrMaker::Set(aclopAttr* attr, const string& name, const char* value) { aclopSetAttrString(attr, name.c_str(), value); }

void OpAttrMaker::Set(aclopAttr* attr, const string& name, c10::IntArrayRef value) { aclopSetAttrListInt(attr, name.c_str(), value.size(), value.data()); }

void OpAttrMaker::Set(aclopAttr* attr, const string& name, at::ArrayRef<float> value) { aclopSetAttrListFloat(attr, name.c_str(), value.size(), value.data()); }

void OpAttrMaker::Set(aclopAttr* attr, const string& name, at::ArrayRef<uint8_t> value) {
    aclopSetAttrListBool(attr, name.c_str(), value.size(), value.data());
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, c10::Scalar value) {
    float val = CalcuOpUtil::GetScalarFloatValue(value);
    aclopSetAttrFloat(attr, name.c_str(), val);
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, at::ScalarType value) {
    aclDataType val = CalcuOpUtil::ConvertToAclDataType(value);
    aclopSetAttrDataType(attr, name.c_str(), val);
}

void OpAttrMaker::Set(aclopAttr* attr, const string& name, at::ArrayRef<c10::IntArrayRef> value) {
    // Pointer to values of each listInt.
    c10::SmallVector<int64_t*, N> attrValue;
    // Pointer to number of each listInt.
    c10::SmallVector<int, N> eachListIntNum;
    // Value of each listInt.
    c10::SmallVector<c10::SmallVector<int64_t, N>, N> eachListIntVal;
    for (const auto i : c10::irange(value.size())) {
        c10::SmallVector<int64_t, N> listInt;
        int64_t valueSize = static_cast<int64_t>(value[i].size());
        listInt.resize(valueSize);
        std::copy(value[i].begin(), value[i].end(), listInt.begin());
        eachListIntVal.emplace_back(listInt);
        attrValue.emplace_back(eachListIntVal.back().data());
        eachListIntNum.emplace_back(valueSize);
    }

    aclopSetAttrListListInt(attr, name.c_str(), attrValue.size(), eachListIntNum.data(), attrValue.data());
}

class AclTensorDescMaker {
public:
    AclTensorDescMaker() {}
    ~AclTensorDescMaker() = default;

    AclTensorDescMaker& Create(aclDataType dataType, torch_npu::NPUStorageDesc storageDesc) {
        c10::SmallVector<int64_t, 5> dims;
        // if aclDataType is ACL_STRING, storageDims is empty.
        if (dataType != ACL_STRING) {
            dims = storageDesc.base_sizes_;
        }
        auto format = storageDesc.origin_format_;
        if (debugLevel()) {
            std::cout << __FUNCTION__ << ":" << dataType << "," << dims << "," << format << std::endl;
        }

        desc = aclCreateTensorDesc(dataType, dims.size(), dims.data(), format);
        return *this;
    }

    inline AclTensorDescMaker& Create(aclDataType dataType, c10::IntArrayRef dims, aclFormat format) {
        if (debugLevel()) {
            std::cout << __FUNCTION__ << ":" << dataType << "," << dims << "," << format << std::endl;
        }
        desc = aclCreateTensorDesc(dataType, dims.size(), dims.data(), format);
        return *this;
    }

    inline AclTensorDescMaker& Create(aclDataType dataType, aclFormat format) {
        if (debugLevel()) {
            std::cout << __FUNCTION__ << ":" << dataType << "," << format << std::endl;
        }
        desc = aclCreateTensorDesc(dataType, 0, nullptr, format);
        return *this;
    }

    inline AclTensorDescMaker& SetFormat(aclFormat format) {
        aclSetTensorFormat(desc, format);
        return *this;
    }

    inline AclTensorDescMaker& SetPlacement(aclMemType memType) {
        aclSetTensorPlaceMent(desc, memType);
        return *this;
    }

    template <unsigned int N>
    inline AclTensorDescMaker& SetShape(const c10::SmallVector<int64_t, N>& dims) {
        aclSetTensorShape(desc, dims.size(), dims.data());
        return *this;
    }

    template <unsigned int N>
    AclTensorDescMaker& SetRange(const c10::SmallVector<int64_t, N>& rangs) {
        int arryDim = rangs.size() == 0 ? 0 : rangs.size() / 2;

        int64_t range[arryDim][2];
        for (int i = 0, j = 0; i < arryDim; i++, j += 2) {
            range[i][0] = rangs[j];
            range[i][1] = rangs[j + 1];
        }

        aclSetTensorShapeRange(desc, arryDim, range);
        return *this;
    }

    inline AclTensorDescMaker& SetName(const std::string& name) {
        if (!name.empty()) {
            aclSetTensorDescName(desc, name.c_str());
        }
        return *this;
    }

    inline AclTensorDescMaker& SetConstAttr(c10::optional<at::Tensor> cpu_tensor) {
        if (cpu_tensor.has_value() && cpu_tensor.value().defined()) {
            aclSetTensorConst(desc, cpu_tensor.value().data_ptr(), cpu_tensor.value().itemsize() * cpu_tensor.value().numel());
        }

        return *this;
    }

    inline aclTensorDesc* Get() const { return desc; }

private:
    aclTensorDesc* desc = nullptr;
};  // class AclTensorDescMaker

//
class AclTensorBufferMaker {
public:
    // base of Ctr
    // params: tensor, offset, remained size
    AclTensorBufferMaker(const at::Tensor* tensor, int64_t offset, int64_t n) {
        uint8_t* header = reinterpret_cast<uint8_t*>(tensor->data_ptr()) - tensor->itemsize() * static_cast<uint8_t>(offset);
        size_t bufferSize = tensor->itemsize() * static_cast<size_t>(n);
        ptr = aclCreateDataBuffer(header, bufferSize);
    }

    // offset = 0
    explicit AclTensorBufferMaker(const at::Tensor* tensor, int64_t n = 1) {
        if (tensor == nullptr || n == 0) {
            ptr = aclCreateDataBuffer(nullptr, 0);
        } else {
            ptr = aclCreateDataBuffer(reinterpret_cast<void*>(tensor->data_ptr()), tensor->itemsize() * n);
        }
    }

    // offset = 0
    explicit AclTensorBufferMaker(const at::Tensor& tensor, int64_t n = 1) {
        ptr = aclCreateDataBuffer(reinterpret_cast<void*>(tensor.data_ptr()), tensor.itemsize() * n);
    }

    ~AclTensorBufferMaker() = default;

    inline aclDataBuffer* Get() const { return ptr; }

private:
    aclDataBuffer* ptr = nullptr;
};  // class AclTensorBufferMaker

struct ACL_PARAMS {
    ACL_PARAMS() {
        input_desc = nullptr;
        input_data_buf = nullptr;
        output_desc = nullptr;
        output_data_buf = nullptr;
    }

    int input_num{0};
    const aclTensorDesc** input_desc;
    const aclDataBuffer** input_data_buf;
    int output_num{0};
    const aclTensorDesc** output_desc;
    aclDataBuffer** output_data_buf;
};

struct ACL_DYNAMIC_PARAMS {
    ACL_DYNAMIC_PARAMS() {
        input_desc = nullptr;
        input_data_buf = nullptr;
        output_desc = nullptr;
        output_data_buf = nullptr;
        inputDims = nullptr;
        outputDims = nullptr;
        inputFormats = nullptr;
        outputFormats = nullptr;
        compile_input_desc = nullptr;
        compile_output_desc = nullptr;

        hasAttr = false;
    }

    int input_num = 0;
    const aclTensorDesc** input_desc;
    const aclDataBuffer** input_data_buf;
    int output_num = 0;
    const aclTensorDesc** output_desc;
    aclDataBuffer** output_data_buf;
    int64_t* inputDims;
    int64_t* outputDims;
    aclFormat* inputFormats;
    aclFormat* outputFormats;
    const aclTensorDesc** compile_input_desc;
    const aclTensorDesc** compile_output_desc;
    bool hasAttr;
    std::string dynamicKey;
};

struct CONST_PARAMS {
    int constNum = 0;
    const int64_t** constList = nullptr;
    const int64_t* constIdx = nullptr;
    CONST_PARAMS() = default;
};

struct ExecuteParas {
    using PROCESS_FUNC = std::function<int()>;
    char opType[50]{};
    bool isJitDisable = false;
    ACL_PARAMS paras;
    CONST_PARAMS constParams;
    const aclopAttr* attr;
    int64_t constIdx = -1;
    static std::atomic<uint64_t> g_pta_correlation_id;
    uint64_t pta_correlation_id = 0;
    c10::SmallVector<at::Tensor, N> hostMemory;
    ExecuteParas() = default;
    void Release();
    void Copy(ExecuteParas& other);
    void CopyEx(ExecuteParas& other);
    PROCESS_FUNC customHandler;
};

std::atomic<uint64_t> ExecuteParas::g_pta_correlation_id{0};

NPUStatus DestroyAclParams(ACL_PARAMS& params) {
    if (params.input_num != 0) {
        if (params.input_desc != nullptr) {
            for (int i = 0; i < params.input_num; ++i) {
                aclDestroyTensorDesc(params.input_desc[i]);
            }
        }
        if (params.input_data_buf != nullptr) {
            for (int i = 0; i < params.input_num; ++i) {
                NPU_CHECK_ERROR(aclDestroyDataBuffer(params.input_data_buf[i]));
            }
        }
        params.input_num = 0;
    }
    if (params.output_num != 0) {
        if (params.output_desc != nullptr) {
            for (int i = 0; i < params.output_num; ++i) {
                aclDestroyTensorDesc(params.output_desc[i]);
            }
        }
        if (params.output_data_buf != nullptr) {
            for (int i = 0; i < params.output_num; ++i) {
                NPU_CHECK_ERROR(aclDestroyDataBuffer(params.output_data_buf[i]));
            }
        }
        params.output_num = 0;
    }
    free(params.input_desc);
    params.input_desc = nullptr;
    params.input_data_buf = nullptr;
    params.output_desc = nullptr;
    params.output_data_buf = nullptr;
    return SUCCESS;
}

void DestroyConstParams(CONST_PARAMS& params) {
    if (params.constList != nullptr) {
        for (int i = 0; i < params.constNum; ++i) {
            if (params.constList[i] != nullptr) {
                delete[] params.constList[i];
            }
        }
    }
    params.constList = nullptr;
    params.constIdx = nullptr;
}

void ExecuteParas::Release() {
    // if useDynamicCompile, this attr will be freed in dynamic compile.
    if (attr != nullptr) {
        aclopDestroyAttr(attr);
    }
    DestroyConstParams(constParams);
    NPUStatus ret = DestroyAclParams(paras);
    if (ret != SUCCESS) {
        NPU_LOGE("DestroyAclParams fail, ret: %s", ret.c_str());
    }
    hostMemory.clear();
    customHandler = nullptr;
    return;
}

void ExecuteParas::Copy(ExecuteParas& other) {
    strncpy(this->opType, other.opType, sizeof(ExecuteParas::opType) - 1);
    this->paras = other.paras;
    this->attr = other.attr;
    this->constParams = other.constParams;
    this->hostMemory = other.hostMemory;
    this->isJitDisable = other.isJitDisable;
    this->customHandler = other.customHandler;
    this->pta_correlation_id = other.pta_correlation_id;
}

void ExecuteParas::CopyEx(ExecuteParas& other) {
    this->paras = other.paras;
    this->attr = other.attr;
    this->constParams = other.constParams;
}

// the member in AclExecParam is create by :
// aclCreateDataBuffer and aclCreateTensorDesc
// so aclDestroyTensorDesc and aclDestroyDataBuffer should be called when dtr
// aclopDestroyAttr
class OpCommandImpl {
public:
    OpCommandImpl() {}
    ~OpCommandImpl() {
        // do nothing, can not release resource, because of multi-thread or
        // queue-enable
    }

    void SetName(const string& name) { opName = name; }

    void SetCustomHandler(PROC_FUNC func) { execParam.customHandler = func; }

    const string& GetName() const { return opName; }

    void AddInput(const aclTensorDesc* desc, const aclDataBuffer* buffer) {
        execParam.inDesc.emplace_back(std::move(desc));
        execParam.inBuffer.emplace_back(std::move(buffer));
    }

    void AddInput(const aclTensorDesc* desc, const aclDataBuffer* buffer, const at::Tensor& hostTensor) {
        AddInput(desc, buffer);
        execParam.hostMem.emplace_back(hostTensor);
    }

    void AddInput(const string& str);

    void AddOutput(const aclTensorDesc* desc, aclDataBuffer* buffer) {
        execParam.outDesc.emplace_back(std::move(desc));
        execParam.outBuffer.emplace_back(std::move(buffer));
    }

    template <typename dataType>
    void AddAttr(const string& attrName, dataType value) {
        InitAttr();
        OpAttrMaker::Set(execParam.attr, attrName, value);
    }

    // export op execute params
    void ExportParams(ExecuteParas& params) {
        TORCH_CHECK(sizeof(ExecuteParas::opType) >= opName.length() + 1, "Too long Ascend IR Name: ", opName);
        memset(params.opType, '\0', sizeof(params.opType));
        opName.copy(params.opType, opName.length() + 1);
        params.attr = execParam.attr;
        // make params
        int inputNum = static_cast<int>(execParam.inDesc.size());
        int outputNum = static_cast<int>(execParam.outDesc.size());

        size_t inputTensorDescArrLen = inputNum * sizeof(uintptr_t);
        size_t inputDataBuffArrLen = inputNum * sizeof(uintptr_t);

        size_t outputTensorDescArrLen = outputNum * sizeof(uintptr_t);
        size_t outputDataBuffArrLen = outputNum * sizeof(uintptr_t);

        size_t totalMemLen = inputTensorDescArrLen + inputDataBuffArrLen + outputTensorDescArrLen + outputDataBuffArrLen;
        char* basePtr = static_cast<char*>(malloc(totalMemLen));
        AT_ASSERT(basePtr != nullptr);
        const aclTensorDesc** aclTensorInputDescArr = reinterpret_cast<const aclTensorDesc**>(basePtr);
        basePtr += inputTensorDescArrLen;
        const aclDataBuffer** aclDataInputBuffArr = reinterpret_cast<const aclDataBuffer**>(basePtr);
        basePtr += inputDataBuffArrLen;

        const aclTensorDesc** aclTensorOutputDescArr = reinterpret_cast<const aclTensorDesc**>(basePtr);
        basePtr += outputTensorDescArrLen;
        aclDataBuffer** aclDataOutputBuffArr = reinterpret_cast<aclDataBuffer**>(basePtr);

        std::copy(execParam.inDesc.begin(), execParam.inDesc.end(), aclTensorInputDescArr);
        std::copy(execParam.inBuffer.begin(), execParam.inBuffer.end(), aclDataInputBuffArr);
        std::copy(execParam.outDesc.begin(), execParam.outDesc.end(), aclTensorOutputDescArr);
        std::copy(execParam.outBuffer.begin(), execParam.outBuffer.end(), aclDataOutputBuffArr);

        params.paras.input_num = inputNum;
        params.paras.output_num = outputNum;
        params.paras.input_desc = aclTensorInputDescArr;
        params.paras.input_data_buf = aclDataInputBuffArr;
        params.paras.output_desc = aclTensorOutputDescArr;
        params.paras.output_data_buf = aclDataOutputBuffArr;
        params.hostMemory = execParam.hostMem;
        params.customHandler = execParam.customHandler;
        params.pta_correlation_id = ExecuteParas::g_pta_correlation_id++;
#if 0
        if (!ForceJitCompileList::GetInstance().Inlist(opName) && env::CheckJitDisable()) {
            params.isJitDisable = true;
        }
#endif
    }

    // Set engine priority for op on data preprocessing stream
    void SetEnginePriority() {
        auto stream = c10_npu::getCurrentNPUStream();
        // if (stream.isDataPreprocessStream()) {
        if (0) {
            AddAttr("_performance_prior", true);
            AddAttr<std::string>("_exclude_engines", "AiCore");
        }
    }

    void Run(bool sync, c10::SmallVector<int64_t, N>& sync_index, c10::SmallVector<at::Tensor, N>& outputTensor);

    void releaseSource(bool no_blocking = true) {
        if (no_blocking) {
            std::for_each(execParam.inDesc.begin(), execParam.inDesc.end(), aclDestroyTensorDesc);
            std::for_each(execParam.outDesc.begin(), execParam.outDesc.end(), aclDestroyTensorDesc);
            std::for_each(execParam.inBuffer.begin(), execParam.inBuffer.end(), aclDestroyDataBuffer);
            std::for_each(execParam.outBuffer.begin(), execParam.outBuffer.end(), aclDestroyDataBuffer);
            if (execParam.attr != nullptr) {
                aclopDestroyAttr(execParam.attr);
                execParam.attr = nullptr;
            }
        }

        execParam.inDesc.clear();
        execParam.inBuffer.clear();

        execParam.outDesc.clear();
        execParam.outBuffer.clear();

        execParam.hostMem.clear();

        // recover
        execParam.attr = nullptr;
        execParam.customHandler = nullptr;
        opName = "";
    }

private:
    struct AclExecParam {
        c10::SmallVector<const aclTensorDesc*, N> inDesc;    // owned
        c10::SmallVector<const aclDataBuffer*, N> inBuffer;  // owned
        c10::SmallVector<const aclTensorDesc*, N> outDesc;   // owned
        c10::SmallVector<aclDataBuffer*, N> outBuffer;       // owned
        c10::SmallVector<at::Tensor, N> hostMem;
        aclopAttr* attr = nullptr;
        PROC_FUNC customHandler = nullptr;
    };

    void InitAttr() {
        if (execParam.attr == nullptr) {
            execParam.attr = aclopCreateAttr();
        }
    }

    aclError InnerRun(const string& name, AclExecParam& params, bool sync, c10::SmallVector<int64_t, N>& sync_index,
                      c10::SmallVector<at::Tensor, N>& outputTensor);

    void SetDeterministic();

private:
    string opName;
    AclExecParam execParam;
};  // class OpCommandImpl

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define ASCEND_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define ASCEND_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define ASCEND_LIKELY(expr) (expr)
#define ASCEND_UNLIKELY(expr) (expr)
#endif

#if __has_attribute(always_inline) || defined(__GNUC__)
#define ASCEND_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#elif defined(_MSC_VER)
#define ASCEND_ALWAYS_INLINE __forceinline
#else
#define ASCEND_ALWAYS_INLINE inline
#endif

#define ACL_REQUIRE_OK_OP(expr, opstr)                                                                                                                   \
    do {                                                                                                                                                 \
        if (ASCEND_UNLIKELY((expr) != 0)) {                                                                                                              \
            printf("%s\n", opstr);                                                                                                                       \
            TORCH_CHECK((expr) == 0, __func__, ":", __FILE__, ":", __LINE__, " NPU error,NPU error code is:", expr, "\n", c10_npu::acl::AclGetErrMsg()); \
        }                                                                                                                                                \
    } while (0)

void OpCommandImpl::Run(bool sync, c10::SmallVector<int64_t, N>& sync_index, c10::SmallVector<at::Tensor, N>& outputTensor) {
    NPU_LOGD("Op %s start run.", opName.c_str());
    // RECORD_FUNCTION(opName, std::vector<c10::IValue>({}));
    diopi::GilScopedRelease gilReleaeGuard;
    ACL_REQUIRE_OK_OP(InnerRun(opName, execParam, sync, sync_index, outputTensor), opName.c_str());
    NPU_LOGD("Op %s run over.", opName.c_str());
}

aclError OpCommandImpl::InnerRun(const string& name, AclExecParam& params, bool sync, c10::SmallVector<int64_t, N>& sync_index,
                                 c10::SmallVector<at::Tensor, N>& outputTensor) {
    aclError ret;
    // at_npu::native::NpuUtils::ProfReportMarkData(name);
    auto stream = c10_npu::getCurrentNPUStream();
    auto inputSize = params.inBuffer.size();
    auto outputSize = params.outBuffer.size();
    // open the deterministicAlgorithms config
    SetDeterministic();
    bool reset_flag = false;
#if 0
    if (ForceJitCompileList::GetInstance().Inlist(name) && env::CheckJitDisable()) {
        AclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, "enable");
        reset_flag = true;
    }
#endif
    int index = 0;
    do {
        if (params.customHandler) {
            ret = params.customHandler();
            if (ret != ACL_ERROR_NONE) {
                C10_NPU_SHOW_ERR_MSG();
                TORCH_CHECK(false, "Custom hand fail!");
            }
            index++;
            continue;
        }
#if 0
        if (at_npu::native::aoe::aoe_manager().IsAoeEnabled() &&
            at_npu::native::aoe::aoe_manager().IsInWhitelist(name)) {
          ret = at_npu::native::AclGenGraphAndDumpForOp(
              name.c_str(),
              inputSize,
              params.inDesc.data(),
              params.inBuffer.data(),
              outputSize,
              params.outDesc.data(),
              params.outBuffer.data(),
              params.attr,
              ACL_ENGINE_SYS,
              at_npu::native::aoe::aoe_manager().GetDumpGraphPath().c_str(),
              nullptr);
          if (ret != ACL_ERROR_NONE) {
            C10_NPU_SHOW_ERR_MSG();
            TORCH_CHECK(false, "In aoe mode, AclGenGraphAndDumpForOp failed!");
          }
        }
#endif
        {
            diopi::StreamLockGuard streamLockGuard(stream.stream());
            ret = AclopCompileAndExecuteV2(name.c_str(),
                                           inputSize,
                                           const_cast<aclTensorDesc**>(params.inDesc.data()),
                                           const_cast<aclDataBuffer**>(params.inBuffer.data()),
                                           outputSize,
                                           const_cast<aclTensorDesc**>(params.outDesc.data()),
                                           params.outBuffer.data(),
                                           params.attr,
                                           ACL_ENGINE_SYS,
                                           ACL_COMPILE_SYS,
                                           NULL,
                                           stream);
        }
        NPU_CHECK_ERROR(ret);
        if (sync) {
            int64_t dimSize;
            for (size_t i = 0; i < sync_index.size(); i++) {
                c10::SmallVector<int64_t, N> real_shape;
                for (int64_t j = 0; j < outputTensor[sync_index[i]].dim(); j++) {
                    NPU_CHECK_ERROR(aclGetTensorDescDimV2(params.outDesc[sync_index[i]], j, &dimSize));
                    real_shape.emplace_back(dimSize);
                }
                outputTensor[sync_index[i]].resize_(real_shape);
            }
        }
        ++index;
    } while (NpuUtils::IsOomError(ret, index) && (index < NPU_MAX_OP_EXEC_TRY_NUM));
    if (reset_flag) {
        AclSetCompileopt(aclCompileOpt::ACL_OP_JIT_COMPILE, "disable");
    }
    return ret;
}

std::tuple<aclTensorDesc*, aclDataBuffer*> CovertNPUTensorWithZeroDimToAclInput(const at::Tensor& tensor, const string& descName) {
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(tensor.scalar_type());
    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(aclDataType, ACL_FORMAT_ND).SetName(descName).Get();
    AclTensorBufferMaker buffer(tensor);
    auto aclBuff = buffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc*, aclDataBuffer*> CovertTensorWithZeroDimToAclInput(const at::Tensor& tensor, at::ScalarType type) {
    at::ScalarType scalarDataType = type;
    if (!tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
        scalarDataType = tensor.scalar_type();
    }
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(scalarDataType);
    c10::Scalar expScalar = tensor.item();
    at::Tensor aclInput = CalcuOpUtil::CopyScalarToDevice(expScalar, scalarDataType);

    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(aclDataType, ACL_FORMAT_ND).Get();
    AclTensorBufferMaker buffer(aclInput);
    auto aclBuff = buffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc*, aclDataBuffer*> CovertTensorToAclInput(const at::Tensor& tensor, const string& descName, const string& forceDataType) {
    at::ScalarType scalarDataType = tensor.scalar_type();
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(scalarDataType, forceDataType);
    auto format = CalcuOpUtil::GetTensorNpuFormat(tensor);
    // const auto &npuDesc = torch_npu::NPUBridge::GetNpuStorageImplDesc(tensor);

    AclTensorDescMaker desc;

    auto aclDesc = desc.Create(aclDataType, tensor.sizes(), static_cast<aclFormat>(format)).SetName(descName).Get();

    // if aclDataType != ACL_STRING, we use storageDims to calculate nums and use
    // nums * tensor element size to calculate buffer size. But if aclDataType =
    // ACL_STRING, STRING tensor size = 1 and storageDims = 0, we can not use it
    // to calculate size, we need from storage_sizes_ to calculate STRING element
    // real size.
    AclTensorBufferMaker buffer(tensor, tensor.numel());
    auto aclBuff = buffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc*, aclDataBuffer*> CovertHostTensorToAclInput(const at::Tensor& tensor, at::ScalarType type, CompileType compileType,
                                                                      const string& forceDataType, const string& descName) {
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(type, forceDataType);
    const auto& dims = tensor.sizes();
    AclTensorDescMaker desc;
    aclFormat format = ACL_FORMAT_ND;
    auto aclDesc = desc.Create(aclDataType, dims, format).SetPlacement(static_cast<aclMemType>(compileType)).SetName(descName).Get();
    AclTensorBufferMaker buffer(tensor, tensor.numel());
    auto aclBuff = buffer.Get();
    return std::tie(aclDesc, aclBuff);
}

std::tuple<aclTensorDesc*, aclDataBuffer*> CovertToAclOutput(const at::Tensor& tensor, const string& forceDataType) {
    aclDataType aclDataType = CalcuOpUtil::ConvertToAclDataType(tensor.scalar_type(), forceDataType);
    auto format = CalcuOpUtil::GetTensorNpuFormat(tensor);
    AclTensorDescMaker desc;
    aclTensorDesc* aclDesc = nullptr;
    if (tensor.sizes().size() > 0 && tensor.numel() == 0) {
        aclDesc = desc.Create(aclDataType, tensor.sizes(), static_cast<aclFormat>(format)).Get();
    } else {
        aclDesc = desc.Create(aclDataType, ACL_FORMAT_ND).Get();
    }
    AclTensorBufferMaker aclBuffer(tensor, tensor.numel());
    auto aclBuff = aclBuffer.Get();
    return std::tie(aclDesc, aclBuff);
}

// This class maintain the position of the current
// OpCommandImpl object in vector, the resources in
// the object is

static bool deterministicaclnn_oldstatus = false;

void OpCommandImpl::SetDeterministic() {
    auto deterministicAlgorithmsStatus = at::globalContext().deterministicAlgorithms();
    if (deterministicaclnn_oldstatus != deterministicAlgorithmsStatus) {
        NPU_CHECK_ERROR(AclSetCompileopt(aclCompileOpt::ACL_OP_DETERMINISTIC, deterministicAlgorithmsStatus ? "1" : "0"));
        NPU_CHECK_ERROR(AclrtCtxSetSysParamOpt(aclSysParamOpt::ACL_OPT_DETERMINISTIC, deterministicAlgorithmsStatus ? 1 : 0));
        // HcclConfigValue configValue = {deterministicAlgorithmsStatus ? 1 : 0};
        // HCCL_CHECK_ERROR(hccl::HcclSetConfig(HcclConfig::HCCL_DETERMINISTIC, configValue));
        deterministicaclnn_oldstatus = deterministicAlgorithmsStatus;
    }
}

OpCommand::OpCommand() {
    aclCmd = new OpCommandImpl();
    aclCmd->SetCustomHandler(nullptr);
}

OpCommand::~OpCommand() { delete aclCmd; }

OpCommand& OpCommand::Name(const string& name) {
    aclCmd->SetName(name);
    return *this;
}

void OpCommand::SetCustomHandler(PROC_FUNC func) { INTERFACE_NOT_IMPL; }

OpCommand& OpCommand::Expect(UnifiedResult unified_result) {
    commonType = unified_result.common_type;
    resultTypeDefined = unified_result.result_type_defined;
    commonShape = unified_result.common_shape;
    return *this;
}

// None Input
OpCommand& OpCommand::Input() {
    AclTensorDescMaker desc;
    auto aclDesc = desc.Create(ACL_DT_UNDEFINED, ACL_FORMAT_UNDEFINED).Get();
    AclTensorBufferMaker buffer(nullptr, 0);
    aclCmd->AddInput(aclDesc, buffer.Get());
    return *this;
}

OpCommand& OpCommand::AddTensorInput(at::Tensor& tensor, at::ScalarType forceScaleType, const string& descName, const string& realData) {
    if (enableDumpArgs()) {
        std::cout << aclCmd->GetName() << ":descName:" << descName << ",input:" << impl::aten::dumpArgs(tensor) << " " << realData << std::endl;
    }
    std::tuple<aclTensorDesc*, aclDataBuffer*> res;
    if (commonType.has_value() && commonType.value() != tensor.scalar_type()) {
        tensor = acl_op::npu_dtype_cast(tensor, commonType.value());
    }

    if (tensor.sizes().size() == 0) {
        if (at_npu::key::isDeviceTensor(tensor)) {
            res = CovertNPUTensorWithZeroDimToAclInput(tensor, descName);
        } else {
            res = CovertTensorWithZeroDimToAclInput(tensor, forceScaleType);
        }
    } else {
        res = CovertTensorToAclInput(tensor, descName, realData);
    }
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res));
    return *this;
}

at::Tensor& OpCommand::Contiguous(const at::Tensor& input) {
    storage.emplace_back(std::move(NpuUtils::format_contiguous_add_copy_optimize(input)));
    return storage.back();
}

// Tensor Input which need contiguous
OpCommand& OpCommand::Input(const at::Tensor& input, const string& descName, const c10::optional<aclFormat>& sensitive_format, const string& realData) {
    return AddTensorInput(Contiguous(input), c10::ScalarType::Undefined, descName, realData);
}

template <typename T>
OpCommand& OpCommand::Input(const c10::ArrayRef<T>& dimListRef, at::IntArrayRef realShape, at::ScalarType toType, CompileType compileType,
                            const string& realDtype, const string& descName) {
    // at::Tensor &tensor = CreateHostTensor((void *)dimListRef.data(), realShape,
    // c10::TensorOptions(at::kCPU).dtype(c10::CppTypeToScalarType<T>::value),toType);
    //  AddHostTensorInput(tensor, compileType, realDtype, descName);
    auto cpuTensor = at::empty(realShape, c10::TensorOptions(at::kCPU).dtype(c10::CppTypeToScalarType<T>::value));
    std::memcpy(cpuTensor.data_ptr(), reinterpret_cast<const void*>(dimListRef.data()), cpuTensor.itemsize() * cpuTensor.numel());
    if (toType != cpuTensor.dtype()) {
        cpuTensor = cpuTensor.to(toType);
    }
    std::tuple<aclTensorDesc*, aclDataBuffer*> res = CovertHostTensorToAclInput(cpuTensor, cpuTensor.scalar_type(), compileType, realDtype, descName);
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res), cpuTensor);

    return *this;
}

template OpCommand& OpCommand::Input(const c10::ArrayRef<double>& dimListRef, at::IntArrayRef realShape, at::ScalarType toType, CompileType compileType,
                                     const string& realDtype, const string& descName);

// IntArrayRef/SmallVector Input, usually hostmemory input, we will do h2d in
// launch kernel
OpCommand& OpCommand::Input(const c10::IntArrayRef& dimListRef, at::ScalarType toType, CompileType compileType, const string& realDtype,
                            const string& descName) {
    Input<int64_t>(dimListRef, dimListRef.size(), toType, compileType, realDtype, descName);
    if (enableDumpArgs()) {
        std::cout << aclCmd->GetName() << ":descName:" << descName << ",input:" << dimListRef << " " << toType << " " << compileType << " " << realDtype
                  << std::endl;
    }
    return *this;
}

namespace {
const uint64_t kStringOffset = 16UL;
const std::string kStringDType = "string";  // NOLINT
static std::unordered_map<at::ScalarType, std::vector<double>> floating_limits_map{
    {at::ScalarType::Double, {std::numeric_limits<double>::max(), std::numeric_limits<double>::min()}},
    {at::ScalarType::Float, {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()}},
    {at::ScalarType::BFloat16, {std::numeric_limits<float>::max(), std::numeric_limits<float>::min()}},
    {at::ScalarType::Half, {65504, -65504}}};
static std::unordered_map<at::ScalarType, std::vector<long>> integral_limits_map{
    {at::ScalarType::Long, {std::numeric_limits<long>::max(), std::numeric_limits<long>::min()}},
    {at::ScalarType::Int, {std::numeric_limits<int>::max(), std::numeric_limits<int>::min()}},
    {at::ScalarType::Byte, {std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::min()}},
    {at::ScalarType::Char, {std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::min()}},
    {at::ScalarType::Short, {std::numeric_limits<int16_t>::max(), std::numeric_limits<int16_t>::min()}}};
}  // namespace

bool ScalarIsInLimits(const c10::Scalar& scalar, at::ScalarType type) {
    bool scalar_flag = false;
    if (at::isFloatingType(type)) {
        auto value = scalar.to<double>();
        scalar_flag = value <= floating_limits_map[type][0] && value >= floating_limits_map[type][1];
    } else if (at::isIntegralType(type)) {
        auto value = scalar.to<long>();
        scalar_flag = value <= integral_limits_map[type][0] && value >= integral_limits_map[type][1];
    }
    return scalar_flag;
}

// Scalar Input, we will do h2d in launch kernel
OpCommand& OpCommand::Input(const c10::Scalar& input, const at::ScalarType type, CompileType compileType) {
    if (enableDumpArgs()) {
        std::cout << aclCmd->GetName() << ":input:" << impl::aten::dumpArgs(input) << " " << type << " " << compileType << std::endl;
    }
    at::ScalarType scalar_type = type;
    if (commonType.has_value()) {
        scalar_type = commonType.value();
    }

    at::Tensor tensor =
        ScalarIsInLimits(input, scalar_type) ? at::detail::scalar_tensor_static(input, scalar_type, at::kCPU) : at::scalar_to_tensor(input).to(scalar_type);
    std::tuple<aclTensorDesc*, aclDataBuffer*> res = CovertHostTensorToAclInput(tensor, tensor.scalar_type(), compileType, "", "");
    aclCmd->AddInput(std::get<0>(res), std::get<1>(res), tensor);
    return *this;
}

// Tensor Input which no need contiguous
OpCommand& OpCommand::InputWithoutContiguous(const at::Tensor& input, const string& descName, const string& realData) {
    if (enableDumpArgs()) {
        std::cout << aclCmd->GetName() << ":InputWithoutContiguous input:" << impl::aten::dumpArgs(input) << descName << realData << std::endl;
    }
    if (input.storage_offset() != 0) {
        TORCH_WARN_ONCE("[Check][offset] Check input storage_offset[%ld] = 0 failed, result is untrustworthy", input.storage_offset());
    }
    return AddTensorInput(const_cast<at::Tensor&>(input));
}

// Output Tensor
OpCommand& OpCommand::Output(at::Tensor& output, const string& descName, const c10::optional<aclFormat>& sensitive_format, const string& realType) {
    if (enableDumpArgs()) {
        std::cout << aclCmd->GetName() << ":descName:" << descName << ",output:" << impl::aten::dumpArgs(output) << std::endl;
    }
    if (resultTypeDefined == false && commonType.has_value() && commonType.value() != output.scalar_type()) {
        output = acl_op::npu_dtype_cast(output, commonType.value());
    }
    auto res = CovertToAclOutput(output, realType);
    aclCmd->AddOutput(std::get<0>(res), std::get<1>(res));
    outputTensor.emplace_back(output);
    return *this;
}

template <typename dataType>
OpCommand& OpCommand::Attr(const string& name, dataType value) {
    if (enableDumpArgs()) {
        std::cout << aclCmd->GetName() << ":Attr:" << name << ":" << value << std::endl;
    }
    aclCmd->AddAttr(name, value);
    return *this;
}

template OpCommand& OpCommand::OpCommand::Attr<string>(const string& name, string value);
template OpCommand& OpCommand::OpCommand::Attr<const char*>(const string& name, const char* value);
template OpCommand& OpCommand::OpCommand::Attr<bool>(const string& name, bool value);
template OpCommand& OpCommand::OpCommand::Attr<float>(const string& name, float value);
template OpCommand& OpCommand::OpCommand::Attr<int64_t>(const string& name, int64_t value);
template OpCommand& OpCommand::OpCommand::Attr<c10::SmallVector<int64_t, N>>(const string& name, c10::SmallVector<int64_t, N> value);
template OpCommand& OpCommand::OpCommand::Attr<c10::SmallVector<int64_t, 8>>(const string& name, c10::SmallVector<int64_t, 8> value);
template OpCommand& OpCommand::OpCommand::Attr<c10::SmallVector<float, 8>>(const string& name, c10::SmallVector<float, 8> value);
template OpCommand& OpCommand::OpCommand::Attr<c10::SmallVector<float, 32>>(const string& name, c10::SmallVector<float, 32> value);
template OpCommand& OpCommand::OpCommand::Attr<c10::ArrayRef<long>>(const string& name, c10::ArrayRef<long> value);
template OpCommand& OpCommand::OpCommand::Attr<c10::ArrayRef<float>>(const string& name, c10::ArrayRef<float> value);
template OpCommand& OpCommand::OpCommand::Attr<c10::ArrayRef<unsigned char>>(const string& name, c10::ArrayRef<unsigned char> value);
template OpCommand& OpCommand::OpCommand::Attr<c10::ScalarType>(const string& name, c10::ScalarType value);
template OpCommand& OpCommand::OpCommand::Attr<c10::Scalar>(const string& name, c10::Scalar value);

// Run a single op
void OpCommand::Run() {
    aclCmd->SetEnginePriority();
    const string& op_name = aclCmd->GetName();
    aclCmd->Run(sync, sync_index, outputTensor);
    if (sync) {
        Sync();
    }
    aclCmd->releaseSource();
}

OpCommand& OpCommand::Sync(c10::SmallVector<int64_t, N>& index) {
    sync_index = index;
    if (!index.empty()) {
        sync = true;
    }
    Sync();
    return *this;
}

OpCommand& OpCommand::Sync() {
    c10_npu::getCurrentNPUStream().synchronize();
    return *this;
}

void npu_fast_reshape_(at::Tensor& tensor) {
    /**
      [NOTE] For some reshape cases such as view, unsqueeze, squeeze, flatten,
      storages of them remain unchanged. So we can refresh reshape tensor's
      metadata to obtain matched tensor.
      */

    // restriction 1
    if (!tensor.is_contiguous()) {
        return;
    }
    // restriction 2
    if (!FormatHelper::IsBaseFormatType(tensor)) {
        return;
    }
#if 0
  // restriction 3: reshape case without any numels change
  if ((tensor.numel() != StorageDescHelper::GetMemorySize(tensor)) ||
      StorageDescHelper::MetaDataAreMatch(&tensor)) {
    return;
  }

  // refresh matadata to input tensor
  StorageDescHelper::ReflushDescBySelf(tensor);
  auto base_format = InferFormat::GuessBaseFormat(tensor.sizes());
  NPUNativeFunctions::npu_format_cast_(tensor, base_format);
#else
    INTERFACE_NOT_IMPL
#endif
}

}  // namespace native

std::pair<uint64_t, uint64_t> NPUGeneratorImpl::philox_engine_inputs(uint64_t increment) {
    diopiTensorHandle_t stateHandle = nullptr;
    auto gen = reinterpret_cast<diopiGeneratorHandle_t>(generator_);
    diopiGeneratorGetState(context, gen, &stateHandle);
    void* statePtr = nullptr;
    diopiGetTensorData(stateHandle, &statePtr);
    PhiloxNpuState* state = reinterpret_cast<PhiloxNpuState*>(statePtr);
    auto ret = std::make_pair(state->seed_, state->offset_.val);
    state->offset_.val += increment;
    diopiGeneratorSetState(gen, stateHandle);
    return ret;
}

namespace detail {

const at::Generator& getDefaultNPUGenerator(c10::DeviceIndex device_index) { INTERFACE_NOT_IMPL }

}  // namespace detail

NPUGeneratorImpl::NPUGeneratorImpl(c10::DeviceIndex device_index)
    : c10::GeneratorImpl{c10::Device(at_npu::key::NativeDeviceType, device_index), c10::DispatchKeySet(at_npu::key::NativeDispatchKey)} {
    // at::npu::assertNotCapturing("Cannot construct a new NPUGeneratorImpl");
}

}  // namespace at_npu

thread_local diopiContextHandle_t context = nullptr;

namespace c10_npu {
namespace acl {

const char* AclGetErrMsg() {
    typedef const char* (*aclGetErrMsg)();
    static aclGetErrMsg func = aclGetRecentErrMsg;
    if (func != nullptr) {
        auto res = func();
        return res != nullptr ? res : "";
    }
    return "";
}

}  // namespace acl

NPUStream getCurrentNPUStream(c10::DeviceIndex device_index) {
    if (device_index == -1) {
        device_index = current_device();
    }
    TORCH_CHECK(context);
    diopiStreamHandle_t stream_handle = nullptr;
    diopiGetStream(context, &stream_handle);
    TORCH_CHECK(stream_handle);
    c10::Device device(c10::DeviceType::XLA, device_index);
    c10::Stream atStream(c10::Stream::Default::DEFAULT, device);
    aclrtStream aclStream = reinterpret_cast<aclrtStream>(stream_handle);
    TORCH_CHECK(aclStream);
    return NPUStream(NPUStream::Unchecked::UNCHECKED, atStream, aclStream);
}

NPUStream getCurrentSecondaryStream(c10::DeviceIndex device_index) { return getCurrentNPUStream(device_index); }

void NPUStream::synchronize() const {
    NPU_CHECK_ERROR(aclrtSynchronizeStream(aclStream_));
    NPU_CHECK_ERROR(aclrtSynchronizeDevice());
}

aclError queue::LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind) {
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    return aclrtMemcpyAsync(dst, dstLen, src, srcLen, kind, stream);
}

}  // namespace c10_npu

namespace torch_npu {

NPUStorageImpl::NPUStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes, at::DataPtr data_ptr, at::Allocator* allocator, bool resizable)
    : c10::StorageImpl(use_byte_size, size_bytes, at::DataPtr(std::move(data_ptr)), allocator, resizable) {}

void NPUStorageImpl::release_resources() { StorageImpl::release_resources(); }

NPUStorageImpl* NPUBridge::GetNpuStorageImpl(c10::StorageImpl* storageImpl) { return static_cast<NPUStorageImpl*>(storageImpl); }

NPUStorageImpl* NPUBridge::GetNpuStorageImpl(c10::Storage&& storage) { return static_cast<NPUStorageImpl*>(storage.unsafeGetStorageImpl()); }

NPUStorageImpl* NPUBridge::GetNpuStorageImpl(const at::Tensor& tensor) { return static_cast<NPUStorageImpl*>(tensor.storage().unsafeGetStorageImpl()); }

NPUStorageDesc& NPUBridge::GetNpuStorageImplDesc(const at::Tensor& tensor) {
    return static_cast<NPUStorageImpl*>(tensor.storage().unsafeGetStorageImpl())->npu_desc_;
}

NPUTensorImpl* NPUBridge::GetNpuTensorImpl(const at::Tensor& tensor) { return static_cast<NPUTensorImpl*>(tensor.unsafeGetTensorImpl()); }

}  // namespace torch_npu

namespace impl {

namespace aten {

// We can use reinterpret_cast directly in the dipu,
// but we cannot use this method directly in the consistency test,
// although the performance will be worse.
#define DIOPI_ADAPTER_BUILD_TENSOR_NOR_USE_CAST 1

#if DIOPI_ADAPTER_BUILD_TENSOR_NOR_USE_CAST

class FakeAllocator : public c10::Allocator {
    void* ptr_ = nullptr;
    size_t size_ = 0;
    c10::Device device_;

public:
    FakeAllocator(void* ptr, size_t size, c10::Device device) : ptr_(ptr), size_(size), device_(device) {}

    FakeAllocator() : device_(c10::DeviceType::CPU) {}

    void set(void* ptr, size_t size, c10::Device device) {
        ptr_ = ptr;
        size_ = size, device_ = device;
    }

    c10::DataPtr allocate(size_t n) const {
        if (n == 0) {
            return c10::InefficientStdFunctionContext::makeDataPtr(nullptr, c10::detail::deleteNothing, device_);
        } else {
            return c10::InefficientStdFunctionContext::makeDataPtr(ptr_, c10::detail::deleteNothing, device_);
        }
    }

    c10::DeleterFnPtr raw_deleter() const { return c10::detail::deleteNothing; }
};

at::Tensor fromPreAllocated(void* data, at::IntArrayRef sizes, at::IntArrayRef strides, const at::TensorOptions& options) {
    auto device = options.device();
    TORCH_CHECK(options.device().has_index());

    size_t nbytes = at::detail::computeStorageNbytes(sizes, strides, options.dtype().itemsize());

    c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::make_intrusive<torch_npu::NPUStorageImpl>(
        at::StorageImpl::use_byte_size_t(), nbytes, c10::InefficientStdFunctionContext::makeDataPtr(data, c10::detail::deleteNothing, device), nullptr, false);
    auto dtype = options.dtype();
    c10::DispatchKeySet ks{c10::DispatchKey::XLA};
    auto tensor = at::detail::make_tensor<at::TensorImpl>(std::move(storage_impl), ks, dtype);
    if (strides.size() > 0) {
        tensor.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    } else {
        tensor.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
    }

    at_npu::native::StorageDescHelper::SetDesc(tensor, sizes, tensor.strides());
    return tensor;
}

const at::Tensor buildATen(diopiConstTensorHandle_t tensor) {
    if (tensor == nullptr) return at::Tensor();

    diopiDtype_t dtype;
    diopiGetTensorDtype(tensor, &dtype);
    caffe2::TypeMeta atType = getATenType(dtype);
    diopiDevice_t device;
    diopiGetTensorDevice(tensor, &device);
    c10::DeviceType atDevice = getATenDevice(device);
    int devId_ = 0;
    ::aclrtGetDevice(&devId_);
    void* data = nullptr;
    diopiGetTensorData(const_cast<diopiTensorHandle_t>(tensor), &data);

    diopiSize_t shape;
    diopiGetTensorShape(tensor, &shape);
    at::IntArrayRef atDims(shape.data, shape.len);

    diopiSize_t stride;
    diopiGetTensorStride(tensor, &stride);
    at::IntArrayRef atStrides(stride.data, stride.len);

    auto options = at::TensorOptions(c10::Device(atDevice, devId_)).dtype(atType);
    return fromPreAllocated(data, atDims, atStrides, options);
}

at::Tensor buildATen(diopiTensorHandle_t tensor) { return buildATen(static_cast<diopiConstTensorHandle_t>(tensor)); }

#else

inline at::Tensor buildATen(diopiTensorHandle_t tensor) {
    if (tensor == nullptr) return at::Tensor();
    return *reinterpret_cast<at::Tensor*>(tensor);
}

inline const at::Tensor buildATen(diopiConstTensorHandle_t tensor) {
    if (tensor == nullptr) return at::Tensor();
    return *reinterpret_cast<const at::Tensor*>(tensor);
}

#endif

at::Generator buildATen(diopiGeneratorHandle_t generator) {
    auto gen = at::make_generator<at_npu::NPUGeneratorImpl>(current_device());
    auto impl = static_cast<at_npu::NPUGeneratorImpl*>(gen.unsafeGetGeneratorImpl());
    impl->generator_ = generator;
    return gen;
}

at::Tensor viewStorage(const at::Tensor input, const c10::IntArrayRef sizes, const c10::IntArrayRef strides, const int64_t storageOffset) {
    TORCH_CHECK(c10::multiply_integers(sizes) <= input.numel());
    TORCH_CHECK(!input.is_cpu());
    std::vector<int64_t> stridesVec(sizes.size(), 1);
    if (strides.size() > 0) {
        std::copy(strides.begin(), strides.end(), stridesVec.begin());
    } else {
        int st = 1;
        for (int64_t i = sizes.size(); i > 0; --i) {
            stridesVec[i - 1] = st;
            if (sizes[i - 1] == 0) continue;
            if (sizes[i - 1] == -1) st = -1;
            if (st != -1) st *= sizes[i - 1];
        }
    }
    return fromPreAllocated(input.data_ptr() + storageOffset * input.itemsize(), sizes, stridesVec, input.options());
}

c10::List<c10::optional<at::Tensor>> castIntIndicesToLongIndices(const c10::List<c10::optional<at::Tensor>>& indices) {
    c10::List<c10::optional<at::Tensor>> result;
    for (c10::optional<at::Tensor> indexOpt : indices) {
        if (!indexOpt.has_value()) {
            result.emplace_back();
        } else {
            at::Tensor index = std::move(*indexOpt);
            result.emplace_back(index.scalar_type() == at::kInt ? index.toType(at::kLong) : index);
        }
    }
    return result;
}

void setCurCtx(diopiContextHandle_t ctx) {
    context = ctx;
    at_npu::native::markedOutputs.clear();
}

void unsetCurCtx() { context = nullptr; }

}  // namespace aten

}  // namespace impl

namespace {

at::Tensor& wrapper_Tensor_fill_(at::Tensor& self, const at::Tensor& value) { return acl_op::fill_(self, value); }

at::Tensor& wrapper__copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    return at_npu::native::NPUNativeFunctions::copy_(self, src, non_blocking);
}

at::Tensor wrapper__view(const at::Tensor& self, at::IntArrayRef size) { return impl::aten::viewStorage(self, size); }

at::Tensor wrapper__as_strided(const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    return at_npu::native::NPUNativeFunctions::as_strided(self, size, stride, storage_offset.value_or(0));
}

const at::Tensor& wrapper__resize_(const at::Tensor& self, at::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format) {
    DEBUG_ARGS(self);
    DEBUG_ARGS(size);
    auto* selfImpl = self.unsafeGetTensorImpl();
    const auto itemsize = self.dtype().itemsize();
    const auto storage_offset = self.storage_offset();

    int64_t new_storage_size = at::detail::computeStorageNbytesContiguous(size, itemsize, storage_offset);

    if (self.numel() >= c10::multiply_integers(size)) {
        auto out = impl::aten::viewStorage(self, size);
    } else {
        auto out = at_npu::native::empty_npu(size, self.options());
        auto storage = selfImpl->unsafe_storage();
        auto storageImpl = storage.unsafeGetStorageImpl();
        storageImpl->set_data_ptr_noswap(std::move(c10::InefficientStdFunctionContext::makeDataPtr(out.data_ptr(), c10::detail::deleteNothing, self.device())));
    }
    selfImpl->set_sizes_contiguous(size);
    return self;
}

void ascend_diopi_fallback(const c10::OperatorHandle& op, at::DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    const auto name = c10::toString(op.operator_name());
    std::cout << __FUNCTION__ << ": op " << name << " fallbacked, must be processed!!!" << std::endl;
    at::native::cpu_fallback(op, stack);
}

at::Tensor wrapper__contiguous(const at::Tensor& self, at::MemoryFormat memory_format) {
    return at_npu::native::NPUNativeFunctions::contiguous(self, memory_format);
}

at::Tensor wrapper__empty_strided(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                                  c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at_npu::native::NPUNativeFunctions::empty_strided(size, stride, dtype, layout, device, pin_memory);
}

at::Tensor wrapper_memory_format_empty(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                                       c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    return at_npu::native::NPUNativeFunctions::empty(size, dtype, layout, device, pin_memory, memory_format);
}

at::Tensor wrapper__clone(const at::Tensor& self, c10::optional<at::MemoryFormat> memory_format) {
    return at_npu::native::NPUNativeFunctions::clone(self, memory_format);
}

at::Tensor& wrapper_source_Storage_set_(at::Tensor& self, at::Storage src) {
    auto* selfImpl = self.unsafeGetTensorImpl();
    auto storage = selfImpl->unsafe_storage();
    auto storageImpl = storage.unsafeGetStorageImpl();
    storageImpl->set_data_ptr(std::move(src.data_ptr()));
    return self;
}

at::Tensor& wrapper_source_Storage_storage_offset_set_(at::Tensor& self, at::Storage source, int64_t storage_offset, at::IntArrayRef size,
                                                       at::IntArrayRef stride) {
    auto* selfImpl = self.unsafeGetTensorImpl();
    auto storage = selfImpl->unsafe_storage();
    auto storageImpl = storage.unsafeGetStorageImpl();
    storageImpl->set_data_ptr(std::move(source.data_ptr()));
    selfImpl->set_storage_offset(storage_offset);
    if (stride.size() > 0) {
        selfImpl->set_sizes_and_strides(size, stride);
    } else {
        selfImpl->set_sizes_contiguous(size);
    }

    return self;
}

at::Tensor wrapper__cat(const at::ITensorListRef& tensors, int64_t dim) { return acl_op::cat(tensors, dim); }

at::Tensor& wrapper__index_put_(at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices, const at::Tensor& values, bool accumulate) {
    auto indicesCast = impl::aten::castIntIndicesToLongIndices(indices);
    return acl_op::_index_put_impl_(self, indicesCast, values, accumulate, false);
}

at::Tensor& wrapper___index_put_impl_(at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices, const at::Tensor& values, bool accumulate,
                                      bool unsafe) {
    auto indicesCast = impl::aten::castIntIndicesToLongIndices(indices);
    return acl_op::_index_put_impl_(self, indicesCast, values, accumulate, unsafe);
}

at::Tensor wrapper_Tensor_index(const at::Tensor& self, const c10::List<c10::optional<at::Tensor>>& indices) {
    auto indicesCast = impl::aten::castIntIndicesToLongIndices(indices);
    return acl_op::index(self, indicesCast);
}

at::Tensor wrapper__bmm(const at::Tensor& self, const at::Tensor& mat2) { return acl_op::bmm(self, mat2); }

at::Tensor wrapper_Tensor_div(const at::Tensor& self, const at::Tensor& other) { return acl_op::div(self, other); }

at::Tensor wrapper_Tensor_mul(const at::Tensor& self, const at::Tensor& other) { return acl_op::mul(self, other); }

at::Tensor wrapper_Scalar_mul(const at::Tensor& self, const at::Scalar& other) { return acl_op::mul(self, other); }

at::Tensor wrapper_Tensor_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) { return acl_op::add(self, other, alpha); }

at::Tensor wrapper_Tensor_sub(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) { return acl_op::sub(self, other, alpha); }

at::Tensor wrapper__index_select(const at::Tensor& self, int64_t dim, const at::Tensor& index) { return acl_op::index_select(self, dim, index); }

at::Tensor wrapper___softmax(const at::Tensor& self, int64_t dim, bool half_to_float) { return acl_op::_softmax(self, dim, half_to_float); }

at::Tensor wrapper_Scalar_eq(const at::Tensor& self, const at::Scalar& other) { return acl_op::eq(self, other); }

at::Tensor& wrapper_Scalar_masked_fill_(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) { return acl_op::masked_fill_(self, mask, value); }

at::Tensor wrapper__repeat(const at::Tensor& self, at::IntArrayRef repeats) { return acl_op::repeat(self, repeats); }

at::Tensor wrapper__transpose(const at::Tensor& self, int64_t dim0, int64_t dim1) {
    int64_t inputSize = self.dim();
    if (dim0 < 0) dim0 = dim0 + inputSize;
    if (dim1 < 0) dim1 = dim1 + inputSize;
    std::vector<int64_t> perms(inputSize);
    std::iota(perms.begin(), perms.end(), 0);
    perms[dim0] = dim1;
    perms[dim1] = dim0;
    return acl_op::npu_transpose(self, perms);
}

at::Scalar wrapper___local_scalar_dense(const at::Tensor& self) { return at_npu::native::NPUNativeFunctions::_local_scalar_dense(self); }

}  // namespace

namespace at {

TORCH_LIBRARY_IMPL(aten, XLA, m) {
    m.impl("fill_.Tensor", TORCH_FN(wrapper_Tensor_fill_));
    m.impl("copy_", TORCH_FN(wrapper__copy_));
    m.impl("reshape", TORCH_FN(wrapper__view));
    m.impl("view", TORCH_FN(wrapper__view));
    m.impl("as_strided", TORCH_FN(wrapper__as_strided));
    m.impl("resize_", TORCH_FN(wrapper__resize_));
    m.impl("contiguous", TORCH_FN(wrapper__contiguous));
    m.impl("empty_strided", TORCH_FN(wrapper__empty_strided));
    m.impl("empty.memory_format", TORCH_FN(wrapper_memory_format_empty));
    m.impl("clone", TORCH_FN(wrapper__clone));
    m.impl("set_.source_Storage", TORCH_FN(wrapper_source_Storage_set_));
    m.impl("set_.source_Storage_storage_offset", TORCH_FN(wrapper_source_Storage_storage_offset_set_));
    m.impl("cat", TORCH_FN(wrapper__cat));
    m.impl("index_put_", TORCH_FN(wrapper__index_put_));
    m.impl("_index_put_impl_", TORCH_FN(wrapper___index_put_impl_));
    m.impl("index.Tensor", TORCH_FN(wrapper_Tensor_index));
    m.impl("bmm", TORCH_FN(wrapper__bmm));
    m.impl("div.Tensor", TORCH_FN(wrapper_Tensor_div));
    m.impl("mul.Tensor", TORCH_FN(wrapper_Tensor_mul));
    m.impl("mul.Scalar", TORCH_FN(wrapper_Scalar_mul));
    m.impl("add.Tensor", TORCH_FN(wrapper_Tensor_add));
    m.impl("sub.Tensor", TORCH_FN(wrapper_Tensor_sub));
    m.impl("index_select", TORCH_FN(wrapper__index_select));
    m.impl("_softmax", TORCH_FN(wrapper___softmax));
    // m.impl("eq.Scalar", TORCH_FN(wrapper_Scalar_eq));
    m.impl("masked_fill_.Scalar", TORCH_FN(wrapper_Scalar_masked_fill_));
    m.impl("repeat", TORCH_FN(wrapper__repeat));
    m.impl("transpose.int", TORCH_FN(wrapper__transpose));
    m.impl("_local_scalar_dense", TORCH_FN(wrapper___local_scalar_dense));
};

TORCH_LIBRARY_IMPL(_, XLA, m) { m.fallback(torch::CppFunction::makeFromBoxedFunction<&ascend_diopi_fallback>()); }

}  // namespace at
