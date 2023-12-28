// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "diopi_impl/helper.hpp"
#include "op_plugin/AclOpsInterface.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

namespace {
// NOTE: helper function of copy, the input parameter is not checked, The caller
// needs to ensure that the parameters are correct.

// the caller should ensure the tensor.is_npu == true
bool is_same_format(const at::Tensor& a, const at::Tensor& b) {
    bool isSameFormat = FormatHelper::GetFormat(a) == FormatHelper::GetFormat(b);
    if (!isSameFormat) {
        bool isBaseFormat = FormatHelper::IsBaseFormatType(a) && FormatHelper::IsBaseFormatType(b);
        return isBaseFormat;
    }
    return true;
}

bool try_to_optimize_copy_with_any_format(at::Tensor& self, const at::Tensor& src) {
    // Some Ops support inputs with 5HD/NZ format, Transdata is redundant
    // Record:
    // Op:Reshape; SliceD || Supportformat: 5HD/NZ
    return TransContiguous::ContiguousOptimizeWithAnyFormat(self, src);
}

// the dst and src are same format now
// the dst and src are base format now
// the dst and src maybe non-contiguous
void copy_d2d_last_method(at::Tensor& self, const at::Tensor& src, bool same_type, bool non_blocking) {
    // general copy method but Low performance
    RECORD_FUNCTION("contiguous_d_ViewCopy", std::vector<c10::IValue>({src}));
    custom_ops::npu_view_copy(self, src, non_blocking);
}

// the dst and src are same format now
void copy_d2d_dtype_format(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    // Note: Src & Self have the same format.
    if (try_to_optimize_copy_with_any_format(self, src)) {
        return;
    }

    if (!FormatHelper::IsBaseFormatType(self)) {  // 必须要非NCHW的才行？
        if (can_use_memcpy(self, src)) {
            RECORD_FUNCTION("d2dCopyAsync with format", std::vector<c10::IValue>({src}));
            return copy_d2d_by_memcpy(self, src);
        }
    }

    if (!FormatHelper::IsBaseFormatType(self)) {
        at::Tensor src_4D = FormatCastHelper::ApplyBaseFormatTensorBy(src);
        at::Tensor dst_4D = FormatCastHelper::ApplyBaseFormatTensorBy(self);
        copy_d2d_dtype_baseformat(dst_4D, src_4D, non_blocking);
        NPUNativeFunctions::npu_format_cast_(self, dst_4D);
        return;
    }
    copy_d2d_dtype_baseformat(self, src, non_blocking);
}

void copy_d2d(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    if (self.dtype() != src.dtype()) {
        custom_ops::npu_dtype_cast_(self, src.contiguous(at::MemoryFormat::Contiguous));  // npu_dtype_cast_ will call copy function.
        return;
    }
    copy_d2d_dtype(self, src, non_blocking);
}

// the format of dst and src is base format now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_between_host_and_device(at::Tensor& dst, const at::Tensor& src, aclrtMemcpyKind kind, bool non_blocking) {
    int64_t nbytes = dst.numel() * dst.element_size();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();

    if (non_blocking) {
        auto ret = CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(dst, nbytes, src, nbytes, kind);
        NPU_CHECK_ERROR(ret);
        NPU_LOGD("non_blocking copy without StreamSynchronize.");
        void* ptr = at_npu::key::isDeviceTensor(dst) ? src.data_ptr() : dst.data_ptr();
        NPU_CHECK_ERROR(THNPUCachingHostAllocator_recordEvent(ptr, stream));
    } else {
        aclError error = c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream);
        auto ret = CalcuOpUtil::AclrtMemcpyWithModeSwitch(std::make_pair(dst.storage().unsafeGetStorageImpl(), dst.storage_offset() * dst.itemsize()),
                                                          nbytes,
                                                          std::make_pair(src.storage().unsafeGetStorageImpl(), src.storage_offset() * src.itemsize()),
                                                          nbytes,
                                                          kind);
        NPU_CHECK_ERROR(ret);
        if (error != ACL_ERROR_NONE) {
            C10_NPU_SHOW_ERR_MSG();
            AT_ERROR("ACL stream synchronize failed, error code:", error);
        }
    }
}

// the format of dst and src is base format now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_h2d_baseformat_dtype_contigous(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
    c10_npu::OptionalNPUGuard device_guard;
    device_guard.set_device(dst.device());
    aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE;
    copy_between_host_and_device(dst, src, kind, non_blocking);
}

// the format of dst and src is baseformat now
// the dtype of dst and src is same
// and src and dst are contiguous
void copy_d2h_baseformat_dtype_contigous(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
    c10_npu::OptionalNPUGuard device_guard;
    device_guard.set_device(src.device());
    aclrtMemcpyKind kind = aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST;
    copy_between_host_and_device(dst, src, kind, non_blocking);
}

// the format of dst and src is baseformat now
void copy_h2d_baseformat(at::Tensor& dst, const at::Tensor& src, bool non_blocking, bool dst_must_be_contiguous = false) {
    bool same_type = (src.dtype() == dst.dtype());
    bool dst_is_contiguous = dst_must_be_contiguous ? true : dst.is_contiguous();
    if (same_type && dst_is_contiguous && src.is_contiguous()) {
        copy_h2d_baseformat_dtype_contigous(dst, src, non_blocking);
        return;
    }

    at::Tensor dst_contig = dst_is_contiguous ? dst : at::empty_like(dst);
    at::Tensor src_contig;
    if (!same_type) {
        src_contig = src.to(dst.dtype()).expand_as(dst).contiguous();
    } else {
        src_contig = src.expand_as(dst).contiguous();
    }
    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    copy_h2d_baseformat_dtype_contigous(dst_contig, src_contig, non_blocking);
    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
        TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
        copy_d2d_dtype(dst, dst_contig, non_blocking);
    }
}

// the format of dst and src is baseformat now
void copy_d2h_baseformat(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
    bool same_type = (src.dtype() == dst.dtype());
    bool dst_is_contiguous = dst.is_contiguous();
    if (same_type && dst_is_contiguous && src.is_contiguous()) {
        copy_d2h_baseformat_dtype_contigous(dst, src, non_blocking);
        return;
    }
    at::Tensor dst_contig = (dst_is_contiguous && same_type) ? dst : at::empty_like(dst, src.dtype());
    at::Tensor src_contig = src.expand_as(dst).contiguous();
    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    copy_d2h_baseformat_dtype_contigous(dst_contig, src_contig, non_blocking);
    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
        TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
        dst.copy_(dst_contig, non_blocking);  // h2h, use cpu copy
    }
}

void copy_h2d(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    if (!FormatHelper::IsBaseFormatType(self)) {
        at::Tensor dst = OpPreparation::ApplyTensorWithSizes(self.sizes(), self.options());
        copy_h2d_baseformat(dst, src, non_blocking, true);
        NPUNativeFunctions::npu_format_cast_(self, dst);
        return;
    }
    copy_h2d_baseformat(self, src, non_blocking);
}

void copy_d2h(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    if (!FormatHelper::IsBaseFormatType(src)) {
        at::Tensor src_4D = FormatCastHelper::ApplyBaseFormatTensorBy(src);
        copy_d2h_baseformat(self, src_4D, non_blocking);
        return;
    }
    copy_d2h_baseformat(self, src, non_blocking);
}
}  // namespace

// the caller should guarantee that the format and dtype are same
bool can_use_memcpy(at::Tensor& dst, const at::Tensor& src) {
    if (StorageDescHelper::IsSameDesc(dst, src)) {
        // Make sure that the metadata are same.
        if (!dst.sizes().equals(src.sizes())) {
            return false;
        }
        if (!dst.strides().equals(src.strides())) {
            return false;
        }
        // Make sure that copy the whole memory.
        // we just need to compare one of them, because of the NpuStorageDesc
        // and metadata(sizes and stride) of src and dst are same.
        if (StorageDescHelper::GetValidMemorySize(src) != src.numel()) {
            return false;
        }
        if ((dst.storage_offset() != 0) || (src.storage_offset() != 0)) {
            return false;
        }
        return true;
    }
    return false;
}

// the dst and src are same dtype now
void copy_d2d_dtype(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    if (!is_same_format(self, src)) {
        at::Tensor src_4D = FormatCastHelper::ApplyBaseFormatTensorBy(src);
        // ApplyBaseFormatTensorBy is redundant for self tensor with base format.
        if (FormatHelper::IsBaseFormatType(self)) {
            copy_d2d_dtype_baseformat(self, src_4D, non_blocking);
            return;
        }
        at::Tensor dst_4D = FormatCastHelper::ApplyBaseFormatTensorBy(self);
        copy_d2d_dtype_baseformat(dst_4D, src_4D, non_blocking);
        NPUNativeFunctions::npu_format_cast_(self, dst_4D);
        return;
    }
    copy_d2d_dtype_format(self, src, non_blocking);
}

// the dst and src are same format now
// the dst and src are base format now
// the dst and src maybe non-contiguous
void copy_d2d_dtype_baseformat(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    if (!self.is_contiguous()) {
        // Contiguous/discontiguous source tensor copy to discontiguous self tensor
        return copy_d2d_last_method(self, src, true, non_blocking);
    }

    if (!src.is_contiguous()) {
        // Discontiguous source tensor copy to contiguous self tensor
        if (TransContiguous::ContiguousOptimizeWithBaseFormat(self, src)) {
            // Optimized trans-contiguous method
            return;
        } else {
            // General trans-contiguous method
            RECORD_FUNCTION("contiguous_d_AsStrided", std::vector<c10::IValue>({src}));
            custom_ops::npu_stride_copy_out(src, src.sizes(), src.strides(), src.storage_offset(), self);
            return;
        }
    } else {
        // Contiguous source tensor copy to contiguous self tensor
        int64_t numel = self.numel();
        if (numel == src.numel()) {
            RECORD_FUNCTION("d2dCopyAsync", std::vector<c10::IValue>({src}));
            NPU_LOGD("copy contiguous tensor inside device");
            return copy_d2d_by_memcpy(self, src, numel);
        }
    }
    // such as discontiguous tensor copy to unmatched tensor
    copy_d2d_last_method(self, src, true, non_blocking);
}

bool try_to_optimize_copy_with_any_format(at::Tensor& self, const at::Tensor& src) {
    // Some Ops support inputs with 5HD/NZ format, Transdata is redundant
    // Record:
    // Op:Reshape; SliceD || Supportformat: 5HD/NZ
    return TransContiguous::ContiguousOptimizeWithAnyFormat(self, src);
}

at::Tensor& NPUNativeFunctions::copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    if (!self.defined() || self.numel() == 0) {
        return self;
    }
    // save tensor dim name
    c10::optional<at::DimnameList> names = src.opt_names();
    if (names.has_value()) {
        internal_set_names_inplace(self, names);
    }

    if (at_npu::key::isDeviceTensor(self)) {
        if (at_npu::key::isDeviceTensor(src)) {
            copy_d2d(self, src, non_blocking);
        } else {
            copy_h2d(self, src, non_blocking);
        }
    } else {
        if (at_npu::key::isDeviceTensor(src)) {
            copy_d2h(self, src, non_blocking);
        }
    }
    if (!non_blocking) {
        c10_npu::getCurrentNPUStream().synchronize();
    }
    c10_npu::getCurrentNPUStream().synchronize();
    return self;
}

class BroadcastContiguousOpt : public ContiguousOpt {
public:
    bool Optimizer(at::Tensor& self, const at::Tensor& src, const ContiguousTensorDesc& src_desc) override {
        if (self.dim() != src.dim()) {
            return false;
        }

        if (can_use_broadcast(src_desc)) {
            RECORD_FUNCTION("contiguous_d_BroadcastTo", std::vector<c10::IValue>({src}));
            bool can_contiguous = broadcast_to_contiguous(self, src, src_desc);
            return can_contiguous;
        }
        return false;
    }

private:
    bool can_use_broadcast(const ContiguousTensorDesc& src_desc) {
        // Reshape is used to process dimension addition cases for expand/expand_as.
        // Here, dimension expansion cases of expand/expand_as are processed.
        const auto& base_sizes = src_desc.base_sizes_;
        const auto& base_strides = src_desc.base_strides_;
        const auto& view_sizes = src_desc.sizes_;
        const auto& view_strides = src_desc.strides_;

        // The new ones will be appended at the front.
        // Any dimension of size 1 can be expanded to an arbitrary value.
        auto base_dim = static_cast<int64_t>(base_sizes.size());
        auto view_dim = static_cast<int64_t>(view_sizes.size());
        auto expand_dims = view_dim - base_dim;
        if (expand_dims < 0) {
            return false;
        }

        bool has_zero_in_stride = false;
        for (int64_t i = 0; i < base_dim; i++) {
            if (view_strides[i + expand_dims] == 0) {
                has_zero_in_stride = true;
                if (base_sizes[i] != 1 || view_sizes[i + expand_dims] == 1) {
                    return false;
                }
            } else {
                if (view_sizes[i + expand_dims] != base_sizes[i] || view_strides[i + expand_dims] != base_strides[i]) {
                    return false;
                }
            }
        }

        for (auto i = 0; i < expand_dims; i++) {
            if (view_sizes[i] != 1 && view_strides[i] != 0) {
                return false;
            }
            has_zero_in_stride = true;
        }
        return has_zero_in_stride;
    }

    bool broadcast_to_contiguous(at::Tensor& self, const at::Tensor& src, const ContiguousTensorDesc& src_desc) {
        std::vector<int64_t> src_size(src.dim());
        for (const auto i : c10::irange(src_desc.sizes_.size())) {
            if (src_desc.strides_[i] == 0) {
                src_size[i] = 1;
            } else {
                src_size[i] = src_desc.sizes_[i];
            }
        }

        // create contiguous tensor for npu BroadcastToD
        at::Tensor temp_src = at::empty({0}, src.options());
        temp_src.set_(src);
        temp_src.unsafeGetTensorImpl()->set_sizes_and_strides(src_size, src.strides());

        if (temp_src.is_contiguous()) {
            // NPU op BroadcastTo not supports dtype of bool yet.
            if (self.dtype() == at::kBool) {
                auto temp_dst = custom_ops::npu_broadcast(temp_src, self.sizes());
                // The current logic is only used in single op mode.
                c10_npu::queue::LaunchAsyncCopyTask(self.data_ptr(), self.nbytes(), temp_dst.data_ptr(), self.nbytes(), ACL_MEMCPY_DEVICE_TO_DEVICE);
                return true;
            }
            custom_ops::npu_broadcast_out(temp_src, self.sizes(), self);
            return true;
        }
        return false;
    }
};  // class BroadcastContiguousOpt

REGISTER_COPY_OPT(broadcast, BroadcastContiguousOpt)

constexpr int MaxCombinedCasesNum = 2;
constexpr int ViewAndBaseInfoStackNum = 2;
// Stacks used for storing inferred infos about shape, stride, offset
// "shape_stride_stacks": [[[shape1],[stride1];[[shape2],[stride2]];...]
// "offset_stack": [storage_offset1, storage_offset2,...]
using ShapeStrideStack = c10::SmallVector<c10::SmallVector<FormatShape, ViewAndBaseInfoStackNum>, MaxCombinedCasesNum>;
using OffsetStack = c10::SmallVector<int64_t, MaxCombinedCasesNum>;

class CombinedContiguousOpt : public ContiguousOpt {
public:
    // Combined tensor == discontiguous tensor caused by combined view operators.
    bool Optimizer(at::Tensor& self, const at::Tensor& src, const ContiguousTensorDesc& src_desc) override {
        // Maximum combined operators suggested: combined_cases_num = 2
        // NOTE: n-cmobined(n>2) can also be supported
        int combined_cases_num = MaxCombinedCasesNum;

        ShapeStrideStack shape_stride_stacks;
        OffsetStack offset_stack;

        if (can_use_combined(shape_stride_stacks, offset_stack, src_desc, combined_cases_num)) {
            RECORD_FUNCTION("contiguous_h_combined", std::vector<c10::IValue>({src}));
            // Record src infos for recovering after trans-contiguous
            auto src_storage_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc();

            at::Tensor base_tensor = at::empty(src_storage_desc.base_sizes_, src.options());
            base_tensor.set_(src.storage());

            // Reconstruct combined discontiguous tensor ==trans==> contiguous tensor
            bool contiguousOrNot = combined_to_contiguous(self, base_tensor, shape_stride_stacks, offset_stack);

            // Recover modified tensor infos of src after trans-contiguous
            StorageDescHelper::CopyDesc(base_tensor, src_storage_desc);
            return contiguousOrNot;
        }
        return false;
    }

private:
    bool cases_avoid(const ContiguousTensorDesc& tensor_desc) {
        for (const auto i : c10::irange(tensor_desc.sizes_.size())) {
            // expand+x,x+expand
            if (tensor_desc.strides_[i] == 0) {
                return true;
            }
        }
        return false;
    }

    // Unmatched tensor ==refresh(no copy)==> macthed tensor
    bool reshape_without_copy_match(at::Tensor& tensor) {
        if (!tensor.is_contiguous()) {
            return false;
        }
        auto npu_desc = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->get_npu_desc();
        if ((c10::multiply_integers(tensor.sizes()) != c10::multiply_integers(npu_desc.base_sizes_)) || (tensor.storage_offset() != npu_desc.base_offset_)) {
            return false;
        }
        RECORD_FUNCTION("contiguous_h_match", std::vector<c10::IValue>({tensor}));
        StorageDescHelper::SetDesc(
            tensor, CalcuOpUtil::ConvertIntArrayRefToSmallVector(tensor.sizes()), CalcuOpUtil::ConvertIntArrayRefToSmallVector(tensor.strides()));
        return true;
    }

    // Whether tensor can be optimized(no optimization).
    bool can_be_optimize_from_default_cases(ContiguousTensorDesc& tensor_desc) {
        OptimizationCases opt_cases{"reshape", "slice", "select"};
        tensor_desc.reset_optimization_cases(opt_cases);
        return TransContiguous::CanOptimize(tensor_desc);
    }

    // Conduct trans-contiguous for given optimization cases.
    bool copy_optimize_contiguous_by_given_cases(at::Tensor& self, const at::Tensor& tensor, OptimizationCases& optimizations) {
        // Set "OpenCombined = false" to avoid recursion.
        return TransContiguous::ContiguousOptimizeWithBaseFormat(self, tensor, optimizations, false);
    }

    // Weak constrains for transpose cases
    bool maybe_permute(const ContiguousTensorDesc& tensor_desc) {
        // tensors with nonmonotonic strides will be taken into consideration
        // (Ascend): 对于特殊stride的情况例如：[*,*,1,1]这种，需要进一步分析影响
        for (const auto i : c10::irange(tensor_desc.sizes_.size() - 1)) {
            if (tensor_desc.strides_[i] < tensor_desc.strides_[i + 1]) {
                return true;
            }
        }
        return false;
    }

    // Weak constrains for select cases
    bool maybe_select(const ContiguousTensorDesc& tensor_desc) {
        for (auto i = tensor_desc.sizes_.size() - 1; i > 0; i--) {
            if (tensor_desc.strides_[i - 1] % (tensor_desc.sizes_[i] * tensor_desc.strides_[i]) != 0) {
                return false;
            }
            if (tensor_desc.strides_[i - 1] / (tensor_desc.sizes_[i] * tensor_desc.strides_[i]) != 1) {
                if (tensor_desc.offset_ % (tensor_desc.sizes_[i] * tensor_desc.strides_[i]) != 0) {
                    return false;
                }
                // Avoid combined-cases such as squeeze+indexing at the first axis.
                if (tensor_desc.strides_[0] != tensor_desc.base_strides_[0]) {
                    return false;
                }
            }
        }
        return true;
    }

    // Weak constrains for slice cases
    bool maybe_slice(const ContiguousTensorDesc& tensor_desc) {
        // tensors with reduced numel will be taken into consideration.
        if (c10::multiply_integers(tensor_desc.sizes_) < c10::multiply_integers(tensor_desc.base_sizes_)) {
            for (const auto i : c10::irange(tensor_desc.sizes_.size() - 2)) {
                if (tensor_desc.strides_[i] % tensor_desc.strides_[i + 1] != 0) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    /*
  Kernel function of "Inference",
  Key inferred infos: infer_size,infer_stride and infer_offset,
  Inference order: permute, select, slice.
  */
    bool can_infer_view_tensor(ContiguousTensorDesc& tensor_desc, FormatShape& infer_size, FormatShape& infer_stride, int64_t& infer_offset) {
        const auto& view_sizes = tensor_desc.sizes_;
        const auto& view_strides = tensor_desc.strides_;

        if (maybe_permute(tensor_desc)) {
            FormatShape& permute_size_sorted = infer_size;
            FormatShape& permute_stride_sorted = infer_stride;
            permute_size_sorted = view_sizes;
            permute_stride_sorted = view_strides;

            // Sort stride
            std::sort(permute_stride_sorted.rbegin(), permute_stride_sorted.rend());

            // Map stride to shape
            std::map<int64_t, int64_t> map_shape_stride;
            std::map<int64_t, int64_t> label_map_shape_stride;
            for (const auto i : c10::irange(view_sizes.size())) {
                map_shape_stride[view_strides[i]] = view_sizes[i];
            }
            // 除去第0维，其他维shape为1时，不记录对应的stride值，该stride的值会和其他维的stride有重复
            for (const auto i : c10::irange(view_sizes.size())) {
                if (i == 0) {
                    map_shape_stride[view_strides[0]] = view_sizes[0];
                } else if (i != 0 && view_sizes[i] != 1) {
                    map_shape_stride[view_strides[i]] = view_sizes[i];
                }
            }
            // stride中有相等的情况，后面相等的stride对应的shape为1
            for (const auto i : c10::irange(view_sizes.size())) {
                if (label_map_shape_stride[permute_stride_sorted[i]] != true) {
                    permute_size_sorted[i] = map_shape_stride[permute_stride_sorted[i]];
                    label_map_shape_stride[permute_stride_sorted[i]] = true;
                } else {
                    permute_size_sorted[i] = 1;
                }
            }
            infer_offset = 0;
            // Refresh tensor's base info to construct transposed tensor
            tensor_desc.base_sizes_ = permute_size_sorted;
            tensor_desc.base_strides_ = permute_stride_sorted;
            // double-checking of may_permute is not required, because view strides
            // does not changed.
            return true;
        }

        if (maybe_select(tensor_desc)) {
            FormatShape& select_size = infer_size;
            FormatShape& select_stride = infer_stride;
            // Infer base shape according to view shape and stride
            select_stride = view_strides;
            select_size = view_sizes;
            // select_size and stride should be one more than view_size
            select_size.emplace_back((int64_t)1);
            select_stride.emplace_back((int64_t)1);

            int64_t i = static_cast<int64_t>(view_sizes.size()) - 1;
            if (view_strides[i] == 1) {
                select_size[i + 1] = view_sizes[i];
                select_stride[i + 1] = 1;

                for (i = i - 1; i >= 0; i--) {
                    if (view_strides[i] != view_strides[i + 1] * view_sizes[i + 1]) {
                        select_size[i + 1] = view_strides[i] / (view_sizes[i + 1] * view_strides[i + 1]);
                        select_stride[i + 1] = view_sizes[i + 1] * view_strides[i + 1];
                        infer_offset = tensor_desc.offset_ % view_strides[i];
                        break;
                    }
                    select_size[i + 1] = view_sizes[i];
                    select_stride[i + 1] = view_strides[i];
                }
            } else {
                select_size[i + 1] = view_strides[i];
                select_stride[i + 1] = 1;
                infer_offset = tensor_desc.offset_ % view_strides[i];
            }
            for (i = i - 1; i >= 0; i--) {
                select_size[i + 1] = view_sizes[i + 1];
                select_stride[i + 1] = view_strides[i + 1];
            }

            select_size[0] = view_sizes[0];
            select_stride[0] = view_strides[0];

            // Refresh tensor's base info to construct selected tensor
            tensor_desc.base_sizes_ = select_size;
            tensor_desc.base_strides_ = select_stride;
            // Whether the construted tensor is selected?
            return maybe_select(tensor_desc);
        }

        if (maybe_slice(tensor_desc)) {
            FormatShape& slice_size = infer_size;
            FormatShape& slice_stride = infer_stride;

            slice_stride = view_strides;
            slice_size = view_sizes;
            // Infer base shape according to base stride
            for (auto i = slice_size.size() - 1; i > 0; i--) {
                // Strides is not divisible means this case cannot be inferred.
                if (view_strides[i] == 0 || view_strides[i - 1] % view_strides[i] != 0) {
                    return false;
                }
                slice_size[i] = (view_strides[i - 1] / view_strides[i]);
            }
            slice_size[0] = 1;
            slice_size[0] = (c10::multiply_integers(tensor_desc.base_sizes_) / c10::multiply_integers(slice_size));
            infer_offset = tensor_desc.offset_;
            // Refresh tensor's base info and storage info to construct sliced tensor
            tensor_desc.base_sizes_ = slice_size;
            tensor_desc.base_strides_ = slice_stride;
            // Whether the construted tensor is sliced?
            return maybe_slice(tensor_desc);
        }
        return false;
    }

    bool stack_infer_info(ShapeStrideStack& shape_stride_stacks, OffsetStack& offset_stacks, int64_t infer_offset, int64_t combined_cases_num,
                          ContiguousTensorDesc& tensor_desc) {
        // Only combined_cases_num-combined Ops cases are taken into consideration
        if (static_cast<int16_t>(shape_stride_stacks.size()) == combined_cases_num) {
            return false;
        }

        c10::SmallVector<FormatShape, 2> stack_shape_stride_part;
        stack_shape_stride_part.emplace_back(CalcuOpUtil::ConvertIntArrayRefToSmallVector(tensor_desc.sizes_));
        stack_shape_stride_part.emplace_back(CalcuOpUtil::ConvertIntArrayRefToSmallVector(tensor_desc.strides_));

        shape_stride_stacks.emplace_back(stack_shape_stride_part);
        offset_stacks.emplace_back(infer_offset);
        return true;
    }

    // Conduct inferring
    bool can_use_combined(ShapeStrideStack& shape_stride_stacks, OffsetStack& offset_stacks, const ContiguousTensorDesc& src_desc, int64_t combined_cases_num) {
        // combined tensor should be discontiguous
        if (src_desc.is_contiguous_ || cases_avoid(src_desc)) {
            return false;
        }

        // Key infos that should be inferred.
        FormatShape infer_size;
        FormatShape infer_stride;
        int64_t infer_offset = 0;

        // Reconstruct "the discontiguous combined tensor desc"
        // viewInfo = combined tensor(src)'s viewInfo
        // baseInfo = combined tensor(src)'s baseInfo
        // src's desc would be modified, so a local struct is created.
        ContiguousTensorDesc local_src_desc = src_desc;

        // Construct "the first inferred tensor" inside "can_infer_view_tensor()"
        // viewInfo = combined tensor(src)'s viewInfo
        // baseInfo = inferred info(infer_size, infer_stride, infer_offset)
        // If the first inferred tensor can be optimized, store its info.
        if (can_infer_view_tensor(local_src_desc, infer_size, infer_stride, infer_offset) &&
            stack_infer_info(shape_stride_stacks, offset_stacks, infer_offset, combined_cases_num, local_src_desc)) {
            // Construct "the second inferred tensor"
            // viewInfo = inferred info(infer_size, infer_stride, infer_offset)
            // baseInfo = combined tensor(src)'s baseInfo
            local_src_desc.sizes_ = infer_size;
            local_src_desc.strides_ = infer_stride;
            local_src_desc.offset_ -= infer_offset;
            local_src_desc.base_sizes_ = src_desc.base_sizes_;
            local_src_desc.base_strides_ = src_desc.base_strides_;
            local_src_desc.refresh_contiguous_using_size_and_stride();
            // The second inferred tensor can be optimized or not
            if (can_be_optimize_from_default_cases(local_src_desc) &&
                stack_infer_info(shape_stride_stacks, offset_stacks, local_src_desc.offset_, combined_cases_num, local_src_desc)) {
                return true;
            }
            // If the second pattern is not inferred successfully, retrun false
            return false;
        }
        // If the first pattern is not inferred successfully, retrun false
        return false;
    }

    // Reconstructing discontiguous tensor at trans-contiguous procedure.
    bool reconstruct_tensor(at::Tensor& src, ShapeStrideStack& shape_stride_stacks, OffsetStack& offset_stacks) {
        auto stack_shape_stride = shape_stride_stacks.pop_back_val();
        auto stack_offset = offset_stacks.pop_back_val();
        // Set view info to make discontiguous tensor.
        // stack_shape_stride[0]: stored shape infos in inferring procedure.
        // stack_shape_stride[1]: stored stride infos in inferring procedure.

        src.set_(src.storage(), stack_offset, stack_shape_stride[0], stack_shape_stride[1]);

        // If current tensor is sliced and the stack is still not empty:
        // stored infos in the stack should be modified.
        if (shape_stride_stacks.size() >= 1 && maybe_slice(TransContiguous::GetTensorDescInfo(src))) {
            auto stack_shape_stride_pre = shape_stride_stacks.pop_back_val();

            std::map<int64_t, int64_t> map_stride_shape;
            auto computed_stride = StorageDescHelper::ComputeStrideFromShape(stack_shape_stride[0]);
            // Adjust shape according to sorted stride
            for (const auto i : c10::irange(stack_shape_stride_pre[0].size())) {
                // if shape_i equals to shape_j, non-unique keys for "map_stride_shape" would be made;
                // Temporarily, making size[i] * stride[i] to obtain unique keys;
                // (Ascend): explore unique keys for any cases when "shape[i] == shape [j]"
                map_stride_shape[stack_shape_stride[0][i] * stack_shape_stride[1][i]] = computed_stride[i];
            }

            for (const auto i : c10::irange(stack_shape_stride_pre[0].size())) {
                stack_shape_stride_pre[1][i] = map_stride_shape[stack_shape_stride_pre[0][i] * stack_shape_stride_pre[1][i]];
            }
            // re-store modified infos
            shape_stride_stacks.emplace_back(stack_shape_stride_pre);
        }
        return true;
    }

    // Conduct trans-contiguous under strict constrains
    bool combined_to_contiguous(at::Tensor& self, at::Tensor& src, ShapeStrideStack& shape_stride_stacks, OffsetStack& offset_stacks) {
        // Base case: the last tensor to be processed.
        if (shape_stride_stacks.size() == 1) {
            if (reconstruct_tensor(src, shape_stride_stacks, offset_stacks)) {
                OptimizationCases opt_cases_last{"reshape", "permute", "slice", "select"};
                return copy_optimize_contiguous_by_given_cases(self, src, opt_cases_last);
            }
            return false;
        }
        // Construct the first tensor and judge whether it can be optimized.
        if (reconstruct_tensor(src, shape_stride_stacks, offset_stacks)) {
            ContiguousTensorDesc src_desc_ = TransContiguous::GetTensorDescInfo(src);
            OptimizationCases opt_cases_first{"reshape", "slice", "select"};
            if (reshape_without_copy_match(src)) {
                // case 1 : The first tensor is reshape-type, refresh its info is enough
                return combined_to_contiguous(self, src, shape_stride_stacks, offset_stacks);
            } else if (can_be_optimize_from_default_cases(src_desc_)) {
                // case 2: The first tensor is discontiguous-type,
                // conduct the standard optimization procedure.
                auto transfer_tensor =
                    OpPreparation::ApplyTensorWithFormat(src.sizes(), src.options(), torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc().npu_format_);
                return (copy_optimize_contiguous_by_given_cases(transfer_tensor, src, opt_cases_first) &&
                        combined_to_contiguous(self, transfer_tensor, shape_stride_stacks, offset_stacks));
            }
            // case3 ： The first tensor is contiguous or cannot be identified==>exit
            return false;
        }
        // If the first tensor cannnot be reconstructed==>exit
        return false;
    }
};  // class combinedContiguousOpt

REGISTER_COPY_OPT(combined, CombinedContiguousOpt)

class IndexingContiguousOpt : public ContiguousOpt {
public:
    bool Optimizer(at::Tensor& self, const at::Tensor& src, const ContiguousTensorDesc& src_desc) override {
        c10::SmallVector<int64_t, MAX_DIM> start;
        c10::SmallVector<int64_t, MAX_DIM> end;
        c10::SmallVector<int64_t, MAX_DIM> step;

        if (can_use_indexing(src_desc, start, end, step)) {
            RECORD_FUNCTION("contiguous_d_StridedSlice", std::vector<c10::IValue>({src}));
            indexing_to_contiguous(self, src, start, end, step, src_desc);
            return true;
        }
        return false;
    }

private:
    bool can_use_indexing(const ContiguousTensorDesc& src_desc, c10::SmallVector<int64_t, MAX_DIM>& start, c10::SmallVector<int64_t, MAX_DIM>& end,
                          c10::SmallVector<int64_t, MAX_DIM>& step) {
        if (c10::multiply_integers(src_desc.sizes_) >= c10::multiply_integers(src_desc.base_sizes_)) {
            return false;
        }

        if (src_desc.sizes_.size() != src_desc.base_sizes_.size()) {
            return false;
        }
        if (src_desc.strides_.size() != src_desc.base_strides_.size()) {
            return false;
        }

        const auto& base_size = src_desc.base_sizes_;
        const auto& base_stride = src_desc.base_strides_;
        const auto& indexing_size = src_desc.sizes_;
        const auto& indexing_stride = src_desc.strides_;

        for (const auto i : c10::irange(indexing_size.size())) {
            // base_stride should not be 0.
            if ((base_stride[i] == 0) || (indexing_stride[i] < base_stride[i]) || ((indexing_stride[i] % base_stride[i]) != 0)) {
                return false;
            }
        }

        // indexing信息获取部分
        // Get step info(for indexing step at index aixs should > 1)
        for (const auto i : c10::irange(indexing_size.size())) {
            step.emplace_back(indexing_stride[i] / base_stride[i]);
        }

        // Get start index based on offset and base stride
        int64_t src_offset = src_desc.offset_;
        for (const auto i : c10::irange(indexing_size.size())) {
            start.emplace_back(src_offset / base_stride[i]);
            src_offset = src_offset % base_stride[i];
        }

        // infer end index
        for (const auto i : c10::irange(indexing_size.size())) {
            int64_t calculate_end = start[i] + indexing_size[i] * step[i];
            if (calculate_end - step[i] > src_desc.base_sizes_[i]) {
                // Op StrideSlice(Slice) don't support span-axis indexing(slice).
                return false;
            }
            end.emplace_back(calculate_end);
        }

        // indexing场景判断: (1) step乘积>1(=1为slice);
        //                  (2) 当前规避最后一轴indexing,
        //                  因为stridedsliceD算子不支持; (3)
        //                  除去step!=1的轴，其他轴size，stride均与base_size,
        //                  base_stride相等(排除非关键轴reshape场景); (4)
        //                  对step!=1的轴，限制stride[i]=step[i]*size[i+1]*stride[i+1];(排除关键轴的reshape场景);
        //                  (5) 对step!=1的轴,
        //                  size(i)不可以为1:主要排除潜在的unsqueeze(0)+select(1,x)等走入indexing分支
        // case 1 & 2
        if (c10::multiply_integers(step) == 1 || step[step.size() - 1] != 1) {
            return false;
        }
        // case 3
        for (const auto i : c10::irange(step.size())) {
            if (step[i] == 1 && indexing_size[i] != base_size[i]) {
                return false;
            }
        }
        // case 4 and 5: step!=1的轴的校验
        for (const auto i : c10::irange(step.size() - 1)) {
            // 对于非最后一轴的indexing，对应的stride[i]=step[i]*size[i+1]*stride[i+1],（此时最后一轴stride限制为1）
            // 不满足上述条件，需要予以剔除，主要干扰：组合类reshape操作。
            if (step[i] != 1) {
                if (indexing_size[i] == 1) {
                    return false;
                }
                if (step[i + 1] == 1 && (indexing_stride[i] != indexing_size[i + 1] * indexing_stride[i + 1] * step[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    void indexing_to_contiguous(at::Tensor& self, const at::Tensor& src, c10::SmallVector<int64_t, MAX_DIM>& start, c10::SmallVector<int64_t, MAX_DIM>& end,
                                c10::SmallVector<int64_t, MAX_DIM>& step, const ContiguousTensorDesc& src_desc) {
        const auto& base_size = src_desc.base_sizes_;
        // recover contiguous base tensor
        at::Tensor temp_src = at::empty(src_desc.base_sizes_, src.options());
        temp_src.set_(src.storage(), temp_src.storage_offset(), temp_src.sizes(), temp_src.strides());

        // call StridedSlice op
        custom_ops::npu_indexing_out(temp_src, start, end, step, 0, 0, 0, 0, 0, self);

        return;
    }
};  // class IndexingContiguousOpt

REGISTER_COPY_OPT(indexing, IndexingContiguousOpt)

class PermuteContiguousOpt : public ContiguousOpt {
public:
    bool Optimizer(at::Tensor& self, const at::Tensor& src, const ContiguousTensorDesc& src_desc) override {
        // pattern permute
        c10::SmallVector<int64_t, MAX_DIM> perm;
        c10::SmallVector<int64_t, 5> sizes;
        if (can_use_permute(src_desc, perm, sizes)) {
            RECORD_FUNCTION("contiguous_d_Transpose", std::vector<c10::IValue>({src}));
            // Refresh src Tensor to match output self Tensor
            auto src_desc_stored = torch_npu::NPUBridge::GetNpuStorageImpl(src)->get_npu_desc();
            auto& src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
            src_desc.base_sizes_ = sizes;
            src_desc.base_strides_ = StorageDescHelper::ComputeStrideFromShape(static_cast<FormatShape>(sizes));
            src_desc.storage_sizes_ = sizes;

            custom_ops::npu_transpose_out(src, perm, false, self);
            src_desc = src_desc_stored;
            return true;
        }
        return false;
    }

    bool CanOptimizer(const ContiguousTensorDesc& src_desc) override {
        c10::SmallVector<int64_t, MAX_DIM> perm;
        c10::SmallVector<int64_t, 5> sizes;
        return can_use_permute(src_desc, perm, sizes);
    }

private:
    bool can_use_permute(const ContiguousTensorDesc& src_desc, c10::SmallVector<int64_t, MAX_DIM>& perm, c10::SmallVector<int64_t, 5>& sizes) {
        const auto& base_sizes = src_desc.base_sizes_;
        const auto& base_strides = src_desc.base_strides_;
        auto view_sizes = src_desc.sizes_;
        auto view_strides = src_desc.strides_;

        c10::SmallVector<int64_t, MAX_DIM> indexes;
        for (const auto i : c10::irange(src_desc.sizes_.size())) {
            indexes.emplace_back(i);
        }

        // After permute or reshape+permute, the total amount of data remains
        // unchanged.
        if (c10::multiply_integers(view_sizes) != c10::multiply_integers(base_sizes)) {
            return false;
        }

        // Reorder axes of shape and stride in descending order
        for (const auto i : c10::irange(src_desc.sizes_.size() - 1)) {
            for (const auto j : c10::irange(i + 1, src_desc.sizes_.size())) {
                bool need_swap = (view_strides[i] < view_strides[j]) || (view_strides[i] == view_strides[j] && view_sizes[i] < view_sizes[j]);
                if (need_swap) {
                    std::swap(view_strides[i], view_strides[j]);
                    std::swap(view_sizes[i], view_sizes[j]);
                    std::swap(indexes[i], indexes[j]);
                }
            }
        }

        // After reordering, check whether the shape and stride match
        auto current_stride = 1;
        int64_t src_desc_sizes = static_cast<int64_t>(src_desc.sizes_.size());
        for (int64_t i = src_desc_sizes - 1; i >= 0; i--) {
            if (current_stride != view_strides[i]) {
                NPU_LOGD(
                    "After reordering, shape and stride still do not match, and "
                    "permute pattern cannot be used.");
                return false;
            }
            current_stride *= view_sizes[i];
        }
        if ((base_sizes.size() - view_sizes.size()) != (base_strides.size() - view_strides.size())) {
            NPU_LOGD(
                "Reordered shape and base shape do not match, and permute "
                "pattern cannot be used.");
            return false;
        }

        // Calculate perm and sizes for permute
        for (const auto ele : view_sizes) {
            sizes.emplace_back(ele);
        }
        perm = indexes;
        for (const auto i : c10::irange(src_desc.sizes_.size())) {
            perm[indexes[i]] = i;
        }
        return true;
    }

    void optimize_permute(c10::SmallVector<int64_t, MAX_DIM>& perm, c10::SmallVector<int64_t, 5>& sizes) {
        c10::SmallVector<int64_t, MAX_DIM> optimized_perm;
        c10::SmallVector<int64_t, 5> optimized_sizes;
        if (perm.size() != sizes.size()) {
            NPU_LOGD("Param perm and sizes do not match.");
            return;
        }

        // Gather index
        int64_t perm_size = static_cast<int64_t>(perm.size());
        for (int64_t i = 0; i < perm_size; i++) {
            auto temp_perm_i = perm[i];
            auto temp_sizes_i = sizes[perm[i]];
            for (const auto j : c10::irange(i + 1, perm_size)) {
                if (perm[i] + 1 == perm[j]) {
                    temp_sizes_i *= sizes[perm[j]];
                    ++i;
                    continue;
                }
                break;
            }
            if (temp_sizes_i == 1) {
                // Optimize permute calculation for better performance, by squeezing
                // permute param.
                continue;
            }
            optimized_perm.emplace_back(temp_perm_i);
            optimized_sizes.emplace_back(temp_sizes_i);
        }
        if (optimized_perm.size() == perm.size()) {
            NPU_LOGD("No adjacent axes, cannot be optimized.");
            return;
        }

        // Calculate new perm and shape
        c10::SmallVector<int64_t, MAX_DIM> perm_indexes;
        for (const auto i : c10::irange(optimized_perm.size())) {
            perm_indexes.emplace_back(i);
        }
        for (const auto i : c10::irange(optimized_perm.size() - 1)) {
            for (const auto j : c10::irange(i + 1, optimized_perm.size())) {
                if (optimized_perm[i] > optimized_perm[j]) {
                    std::swap(optimized_perm[i], optimized_perm[j]);
                    std::swap(perm_indexes[i], perm_indexes[j]);
                }
            }
        }
        perm = perm_indexes;
        for (const auto i : c10::irange(perm_indexes.size())) {
            perm[perm_indexes[i]] = i;
        }
        sizes = optimized_sizes;
        for (const auto i : c10::irange(perm_indexes.size())) {
            sizes[i] = optimized_sizes[perm_indexes[i]];
        }
    }

    template <typename T>
    void squeeze_shape_and_stride(T& shape, T& stride) {
        int64_t shape_size = static_cast<int64_t>(shape.size());
        for (int64_t i = 0; i < shape_size; i++) {
            if (shape[i] == 1) {
                shape.erase(shape.begin() + i);
                stride.erase(stride.begin() + i);
                --i;
            }
        }
    }
};  // class PermuteContiguousOpt

REGISTER_COPY_OPT(permute, PermuteContiguousOpt)

bool can_use_memecpy_for_NZ_format(const ContiguousTensorDesc& tensor_desc) {
    int64_t tensor_shape_size = static_cast<int64_t>(tensor_desc.sizes_.size());
    int64_t base_shape_size = static_cast<int64_t>(tensor_desc.base_sizes_.size());
    // No padding&&offset!=0 at the same time. e.g. x(3, 15, 16)[1:]
    if (((tensor_desc.sizes_[tensor_shape_size - 1] % 16 != 0) || (tensor_desc.sizes_[tensor_shape_size - 2] % 16 != 0)) && tensor_desc.offset_ != 0) {
        return false;
    }
    // Make sure that sizes of last 2 dims don't change
    if (tensor_desc.sizes_[tensor_shape_size - 1] != tensor_desc.base_sizes_[base_shape_size - 1] ||
        tensor_desc.sizes_[tensor_shape_size - 2] != tensor_desc.base_sizes_[base_shape_size - 2]) {
        return false;
    }
    return true;
}

bool can_use_memcpy_for_other_format(const ContiguousTensorDesc& tensor_desc) {
    // torch.flatten(x) case should be removed
    if (tensor_desc.sizes_.size() < 2) {
        return false;
    }

    switch (tensor_desc.npu_format_) {
        case ACL_FORMAT_FRACTAL_NZ:
            return can_use_memecpy_for_NZ_format(tensor_desc);
        // (Ascend): 5HD format can also be optimized likes NZ format
        default:
            // For other format, make sure that copy the whole memory.
            // Moreover, storage size expanding caused by padding could be avoided
            if (!(tensor_desc.base_sizes_ == tensor_desc.sizes_)) {
                return false;
            }
            // Make sure no pandding happens
            if (c10::multiply_integers(tensor_desc.sizes_) != c10::multiply_integers(tensor_desc.storage_sizes_)) {
                return false;
            }
            return true;
    }
}

bool check_reshape_match(const ContiguousTensorDesc& tensor_desc) {
    // (case 1) Reshape tensor should be contiguous
    if (!tensor_desc.is_contiguous_) {
        return false;
    }
    // (case2) for other format, sizes at key dims should remain unchanged
    if (!FormatHelper::IsBaseFormatType(tensor_desc.npu_format_)) {
        return can_use_memcpy_for_other_format(tensor_desc);
    }
    return true;
}

bool check_reshape_match(const ContiguousTensorDesc& self_desc, const ContiguousTensorDesc& src_desc) {
    // For all format, both src and self are taken into consideration
    if (check_reshape_match(src_desc) && check_reshape_match(self_desc)) {
        // tensor numels eqs for self and src tensor. i.e. make sure that storage
        // keep same.
        if (!(self_desc.sizes_ == src_desc.sizes_)) {
            return false;
        }

        return true;
    }
    return false;
}

bool CanUseMemcpyForOtherFormat(const at::Tensor& tensor) {
    ContiguousTensorDesc tensor_desc = TransContiguous::GetTensorDescInfo(tensor);
    return can_use_memcpy_for_other_format(tensor_desc);
}

class ReshapeContiguousOpt : public ContiguousOpt {
public:
    bool Optimizer(at::Tensor& result, const at::Tensor& src, const ContiguousTensorDesc& src_desc) override {
        ContiguousTensorDesc result_desc = TransContiguous::GetTensorDescInfo(result);
        if (check_reshape_match(result_desc, src_desc)) {
            RECORD_FUNCTION("contiguous_d_Reshape", std::vector<c10::IValue>({src}));
            custom_ops::npu_reshape_out(src, src.sizes(), false, result);
            return true;
        }
        return false;
    }

    bool CanOptimizer(const ContiguousTensorDesc& src_desc) override { return check_reshape_match(src_desc); }
};  // class ReshapeContiguousOpt

REGISTER_COPY_OPT(reshape, ReshapeContiguousOpt)

class ReshapeV2ContiguousOpt : public ContiguousOpt {
public:
    bool Optimizer(at::Tensor& result, const at::Tensor& src, const ContiguousTensorDesc& src_desc) override {
        ContiguousTensorDesc result_desc = TransContiguous::GetTensorDescInfo(result);
        if (check_reshape_match(result_desc, src_desc)) {
            if (can_use_memory_repoint(src_desc) && reshape_match_by_memory_repoint(src, result)) {
                return true;
            }
            RECORD_FUNCTION("contiguous_d_Reshape", std::vector<c10::IValue>({src}));
            custom_ops::npu_reshape_out(src, src.sizes(), false, result);
            return true;
        }
        return false;
    }

    bool CanOptimizer(const ContiguousTensorDesc& src_desc) override { return check_reshape_match(src_desc); }

private:
    template <typename dataDtype>
    void ResetDataPtr(const at::Tensor& src, at::Tensor& self, dataDtype* value) {
        dataDtype* src_data_ptr = value + src.storage_offset();
        at::DataPtr self_data_ptr = at::DataPtr(src_data_ptr, self.storage().device());
        self.storage().set_data_ptr(std::move(self_data_ptr));
    }

    bool reshape_match_by_memory_repoint(const at::Tensor& src, at::Tensor& self) {
        RECORD_FUNCTION("contiguous_h_memRepoint", std::vector<c10::IValue>({src}));
        switch (src.scalar_type()) {
            case at::ScalarType::Half:
                ResetDataPtr(src, self, static_cast<at::Half*>(src.storage().data_ptr().get()));
                return true;
            case at::ScalarType::BFloat16:
                ResetDataPtr(src, self, static_cast<at::BFloat16*>(src.storage().data_ptr().get()));
                return true;
            case at::ScalarType::Float:
                ResetDataPtr(src, self, static_cast<float*>(src.storage().data_ptr().get()));
                return true;
            case at::ScalarType::Byte:
                ResetDataPtr(src, self, static_cast<uint8_t*>(src.storage().data_ptr().get()));
                return true;
            case at::ScalarType::Char:
                ResetDataPtr(src, self, static_cast<int8_t*>(src.storage().data_ptr().get()));
                return true;
            case at::ScalarType::Short:
                ResetDataPtr(src, self, static_cast<int16_t*>(src.storage().data_ptr().get()));
                return true;
            case at::ScalarType::Int:
                ResetDataPtr(src, self, static_cast<int*>(src.storage().data_ptr().get()));
                return true;
            case at::ScalarType::Long:
                ResetDataPtr(src, self, static_cast<int64_t*>(src.storage().data_ptr().get()));
                return true;
            default:
                // Turn to conducting d2dCopyAsync for other dtypes.
                return false;
        }
    }

    bool can_use_memory_repoint(const ContiguousTensorDesc& src_desc) {
        if (FormatHelper::IsBaseFormatType(src_desc.npu_format_)) {
            return true;
        }

        if (src_desc.npu_format_ == ACL_FORMAT_FRACTAL_NZ) {
            // No padding
            if ((src_desc.sizes_[src_desc.sizes_.size() - 1] % 16 == 0) && (src_desc.sizes_[src_desc.sizes_.size() - 2] % 16 == 0)) {
                return true;
            }
            return false;
        }
        return false;
    }
};  // class ReshapeV2ContiguousOpt

REGISTER_COPY_OPT(reshapeV2, ReshapeV2ContiguousOpt)

class SelectContiguousOpt : public ContiguousOpt {
public:
    bool Optimizer(at::Tensor& self, const at::Tensor& src, const ContiguousTensorDesc& src_desc) override {
        // select(dim, start), length[dim] == 1
        c10::SmallVector<int64_t, MAX_DIM> start;
        c10::SmallVector<int64_t, MAX_DIM> length;

        if (can_use_select(src_desc, start, length)) {
            RECORD_FUNCTION("contiguous_d_StridedSlice", std::vector<c10::IValue>({src}));
            select_to_contiguous(self, src, start, length, src_desc);
            return true;
        }
        return false;
    }

    bool CanOptimizer(const ContiguousTensorDesc& src_desc) override {
        c10::SmallVector<int64_t, MAX_DIM> start;
        c10::SmallVector<int64_t, MAX_DIM> length;
        return can_use_select(src_desc, start, length);
    }

private:
    bool can_use_select(const ContiguousTensorDesc& src_desc, c10::SmallVector<int64_t, MAX_DIM>& start, c10::SmallVector<int64_t, MAX_DIM>& length) {
        // base info and src info
        const auto& base_size = src_desc.base_sizes_;
        const auto& base_stride = src_desc.base_strides_;
        const auto& select_size = src_desc.sizes_;
        const auto& select_stride = src_desc.strides_;

        // len(base_size) - len(select_size) == 1  && len(base_stride) -
        // len(select_stride) == 1
        if ((base_size.size() - select_size.size() != 1) || (base_stride.size() - select_stride.size() != 1)) {
            return false;
        }

        // recover src tensor info: shape and stride
        c10::SmallVector<int64_t, MAX_DIM> temp_size;
        c10::SmallVector<int64_t, MAX_DIM> temp_stride;
        for (size_t i = 0U; i <= select_size.size(); i++) {
            if (base_size[i] != select_size[i] || base_stride[i] != select_stride[i]) {
                temp_size.emplace_back(base_size[i]);
                temp_stride.emplace_back(base_stride[i]);
                for (size_t j = i + 1U; j <= select_size.size(); j++) {
                    temp_size.emplace_back(select_size[j - 1]);
                    temp_stride.emplace_back(select_stride[j - 1]);
                    i = j + 1;
                }
            } else {
                temp_size.emplace_back(select_size[i]);
                temp_stride.emplace_back(select_stride[i]);
            }
        }

        for (const auto i : c10::irange(select_size.size() + 1)) {
            if (base_size[i] == temp_size[i] && base_stride[i] == temp_stride[i]) {
                continue;
            } else {
                return false;
            }
        }

        // Collect the select infos for SliceD: dim, start, length
        // confirm the selected dim
        int64_t dim = static_cast<int64_t>(base_size.size()) - 1;
        for (const auto i : c10::irange(select_size.size())) {
            if (base_size[i] != select_size[i] || base_stride[i] != select_stride[i]) {
                dim = i;
                break;
            }
        }

        // Obtain start index and select length
        int64_t int_index = src_desc.offset_ / base_stride[dim];
        for (const auto i : c10::irange(base_size.size())) {
            if (i == dim) {
                start.emplace_back(int_index);
                length.emplace_back(1);
            } else {
                start.emplace_back(0);
                length.emplace_back(base_size[i]);
            }
        }
        return true;
    }

    void select_to_contiguous(at::Tensor& self, const at::Tensor& src, c10::SmallVector<int64_t, MAX_DIM>& start, c10::SmallVector<int64_t, MAX_DIM>& length,
                              const ContiguousTensorDesc& src_desc) {
        const auto& base_size = src_desc.base_sizes_;
        // Recover base tensor(necessary) a = b.select(1, 1)
        at::Tensor temp_src = at::empty(base_size, src.options());
        temp_src.set_(src.storage(), temp_src.storage_offset(), temp_src.sizes(), temp_src.strides());

        // construct StridedSlice param
        int64_t axis_size = static_cast<int64_t>(start.size());
        c10::SmallVector<int64_t, MAX_DIM> strides(axis_size, 1);
        c10::SmallVector<int64_t, MAX_DIM> end;
        int64_t shrink_mask = 0;
        for (int64_t i = 0; i < axis_size; ++i) {
            end.emplace_back(start[i] + length[i]);
            if (length[i] == 1 && temp_src.size(i) != 1) {
                shrink_mask += std::pow(2, i);
            }
        }

        // call StridedSlice op to contiguous
        custom_ops::npu_indexing_out(temp_src, start, end, strides, 0, 0, 0, 0, shrink_mask, self);
        return;
    }
};  // class SelectContiguousOpt

REGISTER_COPY_OPT(select, SelectContiguousOpt)

class SliceContiguousOpt : public ContiguousOpt {
public:
    bool Optimizer(at::Tensor& self, const at::Tensor& src, const ContiguousTensorDesc& src_desc) override {
        // Pattern slice.
        // Current pattern does not directly depend on other patterns.
        // The relative sequence of this pattern and other patterns is not
        // important.
        c10::SmallVector<int64_t, MAX_DIM> offsets;
        c10::SmallVector<int64_t, MAX_DIM> size;
        if (can_use_slice(src_desc, offsets, size)) {
            RECORD_FUNCTION("contiguous_d_Slice", std::vector<c10::IValue>({src}));
            slice_to_contiguous(self, src, offsets, size, src_desc);
            return true;
        }
        return false;
    }

    bool CanOptimizer(const ContiguousTensorDesc& src_desc) override {
        c10::SmallVector<int64_t, MAX_DIM> offsets;
        c10::SmallVector<int64_t, MAX_DIM> size;
        return can_use_slice(src_desc, offsets, size);
    }

private:
    // npu-slice pattern cover several view ops, including chunk, split, narrow
    // and part of index. Judgment logic is based on the implement of view ops in
    // adapter layer.
    bool can_use_slice(const ContiguousTensorDesc& src_desc, c10::SmallVector<int64_t, MAX_DIM>& offsets, c10::SmallVector<int64_t, MAX_DIM>& size) {
        const auto& base_sizes = src_desc.base_sizes_;
        const auto& base_strides = src_desc.base_strides_;
        auto view_sizes = src_desc.sizes_;
        auto view_strides = src_desc.strides_;

        // narrow+select(select at last dim) ==> single narrow
        // 限制条件：1. 最后一轴stride非1==>最后一轴select；2.
        // 基础格式；3.非最后一轴发生narrow（元素减少）
        // 最小化影响：仅限最后一轴的select，即tensor.select(-1, 1) ==
        // tensor[**,1:2],select过渡到narrow
        if (view_strides[view_strides.size() - 1] != 1 && FormatHelper::IsBaseFormatType(src_desc.npu_format_) && view_strides.size() < base_strides.size() &&
            c10::multiply_integers(view_sizes) < c10::multiply_integers(base_sizes) / base_sizes[base_sizes.size() - 1]) {
            view_sizes.emplace_back(1);
            view_strides.emplace_back(1);
        }

        // Strides must be the same.
        if (view_strides != base_strides) {
            return false;
        }

        // Only narrow dims are different.
        c10::SmallVector<int64_t, MAX_DIM> narrow_dims;
        if (view_sizes.size() != base_sizes.size()) {
            return false;
        }
        for (const auto i : c10::irange(view_sizes.size())) {
            if (view_sizes[i] == base_sizes[i]) {
                narrow_dims.emplace_back(0);
            } else if (view_sizes[i] < base_sizes[i]) {
                narrow_dims.emplace_back(1);
            } else {
                return false;
            }
        }

        // Calculate npu slice param.
        size = view_sizes;
        offsets.clear();
        int64_t storage_offsets = src_desc.offset_;
        // src.storage_offset() == start[narrow_dims[i]]*stride[narrow_dims[i]]
        for (const auto i : c10::irange(view_strides.size())) {
            offsets.emplace_back(storage_offsets / view_strides[i]);
            storage_offsets = storage_offsets % view_strides[i];
        }
        if (storage_offsets != 0) {
            return false;
        }
        for (const auto i : c10::irange(offsets.size())) {
            if ((offsets[i] + view_sizes[i]) > base_sizes[i]) {
                // In narrow calculation, (start + length) <= cur_size
                return false;
            }
            if (offsets[i] != 0 && narrow_dims[i] == 0) {
                // narrow_dims[i] == 0 means dim i is not involved in narrow
                // calculation. offsets[i] != 0 means dim i has the start of narrow
                // calculation. Two conditions are contradictory.
                return false;
            }
        }
        return true;
    }

    void slice_to_contiguous(at::Tensor& self, const at::Tensor& src, const c10::SmallVector<int64_t, MAX_DIM>& offsets,
                             const c10::SmallVector<int64_t, MAX_DIM>& size, const ContiguousTensorDesc& src_desc) {
        // create contiguous tensor for npu slice
        const auto& temp_tensor_size = src_desc.base_sizes_;
        at::Tensor temp_src = at::empty(temp_tensor_size, src.options());
        temp_src.set_(src.storage(), temp_src.storage_offset(), temp_src.sizes(), temp_src.strides());

        custom_ops::npu_slice_out(temp_src, offsets, size, self);
        return;
    }
};  // class SliceContiguousOpt

REGISTER_COPY_OPT(slice, SliceContiguousOpt)

}  // namespace native
}  // namespace at_npu
