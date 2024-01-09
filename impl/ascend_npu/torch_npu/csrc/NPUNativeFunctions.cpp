#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

#include "diopi_impl/helper.hpp"
#include "op_plugin/AclOpsInterface.h"

namespace at_npu::native {

#define CUSTOM_OP_NOT_IMPL std::cout << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << ": not impled yet" << std::endl;

c10::Scalar NPUNativeFunctions::_local_scalar_dense(const at::Tensor& self) {
    c10::Scalar r;
    AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_local_scalar_dense_npu", [&] {
        scalar_t value = 0;
        c10_npu::NPUStream copy_stream = c10_npu::getCurrentNPUStream();
        // Synchronous copy after stream synchronization
        aclError error = c10_npu::acl::AclrtSynchronizeStreamWithTimeout(copy_stream);
        if (error != ACL_ERROR_NONE) {
            C10_NPU_SHOW_ERR_MSG();
            AT_ERROR("ACL stream synchronize failed.");
            return;
        }

        error = CalcuOpUtil::AclrtMemcpyWithModeSwitch(&value,
                                                       sizeof(scalar_t),
                                                       std::make_pair(self.storage().unsafeGetStorageImpl(), self.storage_offset() * self.itemsize()),
                                                       sizeof(scalar_t),
                                                       ACL_MEMCPY_DEVICE_TO_HOST);
        if (error != ACL_ERROR_NONE) {
            C10_NPU_SHOW_ERR_MSG();
            AT_ERROR("aclrtMemcpy device to host error.");
            return;
        }
        r = c10::Scalar(value);
    });
    return r;
}

at::Tensor format_cast_impl_out_npu(at::Tensor& dst, const at::Tensor& src) {
    string srcFormat = FormatHelper::GetFormatName(src);
    string dstFormat = FormatHelper::GetFormatName(dst);

    if (!FormatCastHelper::IsSameGroupType(src, dst)) {
        bool res = FormatCastHelper::format_cast_between_group(dst, src, format_cast_impl_out_npu);
        if (!res) {
            AT_ERROR("unsupport cast from ", srcFormat, " to ", dstFormat);
        }
        return dst;
    }

    // NpuStorageOffsetGuard guard_input(const_cast<at::Tensor &>(src));
    // NpuStorageOffsetGuard guard_output(dst);
    OpCommand cmd;
    cmd.Name("Identity").InputWithoutContiguous(src).Output(dst).Run();
    return dst;
}

// conver self to acl_format, write the result into new result tensor
at::Tensor npu_format_cast_impl(const at::Tensor& src, int64_t acl_format) {
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    if (src_desc.npu_format_ == acl_format) {
        NPU_LOGD("no need to do format cast");
        return src;
    }
    if (FormatHelper::IsBaseFormatType(src) && FormatHelper::IsBaseFormatType(static_cast<aclFormat>(acl_format))) {
        FormatCastHelper::format_cast_as_base_format(src, static_cast<aclFormat>(acl_format));
        return src;
    }

    at::Tensor dst = OpPreparation::ApplyTensorWithFormat(src_desc.base_sizes_, src.options(), acl_format);

    // calculate the output result of the NPU
    format_cast_impl_out_npu(dst, src);

    // format cast only change physical layout of base tensor and view tensor's
    // metadata remain unchanged
    dst.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());
    return dst;
}

// convert src from src_format to dst_format, write the result into dst
at::Tensor& NPUNativeFunctions::npu_format_cast_(at::Tensor& dst, const at::Tensor& src) {
    torch_npu::utils::torch_check_npu(dst);
    torch_npu::utils::torch_check_npu(src);
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_;
    if (src_desc.npu_format_ == dst_desc.npu_format_) {
        dst.copy_(src);
        return dst;
    }

    // calculate the output result of the NPU
    format_cast_impl_out_npu(dst, src);

    return dst;
}

at::Tensor NPUNativeFunctions::contiguous(const at::Tensor& self, at::MemoryFormat memory_format) {
    if (self.is_contiguous(memory_format)) {
        return self;
    }
    TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous, "NPU contiguous operator only supportted contiguous memory format.");
    return self.clone();
}

int64_t NPUNativeFunctions::get_storage_size(const at::Tensor& self) {
    torch_npu::utils::torch_check_npu(self);
    auto sizes = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_.storage_sizes_;
    int64_t n = 1;
    for (auto s : sizes) {
        n *= s;
    }
    return n;
}

at::Tensor NPUNativeFunctions::clone(const at::Tensor& self, c10::optional<at::MemoryFormat> memory_format) {
    return at_npu::native::clone(self, memory_format);
}

at::Tensor NPUNativeFunctions::empty(c10::SymIntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                                     c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
    return at_npu::native::empty_npu(c10::asIntArrayRefUnchecked(size), dtype, layout, device, pin_memory, memory_format);
}

at::Tensor NPUNativeFunctions::empty_strided(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<at::ScalarType> dtype,
                                             c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    return at_npu::native::empty_strided_npu(size, stride, dtype, layout, device, pin_memory);
}

static inline void checkInBoundsForStorage(c10::IntArrayRef size, c10::IntArrayRef stride, int64_t storage_offset, const caffe2::TypeMeta data_type,
                                           const c10::Storage& new_storage) {
    int64_t storage_size_bytes = static_cast<int64_t>(at::detail::computeStorageNbytes(size, stride, data_type.itemsize()));
    int64_t storage_offset_bytes = storage_offset * static_cast<int64_t>(data_type.itemsize());
    if (storage_size_bytes == 0) {
        // NB: (a tensor with arbitrary 0 dims)'s storage can have any numel.
        return;
    }

    int64_t new_storage_size_bytes = static_cast<int64_t>(new_storage.nbytes());
    TORCH_CHECK(storage_size_bytes + storage_offset_bytes <= new_storage_size_bytes,
                "setStorage: sizes ",
                size,
                ", strides ",
                stride,
                ","
                " storage offset ",
                storage_offset,
                ", and itemsize ",
                data_type.itemsize(),
                " requiring a storage size of ",
                storage_size_bytes,
                " are out of bounds for storage of size ",
                new_storage_size_bytes);
}

inline void setStrided(const at::Tensor& self, c10::IntArrayRef size, c10::IntArrayRef stride, int64_t storage_offset) {
    TORCH_CHECK(size.size() == stride.size(), "mismatch in length of strides and shape");
    auto* self_ = self.unsafeGetTensorImpl();
    checkInBoundsForStorage(size, stride, storage_offset, self_->dtype(), self_->storage());

    /* storage offset */
    TORCH_CHECK(storage_offset >= 0, "Tensor: invalid storage offset ", storage_offset);
    self_->set_storage_offset(storage_offset);

    /* size and stride */
    if (self_->sizes() == size && self_->strides() == stride) {
        return;
    }
    for (auto val : stride) {
        TORCH_CHECK(val >= 0,
                    "as_strided: Negative strides are not supported at the moment, "
                    "got strides: ",
                    stride);
    }
    self_->set_sizes_and_strides(size, stride);
}

at::Tensor NPUNativeFunctions::as_strided(const at::Tensor& self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
    auto dst = self;
    if (InferFormat::IsDefiniteTensorWhenMetaDataChanges(dst, size)) {
        dst = FormatCastHelper::ApplyBaseFormatTensorBy(dst);
    }
    auto storage_offset_new = storage_offset.value_or(dst.storage_offset());
    auto result = at::detail::make_tensor<at::TensorImpl>(c10::TensorImpl::VIEW, c10::Storage(dst.storage()), dst.key_set(), dst.dtype());
    setStrided(result, size, stride, storage_offset_new);
    return result;
}

at::Tensor NPUNativeFunctions::empty_with_format(c10::IntArrayRef size, c10::optional<c10::ScalarType> dtype_opt, c10::optional<c10::Layout> layout_opt,
                                                 c10::optional<c10::Device> device_opt, c10::optional<bool> pin_memory_opt, int64_t dst_format) {
    return at_npu::native::empty_with_format(size, dtype_opt, layout_opt, device_opt, pin_memory_opt, dst_format);
}

at::Tensor NPUNativeFunctions::empty_with_format(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype,
                                                 c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory,
                                                 int64_t acl_format) {
    OP_NOT_IMPL;
}

at::Tensor NPUNativeFunctions::unsafe_empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                                                        c10::optional<at::Device> device, c10::optional<bool> pin_memory, int64_t acl_format,
                                                        bool keep_format) {
    // This is a special interface that can adjust the memory application results. Check before use.

    // Some ops cannot operate directly based on ND format, such as MatMul, BatchMatMul, MaxPoolWithArgmaxV1.
    // For these ops, specify the parameter keep_format to ensure that
    // the specified internal format is preserved.
    if ((!keep_format) && at_npu::native::env::CheckForbidInternalFormat()) {
        acl_format = static_cast<int64_t>(FormatHelper::GetBaseFormat(static_cast<aclFormat>(acl_format)));
    }

    return NPUNativeFunctions::empty_with_format(size, dtype, layout, device, pin_memory, acl_format);
}

namespace custom_ops {

int64_t npu_change_data_ptr(const at::Tensor& dst, const at::Tensor& src, int64_t index) { CUSTOM_OP_NOT_IMPL; }
int64_t get_npu_format(const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }

at::Tensor npu_format_cast(const at::Tensor& self, const at::Tensor& dst) {
    torch_npu::utils::torch_check_npu(dst);
    auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_;
    int64_t dst_format = dst_desc.npu_format_;
    return npu_format_cast_impl(self, dst_format);
}

at::Tensor& npu_format_cast_(at::Tensor& src, int64_t acl_format) {
    torch_npu::utils::torch_check_npu(src);
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    if (src_desc.npu_format_ == acl_format) {
        return src;
    }
    if (FormatHelper::IsBaseFormatType(src) && FormatHelper::IsBaseFormatType(static_cast<aclFormat>(acl_format))) {
        FormatCastHelper::format_cast_as_base_format(src, static_cast<aclFormat>(acl_format));
        return src;
    }

    at::Tensor dst = OpPreparation::ApplyTensorWithFormat(src_desc.base_sizes_, src.options(), acl_format);

    // calculate the output result of the NPU
    format_cast_impl_out_npu(dst, src);

    // format cast only change physical layout of base tensor and view tensor's
    // metadata remain unchanged
    src.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());

    return src;
}

at::Tensor& npu_format_cast_(at::Tensor& self, const at::Tensor& src) {
    torch_npu::utils::torch_check_npu(src);
    auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;
    int64_t dst_format = dst_desc.npu_format_;
    return npu_format_cast_(self, dst_format);
}

at::Tensor npu_format_cast(const at::Tensor& self, int64_t acl_format) { return npu_format_cast_impl(self, acl_format); }

at::Tensor _npu_format_cast(const at::Tensor& self, int64_t acl_format) { return npu_format_cast_impl(self, acl_format); }

at::Tensor empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device,
                             c10::optional<bool> pin_memory, int64_t acl_format) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor unsafe_empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                                    c10::optional<at::Device> device, c10::optional<bool> pin_memory, int64_t acl_format, bool keep_format) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor empty_with_format(at::IntArrayRef size, c10::optional<at::DimnameList> names, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                             c10::optional<at::Device> device, c10::optional<bool> pin_memory, int64_t acl_format) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor& copy_memory_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    AT_ASSERT(at_npu::key::isDeviceTensor(src), "copy_memory_ only support npu tensor");
    AT_ASSERT(src.dtype() == self.dtype(), "input tensors of copy_memory_ should have same dtype");
    // AT_ASSERT(
    //     src.is_contiguous() && self.is_contiguous(),
    //     "input tensors of copy_memory_ should be contiguous");
    AT_ASSERT(src.device().index() == self.device().index(), "input tensors of copy_memory_ should have same device index");
    auto dst_desc = torch_npu::NPUBridge::GetNpuStorageImpl(self)->npu_desc_;
    auto src_desc = torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_;

    int dst_size = 0;
    int src_size = 0;

    if (FormatHelper::IsPadded(&self)) {
        AT_ASSERT(self.storage_offset() == 0);
        dst_size = c10::multiply_integers(dst_desc.storage_sizes_);
    } else {
        auto dst_element = c10::multiply_integers(self.sizes());
        auto dst_storage = c10::multiply_integers(dst_desc.storage_sizes_);
        dst_size = (dst_element > dst_storage) ? dst_storage : dst_element;
    }

    if (FormatHelper::IsPadded(&src)) {
        AT_ASSERT(src.storage_offset() == 0);
        src_size = c10::multiply_integers(src_desc.storage_sizes_);
    } else {
        auto src_element = c10::multiply_integers(src.sizes());
        auto src_storage = c10::multiply_integers(src_desc.storage_sizes_);
        src_size = (src_element > src_storage) ? src_storage : src_element;
    }

    // Designed for the gather of tensors, ignoring npu_format_ and
    // copying continuous memory between npu tensors.
    auto ret = CalcuOpUtil::LaunchAsyncCopyTaskWithModeSwitch(self, dst_size * self.itemsize(), src, dst_size * self.itemsize(), ACL_MEMCPY_DEVICE_TO_DEVICE);
    NPU_CHECK_ERROR(ret);

    if (!non_blocking) {
        c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
    }
    return self;
}

at::Tensor format_contiguous(const at::Tensor& self) { return NpuUtils::format_contiguous(self); }

bool check_match(const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
void check_memory_overlaps(at::TensorList inputs, at::TensorList outputs) { CUSTOM_OP_NOT_IMPL; }
int64_t get_storage_size(const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }

at::Tensor& npu_view_copy(at::Tensor& self, const at::Tensor& other, bool non_blocking) { acl_op::npu_view_copy(self, other, non_blocking); }
at::Tensor npu_transpose(const at::Tensor& self, at::IntArrayRef perm, bool require_contiguous) { CUSTOM_OP_NOT_IMPL; }

at::Tensor& npu_transpose_out(const at::Tensor& self, at::IntArrayRef perm, bool require_contiguous, at::Tensor& out) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_broadcast(const at::Tensor& self, at::IntArrayRef size) { return acl_op::npu_broadcast(self, size); }

at::Tensor& npu_broadcast_out(const at::Tensor& self, at::IntArrayRef size, at::Tensor& out) { return acl_op::npu_broadcast_out(self, size, out); }

at::Tensor npu_dtype_cast(const at::Tensor& self, at::ScalarType dtype) { return acl_op::npu_dtype_cast(self, dtype); }

#if 1
at::Tensor& npu_dtype_cast_(at::Tensor& self, const at::Tensor& src) {
    at::Tensor source = src.contiguous();

    if (src.sizes() != self.sizes()) {
        source = npu_broadcast(source, self.sizes());
    }
    if (source.strides() == self.strides() && self.is_contiguous()) {
        acl_op::npu_dtype_cast_(self, source);
    } else {
        at::Tensor selfTemp = at_npu::native::empty_npu(source.sizes(), self.options());
        acl_op::npu_dtype_cast_(selfTemp, source);
        self.copy_(selfTemp);
    }
    return self;
}
#else

at::Tensor& npu_dtype_cast_(at::Tensor& self, const at::Tensor& src) {
    DEBUG_ARGS(self)
    DEBUG_ARGS(src)
    if (self.sizes() == src.sizes() && self.strides() == src.strides()) {
        acl_op::npu_dtype_cast_(self, src);
        return self;
    }

    if (self.sizes() == src.sizes() && self.strides() != src.strides()) {
        at::Tensor srcContiguous;
        if (src.is_contiguous()) {
            srcContiguous = src;
        } else {
            srcContiguous = at_npu::native::empty_npu(src.sizes(), src.options());
            srcContiguous.copy_(src);
        }
        if (self.is_contiguous()) {
            return npu_dtype_cast_(self, srcContiguous);
        } else {
            auto selfContiguous = at_npu::native::empty_npu(self.sizes(), self.options());
            npu_dtype_cast_(selfContiguous, srcContiguous);
            self.copy_(selfContiguous);
            return self;
        }
    }
    if (self.sizes() != src.sizes()) {
        auto srcBroaded = npu_broadcast(src.contiguous(), self.sizes());
        return npu_dtype_cast_(self, srcBroaded);
    }

    TORCH_CHECK(false, "unhandled situation");
    return self;
}
#endif

at::Tensor npu_alloc_float_status(const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_get_float_status(const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_clear_float_status(const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
at::Tensor& one_(at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
at::Tensor fast_gelu(const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_fast_gelu_backward(const at::Tensor& grad, const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
bool _amp_foreach_non_finite_check(at::TensorList scaled_grads) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_sign_bits_pack(const at::Tensor& self, int64_t size) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_bert_apply_adam(const at::Scalar& lr, const at::Scalar& beta1, const at::Scalar& beta2,
                                                                     const at::Scalar& epsilon, const at::Tensor& grad, const at::Scalar& max_grad_norm,
                                                                     const at::Scalar& global_grad_norm, const at::Scalar& weight_decay,
                                                                     const c10::optional<at::Scalar>& step_size, int64_t adam_mode) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_bert_apply_adam_out(const at::Scalar& lr, const at::Scalar& beta1, const at::Scalar& beta2,
                                                                            const at::Scalar& epsilon, const at::Tensor& grad, const at::Scalar& max_grad_norm,
                                                                            const at::Scalar& global_grad_norm, const at::Scalar& weight_decay,
                                                                            const c10::optional<at::Scalar>& step_size, int64_t adam_mode, at::Tensor& var,
                                                                            at::Tensor& m, at::Tensor& v) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose2d_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
                                                                               at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride,
                                                                               at::IntArrayRef dilation, int64_t groups, ::std::array<bool, 3> output_mask) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv_transpose3d_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
                                                                               at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef stride,
                                                                               at::IntArrayRef dilation, int64_t groups, ::std::array<bool, 3> output_mask) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_convolution_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
                                                                          at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
                                                                          int64_t groups, ::std::array<bool, 3> output_mask) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_conv_transpose2d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef padding,
                                at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_conv2d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding,
                      at::IntArrayRef dilation, int64_t groups) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor& npu_conv2d_out(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
                           at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor& out) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv2d_backward(const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
                                                                     at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups,
                                                                     ::std::array<bool, 3> output_mask) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_conv3d(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride, at::IntArrayRef padding,
                      at::IntArrayRef dilation, int64_t groups) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor& npu_conv3d_out(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
                           at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, at::Tensor& out) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_conv3d_backward(const at::Tensor& input, const at::Tensor& grad, const at::Tensor& weight,
                                                                     at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups,
                                                                     ::std::array<bool, 3> output_mask) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_stride_add(const at::Tensor& self, const at::Tensor& other, const at::Scalar& offset1, const at::Scalar& offset2, const at::Scalar& c1_len) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_slice(const at::Tensor& self, at::IntArrayRef offsets, at::IntArrayRef size) { return acl_op::npu_slice(self, offsets, size); }
at::Tensor& npu_slice_out(const at::Tensor& self, at::IntArrayRef offsets, at::IntArrayRef size, at::Tensor& out) {
    return acl_op::npu_slice_out(self, offsets, size, out);
}
at::Tensor npu_indexing(const at::Tensor& self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask,
                        int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor& npu_indexing_out(const at::Tensor& self, at::IntArrayRef begin, at::IntArrayRef end, at::IntArrayRef strides, int64_t begin_mask, int64_t end_mask,
                             int64_t ellipsis_mask, int64_t new_axis_mask, int64_t shrink_axis_mask, at::Tensor& out) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_softmax_cross_entropy_with_logits_backward(const at::Tensor& grad, const at::Tensor& self, const at::Tensor& labels) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_stride_copy(const at::Tensor& self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar& storage_offset) { CUSTOM_OP_NOT_IMPL; }

at::Tensor& npu_stride_copy_out(const at::Tensor& self, at::IntArrayRef shape, at::IntArrayRef stride, const at::Scalar& storage_offset, at::Tensor& out) {
    return acl_op::npu_stride_copy_out(self, shape, stride, storage_offset, out);
}

at::Tensor npu_roi_align(const at::Tensor& self, const at::Tensor& rois, double spatial_scale, int64_t pooled_height, int64_t pooled_width, int64_t sample_num,
                         int64_t roi_end_mode) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_roi_alignbk(const at::Tensor& self, const at::Tensor& rois, at::IntArrayRef xdiff_shape, int64_t pooled_width, int64_t pooled_height,
                           double spatial_scale, int64_t sample_num, c10::optional<int64_t> roi_end_mode) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor& npu_sort_v2_out(const at::Tensor& self, int64_t dim, bool descending, at::Tensor& out) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_sort_v2(const at::Tensor& self, int64_t dim, bool descending) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_one_hot(const at::Tensor& self, int64_t num_classes, int64_t depth, const at::Scalar& on_value, const at::Scalar& off_value) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor> npu_linear_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_anchor_response_flags(const at::Tensor& self, at::IntArrayRef featmap_size, at::IntArrayRef stride, int64_t num_base_anchors) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_dropout_backward(const at::Tensor& grad_output, const at::Tensor& mask, double p) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> npu_nms_rotated(const at::Tensor& self, const at::Tensor& scores, double iou_threshold, double scores_threshold,
                                                     int64_t max_output_size, int64_t mode) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_masked_fill_range(const at::Tensor& self, const at::Tensor& start, const at::Tensor& end, const at::Tensor& value, int64_t axis) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_sub_sample(const at::Tensor& self, int64_t per_images, double positive_fraction) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_yolo_boxes_encode(const at::Tensor& self, const at::Tensor& gt_bboxes, const at::Tensor& stride, bool performance_mode) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_scatter(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t dim) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_layer_norm_eval(const at::Tensor& input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor>& weight,
                               const c10::optional<at::Tensor>& bias, double eps) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_rotated_box_encode(const at::Tensor& self, const at::Tensor& gt_bboxes, const at::Tensor& weight) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_rotated_box_decode(const at::Tensor& self, const at::Tensor& deltas, const at::Tensor& weight) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_rotated_overlaps(const at::Tensor& self, const at::Tensor& query_boxes, bool trans) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_silu_backward(const at::Tensor& grad_output, const at::Tensor& x0, const at::Tensor& x1) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_rotated_iou(const at::Tensor& self, const at::Tensor& query_boxes, bool trans, int64_t mode, bool is_cross, double v_threshold,
                           double e_threshold) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_nms_with_mask(const at::Tensor& input, const at::Scalar& iou_threshold) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gru_backward(
    const c10::optional<at::Tensor>& grady, const c10::optional<at::Tensor>& gradh, const at::Tensor& input, const at::Tensor& weight_input,
    const at::Tensor& weight_hidden, const at::Tensor& bias_input, const at::Tensor& bias_hidden, const at::Tensor& seq_length, const at::Tensor& hx,
    const at::Tensor& y_output, const at::Tensor& h_output, const at::Tensor& output_updata, const at::Tensor& output_reset, const at::Tensor& output_new,
    const at::Tensor& hidden_new) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_mish_backward(const at::Tensor& grad, const at::Tensor& input) { CUSTOM_OP_NOT_IMPL; }

at::Tensor npu_reshape(const at::Tensor& self, at::IntArrayRef shape, bool can_refresh) { return acl_op::npu_reshape(self, shape, can_refresh); }

at::Tensor& npu_reshape_out(const at::Tensor& self, at::IntArrayRef shape, bool can_refresh, at::Tensor& out) {
    return acl_op::npu_reshape_out(self, shape, can_refresh, out);
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_batch_nms(const at::Tensor& self, const at::Tensor& scores, double score_threshold,
                                                                           double iou_threshold, int64_t max_size_per_class, int64_t max_total_size,
                                                                           bool change_coordinate_frame, bool transpose_box) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_bounding_box_encode(const at::Tensor& anchor_box, const at::Tensor& ground_truth_box, double means0, double means1, double means2, double means3,
                                   double stds0, double stds1, double stds2, double stds3) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_bounding_box_decode(const at::Tensor& rois, const at::Tensor& deltas, double means0, double means1, double means2, double means3, double stds0,
                                   double stds1, double stds2, double stds3, at::IntArrayRef max_shape, double wh_ratio_clip) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_apply_adam(const at::Scalar& beta1_power, const at::Scalar& beta2_power, const at::Scalar& lr,
                                                                const at::Scalar& beta1, const at::Scalar& beta2, const at::Scalar& epsilon,
                                                                const at::Tensor& grad, c10::optional<bool> use_locking, c10::optional<bool> use_nesterov) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_apply_adam_out(const at::Scalar& beta1_power, const at::Scalar& beta2_power, const at::Scalar& lr,
                                                                       const at::Scalar& beta1, const at::Scalar& beta2, const at::Scalar& epsilon,
                                                                       const at::Tensor& grad, c10::optional<bool> use_locking,
                                                                       c10::optional<bool> use_nesterov, at::Tensor& var, at::Tensor& m, at::Tensor& v) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_apply_adam_w(const at::Scalar& beta1_power, const at::Scalar& beta2_power, const at::Scalar& lr,
                                                                  const at::Scalar& weight_decay, const at::Scalar& beta1, const at::Scalar& beta2,
                                                                  const at::Scalar& epsilon, const at::Tensor& grad,
                                                                  const c10::optional<at::Tensor>& max_grad_norm, c10::optional<bool> amsgrad,
                                                                  c10::optional<bool> maximize) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> npu_apply_adam_w_out(const at::Scalar& beta1_power, const at::Scalar& beta2_power, const at::Scalar& lr,
                                                                         const at::Scalar& weight_decay, const at::Scalar& beta1, const at::Scalar& beta2,
                                                                         const at::Scalar& epsilon, const at::Tensor& grad,
                                                                         const c10::optional<at::Tensor>& max_grad_norm, c10::optional<bool> amsgrad,
                                                                         c10::optional<bool> maximize, at::Tensor& var, at::Tensor& m, at::Tensor& v) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_deformable_conv2dbk(const at::Tensor& input, const at::Tensor& grad_output,
                                                                                     const at::Tensor& offset_out, const at::Tensor& weight,
                                                                                     const at::Tensor& offset, at::IntArrayRef kernel_size,
                                                                                     at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
                                                                                     int64_t groups, int64_t deformable_groups, bool modulated) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor> npu_giou_backward(const at::Tensor& grad, const at::Tensor& bboxes, const at::Tensor& gtboxes, bool trans, bool is_cross,
                                                       int64_t mode) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor> npu_diou_backward(const at::Tensor& grad, const at::Tensor& bboxes, const at::Tensor& gtboxes, bool trans, bool is_cross,
                                                       int64_t mode) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_iou(const at::Tensor& bboxes, const at::Tensor& gtboxes, int64_t mode) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> npu_nms_v4(const at::Tensor& self, const at::Tensor& scores, const at::Scalar& max_output_size,
                                                const at::Tensor& iou_threshold, const at::Tensor& scores_threshold, bool pad_to_max_output_size) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_pad(const at::Tensor& input, at::IntArrayRef paddings) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> npu_random_choice_with_mask(const at::Tensor& x, int64_t count, int64_t seed, int64_t seed2) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_normalize_batch(const at::Tensor& self, const at::Tensor& seq_len, int64_t normalize_type) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_ptiou(const at::Tensor& bboxes, const at::Tensor& gtboxes, int64_t mode) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_backward(
    const c10::optional<at::Tensor>& grady, const c10::optional<at::Tensor>& gradh, const c10::optional<at::Tensor>& gradc, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& hx, const at::Tensor& cx, const at::Tensor& y_output, const at::Tensor& h_output,
    const at::Tensor& c_output, const at::Tensor& i, const at::Tensor& j, const at::Tensor& f, const at::Tensor& o, const at::Tensor& tanhc) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor _dropout_with_byte_mask_backward(const at::Tensor& grad_output, const at::Tensor& mask, double p) { CUSTOM_OP_NOT_IMPL; }
at::Tensor dropout_with_byte_mask(const at::Tensor& self, double p, bool train) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> npu_dropout_with_add_softmax_backward(const at::Tensor& grad, const at::Tensor& mask, const at::Tensor& softmax_out,
                                                                           const at::Scalar& alpha, double prob, int64_t dim) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
npu_multi_head_attention_backward(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& query_weight,
                                  const at::Tensor& key_weight, const at::Tensor& value_weight, const at::Tensor& out_proj_weight,
                                  const c10::optional<at::Tensor>& query_bias, const c10::optional<at::Tensor>& key_bias,
                                  const c10::optional<at::Tensor>& value_bias, const c10::optional<at::Tensor>& out_proj_bias, const at::Tensor& query_res,
                                  const at::Tensor& key_res, const at::Tensor& value_res, const at::Tensor& attn_scores, const at::Tensor& attn_res,
                                  const at::Tensor& context, const at::Tensor& y_grad, const at::Tensor& dropout_mask, int64_t attn_head_num,
                                  int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob, bool softmax_use_float) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_dropout_gen_mask(at::IntArrayRef size, double p, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout,
                                c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor> npu_ciou_backward(const at::Tensor& grad, const at::Tensor& bboxes, const at::Tensor& gtboxes,
                                                       const c10::optional<at::Tensor>& atan_sub, bool trans, bool is_cross, int64_t mode) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_sign_bits_unpack(const at::Tensor& input, int64_t size, at::ScalarType dtype) { CUSTOM_OP_NOT_IMPL; }
at::Tensor decode_jpeg(const at::Tensor& self, at::IntArrayRef image_shape, int64_t channels, bool try_recover_truncated) { CUSTOM_OP_NOT_IMPL; }
at::Tensor crop_and_resize(const at::Tensor& self, c10::optional<at::ArrayRef<double>> boxes, at::IntArrayRef box_index, at::IntArrayRef crop_size,
                           double extrapolation_value, c10::string_view method) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor reverse(const at::Tensor& self, at::IntArrayRef axis) { CUSTOM_OP_NOT_IMPL; }
at::Tensor image_normalize(const at::Tensor& self, c10::optional<at::ArrayRef<double>> mean, c10::optional<at::ArrayRef<double>> variance, int64_t dtype) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor& image_normalize_(at::Tensor& self, c10::optional<at::ArrayRef<double>> mean, c10::optional<at::ArrayRef<double>> variance, int64_t dtype) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor img_to_tensor(const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> _conv_depthwise2d_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& weight,
                                                                at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                                                                at::IntArrayRef dilation, ::std::array<bool, 2> output_mask) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> slow_conv_dilated2d_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& weight,
                                                                              at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                                                                              at::IntArrayRef dilation, ::std::array<bool, 3> output_mask) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> slow_conv_transpose2d_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& weight,
                                                                                at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                                                                                at::IntArrayRef output_padding, at::IntArrayRef dilation,
                                                                                ::std::array<bool, 3> output_mask) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_cell_backward(
    const c10::optional<at::Tensor>& grady, const c10::optional<at::Tensor>& gradh, const c10::optional<at::Tensor>& gradc, const at::Tensor& input,
    const at::Tensor& w_ih, const at::Tensor& w_hh, const at::Tensor& h, const at::Tensor& c, const at::Tensor& y_output, const at::Tensor& h_output,
    const at::Tensor& c_output, const at::Tensor& i, const at::Tensor& j, const at::Tensor& f, const at::Tensor& o, const at::Tensor& tanhc) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor> batch_norm_reduce(const at::Tensor& input, double eps) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> batch_norm_gather_stats_update(const at::Tensor& input, const at::Tensor& mean, const at::Tensor& invstd,
                                                                    const c10::optional<at::Tensor>& running_mean, const c10::optional<at::Tensor>& running_var,
                                                                    double momentum, double eps, const at::Tensor& counts) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_backward(const at::Tensor& grad_output, const at::Tensor& softmax_output,
                                                                                    const at::Tensor& query_layer, const at::Tensor& key_layer,
                                                                                    const at::Tensor& value_layer, const at::Tensor& mask,
                                                                                    const at::Scalar& scale, double keep_prob, bool query_transpose,
                                                                                    bool key_transpose, bool value_transpose, bool dx_transpose) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_fwd(const at::Tensor& query_layer, const at::Tensor& key_layer,
                                                                               const at::Tensor& value_layer, const at::Tensor& attention_mask,
                                                                               const at::Scalar& scale, double keep_prob, bool query_transpose,
                                                                               bool key_transpose, bool bmm_score_transpose_a, bool bmm_score_transpose_b,
                                                                               bool value_transpose, bool dx_transpose) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_fused_attention_score_grad(const at::Tensor& grad_output, const at::Tensor& softmax_output,
                                                                                const at::Tensor& query_layer, const at::Tensor& key_layer,
                                                                                const at::Tensor& value_layer, const at::Tensor& mask, const at::Scalar& scale,
                                                                                double keep_prob, bool query_transpose, bool key_transpose,
                                                                                bool value_transpose, bool dx_transpose) {
    CUSTOM_OP_NOT_IMPL;
}
::std::vector<at::Tensor> npu_fused_attention_qkv_grad(const at::Tensor& grad_output_query, const at::Tensor& grad_output_key,
                                                       const at::Tensor& grad_output_value, const at::Tensor& query_kernel, const at::Tensor& key_kernel,
                                                       const at::Tensor& value_kernel, const at::Tensor& hidden_states, const at::Tensor& grad_output_ln) {
    CUSTOM_OP_NOT_IMPL;
}
::std::vector<at::Tensor> npu_fused_attention_layernorm_qkv_fwd(const at::Tensor& x, const at::Tensor& kernel_query, const at::Tensor& kernel_key,
                                                                const at::Tensor& kernel_value, const at::Tensor& gamma, const at::Tensor& beta,
                                                                const c10::optional<at::Tensor>& bias_query, const c10::optional<at::Tensor>& bias_key,
                                                                const c10::optional<at::Tensor>& bias_value, int64_t seq_len, int64_t num_heads, double eps) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_layernorm_grad(const at::Tensor& grad_out, const at::Tensor& input, at::IntArrayRef normalized_shape,
                                                                    const at::Tensor& mean, const at::Tensor& rstd, const c10::optional<at::Tensor>& weight,
                                                                    const c10::optional<at::Tensor>& bias) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor> npu_ifmr(const at::Tensor& data, const at::Tensor& data_min, const at::Tensor& data_max, const at::Tensor& cumsum,
                                              double min_percentile, double max_percentile, double search_start, double search_end, double search_step,
                                              bool with_offset) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_grid_assign_positive(const at::Tensor& self, const at::Tensor& overlaps, const at::Tensor& box_responsible_flags, const at::Tensor& max_overlaps,
                                    const at::Tensor& argmax_overlaps, const at::Tensor& gt_max_overlaps, const at::Tensor& gt_argmax_overlaps, int64_t num_gts,
                                    double pos_iou_thr, double min_pos_iou, bool gt_max_assign_all) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_rotary_mul(const at::Tensor& self, const at::Tensor& r1, const at::Tensor& r2) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rotary_mul_backward(const at::Tensor& grad, const at::Tensor& self, const at::Tensor& r1,
                                                                         const at::Tensor& r2) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_convolution(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
                           at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_convolution_transpose(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias, at::IntArrayRef padding,
                                     at::IntArrayRef output_padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_confusion_transpose(const at::Tensor& self, at::IntArrayRef perm, at::IntArrayRef shape, bool transpose_first) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_ps_roi_pooling(const at::Tensor& self, const at::Tensor& rois, double spatial_scale, int64_t group_size, int64_t output_dim) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_linear(const at::Tensor& input, const at::Tensor& weight, const c10::optional<at::Tensor>& bias) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> _npu_dropout(const at::Tensor& self, double p) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_softmax_cross_entropy_with_logits(const at::Tensor& self, const at::Tensor& labels) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> npu_max(const at::Tensor& self, int64_t dim, bool keepdim) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> npu_max(const at::Tensor& self, at::Dimname dim, bool keepdim) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_bmmV2(const at::Tensor& self, const at::Tensor& mat2, at::IntArrayRef output_sizes) { CUSTOM_OP_NOT_IMPL; }

at::Tensor npu_silu(const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
at::Tensor& npu_silu_(at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gru(const at::Tensor& input, const at::Tensor& hx,
                                                                                             const at::Tensor& weight_input, const at::Tensor& weight_hidden,
                                                                                             const at::Tensor& bias_input, const at::Tensor& bias_hidden,
                                                                                             const at::Tensor& seq_length, bool has_biases, int64_t num_layers,
                                                                                             double dropout, bool train, bool bidirectional, bool batch_first) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_mish(const at::Tensor& self) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> npu_min(const at::Tensor& self, int64_t dim, bool keepdim) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> npu_min(const at::Tensor& self, at::Dimname dim, bool keepdim) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor> npu_deformable_conv2d(const at::Tensor& input, const at::Tensor& weight, const at::Tensor& offset,
                                                           const c10::optional<at::Tensor>& bias, at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                           at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups, int64_t deformable_groups,
                                                           bool modulated) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_giou(const at::Tensor& self, const at::Tensor& gtboxes, bool trans, bool is_cross, int64_t mode) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_diou(const at::Tensor& self, const at::Tensor& gtboxes, bool trans, bool is_cross, int64_t mode) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& seq_mask, const at::Tensor& h, const at::Tensor& c,
    bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_data(
    const at::Tensor& input, const at::Tensor& batch_sizes, const at::Tensor& weight, const at::Tensor& bias, const at::Tensor& seq_mask, const at::Tensor& h,
    const at::Tensor& c, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor> _dropout_with_byte_mask(const at::Tensor& self, double p) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_dropout_with_add_softmax(const at::Tensor& self, const at::Tensor& x1, const at::Scalar& alpha,
                                                                              double prob, int64_t dim) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_scaled_masked_softmax(const at::Tensor& x, const at::Tensor& mask, const at::Scalar& scale, bool fixed_triu_mask) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_multi_head_attention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& query_weight, const at::Tensor& key_weight,
    const at::Tensor& value_weight, const at::Tensor& attn_mask, const at::Tensor& out_proj_weight, const c10::optional<at::Tensor>& query_bias,
    const c10::optional<at::Tensor>& key_bias, const c10::optional<at::Tensor>& value_bias, const c10::optional<at::Tensor>& out_proj_bias,
    const c10::optional<at::Tensor>& dropout_mask, int64_t attn_head_num, int64_t attn_dim_per_head, int64_t src_len, int64_t tgt_len, double dropout_prob,
    bool softmax_use_float) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t> npu_fusion_attention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, int64_t head_num, c10::string_view input_layout,
    const c10::optional<at::Tensor>& pse, const c10::optional<at::Tensor>& padding_mask, const c10::optional<at::Tensor>& atten_mask, double scale,
    double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise, bool gen_mask_parallel, bool sync) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_fusion_attention_grad(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const at::Tensor& dy, int64_t head_num, c10::string_view input_layout,
    const c10::optional<at::Tensor>& pse, const c10::optional<at::Tensor>& padding_mask, const c10::optional<at::Tensor>& atten_mask,
    const c10::optional<at::Tensor>& softmax_max, const c10::optional<at::Tensor>& softmax_sum, const c10::optional<at::Tensor>& softmax_in,
    const c10::optional<at::Tensor>& attention_in, double scale_value, double keep_prob, int64_t pre_tockens, int64_t next_tockens, int64_t inner_precise,
    int64_t seed, int64_t offset, int64_t numels, bool gen_mask_parallel, bool sync) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor> npu_dropout_do_mask(const at::Tensor& self, const at::Tensor& mask, double p) { CUSTOM_OP_NOT_IMPL; }
at::Tensor npu_ciou(const at::Tensor& self, const at::Tensor& gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag) { CUSTOM_OP_NOT_IMPL; }
::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_lstm_cell(
    const at::Tensor& input, const at::Tensor& w_ih, const at::Tensor& w_hh, const at::Tensor& h, const at::Tensor& c, const c10::optional<at::Tensor>& bias) {
    CUSTOM_OP_NOT_IMPL;
}
at::Tensor npu_fused_attention_score(const at::Tensor& query_layer, const at::Tensor& key_layer, const at::Tensor& value_layer,
                                     const at::Tensor& attention_mask, const at::Scalar& scale, double keep_prob, bool query_transpose, bool key_transpose,
                                     bool bmm_score_transpose_a, bool bmm_score_transpose_b, bool value_transpose, bool dx_transpose) {
    CUSTOM_OP_NOT_IMPL;
}
::std::tuple<at::Tensor, at::Tensor> _npu_ciou(const at::Tensor& self, const at::Tensor& gtboxes, bool trans, bool is_cross, int64_t mode, bool atan_sub_flag) {
    CUSTOM_OP_NOT_IMPL;
}

}  // namespace custom_ops

}  // namespace at_npu::native
