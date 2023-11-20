#pragma once

#include <stdio.h>

#include <iostream>
#include <string>

#include "op-plugin/op_plugin/utils/OpConstants.h"
#include"torch_npu/csrc/core/npu/NPUErrorCodes.h"

#include "torch_npu/third_party/acl/inc/acl/acl_base.h"
#include "torch_npu/third_party/acl/inc/acl/acl_rt.h"
#include "torch_npu/third_party/acl/inc/acl/acl.h"
#include "torch_npu/third_party/acl/inc/acl/acl_op_compiler.h"

#define NPUStatus std::string
#define SUCCESS "SUCCESS"
#define INTERNEL_ERROR "INTERNEL_ERROR"
#define PARAM_ERROR "PARAM_ERROR"
#define ALLOC_ERROR "ALLOC_ERROR"
#define FAILED "FAILED"
const int SHAPE_SIZE = 8;

#define __FILENAME__ __FILE__

#define ASCEND_LOGE(fmt, ...) aclAppLog(ACL_ERROR, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__)
#define ASCEND_LOGW(fmt, ...) aclAppLog(ACL_WARNING, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__)
#define ASCEND_LOGI(fmt, ...) aclAppLog(ACL_INFO, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__)
#define ASCEND_LOGD(fmt, ...) aclAppLog(ACL_DEBUG, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__)

#define NPU_LOGE(fmt, ...) printf("[ERROR]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define NPU_LOGW(fmt, ...) printf("[WARN]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define NPU_LOGI(fmt, ...) printf("[INFO]:" #fmt "\n", ##__VA_ARGS__)

#ifdef USE_NPU_LOG
#define NPU_LOGD(fmt, ...) printf("[INFO]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
#else
#define NPU_LOGD(fmt, ...)
#endif

#ifdef _WIN32
#if defined(C10_NPU_BUILD_SHARED_LIBS)
#define C10_NPU_EXPORT __declspec(dllexport)
#define C10_NPU_IMPORT __declspec(dllimport)
#else
#define C10_NPU_EXPORT
#define C10_NPU_IMPORT
#endif
#else  // _WIN32
#if defined(__GNUC__)
#define C10_NPU_EXPORT __attribute__((__visibility__("default")))
#else  // defined(__GNUC__)
#define C10_NPU_EXPORT
#endif  // defined(__GNUC__)
#define C10_NPU_IMPORT C10_NPU_EXPORT
#endif  // _WIN32

// This one is being used by libc10_cuda.so
#ifdef C10_NPU_BUILD_MAIN_LIB
#define C10_NPU_API C10_NPU_EXPORT
#else

#define C10_NPU_API C10_NPU_IMPORT
#endif

#define TORCH_NPU_API C10_NPU_API

#define C10_COMPILE_TIME_MAX_NPUS 16



#define C10_NPU_SHOW_ERR_MSG()                            \
do {                                                      \
  std::cout<<c10_npu::acl::AclGetErrMsg()<<std::endl;    \
} while (0)

#define NPU_CHECK_ERROR(err_code)                                    \
  do {                                                               \
    auto Error = err_code;                                           \
    static c10_npu::acl::AclErrorCode err_map;                       \
    if ((Error) != ACL_ERROR_NONE) {                                 \
      TORCH_CHECK(                                                   \
        false,                                                       \
        __func__,                                                    \
        ":",                                                         \
        __FILE__,                                                    \
        ":",                                                         \
        __LINE__,                                                    \
        " NPU error, error code is ", Error,                         \
        (err_map.error_code_map.find(Error) !=                       \
        err_map.error_code_map.end() ?                               \
        "\n[Error]: " + err_map.error_code_map[Error] : "."),        \
        "\n", c10_npu::acl::AclGetErrMsg());                         \
    }                                                                \
  } while (0)

#define NPU_CHECK_SUPPORTED_OR_ERROR(err_code)                         \
  do {                                                                 \
    auto Error = err_code;                                             \
    static c10_npu::acl::AclErrorCode err_map;                         \
    if ((Error) != ACL_ERROR_NONE) {                                   \
      if ((Error) == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {               \
        static auto feature_not_support_warn_once = []() {             \
          printf("[WARN]%s,%s:%u:%s\n",                                \
                 __FUNCTION__, __FILENAME__, __LINE__,                 \
                 "Feature is not supportted and the possible cause is" \
                 " that driver and firmware packages do not match.");  \
          return true;                                                 \
        }();                                                           \
      } else {                                                         \
        TORCH_CHECK(                                                   \
          false,                                                       \
          __func__,                                                    \
          ":",                                                         \
          __FILE__,                                                    \
          ":",                                                         \
          __LINE__,                                                    \
          " NPU error, error code is ", Error,                         \
          (err_map.error_code_map.find(Error) !=                       \
          err_map.error_code_map.end() ?                               \
          "\n[Error]: " + err_map.error_code_map[Error] : "."),        \
          "\n", c10_npu::acl::AclGetErrMsg());                         \
      }                                                                \
    }                                                                  \
  } while (0)

#define NPU_CHECK_WARN(err_code)                                     \
  do {                                                               \
    auto Error = err_code;                                           \
    static c10_npu::acl::AclErrorCode err_map;                       \
    if ((Error) != ACL_ERROR_NONE) {                                 \
      TORCH_NPU_WARN("NPU warning, error code is ", Error,               \
        "[Error]: ",                                                 \
        (err_map.error_code_map.find(Error) !=                       \
        err_map.error_code_map.end() ?                               \
        "\n[Error]: " + err_map.error_code_map[Error] : "."),        \
        "\n", c10_npu::acl::AclGetErrMsg());                         \
    }                                                                \
  } while (0)

static void warn_(const ::c10::Warning& warning) {

}

#define TORCH_NPU_WARN(...)                                  \
  warn_(::c10::Warning(                                       \
      ::c10::UserWarning(),                                  \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
      ::c10::str(__VA_ARGS__),                               \
      false));

#define TORCH_NPU_WARN_ONCE(...)                                          \
  C10_UNUSED static const auto C10_ANONYMOUS_VARIABLE(TORCH_NPU_WARN_ONCE_) = \
      [&] {                                                               \
        TORCH_NPU_WARN(__VA_ARGS__);                                      \
        return true;                                                      \
      }()

#define INTERFACE_NOT_IMPL std::cout<<__FILE__<<":"<<__LINE__<<":"<<__FUNCTION__<<": not impled yet"<<std::endl;

using std::string;

namespace at_npu {
namespace native {

// get common dtype and shape from op adapter layer
struct UnifiedResult {
    c10::optional<at::ScalarType> common_type = c10::nullopt;
    c10::optional<c10::IntArrayRef> common_shape = c10::nullopt;
    // judge result tensor's dtype is defined or not.
    // if result's dtype is defined, result_type_defined is true and result's dtype remains unchanged.
    bool result_type_defined = false;
};

typedef enum CompileType {
    MEMORY_HOST_COMPILE_DEPENDENT = 1,
    MEMORY_HOST_COMPILE_INDEPENDENT = 2,
} CompileType;

class OpPreparation {
public:
    static UnifiedResult binary_op_check(at::Tensor &out, const at::Tensor &a, const at::Tensor &b, bool check_mem_overlap){INTERFACE_NOT_IMPL}
    static UnifiedResult binary_op_check(at::Tensor &out, const at::Tensor &a, const c10::Scalar b, bool check_mem_overlap) {INTERFACE_NOT_IMPL}
    static UnifiedResult comparison_op_check(at::Tensor &out, const at::Tensor &a, const at::Tensor &b, bool check_mem_overlap) {INTERFACE_NOT_IMPL}
    static UnifiedResult unary_op_check(at::Tensor &out, const at::Tensor &a, bool check_mem_overlap) {INTERFACE_NOT_IMPL}
    static void nullary_op(at::Tensor &out) {INTERFACE_NOT_IMPL}
    static UnifiedResult reduce_op_check(at::Tensor &out, const at::Tensor &a) {INTERFACE_NOT_IMPL}
    static UnifiedResult reduce_op_check(at::Tensor &out1, at::Tensor &out2, const at::Tensor &a) {INTERFACE_NOT_IMPL}
    // From CalcuOpUtil part
    static aclDataType convert_to_acl_data_type(const at::ScalarType &data_type) {INTERFACE_NOT_IMPL}
    static aclDataType convert_to_acl_data_type(const at::ScalarType &data_type, const string &realDataType) {INTERFACE_NOT_IMPL}
    static at::Tensor copy_scalar_to_device(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type) {INTERFACE_NOT_IMPL}
    static at::Tensor copy_tensor_host_to_device(const at::Tensor &cpu_tensor) {INTERFACE_NOT_IMPL}

    static bool is_scalar_wrapped_to_tensor(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    static int64_t get_tensor_npu_format(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    static c10::SmallVector<int64_t, 5> get_tensor_desc_base_sizes(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    // check output tensor
    static void check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst, at::ScalarType expect_dtype, c10::IntArrayRef expect_size) {INTERFACE_NOT_IMPL}
    static void check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst, const at::Tensor &expect_tensor) {INTERFACE_NOT_IMPL}
    static void check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst, c10::IntArrayRef expect_size) {INTERFACE_NOT_IMPL}
    static void check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst, const at::Tensor &expect_tensor, c10::IntArrayRef expect_size) {INTERFACE_NOT_IMPL}
    // check memory overlaps
    static void check_memory(const std::initializer_list<at::Tensor> &inputs, const std::initializer_list<at::Tensor> &outputs) {INTERFACE_NOT_IMPL}
    // cast format
    static at::Tensor cast_to_ori_format(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    static at::Tensor &cast_to_ori_format(at::Tensor &tensor) {INTERFACE_NOT_IMPL}

    static int8_t get_cube_math_type(bool allowHf32) {INTERFACE_NOT_IMPL}

    // used to apply output tensor
    static at::Tensor apply_tensor(const at::Tensor &src) {INTERFACE_NOT_IMPL}
    static at::Tensor apply_tensor(const at::Tensor &src, c10::IntArrayRef sizes) {INTERFACE_NOT_IMPL}
    static at::Tensor apply_tensor(const at::Tensor &src, const c10::TensorOptions &options) {INTERFACE_NOT_IMPL}
    static at::Tensor apply_tensor(c10::IntArrayRef sizes, const c10::TensorOptions &options, const at::Tensor &src) {INTERFACE_NOT_IMPL}
    static at::Tensor apply_tensor_with_format(const at::Tensor &src, int64_t format, bool keep_format = false) {INTERFACE_NOT_IMPL}
    static at::Tensor apply_tensor_with_format(const at::Tensor &src, c10::IntArrayRef sizes, int64_t format, bool keep_format = false) {INTERFACE_NOT_IMPL}
    static at::Tensor apply_tensor_with_format(c10::IntArrayRef sizes, const c10::TensorOptions &options, int64_t format, bool keep_format = false) {INTERFACE_NOT_IMPL}
    static at::Tensor apply_tensor_with_sizes(c10::IntArrayRef sizes, const c10::TensorOptions &options) {INTERFACE_NOT_IMPL}

    // DEPRECATED: CheckOut will be deprecated, please use check_tensor to check output tensor instead.
    static void CheckOut(const std::initializer_list<at::Tensor> &inputs, at::Tensor &output, at::Tensor dst) {INTERFACE_NOT_IMPL}
    static void CheckOut(const std::initializer_list<at::Tensor> &inputs, at::Tensor &output, at::Tensor dst, c10::IntArrayRef shape) {INTERFACE_NOT_IMPL}
    static void CheckOut(const std::initializer_list<at::Tensor> &input, at::Tensor &output, int64_t format, at::ScalarType dtype, c10::IntArrayRef shape) {INTERFACE_NOT_IMPL}
    // DEPRECATED: CastBackToOriFormat will be deprecated, please use cast_to_ori_format instead.
    static at::Tensor CastBackToOriFormat(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    static at::Tensor &CastBackToOriFormat(at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    // DEPRECATED: ApplyTensor will be deprecated, please use apply_tensor instead.
    TORCH_NPU_API static at::Tensor ApplyTensor(const at::Tensor &src) {INTERFACE_NOT_IMPL}
    TORCH_NPU_API static at::Tensor ApplyTensor(const at::Tensor &src, c10::IntArrayRef sizes) {INTERFACE_NOT_IMPL}
    TORCH_NPU_API static at::Tensor ApplyTensor(const at::Tensor &src, const c10::TensorOptions &options) {INTERFACE_NOT_IMPL}
    TORCH_NPU_API static at::Tensor ApplyTensor(c10::IntArrayRef sizes, const c10::TensorOptions &options, const at::Tensor &src) {INTERFACE_NOT_IMPL}
    // DEPRECATED: ApplyTensorWithFormat will be deprecated, please use apply_tensor_with_format instead.
    static at::Tensor ApplyTensorWithFormat(const at::Tensor &src, int64_t format, bool keep_format = false) {INTERFACE_NOT_IMPL}
    static at::Tensor ApplyTensorWithFormat(const at::Tensor &src, c10::IntArrayRef sizes, int64_t format, bool keep_format = false) {INTERFACE_NOT_IMPL}
    static at::Tensor ApplyTensorWithFormat(c10::IntArrayRef sizes, const c10::TensorOptions &options, int64_t format, bool keep_format = false) {INTERFACE_NOT_IMPL}
    static at::Tensor apply_tensor_without_format(const at::Tensor &src) {INTERFACE_NOT_IMPL}
    static at::Tensor apply_tensor_without_format(const at::Tensor &src, c10::IntArrayRef sizes) {INTERFACE_NOT_IMPL}
    static at::Tensor apply_tensor_without_format(c10::IntArrayRef sizes, const c10::TensorOptions &options) {INTERFACE_NOT_IMPL}
    static at::Tensor unsafe_empty_workspace(uint64_t size) {INTERFACE_NOT_IMPL}
    // DEPRECATED: ApplyTensorWithSizes will be deprecated, please use apply_tensor_with_sizes instead.
    static at::Tensor ApplyTensorWithSizes(c10::IntArrayRef sizes, const c10::TensorOptions &options) {INTERFACE_NOT_IMPL}
    // DEPRECATED: CheckMemory will be deprecated, please use check_memory instead.
    static void CheckMemory(const std::initializer_list<at::Tensor> &inputs, const std::initializer_list<at::Tensor> &outputs) {INTERFACE_NOT_IMPL}
    static bool IsCPUScalar(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
};  // namespace OpPreparation

class CalcuOpUtil {
public:
    static aclDataType ConvertToAclDataType(const at::ScalarType &data_type) {INTERFACE_NOT_IMPL}
    static aclDataType ConvertToAclDataType(const at::ScalarType &data_type, const string &realDataType) {INTERFACE_NOT_IMPL}
    static c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    static at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type) {INTERFACE_NOT_IMPL}
    static at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor) {INTERFACE_NOT_IMPL}
    static NPUStatus AclrtMemcpyAsync(const std::pair<at::Tensor, int64_t> &dst, size_t dst_size, const std::pair<at::Tensor, int64_t> &src, size_t src_size,
                                      aclrtMemcpyKind kind) {INTERFACE_NOT_IMPL}
#if 0
  // Add some public interfaces for aclrtmemcpy process,
  // to launch graph in graph mode automatically.
  static aclError
  AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair &dst,
                            size_t dstMax,
                            const StorageAndOffsetMemSizePair &src,
                            size_t count, aclrtMemcpyKind kind);
  static aclError
  AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair &dst,
                            size_t dstMax, const void *src, size_t count,
                            aclrtMemcpyKind kind);
  static aclError
  AclrtMemcpyWithModeSwitch(void *dst, size_t dstMax,
                            const StorageAndOffsetMemSizePair &src,
                            size_t count, aclrtMemcpyKind kind);
  static aclError LaunchAsyncCopyTaskWithModeSwitch(const at::Tensor &dst,
                                                    size_t dstMax,
                                                    const at::Tensor &src,
                                                    size_t count,
                                                    aclrtMemcpyKind kind);
  static aclError LaunchAsyncCopyTaskWithModeSwitch(const c10::StorageImpl &dst,
                                                    size_t dstMax, void *src,
                                                    size_t count,
                                                    aclrtMemcpyKind kind);
#endif
    static void CheckMemoryOverLaps(c10::ArrayRef<at::Tensor> inputs, c10::ArrayRef<at::Tensor> outputs) {INTERFACE_NOT_IMPL}
    static bool IsScalarWrappedToTensor(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    static float GetScalarFloatValue(const c10::Scalar &scalar) {INTERFACE_NOT_IMPL}
    static int64_t GetTensorNpuFormat(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    static c10::SmallVector<int64_t, SHAPE_SIZE> ConvertIntArrayRefToSmallVector(c10::IntArrayRef intArray) {INTERFACE_NOT_IMPL}
    static int8_t GetCubeMathType(bool allowHf32) {INTERFACE_NOT_IMPL}
};  // class CalcuOpUtil

class NpuUtils {
public:
    static bool check_match(const at::Tensor *tensor) {INTERFACE_NOT_IMPL}
    TORCH_NPU_API static at::Tensor format_contiguous(const at::Tensor &src) {INTERFACE_NOT_IMPL}
    static at::Tensor format_contiguous_add_copy_optimize(const at::Tensor &src) {INTERFACE_NOT_IMPL}
    static void RefreshFormat(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    static void format_fresh_view(at::Tensor &x, const at::Tensor &y) {INTERFACE_NOT_IMPL}

    static bool check_5d_5d_match(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}
    static bool IsOomError(aclError ret, int index) {INTERFACE_NOT_IMPL}
    static void check_1d(const at::Tensor &t, const char *arg, const char *fn) {INTERFACE_NOT_IMPL}
}; // class NpuUtils

inline const std::string AclDateTypeToString(aclDataType descDType) {INTERFACE_NOT_IMPL}
inline const std::string AclFormatToString(aclFormat descFormat) {INTERFACE_NOT_IMPL}

class OpCommand {
public:
    TORCH_NPU_API OpCommand() {
        // aclCmds = OpCommandImpls::GetInstanceByTid(std::this_thread::get_id());
        // aclCmds->Push(aclCmd);
        // aclCmd->SetCustomHandler(nullptr);
    }
    TORCH_NPU_API ~OpCommand() {}

    OpCommand(const OpCommand &other) = delete;
    OpCommand(OpCommand &&other) = delete;
    OpCommand &operator=(const OpCommand &) = delete;
    OpCommand &operator=(OpCommand &&) = delete;

    TORCH_NPU_API OpCommand &Name(const string &name) { return *this; }

    OpCommand &Expect(UnifiedResult unified_result) { return *this; }

    // None Input
    TORCH_NPU_API OpCommand &Input() { return *this; }

    // Tensor Input which need contiguous
    TORCH_NPU_API OpCommand &Input(const at::Tensor &input, const string &descName = "", const c10::optional<aclFormat> &sensitive_format = c10::nullopt,
                                   const string &realData = "") {
        return *this;
    }

    // IntArrayRef/SmallVector Input, usually hostmemory input, we will do h2d in launch kernel
    TORCH_NPU_API OpCommand &Input(const c10::IntArrayRef &dimListRef, at::ScalarType toType = at::kLong,
                                   CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT, const string &realDtype = "",
                                   const string &descName = "") {
        return *this;
    }

    // Scalar Input, we will do h2d in launch kernel
    TORCH_NPU_API OpCommand &Input(const c10::Scalar &input, const at::ScalarType type,
                                   CompileType compileType = CompileType::MEMORY_HOST_COMPILE_INDEPENDENT) {
        return *this;
    }

    // ArrayRef Input, usually hostmemory input, we will do h2d in launch kernel
    template <typename T>
    TORCH_NPU_API OpCommand &Input(const c10::ArrayRef<T> &dimListRef, at::IntArrayRef realShape, at::ScalarType toType,
                                   CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT, const string &realDtype = "",
                                   const string &descName = "") {
        return *this;
    }

    // Tensor Input which no need contiguous
    OpCommand &InputWithoutContiguous(const at::Tensor &input, const string &descName = "", const string &realData = "") { return *this; }

    // Output Tensor
    TORCH_NPU_API OpCommand &Output(at::Tensor &output, const string &descName = "", const c10::optional<aclFormat> &sensitive_format = c10::nullopt,
                                    const string &realType = "") {
        return *this;
    }

    // Attr
    template <typename dataType>
    TORCH_NPU_API OpCommand &Attr(const string &name, dataType value) {
        // aclCmd->AddAttr(name, value);
        return *this;
    }

    // Attr depend on condition
    template <typename dataType>
    TORCH_NPU_API OpCommand &Attr(const string &name, dataType value, bool cond) {
        if (!cond) {
            return *this;
        }
        return Attr(name, value);
    }

    // Run a single op
    TORCH_NPU_API void Run() {}

    OpCommand &Sync(c10::SmallVector<int64_t, N> &index) { return *this; }

    OpCommand &Sync() {
        // c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
        // NPU_CHECK_ERROR(c10_npu::acl::AclrtSynchronizeStreamWithTimeout(stream));
        return *this;
    }

};  // class OpCommand

namespace env {

/**
  check if the autotuen is enabled, return true or false.
  */
inline bool AutoTuneEnabled() {INTERFACE_NOT_IMPL}
inline bool CheckBmmV2Enable() {INTERFACE_NOT_IMPL}
inline bool CheckJitDisable() {INTERFACE_NOT_IMPL}
inline bool CheckProfilingEnable() {INTERFACE_NOT_IMPL}
inline bool CheckMmBmmNDDisable() {INTERFACE_NOT_IMPL}
inline bool CheckForbidInternalFormat() {INTERFACE_NOT_IMPL}
inline bool IsAllowFP32ToFP16() {INTERFACE_NOT_IMPL}
inline bool IsAllowConvHF32() {INTERFACE_NOT_IMPL}
inline bool IsAllowMatmulHF32() {INTERFACE_NOT_IMPL}

} // namespace env

}  // namespace native
}  // namespace at_npu


namespace torch_npu {

struct NPUStorageDesc {
public:
    struct use_byte_size_t {};

    c10::SmallVector<int64_t, 5> base_sizes_;
    c10::SmallVector<int64_t, 5> base_strides_;
    c10::SmallVector<int64_t, 5> storage_sizes_;
    int64_t base_offset_ = 0; // no use
    use_byte_size_t base_dtype_ = {}; // no use
    aclFormat origin_format_ = ACL_FORMAT_UNDEFINED;
    aclFormat npu_format_ = ACL_FORMAT_ND;
    // used to make CANN GE tensor from storagImpl
    caffe2::TypeMeta data_type_;
}; // struct NPUStorageDesc

struct NPUStorageImpl : public c10::StorageImpl {
  explicit NPUStorageImpl(use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);
  ~NPUStorageImpl() override = default;

  void release_resources() override;

  // not private
  NPUStorageDesc npu_desc_;

  NPUStorageDesc get_npu_desc() const {
    return npu_desc_;
  }
}; // struct NPUStorageImpl

// NPUTensorImpl class is derived from c10::TensorImpl, and it is only used to handle an NPU tensor.
// Its scope is just to handle an NPUTensor.
class NPUTensorImpl : public c10::TensorImpl {
public:
  explicit NPUTensorImpl(c10::Storage&& storage, const caffe2::TypeMeta& data_type);

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) final {INTERFACE_NOT_IMPL}

  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const final {INTERFACE_NOT_IMPL}
  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const final {INTERFACE_NOT_IMPL}

public:
  NPUTensorImpl(const NPUTensorImpl&) = delete;
  NPUTensorImpl& operator=(const NPUTensorImpl&) = delete;
  NPUTensorImpl(NPUTensorImpl&&) = default;
  NPUTensorImpl& operator=(NPUTensorImpl&&) = default;
  ~NPUTensorImpl() {INTERFACE_NOT_IMPL}
};


class NPUBridge {
public:
  // at::tensor to NPUStorageImpl
  static NPUStorageImpl* GetNpuStorageImpl(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}

  // c10::StorageImpl to NPUStorageImpl
  static NPUStorageImpl* GetNpuStorageImpl(c10::StorageImpl* storageImpl) {INTERFACE_NOT_IMPL}

  // c10::Storage to NPUStorageImpl
  static NPUStorageImpl* GetNpuStorageImpl(c10::Storage&& storage) {INTERFACE_NOT_IMPL}

  // tensor to NPUStorageDesc
  static NPUStorageDesc& GetNpuStorageImplDesc(const at::Tensor &tensor) {INTERFACE_NOT_IMPL}

  // tensor to NPUTensorImpl
  static NPUTensorImpl* GetNpuTensorImpl(const at::Tensor& tensor) {INTERFACE_NOT_IMPL}
}; // class NPUBridge


namespace utils {

inline bool is_npu(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return false;
  }
  return !tensor.device().is_cpu();
}

inline bool is_npu(const at::TensorOptions& options) {
  return !options.device().is_cpu();
}

inline bool is_npu(const at::Device& device) {
  return !device.is_cpu();
}

inline void torch_check_npu(const at::Tensor& tensor) {
  TORCH_CHECK(is_npu(tensor),
              "Expected NPU tensor, please check whether the input tensor device is correct.");
}

inline void torch_check_npu(const at::TensorOptions& options) {
  TORCH_CHECK(is_npu(options),
              "Expected NPU tensor, please check whether the input tensor device is correct.");
}

inline void torch_check_npu(const at::Device& device) {
  TORCH_CHECK(is_npu(device),
              "Expected NPU tensor, please check whether the input tensor device is correct.");
}

inline c10::DeviceType get_npu_device_type() {
  return c10::DeviceType::PrivateUse1;
}

inline void maybe_initialize_npu(const at::TensorOptions& options) {

}

inline void maybe_initialize_npu(const at::Device& device) {

}

inline void maybe_initialize_npu(const c10::optional<at::Device>& device) {

}

} // namespace utils

} // torch_npu


#include "torch_npu/csrc/aten/NPUNativeFunctions.h"