#pragma once

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <stdio.h>

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_base.h"
#include "acl/acl_op_compiler.h"
#include "acl/acl_rt.h"
#include "ge/ge_api.h"
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/utils/OpConstants.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUErrorCodes.h"
#include "torch_npu/csrc/core/npu/NpuVariables.h"
#include "torch_npu/csrc/framework/interface/AclOpCompileInterface.h"

#define NPUStatus std::string
#define SUCCESS "SUCCESS"
#define INTERNEL_ERROR "INTERNEL_ERROR"
#define PARAM_ERROR "PARAM_ERROR"
#define ALLOC_ERROR "ALLOC_ERROR"
#define FAILED "FAILED"

#define __FILENAME__ __FILE__

#define ASCEND_LOGE(fmt, ...) aclAppLog(ACL_ERROR, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__)
#define ASCEND_LOGW(fmt, ...) aclAppLog(ACL_WARNING, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__)
#define ASCEND_LOGI(fmt, ...) aclAppLog(ACL_INFO, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__)
#define ASCEND_LOGD(fmt, ...) aclAppLog(ACL_DEBUG, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__)

#define NPU_LOGE(fmt, ...) printf("[ERROR]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define NPU_LOGW(fmt, ...) printf("[WARN]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
#define NPU_LOGI(fmt, ...) printf("[INFO]:" #fmt "\n", ##__VA_ARGS__)

#if defined(NDEBUG)
#define NPU_LOGD(fmt, ...)
#else
#define NPU_LOGD(fmt, ...) printf("[INFO]%s,%s:%u:" #fmt "\n", __FUNCTION__, __FILENAME__, __LINE__, ##__VA_ARGS__)
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

#define C10_NPU_SHOW_ERR_MSG()                                  \
    do {                                                        \
        std::cout << c10_npu::acl::AclGetErrMsg() << std::endl; \
    } while (0)

#define NPU_CHECK_ERROR(err_code)                                                                                                                   \
    do {                                                                                                                                            \
        auto Error = err_code;                                                                                                                      \
        static c10_npu::acl::AclErrorCode err_map;                                                                                                  \
        if ((Error) != ACL_ERROR_NONE) {                                                                                                            \
            TORCH_CHECK(false,                                                                                                                      \
                        __func__,                                                                                                                   \
                        ":",                                                                                                                        \
                        __FILE__,                                                                                                                   \
                        ":",                                                                                                                        \
                        __LINE__,                                                                                                                   \
                        " NPU error, error code is ",                                                                                               \
                        Error,                                                                                                                      \
                        (err_map.error_code_map.find(Error) != err_map.error_code_map.end() ? "\n[Error]: " + err_map.error_code_map[Error] : "."), \
                        "\n",                                                                                                                       \
                        c10_npu::acl::AclGetErrMsg());                                                                                              \
        }                                                                                                                                           \
    } while (0)

#define NPU_CHECK_SUPPORTED_OR_ERROR(err_code)                                                                                                          \
    do {                                                                                                                                                \
        auto Error = err_code;                                                                                                                          \
        static c10_npu::acl::AclErrorCode err_map;                                                                                                      \
        if ((Error) != ACL_ERROR_NONE) {                                                                                                                \
            if ((Error) == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {                                                                                          \
                static auto feature_not_support_warn_once = []() {                                                                                      \
                    printf("[WARN]%s,%s:%u:%s\n",                                                                                                       \
                           __FUNCTION__,                                                                                                                \
                           __FILENAME__,                                                                                                                \
                           __LINE__,                                                                                                                    \
                           "Feature is not supportted and the possible cause is"                                                                        \
                           " that driver and firmware packages do not match.");                                                                         \
                    return true;                                                                                                                        \
                }();                                                                                                                                    \
            } else {                                                                                                                                    \
                TORCH_CHECK(false,                                                                                                                      \
                            __func__,                                                                                                                   \
                            ":",                                                                                                                        \
                            __FILE__,                                                                                                                   \
                            ":",                                                                                                                        \
                            __LINE__,                                                                                                                   \
                            " NPU error, error code is ",                                                                                               \
                            Error,                                                                                                                      \
                            (err_map.error_code_map.find(Error) != err_map.error_code_map.end() ? "\n[Error]: " + err_map.error_code_map[Error] : "."), \
                            "\n",                                                                                                                       \
                            c10_npu::acl::AclGetErrMsg());                                                                                              \
            }                                                                                                                                           \
        }                                                                                                                                               \
    } while (0)

#define NPU_CHECK_WARN(err_code)                                                                                                                       \
    do {                                                                                                                                               \
        auto Error = err_code;                                                                                                                         \
        static c10_npu::acl::AclErrorCode err_map;                                                                                                     \
        if ((Error) != ACL_ERROR_NONE) {                                                                                                               \
            TORCH_NPU_WARN("NPU warning, error code is ",                                                                                              \
                           Error,                                                                                                                      \
                           "[Error]: ",                                                                                                                \
                           (err_map.error_code_map.find(Error) != err_map.error_code_map.end() ? "\n[Error]: " + err_map.error_code_map[Error] : "."), \
                           "\n",                                                                                                                       \
                           c10_npu::acl::AclGetErrMsg());                                                                                              \
        }                                                                                                                                              \
    } while (0)

#define INTERFACE_NOT_IMPL
// #define INTERFACE_NOT_IMPL std::cout << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << ": not impled yet" << std::endl;

static void warn_(const ::c10::Warning& warning) { INTERFACE_NOT_IMPL; }

#define TORCH_NPU_WARN(...) warn_(::c10::Warning(::c10::UserWarning(), {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, ::c10::str(__VA_ARGS__), false));

#define TORCH_NPU_WARN_ONCE(...)                                                      \
    C10_UNUSED static const auto C10_ANONYMOUS_VARIABLE(TORCH_NPU_WARN_ONCE_) = [&] { \
        TORCH_NPU_WARN(__VA_ARGS__);                                                  \
        return true;                                                                  \
    }()

#define RECORD_FUNCTION(...) \
    {}

namespace at_npu {
namespace key {
static constexpr c10::DeviceType NativeDeviceType = c10::DeviceType::XLA;
static constexpr c10::DispatchKey NativeDispatchKey = c10::DispatchKey::XLA;
static constexpr c10::DispatchKey NativeAutogradDispatchKey = c10::DispatchKey::AutogradXLA;
static constexpr c10::Backend NativeBackend = c10::Backend::XLA;
static const std::string npu_device_str("npu");      // NOLINT
static const std::string default_device_str("xla");  // NOLINT

static bool isDeviceTensor(const at::Tensor& tensor) { return !tensor.is_cpu(); }

}  // namespace key

// Stores state values. Passed as a kernel argument. See "Usage:" above.
struct PhiloxNpuState {
    PhiloxNpuState() = default;
    PhiloxNpuState(const PhiloxNpuState&) = default;
    // Called if graph capture is not underway
    PhiloxNpuState(uint64_t seed, uint64_t offset) {
        seed_ = seed;
        offset_.val = offset;
    }
    // Called if graph capture is underway
    PhiloxNpuState(uint64_t seed, int64_t* offset_extragraph, uint32_t offset_intragraph) {
        seed_ = seed;
        offset_.ptr = offset_extragraph;
        offset_intragraph_ = offset_intragraph;
        captured_ = true;
    }
    // Public members, directly accessible by at::Npu::philox::unpack.
    // If we made them private with getters/setters, the getters/setters
    // would have to be __device__, and we can't declare __device__ in ATen.
    union Payload {
        uint64_t val;
        int64_t* ptr;
    };

    uint64_t seed_;
    Payload offset_;
    uint32_t offset_intragraph_{0};
    bool captured_ = false;
};

struct NPUGeneratorImpl : public c10::GeneratorImpl {
    // Constructors
    explicit NPUGeneratorImpl(c10::DeviceIndex device_index = -1);
    ~NPUGeneratorImpl() = default;

    // NPUGeneratorImpl methods
    std::shared_ptr<NPUGeneratorImpl> clone() const { INTERFACE_NOT_IMPL; }
    virtual GeneratorImpl* clone_impl() const { INTERFACE_NOT_IMPL; }

    void set_current_seed(uint64_t seed) { INTERFACE_NOT_IMPL; }
    uint64_t current_seed() const { INTERFACE_NOT_IMPL; }
    uint64_t seed() { INTERFACE_NOT_IMPL; }
    void set_state(const c10::TensorImpl& new_state) { INTERFACE_NOT_IMPL; }
    c10::intrusive_ptr<c10::TensorImpl> get_state() const { INTERFACE_NOT_IMPL; }
    void set_philox_offset_per_thread(uint64_t offset) { INTERFACE_NOT_IMPL; }
    uint64_t philox_offset_per_thread() const { INTERFACE_NOT_IMPL; }
    void capture_prologue(int64_t* offset_extragraph) { INTERFACE_NOT_IMPL; }
    uint64_t capture_epilogue() { INTERFACE_NOT_IMPL; }
    PhiloxNpuState philox_npu_state(uint64_t increment) { INTERFACE_NOT_IMPL; }

    // Temporarily accommodates call sites that use philox_engine_inputs.
    // Allows incremental refactor of call sites to use philox_npu_state.
    std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
    static c10::DeviceType device_type() { return c10::DeviceType::XLA; }
    void* generator_ = nullptr;
};

namespace detail {

const at::Generator& getDefaultNPUGenerator(c10::DeviceIndex device_index = -1);

}  // namespace detail

}  // namespace at_npu

namespace c10_npu {

namespace acl {

const char* AclGetErrMsg();
static aclError AclrtSynchronizeStreamWithTimeout(aclrtStream stream) { NPU_CHECK_ERROR(aclrtSynchronizeStream(stream)); }

}  // namespace acl

class NPUStream {
public:
    enum Unchecked { UNCHECKED };

    explicit NPUStream(c10::Stream stream) : stream_(stream) {
        // TORCH_CHECK(stream_.device_type() == at_npu::key::NativeDeviceType);
    }

    explicit NPUStream(Unchecked, c10::Stream stream) : stream_(stream) {}

    explicit NPUStream(Unchecked, c10::Stream stream, aclrtStream aclStream) : stream_(stream), aclStream_(aclStream) {}

    ~NPUStream() {}

    bool operator==(const NPUStream& other) const noexcept { return unwrap() == other.unwrap(); }

    bool operator!=(const NPUStream& other) const noexcept { return unwrap() != other.unwrap(); }

    /// Implicit conversion to rtStream_t.
    operator aclrtStream() const { return stream(); }

    /// Implicit conversion to pytorch Stream.
    operator c10::Stream() const { return unwrap(); }

    /// Used to avoid baking in device type explicitly to Python-side API.
    c10::DeviceType device_type() const { INTERFACE_NOT_IMPL; }

    /// Get the NPU device index that this stream is associated with.
    c10::DeviceIndex device_index() const { return stream_.device_index(); }

    /// Get the full Device that this stream is associated with.  The Device
    /// is guaranteed to be a NPU device.
    c10::Device device() const { INTERFACE_NOT_IMPL; }

    c10::StreamId id() const { INTERFACE_NOT_IMPL }

    bool query() const { INTERFACE_NOT_IMPL; }

    void synchronize() const;

    /// Explicit conversion to rtStream_t.
    C10_NPU_API aclrtStream stream() const { return aclStream_; }

    /// Explicit conversion to Stream.
    c10::Stream unwrap() const { return stream_; }

    /// The NPUStream can be unpacked using unpack().
    struct c10::StreamData3 pack3() const { return stream_.pack3(); }

    // Unpack a NPUStream from the 3 fields generated by pack().
    static NPUStream unpack3(c10::StreamId stream_id, c10::DeviceIndex device_index, c10::DeviceType device_type) {
        return NPUStream(c10::Stream::unpack3(stream_id, device_index, device_type));
    }

    /// Explicit conversion to rtStream_t， with out empty taskQ.
    aclrtStream stream(const bool need_empty) const { INTERFACE_NOT_IMPL; }

private:
    c10::Stream stream_;
    aclrtStream aclStream_ = nullptr;
};
#if 0
NPUStream getNPUStreamFromPool(c10::DeviceIndex device = -1);

NPUStream getDefaultNPUStream(c10::DeviceIndex device_index = -1);



NPUStream getCurrentSecondaryStream(c10::DeviceIndex device_index = -1);

aclrtStream getCurrentNPUStreamNoWait(c10::DeviceIndex device_index = -1);

NPUStatus emptyAllNPUStream();

C10_NPU_API bool npuSynchronizeDevice(bool check_error = true);


void setCurrentNPUStream(NPUStream stream);

std::ostream& operator<<(std::ostream& stream, const NPUStream& s);
#endif

C10_NPU_API NPUStream getCurrentNPUStream(c10::DeviceIndex device_index = -1);

C10_NPU_API NPUStream getCurrentSecondaryStream(c10::DeviceIndex device_index = -1);

struct SecondaryStreamGuard {
    SecondaryStreamGuard() = delete;
    explicit SecondaryStreamGuard(c10::Stream stream) {}
    ~SecondaryStreamGuard() {}
};

namespace NPUCachingAllocator {

static void recordStream(const c10::DataPtr& ptr, c10_npu::NPUStream stream) { INTERFACE_NOT_IMPL; }

}  // namespace NPUCachingAllocator

/// A variant of OptionalDeviceGuard that is specialized for NPU.  See
/// NPUGuard for when you can use this.
struct OptionalNPUGuard {
    /// Create an uninitialized OptionalNPUGuard.
    OptionalNPUGuard() {}

    /// Set the current NPU device to the passed Device, if it is not nullopt.
    explicit OptionalNPUGuard(c10::optional<c10::Device> device_opt) {}

    /// Set the current NPU device to the passed device index, if it is not
    /// nullopt
    explicit OptionalNPUGuard(c10::optional<c10::DeviceIndex> device_index_opt) {}

    void set_device(c10::Device device) {}
};

namespace queue {

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind);

}  // namespace queue

}  // namespace c10_npu

using std::string;
using std::vector;

namespace torch_npu {

struct NPUStorageDesc {
public:
    struct use_byte_size_t {};

    c10::SmallVector<int64_t, 5> base_sizes_;
    c10::SmallVector<int64_t, 5> base_strides_;
    c10::SmallVector<int64_t, 5> storage_sizes_;
    int64_t base_offset_ = 0;          // no use
    use_byte_size_t base_dtype_ = {};  // no use
    aclFormat origin_format_ = ACL_FORMAT_UNDEFINED;
    aclFormat npu_format_ = ACL_FORMAT_ND;
    // used to make CANN GE tensor from storagImpl
    caffe2::TypeMeta data_type_;
};  // struct NPUStorageDesc

struct NPUStorageImpl : public c10::StorageImpl {
    explicit NPUStorageImpl(use_byte_size_t use_byte_size, size_t size_bytes, at::DataPtr data_ptr, at::Allocator* allocator, bool resizable);
    ~NPUStorageImpl() override = default;

    void release_resources() override;

    // not private
    NPUStorageDesc npu_desc_;

    NPUStorageDesc get_npu_desc() const { return npu_desc_; }
};  // struct NPUStorageImpl

// NPUTensorImpl class is derived from c10::TensorImpl, and it is only used to handle an NPU tensor.
// Its scope is just to handle an NPUTensor.
class NPUTensorImpl : public c10::TensorImpl {
public:
    explicit NPUTensorImpl(c10::Storage&& storage, const caffe2::TypeMeta& data_type);

    void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) final { INTERFACE_NOT_IMPL; }

    c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(const c10::VariableVersion& version_counter, bool allow_tensor_metadata_change) const final {
        INTERFACE_NOT_IMPL;
    }
    /**
     * Return a TensorImpl that is a shallow-copy of this TensorImpl.
     *
     * For usage of `version_counter` and `allow_tensor_metadata_change`,
     * see NOTE [ TensorImpl Shallow-Copying ].
     */
    c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(c10::VariableVersion&& version_counter, bool allow_tensor_metadata_change) const final {
        INTERFACE_NOT_IMPL;
    }

public:
    NPUTensorImpl(const NPUTensorImpl&) = delete;
    NPUTensorImpl& operator=(const NPUTensorImpl&) = delete;
    NPUTensorImpl(NPUTensorImpl&&) = default;
    NPUTensorImpl& operator=(NPUTensorImpl&&) = default;
    ~NPUTensorImpl() { INTERFACE_NOT_IMPL; }
};

class NPUBridge {
public:
    // at::tensor to NPUStorageImpl
    static NPUStorageImpl* GetNpuStorageImpl(const at::Tensor& tensor);

    // c10::StorageImpl to NPUStorageImpl
    static NPUStorageImpl* GetNpuStorageImpl(c10::StorageImpl* storageImpl);

    // c10::Storage to NPUStorageImpl
    static NPUStorageImpl* GetNpuStorageImpl(c10::Storage&& storage);

    // tensor to NPUStorageDesc
    static NPUStorageDesc& GetNpuStorageImplDesc(const at::Tensor& tensor);

    // tensor to NPUTensorImpl
    static NPUTensorImpl* GetNpuTensorImpl(const at::Tensor& tensor);
};  // class NPUBridge

namespace utils {

inline bool is_npu(const at::Tensor& tensor) {
    if (!tensor.defined()) {
        return false;
    }
    return !tensor.device().is_cpu();
}

inline bool is_npu(const at::TensorOptions& options) { return !options.device().is_cpu(); }

inline bool is_npu(const at::Device& device) { return !device.is_cpu(); }

inline void torch_check_npu(const at::Tensor& tensor) {
    TORCH_CHECK(is_npu(tensor), "Expected NPU tensor, please check whether the input tensor device is correct.");
}

inline void torch_check_npu(const at::TensorOptions& options) {
    TORCH_CHECK(is_npu(options), "Expected NPU tensor, please check whether the input tensor device is correct.");
}

inline void torch_check_npu(const at::Device& device) {
    TORCH_CHECK(is_npu(device), "Expected NPU tensor, please check whether the input tensor device is correct.");
}

inline c10::DeviceType get_npu_device_type() { return c10::DeviceType::PrivateUse1; }

inline void maybe_initialize_npu(const at::TensorOptions& options) {}

inline void maybe_initialize_npu(const at::Device& device) {}

inline void maybe_initialize_npu(const c10::optional<at::Device>& device) {}

}  // namespace utils

}  // namespace torch_npu

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

// smallvector max size
const int N = 32;
// npu tensor max size
const int SHAPE_SIZE = 8;
// HALF_MAX and HALF_MIN of NPU support
const int NPU_HALF_MAX = 65504;
const int NPU_HALF_MIN = -65504;
const int NPU_MAX_OP_EXEC_TRY_NUM = 2;

typedef enum CompileType {
    MEMORY_HOST_COMPILE_DEPENDENT = 1,
    MEMORY_HOST_COMPILE_INDEPENDENT = 2,
} CompileType;

class OpPreparation {
public:
    static UnifiedResult binary_op_check(at::Tensor& out, const at::Tensor& a, const at::Tensor& b, bool check_mem_overlap);
    static UnifiedResult binary_op_check(at::Tensor& out, const at::Tensor& a, const c10::Scalar b, bool check_mem_overlap);
    static UnifiedResult comparison_op_check(at::Tensor& out, const at::Tensor& a, const at::Tensor& b, bool check_mem_overlap) { INTERFACE_NOT_IMPL; }
    static UnifiedResult unary_op_check(at::Tensor& out, const at::Tensor& a, bool check_mem_overlap) { INTERFACE_NOT_IMPL; }
    static void nullary_op(at::Tensor& out) { INTERFACE_NOT_IMPL; }
    static UnifiedResult reduce_op_check(at::Tensor& out, const at::Tensor& a) { INTERFACE_NOT_IMPL; }
    static UnifiedResult reduce_op_check(at::Tensor& out1, at::Tensor& out2, const at::Tensor& a) { INTERFACE_NOT_IMPL; }
    // From CalcuOpUtil part
    static aclDataType convert_to_acl_data_type(const at::ScalarType& data_type) { INTERFACE_NOT_IMPL; }
    static aclDataType convert_to_acl_data_type(const at::ScalarType& data_type, const string& realDataType) { INTERFACE_NOT_IMPL; }
    static at::Tensor copy_scalar_to_device(const c10::Scalar& cpu_scalar, at::ScalarType scalar_data_type) { INTERFACE_NOT_IMPL; }
    static at::Tensor copy_tensor_host_to_device(const at::Tensor& cpu_tensor) { INTERFACE_NOT_IMPL; }

    static bool is_scalar_wrapped_to_tensor(const at::Tensor& tensor) { INTERFACE_NOT_IMPL; }
    static int64_t get_tensor_npu_format(const at::Tensor& tensor) { INTERFACE_NOT_IMPL; }
    static c10::SmallVector<int64_t, 5> get_tensor_desc_base_sizes(const at::Tensor& tensor);
    // check output tensor
    static void check_tensor(const std::initializer_list<at::Tensor>& src_list, at::Tensor& dst, at::ScalarType expect_dtype, c10::IntArrayRef expect_size) {
        INTERFACE_NOT_IMPL
    }
    static void check_tensor(const std::initializer_list<at::Tensor>& src_list, at::Tensor& dst, const at::Tensor& expect_tensor) { INTERFACE_NOT_IMPL; }
    static void check_tensor(const std::initializer_list<at::Tensor>& src_list, at::Tensor& dst, c10::IntArrayRef expect_size) { INTERFACE_NOT_IMPL; }
    static void check_tensor(const std::initializer_list<at::Tensor>& src_list, at::Tensor& dst, const at::Tensor& expect_tensor,
                             c10::IntArrayRef expect_size) {
        INTERFACE_NOT_IMPL
    }
    // check memory overlaps
    static void check_memory(const std::initializer_list<at::Tensor>& inputs, const std::initializer_list<at::Tensor>& outputs) { INTERFACE_NOT_IMPL; }

    // cast format
    static at::Tensor cast_to_ori_format(const at::Tensor& tensor) { INTERFACE_NOT_IMPL; }
    static at::Tensor& cast_to_ori_format(at::Tensor& tensor) { INTERFACE_NOT_IMPL; }

    static int8_t get_cube_math_type(bool allowHf32) { INTERFACE_NOT_IMPL; }
    static void markAsOutputForApplyTensor(at::Tensor& src);
    // used to apply output tensor
    static at::Tensor apply_tensor(const at::Tensor& src);
    static at::Tensor apply_tensor(const at::Tensor& src, c10::IntArrayRef sizes);
    static at::Tensor apply_tensor(const at::Tensor& src, const c10::TensorOptions& options);
    static at::Tensor apply_tensor(c10::IntArrayRef sizes, const c10::TensorOptions& options, const at::Tensor& src);
    static at::Tensor apply_tensor_with_format(const at::Tensor& src, int64_t format, bool keep_format = false);
    static at::Tensor apply_tensor_with_format(const at::Tensor& src, c10::IntArrayRef sizes, int64_t format, bool keep_format = false);
    static at::Tensor apply_tensor_with_format(c10::IntArrayRef sizes, const c10::TensorOptions& options, int64_t format, bool keep_format = false);
    static at::Tensor apply_tensor_with_sizes(c10::IntArrayRef sizes, const c10::TensorOptions& options);
    static at::Tensor apply_tensor_without_format(const at::Tensor& src) { INTERFACE_NOT_IMPL; }
    static at::Tensor apply_tensor_without_format(const at::Tensor& src, c10::IntArrayRef sizes) { INTERFACE_NOT_IMPL; }
    static at::Tensor apply_tensor_without_format(c10::IntArrayRef sizes, const c10::TensorOptions& options) { INTERFACE_NOT_IMPL; }

    // DEPRECATED: CheckOut will be deprecated, please use check_tensor to check output tensor instead.
    static void CheckOut(const std::initializer_list<at::Tensor>& inputs, at::Tensor& output, at::Tensor dst);
    static void CheckOut(const std::initializer_list<at::Tensor>& inputs, at::Tensor& output, at::Tensor dst, c10::IntArrayRef shape);
    static void CheckOut(const std::initializer_list<at::Tensor>& input, at::Tensor& output, int64_t format, at::ScalarType dtype, c10::IntArrayRef shape);
    // DEPRECATED: CastBackToOriFormat will be deprecated, please use cast_to_ori_format instead.
    static at::Tensor CastBackToOriFormat(const at::Tensor& tensor);
    static at::Tensor& CastBackToOriFormat(at::Tensor& tensor);
    // DEPRECATED: ApplyTensor will be deprecated, please use apply_tensor instead.
    TORCH_NPU_API static at::Tensor ApplyTensor(const at::Tensor& src) { return apply_tensor(src); }
    TORCH_NPU_API static at::Tensor ApplyTensor(const at::Tensor& src, c10::IntArrayRef sizes) { return apply_tensor(src, sizes); }
    TORCH_NPU_API static at::Tensor ApplyTensor(const at::Tensor& src, const c10::TensorOptions& options) { return apply_tensor(src, options); }
    TORCH_NPU_API static at::Tensor ApplyTensor(c10::IntArrayRef sizes, const c10::TensorOptions& options, const at::Tensor& src) {
        return apply_tensor(sizes, options, src);
    }
    // DEPRECATED: ApplyTensorWithFormat will be deprecated, please use apply_tensor_with_format instead.
    static at::Tensor ApplyTensorWithFormat(const at::Tensor& src, int64_t format, bool keep_format = false) {
        return apply_tensor_with_format(src, format, keep_format);
    }
    static at::Tensor ApplyTensorWithFormat(const at::Tensor& src, c10::IntArrayRef sizes, int64_t format, bool keep_format = false) {
        return apply_tensor_with_format(src, sizes, format, keep_format);
    }
    static at::Tensor ApplyTensorWithFormat(c10::IntArrayRef sizes, const c10::TensorOptions& options, int64_t format, bool keep_format = false) {
        return apply_tensor_with_format(sizes, options, format, keep_format);
    }

    static at::Tensor unsafe_empty_workspace(uint64_t size) { INTERFACE_NOT_IMPL; }
    // DEPRECATED: ApplyTensorWithSizes will be deprecated, please use apply_tensor_with_sizes instead.
    static at::Tensor ApplyTensorWithSizes(c10::IntArrayRef sizes, const c10::TensorOptions& options) { return apply_tensor_with_sizes(sizes, options); }
    // DEPRECATED: CheckMemory will be deprecated, please use check_memory instead.
    static void CheckMemory(const std::initializer_list<at::Tensor>& inputs, const std::initializer_list<at::Tensor>& outputs) { INTERFACE_NOT_IMPL; }
    static bool IsCPUScalar(const at::Tensor& tensor) { return tensor.is_cpu() && tensor.numel() == 1; }
};  // namespace OpPreparation

using StorageAndOffsetMemSizePair = std::pair<const c10::StorageImpl*, int64_t>;
class CalcuOpUtil {
public:
    static aclDataType ConvertToAclDataType(const at::ScalarType& data_type);
    static aclDataType ConvertToAclDataType(const at::ScalarType& data_type, const string& realDataType);
    static c10::Scalar ConvertTensorToScalar(const at::Tensor& tensor);
    static at::Tensor CopyScalarToDevice(const c10::Scalar& cpu_scalar, at::ScalarType scalar_data_type);
    static at::Tensor CopyTensorHostToDevice(const at::Tensor& cpu_tensor);
    static NPUStatus AclrtMemcpyAsync(const std::pair<at::Tensor, int64_t>& dst, size_t dst_size, const std::pair<at::Tensor, int64_t>& src, size_t src_size,
                                      aclrtMemcpyKind kind);
    // Add some public interfaces for aclrtmemcpy process,
    // to launch graph in graph mode automatically.
    static aclError AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair& dst, size_t dstMax, const StorageAndOffsetMemSizePair& src, size_t count,
                                              aclrtMemcpyKind kind);
    static aclError AclrtMemcpyWithModeSwitch(const StorageAndOffsetMemSizePair& dst, size_t dstMax, const void* src, size_t count, aclrtMemcpyKind kind);
    static aclError AclrtMemcpyWithModeSwitch(void* dst, size_t dstMax, const StorageAndOffsetMemSizePair& src, size_t count, aclrtMemcpyKind kind);

    static aclError LaunchAsyncCopyTaskWithModeSwitch(const c10::StorageImpl& dst, size_t dstMax, void* src, size_t count, aclrtMemcpyKind kind);
    static aclError LaunchAsyncCopyTaskWithModeSwitch(const at::Tensor& dst, size_t dstMax, const at::Tensor& src, size_t count, aclrtMemcpyKind kind);
    static void CheckMemoryOverLaps(c10::ArrayRef<at::Tensor> inputs, c10::ArrayRef<at::Tensor> outputs);
    static bool IsScalarWrappedToTensor(const at::Tensor& tensor) { return tensor.is_cpu() && tensor.numel() == 1; }
    static float GetScalarFloatValue(const c10::Scalar& scalar);
    static int64_t GetTensorNpuFormat(const at::Tensor& tensor);
    static c10::SmallVector<int64_t, SHAPE_SIZE> ConvertIntArrayRefToSmallVector(c10::IntArrayRef intArray);
    static int8_t GetCubeMathType(bool allowHf32);
};  // class CalcuOpUtil

// Max size of discontiguous cases vector
constexpr int MAX_CASES = 8;
// Max size of shape size
constexpr int MAX_DIM = 5;

// Define the discontiguous cases vector to be optimized
using OptimizationCases = c10::SmallVector<std::string, MAX_CASES>;

struct ContiguousTensorDesc {
    bool is_contiguous_;
    c10::SmallVector<int64_t, MAX_DIM> sizes_;
    c10::SmallVector<int64_t, MAX_DIM> strides_;
    int64_t offset_;
    c10::SmallVector<int64_t, MAX_DIM> base_sizes_;
    c10::SmallVector<int64_t, MAX_DIM> base_strides_;
    c10::SmallVector<int64_t, MAX_DIM> storage_sizes_;
    int64_t base_offset_;
    aclFormat npu_format_;
    OptimizationCases opt_cases_;
    void refresh_contiguous_using_size_and_stride();
    void reset_optimization_cases(const OptimizationCases& opt_cases);
    void add_optimization_case(const std::string& opt_case);
    void find_match_optimization_cases();
};

class TransContiguous {
public:
    TransContiguous() {}
    virtual ~TransContiguous() {}
    static bool CheckClone(const at::Tensor& src, at::Tensor& self);
    static ContiguousTensorDesc GetTensorDescInfo(const at::Tensor& src, const OptimizationCases& opt_cases = optCasesDefault);
    static bool can_optimize_(ContiguousTensorDesc& tensor_desc);
    static bool CanOptimize(ContiguousTensorDesc& tensor_desc);
    static bool CanOptimize(const at::Tensor& tensor, const OptimizationCases& opt_cases);
    static bool contiguous_optimize_with_anyformat_(at::Tensor& self, const at::Tensor& src, ContiguousTensorDesc& src_desc);
    static bool ContiguousOptimizeWithAnyFormat(at::Tensor& self, const at::Tensor& src, const OptimizationCases& opt_cases = optCasesAnyFormat);
    static c10::optional<at::Tensor> ContiguousOptimizeWithAnyFormat(const at::Tensor& src, const OptimizationCases& opt_cases = optCasesAnyFormat);
    static bool ContiguousOptimizeWithBaseFormat(at::Tensor& self, const at::Tensor& src, const OptimizationCases& opt_cases = optCasesDefault,
                                                 bool OpenCombined = true);

private:
    static OptimizationCases optCasesDefault;
    static OptimizationCases optCasesAnyFormat;
};

class ContiguousOpt {
public:
    ContiguousOpt() {}
    virtual ~ContiguousOpt() = default;
    virtual bool Optimizer(at::Tensor& self, const at::Tensor& src, const ContiguousTensorDesc& src_desc) = 0;
    virtual bool CanOptimizer(const ContiguousTensorDesc& src_desc) { return false; }
};

namespace register_opt {
class CopyOptRegister {
public:
    ~CopyOptRegister() = default;
    static CopyOptRegister* GetInstance() {
        static CopyOptRegister instance;
        return &instance;
    }
    void Register(std::string& name, ::std::unique_ptr<ContiguousOpt>& ptr) {
        std::lock_guard<std::mutex> lock(mu_);
        registry.emplace(name, std::move(ptr));
    }

    bool CanOptimize(std::string& name, const ContiguousTensorDesc& src_desc) {
        auto itr = registry.find(name);
        if (itr != registry.end()) {
            return itr->second->CanOptimizer(src_desc);
        }
        return false;
    }

    bool Run(const std::string& name, at::Tensor& self, const at::Tensor& src, const ContiguousTensorDesc& src_desc) {
        auto itr = registry.find(name);
        if (itr != registry.end()) {
            return itr->second->Optimizer(self, src, src_desc);
        }
        return false;
    }

private:
    CopyOptRegister() {}
    mutable std::mutex mu_;
    mutable std::map<std::string, ::std::unique_ptr<ContiguousOpt>> registry;
};  // class CopyOptRegister

class CopyOptBuilder {
public:
    CopyOptBuilder(std::string name, ::std::unique_ptr<ContiguousOpt>& ptr) { CopyOptRegister::GetInstance()->Register(name, ptr); }
    ~CopyOptBuilder() = default;
};  // class CopyOptBuilder
}  // namespace register_opt

#define REGISTER_COPY_OPT(name, optimization) REGISTER_COPY_OPT_UNIQ(name, name, optimization)
#define REGISTER_COPY_OPT_UNIQ(id, name, optimization)                         \
    auto copy_opt_##id = ::std::unique_ptr<ContiguousOpt>(new optimization()); \
    static register_opt::CopyOptBuilder register_copy_opt##id(#name, copy_opt_##id);

class FormatCastHelper {
public:
    static bool IsSameGroupType(const at::Tensor& src, const at::Tensor& dst);
    static void format_cast_as_base_format(const at::Tensor& src, aclFormat format);
    using FormatCastFunc = std::function<at::Tensor(at::Tensor&, const at::Tensor&)>;
    static bool format_cast_between_group(at::Tensor& dst, const at::Tensor& src, FormatCastFunc format_cast_inside_group);
    // this interface is similar to CastBackToOriFormat, but CastBackToOriFormat may have overload problem.
    static at::Tensor ApplyBaseFormatTensorBy(const at::Tensor& src);
    static at::Tensor& CovertSelfToBaseFormat(at::Tensor& src);

private:
    // help function of format_cast_between_group
    static void base_format_cast_nocheck(at::Tensor& dst, const at::Tensor& src);
};  // class FormatCastHelper

class NpuUtils {
public:
    static bool check_match(const at::Tensor* tensor);
    TORCH_NPU_API static at::Tensor format_contiguous(const at::Tensor& src);
    static at::Tensor format_contiguous_add_copy_optimize(const at::Tensor& src);
    static void RefreshFormat(const at::Tensor& tensor);
    static void format_fresh_view(at::Tensor& x, const at::Tensor& y);

    static bool check_5d_5d_match(const at::Tensor& tensor);
    static bool IsOomError(aclError ret, int index);
    static void check_1d(const at::Tensor& t, const char* arg, const char* fn);
    static void ProfReportMarkData(const std::string& msg);
    static void ProfReportMarkDataToNpuProfiler(uint32_t category, const std::string& data, uint64_t correlation_id = 0);
    static void ProfReportMarkDataToNpuProfiler(uint32_t category, void* data, size_t offset);
};  // class NpuUtils

inline const std::string AclDateTypeToString(aclDataType descDType) { INTERFACE_NOT_IMPL; }
inline const std::string AclFormatToString(aclFormat descFormat) { INTERFACE_NOT_IMPL; }

using PROC_FUNC = std::function<int()>;
// in npu device, the max shape size is 8
constexpr int MAX_FORMAT_SHAPE_SIZE = 8;
using FormatShape = c10::SmallVector<int64_t, MAX_FORMAT_SHAPE_SIZE>;

using DyNumAndIndex = std::vector<std::pair<uint32_t, uint32_t>>;
using DynamicInputRegFunc = std::function<ge::OperatorPtr(DyNumAndIndex, std::string)>;

using baseFormatConverter = std::function<FormatShape(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims)>;

class OpCommand {
    class OpCommandImpl* aclCmd = nullptr;
    c10::SmallVector<at::Tensor, N> storage;
    c10::optional<at::ScalarType> commonType = c10::nullopt;
    c10::optional<c10::IntArrayRef> commonShape = c10::nullopt;
    bool resultTypeDefined = false;
    bool sync = false;
    c10::SmallVector<int64_t, N> sync_index;
    c10::SmallVector<at::Tensor, N> outputTensor;
    c10::SmallVector<at::Tensor, N> inputTensor;

public:
    TORCH_NPU_API OpCommand();
    TORCH_NPU_API ~OpCommand();

    OpCommand(const OpCommand& other) = delete;
    OpCommand(OpCommand&& other) = delete;
    OpCommand& operator=(const OpCommand&) = delete;
    OpCommand& operator=(OpCommand&&) = delete;

    TORCH_NPU_API OpCommand& Name(const string& name);
    void SetCustomHandler(PROC_FUNC func);

    OpCommand& DynamicInputReg(DynamicInputRegFunc func, DyNumAndIndex num_and_index) { return *this; }

    OpCommand& Expect(UnifiedResult unified_result);
    // None Input
    TORCH_NPU_API OpCommand& Input();

    // Tensor Input which need contiguous
    TORCH_NPU_API OpCommand& Input(const at::Tensor& input, const string& descName = "", const c10::optional<aclFormat>& sensitive_format = c10::nullopt,
                                   const string& realData = "");

    // IntArrayRef/SmallVector Input, usually hostmemory input, we will do h2d in launch kernel
    TORCH_NPU_API OpCommand& Input(const c10::IntArrayRef& dimListRef, at::ScalarType toType = at::kLong,
                                   CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT, const string& realDtype = "",
                                   const string& descName = "");

    // Scalar Input, we will do h2d in launch kernel
    TORCH_NPU_API OpCommand& Input(const c10::Scalar& input, const at::ScalarType type, CompileType compileType = CompileType::MEMORY_HOST_COMPILE_INDEPENDENT);

    // ArrayRef Input, usually hostmemory input, we will do h2d in launch kernel
    template <typename T>
    TORCH_NPU_API OpCommand& Input(const c10::ArrayRef<T>& dimListRef, at::IntArrayRef realShape, at::ScalarType toType,
                                   CompileType compileType = CompileType::MEMORY_HOST_COMPILE_DEPENDENT, const string& realDtype = "",
                                   const string& descName = "");

    // Tensor Input which no need contiguous
    OpCommand& InputWithoutContiguous(const at::Tensor& input, const string& descName = "", const string& realData = "");
    // Output Tensor
    TORCH_NPU_API OpCommand& Output(at::Tensor& output, const string& descName = "", const c10::optional<aclFormat>& sensitive_format = c10::nullopt,
                                    const string& realType = "");
    // Attr
    template <typename dataType>
    TORCH_NPU_API OpCommand& Attr(const string& name, dataType value);

    // Attr depend on condition
    template <typename dataType>
    TORCH_NPU_API OpCommand& Attr(const string& name, dataType value, bool cond) {
        if (!cond) {
            return *this;
        }
        return Attr(name, value);
    }

    // Run a single op
    TORCH_NPU_API void Run();

    OpCommand& Sync(c10::SmallVector<int64_t, N>& index);

    OpCommand& Sync();

private:
    OpCommand& AddTensorInput(at::Tensor& tensor, at::ScalarType forceScaleType = at::ScalarType::Undefined, const string& descName = "",
                              const string& realData = "");
    at::Tensor& Contiguous(const at::Tensor& input);
};  // class OpCommand

namespace env {

/**
  check if the autotuen is enabled, return true or false.
  */
inline bool AutoTuneEnabled() {
    INTERFACE_NOT_IMPL;
    return false;
}
inline bool CheckBmmV2Enable() {
    INTERFACE_NOT_IMPL;
    return false;
}
inline bool CheckJitDisable() {
    INTERFACE_NOT_IMPL;
    return true;
}
inline bool CheckProfilingEnable() {
    INTERFACE_NOT_IMPL;
    return false;
}
inline bool CheckMmBmmNDDisable() {
    INTERFACE_NOT_IMPL;
    return true;
}
inline bool CheckForbidInternalFormat() {
    INTERFACE_NOT_IMPL;
    return true;
}
inline bool IsAllowFP32ToFP16() {
    INTERFACE_NOT_IMPL;
    return false;
}
inline bool IsAllowConvHF32() {
    INTERFACE_NOT_IMPL;
    return false;
}
inline bool IsAllowMatmulHF32() {
    INTERFACE_NOT_IMPL;
    return false;
}

}  // namespace env

// helper function of storage format
class FormatHelper {
public:
    // helper function of copy, because of padding will change the physical size.
    static bool IsPadded(const at::Tensor* tensor);
    static char* GetFormatName(const at::Tensor& tensor);
    static aclFormat GetBaseFormat(const at::Tensor& tensor);
    static aclFormat GetBaseFormat(aclFormat format);
    static aclFormat GetFormat(const at::Tensor& tensor);

    static bool IsBaseFormatType(aclFormat format);
    static bool IsBaseFormatType(const at::Tensor& tensor);

    // Default assumption: the original format are ND, NCHW or NDHWC.
    // So, if original size are 4D, it maybe NCHW or ND and so on.
    // The format can be split into two parts:
    // 1. The storage size can be infered between NC1HWC0, NHWC, NC1HWC0_C04, NCHW.
    // 2. The storage size can be infered between NDC1HWC0 and NDHWC/NCDHW.
    // The storage size can not be infered between different groups.
    template <typename sizeType>
    static FormatShape GetStorageSizes(aclFormat format, sizeType ori_size);
    // GetStorageSizes used to calculate the storage sizes of op at npu device at different format.
    static FormatShape GetStorageSizes(const torch_npu::NPUStorageDesc& desc);
    static at::Tensor& unsafe_format_cast(at::Tensor& self, int64_t self_format, int64_t result_format);

    static bool IsOpInputBaseFormat(const at::Tensor& tensor);
    static bool IsOpInputBaseFormat(const c10::optional<at::Tensor>& tensor);
    static bool IsOpInputBaseFormat(const c10::List<c10::optional<at::Tensor>>& tensors);
    static bool IsOpInputBaseFormat(const at::TensorList& tensors);
    static bool IsOpInputBaseFormat(const at::ITensorListRef& tensors);

private:
    static bool IsPadded(aclFormat format);
    static char* GetFormatName(aclFormat format);

private:
    using shapeInfer = std::function<FormatShape(c10::IntArrayRef dims)>;
    typedef struct FormatInfo_ {
        aclFormat format = ACL_FORMAT_ND;
        aclFormat baseFormat = ACL_FORMAT_ND;
        shapeInfer func = nullptr;
        char formatName[30] = {0};
        bool isPadded = false;
    } FormatInfo;
    static std::unordered_map<aclFormat, FormatInfo> info;
};  // class FormatHelper

// template impl
template <typename sizeType>
FormatShape FormatHelper::GetStorageSizes(aclFormat format, sizeType ori_size) {
    auto itr = info.find(format);
    if (itr != info.end()) {
        if (itr->second.func) {
            return itr->second.func(ori_size);
        }
    }
    AT_ERROR("unsupport InferShape with format ", GetFormatName(format), "with shape", ori_size);
    return {};
}

// Format is the property of tensor storage. Format is the way to tell an
// operator how the result should be organized in memory and nothing more.
// Storage format collect the helper functions of npu's format. It tell the
// relationship between format and storage.
//
class InferFormat {
public:
    // Feature: The function is used to guess base format
    // The base formats are NCHW, NCDHW, ND, who is not padding.
    // The format transform between other formats should be based
    // on these base formats.(their should convert to base format first.)
    // This function will be called at new, reset, set and so on.
    static std::tuple<aclFormat, aclFormat> GuessFormatUnit(const c10::IntArrayRef& size, aclFormat format);
    // GuessBaseFormat is the base of the format assumption
    // this function is called when apply the new tensor
    static aclFormat GuessBaseFormat(const c10::IntArrayRef& size);
    // this function used to fix format when format and size is not match
    static aclFormat GuessStorageFormat(const c10::IntArrayRef& size, aclFormat format);
    // Features: guess the format of tensor after it called format_contiguous().
    // According to the law of continuity, the output format is same as input format,
    // this function is called to guess the input format, so it also the output format.
    // NOTE: The caller should make sure that the tensor is non-contigous
    static aclFormat GuessFormatWhenContiguous(const at::Tensor& tensor);
    // This api is used to infer storage size when called transdata
    // fix: ND->NZ when dim < 2
    // not effect the storage data.
    static FormatShape GuessStorageSizeWhenConvertFormat(const at::Tensor& tensor);
    // This api is used to judge if tensor is reasonable when size changes.
    // solution: tranform to base format to fix it.
    // fix: NCHW | 5HD -> NCDHW | NCDHW or ND | ND
    // unsqueeze/squeeze/select/flatten/view will change meta data, they will call
    // as_strided and view
    static bool IsDefiniteTensorWhenMetaDataChanges(const at::Tensor& tensor, const c10::IntArrayRef& size);
};  // class InferFormat

class StorageDescHelper {
public:
    // Get Part
    // sizes, strides in StorageDesc are same as those in MetaData
    static bool MetaDataAreMatch(const at::Tensor* tensor);
    // storage offset are match, the npu only support offset == 0
    static inline bool OffsetAreMatch(const at::Tensor* tensor) { return tensor->storage_offset() == 0; }

    // helper function of transdata op.
    static bool IsSameDesc(const torch_npu::NPUStorageDesc& a, const torch_npu::NPUStorageDesc& b);
    static bool IsSameDesc(const at::Tensor& a, const at::Tensor& b);

    // calculate storage size need by npu memory
    static int64_t GetMemorySize(const at::Tensor& dst);
    static int64_t GetMemorySize(const c10::IntArrayRef& size, aclFormat format);
    // Calculate the valid memory size of the tensor, because of view operator and so on.
    static int64_t GetValidMemorySize(const at::Tensor& tensor);

    // Set Part
    // StorageDesc Init/Set
    static void SetDesc(at::Tensor& dst);
    static void SetDesc(at::Tensor& dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides);
    static void SetDesc(at::Tensor& dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides, aclFormat format);
    static bool CheckDescInit(const c10::Storage& storage);

    // For Serialization to Get and Set NpuStorageDesc
    static void GetDescForSerialization(const at::Tensor& dst, std::unordered_map<std::string, bool>& desc_map);
    static void SetDescForSerialization(const at::Tensor& dst, std::unordered_map<std::string, bool>& desc_map);

    static void CopyDesc(at::Tensor& dst, const at::Tensor& src);
    static void CopyDesc(at::Tensor& dst, const c10::Storage& src);
    static void CopyDesc(const at::Tensor& dst, const torch_npu::NPUStorageDesc& src_desc);

    static void UpdateDesc(torch_npu::NPUStorageDesc& npuDesc, const c10::IntArrayRef& new_data_sizes, const c10::IntArrayRef& new_shape_sizes);
    static FormatShape ComputeStrideFromShape(const FormatShape& shape);

    // need to remove later
    static void ReflushDescBySelf(const at::Tensor& src);

private:
    // Get Part
    static bool IsSameSize(const c10::SmallVector<int64_t, 5>& a, const c10::IntArrayRef& b);
    static int64_t GetMemorySize(const torch_npu::NPUStorageDesc& dst);
    // Set Part
    static torch_npu::NPUStorageDesc SetDesc(const caffe2::TypeMeta& dtype);
    static torch_npu::NPUStorageDesc SetDesc(const caffe2::TypeMeta& dtype, const c10::IntArrayRef& size, const c10::IntArrayRef& strides);
    static torch_npu::NPUStorageDesc SetDesc(const caffe2::TypeMeta& dtype, const c10::IntArrayRef& size, const c10::IntArrayRef& strides, aclFormat format);
};  // class StorageDescHelper

bool can_use_memcpy(at::Tensor& dst, const at::Tensor& src);
void copy_d2d_by_memcpy(at::Tensor& dst, const at::Tensor& src, int64_t exceptSize = 0);
void copy_d2d_dtype(at::Tensor& self, const at::Tensor& src, bool non_blocking);
void copy_d2d_dtype_baseformat(at::Tensor& self, const at::Tensor& src, bool non_blocking);
bool try_to_optimize_copy_with_any_format(at::Tensor& self, const at::Tensor& src);
static at::Tensor matmul_by_bmmV2(const at::Tensor& tensor1, const at::Tensor& tensor2) { INTERFACE_NOT_IMPL; }
void npu_fast_reshape_(at::Tensor& tensor);

at::Tensor empty_npu(at::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt = c10::nullopt,
                     c10::optional<at::Device> device_opt = c10::nullopt, c10::optional<bool> pin_memory_opt = c10::nullopt,
                     c10::optional<at::MemoryFormat> memory_format_opt = c10::nullopt);

at::Tensor empty_npu(at::IntArrayRef size, const at::TensorOptions& options);

at::Tensor empty_strided_npu(c10::SymIntArrayRef size, c10::SymIntArrayRef stride, c10::optional<at::ScalarType> dtype,
                             c10::optional<at::Layout> layout = c10::nullopt, c10::optional<at::Device> device = c10::nullopt,
                             c10::optional<bool> pin_memory = c10::nullopt);

at::Tensor empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device,
                             c10::optional<bool> pin_memory, int64_t acl_format);

at::Tensor clone(const at::Tensor& self, c10::optional<at::MemoryFormat> memory_format);

}  // namespace native

}  // namespace at_npu

inline aclError THNPUCachingHostAllocator_recordEvent(void* ptr, c10_npu::NPUStream stream) { return ACL_SUCCESS; }
