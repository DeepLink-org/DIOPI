/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#ifndef IMPL_ASCEND_ACLNN_ADAPTOR_HPP_
#define IMPL_ASCEND_ACLNN_ADAPTOR_HPP_

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <dlfcn.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../ascend_tensor.hpp"
#include "../common/utils.hpp"
#include "../env_vars.hpp"

namespace impl {
namespace ascend {
namespace aclnn_adaptor {

inline void* getOpApiFuncAddrInLib(void* handler, const char* libName, const char* apiName) {
    void* funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        warning(__FILE__, __LINE__, __FUNCTION__, "dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void* getOpApiLibHandler(const char* libName) {
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        warning(__FILE__, __LINE__, __FUNCTION__, "dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

inline void* getOpApiFuncAddr(const char* apiName) {
    constexpr const char kOpApiLibName[] = "libopapi.so";
    static void* opApiHandler = getOpApiLibHandler(kOpApiLibName);
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return getOpApiFuncAddrInLib(opApiHandler, kOpApiLibName, apiName);
}

inline aclTensor* createAclTensorFromAscendTensor(const AscendTensor& input) {
    const auto& shape = input.shape();
    const auto& stride = input.stride();
    const auto storageSize = static_cast<int64_t>(input.storageNbytes() / input.elemsize());
    return ::aclCreateTensor(shape.data(),
                             shape.size(),
                             input.getAclDataType(),
                             stride.data(),
                             input.storageOffset(),
                             input.getAclDataFormat(),  // TODO(lljbash): op_plugin assume non-channel-last, why?
                             &storageSize,
                             /*storageDimsNum=*/1,
                             const_cast<void*>(input.data()));
}

inline aclTensor* createAclTensorFromDiopiTensor(diopiConstTensorHandle_t tensor) {
    ASCEND_CHECK_NULLPTR_ABORT(tensor);
    diopiSize_t shape{};
    diopiGetTensorShape(tensor, &shape);
    diopiSize_t stride{};
    diopiGetTensorStride(tensor, &stride);
    ASCEND_CHECK_ABORT(shape.len == stride.len, "shape.len != stride.len");
    diopiDtype_t dtype{};
    diopiGetTensorDtype(tensor, &dtype);
    int64_t elemsize{};
    diopiGetTensorElemSize(tensor, &elemsize);
    int64_t storageOffset{};
    diopiGetTensorStorageOffset(tensor, &storageOffset);
    std::size_t storageNbytes{};
    diopiGetTensorStorageNbytes(tensor, &storageNbytes);
    const void* tensorData = nullptr;
    diopiGetTensorDataConst(tensor, &tensorData);
    auto type = diopiDtypeToAclDataType(dtype);
    auto format = inferAclDataFormat(shape.len, shape.data, stride.data);
    auto storageSize = static_cast<int64_t>(storageNbytes / elemsize);
    return ::aclCreateTensor(shape.data,
                             shape.len,
                             type,
                             stride.data,
                             storageOffset,
                             format,
                             &storageSize,
                             /*storageDimsNum=*/1,
                             const_cast<void*>(tensorData));
}

inline aclScalar* createAclScalarFromDiopiScalar(const diopiScalar_t* scalar) {
    auto [bytes, nbytes] = getScalarBytes(scalar);
    return ::aclCreateScalar(bytes.data(), diopiDtypeToAclDataType(scalar->stype));
}

inline aclIntArray* createAclIntArrayFromDiopiSize(const diopiSize_t size) { return ::aclCreateIntArray(size.data, size.len); }

template <class T, class U = std::remove_cv_t<std::remove_reference_t<T>>>
decltype(auto) convertType(T&& param) {
    if constexpr (std::is_same_v<U, AscendTensor>) {
        return createAclTensorFromAscendTensor(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, diopiTensorHandle_t> || std::is_same_v<U, diopiConstTensorHandle_t>) {
        return createAclTensorFromDiopiTensor(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, diopiScalar_t*> || std::is_same_v<U, const diopiScalar_t*>) {
        return createAclScalarFromDiopiScalar(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, diopiSize_t> || std::is_same_v<U, const diopiSize_t>) {
        return createAclIntArrayFromDiopiSize(std::forward<T>(param));
    } else {
        static_assert(!std::is_class_v<U> && !std::is_pointer_v<U>);
        return std::forward<T>(param);
    }
}

template <class T, class U = std::remove_reference_t<T>, std::enable_if_t<!std::is_class_v<U> && !std::is_pointer_v<U>, int> = 0>
void releaseConverted(T&& param [[maybe_unused]]) {}  // no conversion, do nothing

#define IMPL_ASCEND_ACLNN_REGISTER_DESTRUCTOR(Type)        \
    inline void releaseConverted(const acl##Type* param) { \
        if (param != nullptr) {                            \
            ::aclDestroy##Type(param);                     \
        }                                                  \
    }
IMPL_ASCEND_ACLNN_REGISTER_DESTRUCTOR(Tensor)
IMPL_ASCEND_ACLNN_REGISTER_DESTRUCTOR(Scalar)
IMPL_ASCEND_ACLNN_REGISTER_DESTRUCTOR(TensorList)
IMPL_ASCEND_ACLNN_REGISTER_DESTRUCTOR(ScalarList)
IMPL_ASCEND_ACLNN_REGISTER_DESTRUCTOR(IntArray)
IMPL_ASCEND_ACLNN_REGISTER_DESTRUCTOR(BoolArray)
IMPL_ASCEND_ACLNN_REGISTER_DESTRUCTOR(FloatArray)
#undef IMPL_ASCEND_ACLNN_REGISTER_DESTRUCTOR

// A class to hold the converted parameters and release them when the object is destroyed.
template <class Tuple>
class ConvertedParamsHolder final {
public:
    explicit ConvertedParamsHolder(Tuple&& params) noexcept : convertedParams_(std::forward<Tuple>(params)) {}
    ~ConvertedParamsHolder() {
        std::apply([](const auto&... params) { (releaseConverted(params), ...); }, convertedParams_);
    }
    ConvertedParamsHolder(const ConvertedParamsHolder&) = delete;
    ConvertedParamsHolder& operator=(const ConvertedParamsHolder&) = delete;
    ConvertedParamsHolder(ConvertedParamsHolder&&) = delete;
    ConvertedParamsHolder& operator=(ConvertedParamsHolder&&) = delete;
    const auto& params() const noexcept { return convertedParams_; }

private:
    Tuple convertedParams_;
};

template <class Tuple>
ConvertedParamsHolder(Tuple&&) -> ConvertedParamsHolder<std::remove_reference_t<Tuple>>;

template <class... Args>
constexpr auto convertParams(const Args&... args) {
    return ConvertedParamsHolder(std::make_tuple(convertType(args)...));
}

typedef int (*InitHugeMemThreadLocal)(void*, bool);
typedef void (*UnInitHugeMemThreadLocal)(void*, bool);
typedef void (*ReleaseHugeMem)(void*, bool);

class AclHugeMem final {
public:
    explicit AclHugeMem(InitHugeMemThreadLocal initFunc, UnInitHugeMemThreadLocal unInitFunc, ReleaseHugeMem releaseFunc)
        : initMemFunc(initFunc), unInitMemFunc(unInitFunc), releaseMemFunc(releaseFunc) {
        if (initMemFunc) {
            initMemFunc(nullptr, false);
        }
    }

    ~AclHugeMem() {
        if (releaseMemFunc) {
            releaseMemFunc(nullptr, false);
        }
        if (unInitMemFunc) {
            unInitMemFunc(nullptr, false);
        }
    }

private:
    InitHugeMemThreadLocal initMemFunc = nullptr;
    UnInitHugeMemThreadLocal unInitMemFunc = nullptr;
    ReleaseHugeMem releaseMemFunc = nullptr;
};

// A class to alloc acl workspace and release it when the object is destroyed.
class AclWorkspace final {
public:
    explicit AclWorkspace(diopiContextHandle_t ctx, std::size_t workspaceSize) noexcept {
        if (workspaceSize > 0) {
            diopiTensorHandle_t bufHandle;
            auto ret = diopiRequireBuffer(ctx, &bufHandle, workspaceSize, diopi_device);
            ASCEND_CHECK(ret == diopiSuccess, "[AclWorkspace] Require workspace size %lld failed.", static_cast<uint64_t>(workspaceSize));
            AscendTensor buf(bufHandle);
            workspaceAddr_ = const_cast<void*>(buf.data());
        }
    }
    ~AclWorkspace() {}
    AclWorkspace(const AclWorkspace&) = delete;
    AclWorkspace& operator=(const AclWorkspace&) = delete;
    AclWorkspace(AclWorkspace&&) = delete;
    AclWorkspace& operator=(AclWorkspace&&) = delete;
    void* addr() const noexcept { return workspaceAddr_; }

private:
    void* workspaceAddr_ = nullptr;
};

template <const char* api, const char* workspaceApi, class... Args>
void callAclnnImpl(diopiContextHandle_t ctx, const Args&... args) {
    if (isDebugAclOpRunnerOn()) {
        std::cout << "ACLNN_ADAPTOR for " << api << '\n';
    }

    /* 0. get aclrtStream */
    aclrtStream stream = nullptr;
    diopiGetStream(ctx, &stream);

    /* 1. call xxxGetWorkspaceSize function. */
    static const auto workspaceSizeFuncAddr = getOpApiFuncAddr(workspaceApi);
    ASCEND_CHECK(workspaceSizeFuncAddr != nullptr, "can't get workSpaceName function.");
    using WorkspaceSizeFuncType = int (*)(std::decay_t<decltype(convertType(std::declval<Args>()))>..., uint64_t*, aclOpExecutor**);
    static const auto workspaceSizeFunc = reinterpret_cast<WorkspaceSizeFuncType>(workspaceSizeFuncAddr);

    static const auto initFunc = reinterpret_cast<InitHugeMemThreadLocal>(getOpApiFuncAddr("InitHugeMemThreadLocal"));
    static const auto unInitFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(getOpApiFuncAddr("UnInitHugeMemThreadLocal"));
    static const auto releaseFunc = reinterpret_cast<ReleaseHugeMem>(getOpApiFuncAddr("ReleaseHugeMem"));
    AclHugeMem aclHugeMem(initFunc, unInitFunc, releaseFunc);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto convertedParams = convertParams(args...);
    auto workspaceStatus = std::apply(workspaceSizeFunc, std::tuple_cat(convertedParams.params(), std::make_tuple(&workspaceSize, &executor)));
    ASCEND_CHECK(workspaceStatus == ACL_SUCCESS, "workspaceStatus not equal ACL_SUCCESS.");

    AclWorkspace workspace(ctx, workspaceSize);

    /* 2. call aclnnXXX function */
    static const auto opApiFuncAddr = getOpApiFuncAddr(api);
    ASCEND_CHECK(opApiFuncAddr != nullptr, "can't get op function.");
    using OpApiFuncType = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    static const auto opApiFunc = reinterpret_cast<OpApiFuncType>(opApiFuncAddr);

    auto ret = opApiFunc(workspace.addr(), workspaceSize, executor, stream);
    ASCEND_CHECK(ret == ACL_SUCCESS, "%s failed. ERROR: %d\n", api, ret);
}

#define DIOPI_ASCEND_CALL_ACLNN(api, ctx, ...)                                                       \
    do {                                                                                             \
        static constexpr const char kApiName[] = #api;                                               \
        static constexpr const char kWorkspaceApiName[] = #api "GetWorkspaceSize";                   \
        ::impl::ascend::aclnn_adaptor::callAclnnImpl<kApiName, kWorkspaceApiName>(ctx, __VA_ARGS__); \
    } while (false)

}  // namespace aclnn_adaptor

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ACLNN_ADAPTOR_HPP_
