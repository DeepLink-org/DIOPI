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
#include <cassert>
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
    // The Ascend kernel can handle the case that the input tensor is nullptr.
    if (tensor == nullptr) {
        return nullptr;
    }
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
    return ::aclCreateTensor(shape.data, shape.len, type, stride.data, storageOffset, format, shape.data, shape.len, const_cast<void*>(tensorData));
}

inline aclScalar* createAclScalarFromDiopiScalar(const diopiScalar_t* scalar) {
    auto [bytes, nbytes] = getScalarBytes(scalar);
    return ::aclCreateScalar(bytes.data(), diopiDtypeToAclDataType(scalar->stype));
}

inline aclIntArray* createAclIntArrayFromDiopiSize(const diopiSize_t size) { return ::aclCreateIntArray(size.data, size.len); }

template <size_t N>
inline aclBoolArray* createAclBoolArrayFromVector(const std::array<bool, N>& vec) {
    return ::aclCreateBoolArray(vec.data(), vec.size());
}

template <typename T>
struct IsBoolStdArray : std::false_type {};

template <std::size_t N>
struct IsBoolStdArray<std::array<bool, N>> : std::true_type {};

inline aclIntArray* createAclIntArrayFromIntVector(const std::vector<int64_t>& vec) { return ::aclCreateIntArray(vec.data(), vec.size()); }

inline aclTensorList* createAclTensorListFromDiopiTensorVector(const std::vector<diopiTensorHandle_t>& tensorsVec) {
    std::vector<const aclTensor*> tensorList(tensorsVec.size());
    for (size_t i = 0; i < tensorsVec.size(); i++) {
        tensorList[i] = createAclTensorFromDiopiTensor(tensorsVec[i]);
    }
    return ::aclCreateTensorList(tensorList.data(), tensorList.size());
}

inline aclTensorList* createAclTensorListFromConstDiopiTensorVector(const std::vector<diopiConstTensorHandle_t>& tensorsVec) {
    std::vector<const aclTensor*> tensorList(tensorsVec.size());
    for (size_t i = 0; i < tensorsVec.size(); i++) {
        tensorList[i] = createAclTensorFromDiopiTensor(tensorsVec[i]);
    }
    return ::aclCreateTensorList(tensorList.data(), tensorList.size());
}

template <class T, class U = std::remove_cv_t<std::remove_reference_t<T>>>
decltype(auto) convertType(T&& param) {
    if constexpr (std::is_same_v<U, AscendTensor>) {
        return createAclTensorFromAscendTensor(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, diopiTensorHandle_t> || std::is_same_v<U, diopiConstTensorHandle_t>) {
        return createAclTensorFromDiopiTensor(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, std::vector<diopiConstTensorHandle_t>>) {
        return createAclTensorListFromConstDiopiTensorVector(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, std::vector<diopiTensorHandle_t>>) {
        return createAclTensorListFromDiopiTensorVector(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, diopiScalar_t*> || std::is_same_v<U, const diopiScalar_t*>) {
        return createAclScalarFromDiopiScalar(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, diopiSize_t>) {
        return createAclIntArrayFromDiopiSize(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, std::vector<int64_t>>) {
        return createAclIntArrayFromIntVector(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, diopiDtype_t>) {
        return diopiDtypeToAclDataType(std::forward<T>(param));
    } else if constexpr (IsBoolStdArray<U>::value) {
        return createAclBoolArrayFromVector<std::tuple_size_v<U>>(std::forward<T>(param));
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
        : initMemFunc_(initFunc), unInitMemFunc_(unInitFunc), releaseMemFunc_(releaseFunc) {
        if (initMemFunc_) {
            initMemFunc_(nullptr, false);
        }
    }

    ~AclHugeMem() {
        if (releaseMemFunc_) {
            releaseMemFunc_(nullptr, false);
        }
        if (unInitMemFunc_) {
            unInitMemFunc_(nullptr, false);
        }
    }

private:
    InitHugeMemThreadLocal initMemFunc_ = nullptr;
    UnInitHugeMemThreadLocal unInitMemFunc_ = nullptr;
    ReleaseHugeMem releaseMemFunc_ = nullptr;
};

// A class to alloc acl workspace and release it when the object is destroyed.
class AclWorkspace final {
public:
    explicit AclWorkspace(diopiContextHandle_t ctx, std::size_t workspaceSize) noexcept {
        if (workspaceSize > 0) {
            diopiTensorHandle_t bufHandle;
            auto ret = diopiRequireBuffer(ctx, &bufHandle, workspaceSize, diopi_device);
            ASCEND_CHECK_ABORT(ret == diopiSuccess, "[AclWorkspace] Require workspace size %ld failed.", static_cast<uint64_t>(workspaceSize));
            AscendTensor buf(bufHandle);
            workspaceAddr_ = const_cast<void*>(buf.data());
        }
    }
    ~AclWorkspace() = default;
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
    ASCEND_CHECK_ABORT(workspaceSizeFuncAddr != nullptr, "[%s] can't get workSpaceName function.", api);
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
    ASCEND_CHECK_ABORT(workspaceStatus == ACL_SUCCESS, "[%s]'s workspaceStatus is not equal to ACL_SUCCESS. aclnnStatus is %d.", api, workspaceStatus);

    AclWorkspace workspace(ctx, workspaceSize);

    /* 2. call aclnnXXX function */
    static const auto opApiFuncAddr = getOpApiFuncAddr(api);
    ASCEND_CHECK_ABORT(opApiFuncAddr != nullptr, "[%s] can't get op function.", api);
    using OpApiFuncType = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    static const auto opApiFunc = reinterpret_cast<OpApiFuncType>(opApiFuncAddr);

    auto ret = opApiFunc(workspace.addr(), workspaceSize, executor, stream);
    ASCEND_CHECK_ABORT(ret == ACL_SUCCESS, "[%s] failed. aclnnStatus is %d.", api, ret);
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
