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

template <class T, class U = std::remove_cv_t<std::remove_reference_t<T>>>
decltype(auto) convertType(T&& param) {
    if constexpr (std::is_same_v<U, AscendTensor>) {
        return createAclTensorFromAscendTensor(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, diopiTensorHandle_t> || std::is_same_v<U, diopiConstTensorHandle_t>) {
        return createAclTensorFromDiopiTensor(std::forward<T>(param));
    } else if constexpr (std::is_same_v<U, diopiScalar_t*> || std::is_same_v<U, const diopiScalar_t*>) {
        return createAclScalarFromDiopiScalar(std::forward<T>(param));
    } else {
        return std::forward<T>(param);
    }
}

template <class... Args>
constexpr auto convertParams(const Args&... args) {
    return std::make_tuple(convertType(args)...);
}

template <class T>
inline void releaseConverted(T&& param [[maybe_unused]]) {}  // no conversion, do nothing

inline void releaseConverted(const aclTensor* tensor) {
    if (tensor != nullptr) {
        ::aclDestroyTensor(tensor);
    }
}

inline void releaseConverted(const aclScalar* scalar) {
    if (scalar != nullptr) {
        ::aclDestroyScalar(scalar);
    }
}

template <class... Args>
void callAclnnImpl(const char* api, const char* workspaceApi, diopiContextHandle_t ctx, const Args&... args) {
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

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto convertedParams = convertParams(args..., &workspaceSize, &executor);
    auto workspaceStatus = std::apply(workspaceSizeFunc, convertedParams);
    ASCEND_CHECK(workspaceStatus == ACL_SUCCESS, "workspaceStatus not equal ACL_SUCCESS.");

    void* workspaceAddr = nullptr;
    if (workspaceSize != 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ASCEND_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d\n", ret);
    }

    /* 2. call aclnnXXX function */
    static const auto opApiFuncAddr = getOpApiFuncAddr(api);
    ASCEND_CHECK(opApiFuncAddr != nullptr, "can't get op function.");
    using OpApiFuncType = int (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    static const auto opApiFunc = reinterpret_cast<OpApiFuncType>(opApiFuncAddr);

    auto ret = opApiFunc(workspaceAddr, workspaceSize, executor, stream);
    ASCEND_CHECK(ret == ACL_SUCCESS, "%s failed. ERROR: %d\n", api, ret);

    /* 3. clean up */
    std::apply([](const auto&... args) { (releaseConverted(args), ...); }, convertedParams);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
}

#define DIOPI_ASCEND_CALL_ACLNN(api, ctx, ...) ::impl::ascend::aclnn_adaptor::callAclnnImpl(#api, #api "GetWorkspaceSize", ctx, __VA_ARGS__)

}  // namespace aclnn_adaptor

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ACLNN_ADAPTOR_HPP_
