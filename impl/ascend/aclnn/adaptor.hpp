/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#ifndef IMPL_ASCEND_ACLNN_ADAPTOR_HPP_
#define IMPL_ASCEND_ACLNN_ADAPTOR_HPP_

#include <dlfcn.h>

#include <functional>
#include <iostream>
#include <numeric>
#include <string>

#include "../ascend_tensor.hpp"
#include "../common/utils.hpp"
#include "../env_vars.hpp"
#include "acl/acl.h"
#include "acl_tensor.hpp"
#include "aclnn/acl_meta.h"

namespace impl {
namespace ascend {

constexpr const char kWorkspaceSizeSuffix[] = "GetWorkspaceSize";

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

inline const char* getOpApiLibName() { return "libopapi.so"; }

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
    static void* opApiHandler = getOpApiLibHandler(getOpApiLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return getOpApiFuncAddrInLib(opApiHandler, getOpApiLibName(), apiName);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
    // static_assert(std::is_same<decltype(f(std::get<I>(t)...)), void>::value,
    //               "call_impl: f(std::get<I>(t)...)");
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) {
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

template <typename Tuple, size_t... I>
auto convertToOpApiFunc(const Tuple& params, void* opApiAddr, std::index_sequence<I...>) {
    typedef int (*OpApiFunc)(typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple>
auto convertToOpApiFunc(const Tuple& params, void* opApiAddr) {
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return convertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

inline aclTensor* convertType(AclTensor& value) { return static_cast<aclTensor*>(value); }

template <typename T>
T convertType(T value) {
    return value;
}

template <typename... Ts>
constexpr auto convertTypes(Ts&... args) {
    return std::make_tuple(convertType(args)...);
}

#define ACLNN_ADAPTOR(api, ctx, ...)                                                                      \
    do {                                                                                                  \
        std::string name = #api;                                                                          \
        static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;              \
        if (aclDebugFlag) {                                                                               \
            std::cout << "ACLNN_ADAPTOR for " << name << std::endl;                                       \
        }                                                                                                 \
                                                                                                          \
        /* 0. get aclrtStream */                                                                          \
        aclrtStream stream;                                                                               \
        diopiGetStream(ctx, &stream);                                                                     \
        std::string workSpaceName = name + kWorkspaceSizeSuffix;                                          \
        volatile auto getWorkspaceSizeFuncAddr = getOpApiFuncAddr(workSpaceName.c_str());                 \
        ASCEND_CHECK(getWorkspaceSizeFuncAddr != nullptr, "can't get workSpaceName function.");           \
                                                                                                          \
        /* 1. call xxxGetWorkspaceSize function. */                                                       \
        uint64_t workspaceSize = 0;                                                                       \
        uint64_t* workspaceSizeAddr = &workspaceSize;                                                     \
        aclOpExecutor* executor = nullptr;                                                                \
        aclOpExecutor** executorAddr = &executor;                                                         \
        auto convertedParams = convertTypes(__VA_ARGS__, workspaceSizeAddr, executorAddr);                \
        static auto getWorkspaceSizeFunc = convertToOpApiFunc(convertedParams, getWorkspaceSizeFuncAddr); \
                                                                                                          \
        auto workspaceStatus = call(getWorkspaceSizeFunc, convertedParams);                               \
        ASCEND_CHECK(workspaceStatus == ACL_SUCCESS, "workspaceStatus not equal ACL_SUCCESS.");           \
                                                                                                          \
        void* workspaceAddr = nullptr;                                                                    \
        if (workspaceSize != 0) {                                                                         \
            auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);             \
            ASCEND_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d\n", ret);              \
        }                                                                                                 \
                                                                                                          \
        /* 2. call aclnnXXX function */                                                                   \
        volatile auto opApiFuncAddr = getOpApiFuncAddr(name.c_str());                                     \
        ASCEND_CHECK(opApiFuncAddr != nullptr, "can't get op function.");                                 \
                                                                                                          \
        typedef int (*OpApiFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);                           \
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                 \
        auto ret = opApiFunc(workspaceAddr, workspaceSize, executor, stream);                             \
        ASCEND_CHECK(ret == ACL_SUCCESS, "%s failed. ERROR: %d\n", name.c_str(), ret);                    \
                                                                                                          \
        if (workspaceSize > 0) {                                                                          \
            aclrtFree(workspaceAddr);                                                                     \
        }                                                                                                 \
    } while (false)

int createAclTensor(diopiConstTensorHandle_t input, aclTensor** tensor);

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ACLNN_ADAPTOR_HPP_
