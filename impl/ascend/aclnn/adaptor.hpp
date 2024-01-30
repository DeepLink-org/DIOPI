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
#include "../common/acloprunner.hpp"
#include "../common/utils.hpp"
// #include "acl_tensor.hpp"
#include <iostream>
#include <typeinfo>

#include "acl/acl.h"
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

template <typename T>
void printType(const T& value) {
    std::cout << "Type of " << value << " is: " << typeid(value).name() << std::endl;
}
inline const char* getOpApiLibName() { return "libopapi.so"; }

inline bool useAclnn() {
    static bool enable = std::getenv("DIOPI_USE_ACLNN") != nullptr;
    return enable;
}
int createAclTensor1(diopiConstTensorHandle_t input, aclTensor** tensor);

inline void* getOpApiFuncAddrInLib(void* handler, const char* libName, const char* apiName) {
    warning(__FILE__, __LINE__, __FUNCTION__, "getOpApiFuncAddrInLib DEBUG DEBG: dlsym call function name %s from %s.", apiName, libName);
    auto funcAddr = dlsym(handler, apiName);        // TODO: 输出每个结果
    warning(__FILE__, __LINE__, __FUNCTION__, "JUSTDEBUG inner %s from %s.handler=%ld, ", apiName, libName, funcAddr);
    if (funcAddr == nullptr) {
        warning(__FILE__, __LINE__, __FUNCTION__, "dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void* getOpApiLibHandler(const char* libName) {
    warning(__FILE__, __LINE__, __FUNCTION__, "getOpApiFuncAddr DEBUG DEBG: dlsym call function from %s.", libName);
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        warning(__FILE__, __LINE__, __FUNCTION__, "dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

inline void* getOpApiFuncAddr(const char* apiName) {
    warning(__FILE__, __LINE__, __FUNCTION__, "getOpApiFuncAddr DEBUG DEBG: dlsym call function name %s.", apiName);
    static auto opApiHandler = getOpApiLibHandler(getOpApiLibName());
    if (opApiHandler == nullptr) {
        return nullptr;
    }
    return getOpApiFuncAddrInLib(opApiHandler, getOpApiLibName(), apiName);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
    // static_assert(std::is_same<decltype(f(std::get<I>(t)...)), void>::value,
    //               "call_impl: f(std::get<I>(t)...)");
    warning(__FILE__, __LINE__, __FUNCTION__, "call function ptr %ld.", f);
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

void printContiTensor(const aclTensor& tensor, const void* tensorPtr);

void printContiTensor(const aclTensor& tensor, diopiConstTensorHandle_t diopi);

inline aclTensor* convertType(AclTensor& value) {
    static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1;
    if (aclDebugFlag) {
        std::cout << "aclTensor ptr=" << value.data() << std::endl;
        std::cout << "ptr()=" << value.ptr() << std::endl;
        printType(&value);
        // printContiTensor(*(value.ptr()), value.data());
    }

    return static_cast<aclTensor*>(value);
}

template <typename T>
T convertType(T value) {
    printType(value);
    return value;
}

template <typename... Ts>
constexpr auto convertTypes(Ts&... args) {
    return std::make_tuple(convertType(args)...);
}


template <typename T>
void printValue(T value) {
    std::cout << "printValue=" << value << std::endl;
    return;
}

typedef int (*InitHugeMemThreadLocal)(void *, bool);
typedef void (*UnInitHugeMemThreadLocal)(void *, bool);
typedef void (*ReleaseHugeMem)(void *, bool);
typedef aclOpExecutor *(*PTAGetExecCache)(uint64_t, uint64_t *);
typedef void (*InitPTACacheThreadLocal)();
typedef void (*SetPTAHashKey)(uint64_t);
typedef bool (*CanUsePTACache)(const char *);
typedef void (*UnInitPTACacheThreadLocal)();

inline void UnInitCacheThreadLocal()
{
    static const auto unInitPTACacheThreadLocalAddr = getOpApiFuncAddr("UnInitPTACacheThreadLocal");
    UnInitPTACacheThreadLocal unInitPTACacheThreadLocalFunc =
        reinterpret_cast<UnInitPTACacheThreadLocal>(unInitPTACacheThreadLocalAddr);
    if (unInitPTACacheThreadLocalFunc) {
        unInitPTACacheThreadLocalFunc();
    }
}

// template <typename... Args>
// int aclnnAdaptor(const std::string& name, diopiContextHandle_t ctx, Args... args) {

#define  aclnnAdaptor(n, ctx, ...) \
    do {                                                                             \
    std::string name = #n; \
    static int aclDebugFlag = std::getenv("DIOPI_DEBUG_ACLOPRUNNER") == nullptr ? 0 : 1; \
    if (aclDebugFlag) { \
        std::cout << "aclnnAdaptor for " << name << std::endl; \
    } \
    aclrtStream stream; \
    diopiGetStream(ctx, &stream); \
    std::string workSpaceName = name + kWorkspaceSizeSuffix; \
    volatile auto getWorkspaceSizeFuncAddr = getOpApiFuncAddr(workSpaceName.c_str()); \
    ASCEND_CHECK_ABORT(getWorkspaceSizeFuncAddr != nullptr, "can't get workSpaceName function."); \
 \
    uint64_t workspaceSize = 0; \
    uint64_t* workspaceSizeAddr = &workspaceSize; \
    aclOpExecutor* executor = nullptr; \
    aclOpExecutor** executorAddr = &executor; \
    auto convertedParams = convertTypes(__VA_ARGS__, workspaceSizeAddr, executorAddr); \
    static auto getWorkspaceSizeFunc = convertToOpApiFunc(convertedParams, getWorkspaceSizeFuncAddr); \
    std::cout << "getWorkspaceSizeFunc ptr = " << getWorkspaceSizeFunc << std::endl; \
 \
    auto workspaceStatus = call(getWorkspaceSizeFunc, convertedParams); \
    ASCEND_CHECK_ABORT(workspaceStatus == ACL_SUCCESS, "workspaceStatus not equal ACL_SUCCESS."); \
 \
    void* workspaceAddr = nullptr; \
    if (workspaceSize != 0) { \
        std::cout << "DEBUG workspaceSize = " << workspaceSize << std::endl; \
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST); \
        ASCEND_CHECK_ABORT(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: %d\n", ret); \
    } \
    volatile auto opApiFuncAddr = getOpApiFuncAddr(name.c_str()); \
    ASCEND_CHECK_ABORT(opApiFuncAddr != nullptr, "can't get op function."); \
 \
    typedef int (*OpApiFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream); \
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr); \
    auto ret = opApiFunc(workspaceAddr, workspaceSize, executor, stream); \
    ASCEND_CHECK_ABORT(ret == ACL_SUCCESS, "%s failed. ERROR: %d\n", name, ret ); \
 \
    ret = aclrtSynchronizeStream(stream); \
    ASCEND_CHECK_ABORT(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed. ERROR: %d\n", ret ); \
 \
    if (workspaceSize > 0) { \
        aclrtFree(workspaceAddr); \
    } \
 \
    } while (false)

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ACLNN_ADAPTOR_HPP_
