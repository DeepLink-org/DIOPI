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

namespace impl {
namespace ascend {

#define ASCEND_CHECK(condition, ...)                                  \
    do {                                                              \
        if (!(condition)) {                                           \
            printf("[%s:%s:%d]: ", __FILE__, __FUNCTION__, __LINE__); \
            printf(__VA_ARGS__);                                      \
            printf("\n");                                             \
        }                                                             \
    } while (0);

#define __FILENAME__ __FILE__

#define ASCEND_LOGE(fmt, ...)                                                                 \
    aclAppLog(ACL_ERROR, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__); \
    printf("%s:%s:%d \n[PTA]:" #fmt, __FILENAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__);

#define ASCEND_LOGW(fmt, ...)                                                                   \
    aclAppLog(ACL_WARNING, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__); \
    printf("%s:%s:%d [PTA]:" #fmt, __FILENAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__);

#define ASCEND_LOGI(fmt, ...)                                                                \
    aclAppLog(ACL_INFO, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__); \
    printf("%s:%s:%d [PTA]:" #fmt, __FILENAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__);

#define ASCEND_LOGD(fmt, ...)                                                                 \
    aclAppLog(ACL_DEBUG, __FILENAME__, __FUNCTION__, __LINE__, "[PTA]:" #fmt, ##__VA_ARGS__); \
    printf("%s:%s:%d [PTA]:" #fmt, __FILENAME__, __FUNCTION__, __LINE__, ##__VA_ARGS__);

inline const char* getOpApiLibName(void) { return "libopapi.so"; }

inline const char* getCustOpApiLibName(void) { return "libcust_opapi.so"; }

inline void* getOpApiFuncAddrInLib(void* handler, const char* libName, const char* apiName) {
    auto funcAddr = dlsym(handler, apiName);
    if (funcAddr == nullptr) {
        ASCEND_LOGW("dlsym %s from %s failed, error:%s.", apiName, libName, dlerror());
    }
    return funcAddr;
}

inline void* getOpApiLibHandler(const char* libName) {
    auto handler = dlopen(libName, RTLD_LAZY);
    if (handler == nullptr) {
        ASCEND_LOGW("dlopen %s failed, error:%s.", libName, dlerror());
    }
    return handler;
}

inline void* getOpApiFuncAddr(const char* apiName) {
    static auto custOpApiHandler = getOpApiLibHandler(getCustOpApiLibName());
    if (custOpApiHandler != nullptr) {
        auto funcAddr = getOpApiFuncAddrInLib(custOpApiHandler, getCustOpApiLibName(), apiName);
        if (funcAddr != nullptr) {
            return funcAddr;
        }
    }

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

template <typename T>
T convertType(T value) {
    return value;
}

template <typename... Ts>
constexpr auto convertTypes(Ts&... args) {
    return std::make_tuple(convertType(args)...);
}

int test(const std::string& name, diopiContextHandle_t ctx, aclTensor* self, aclTensor* out) {
    aclrtStream stream;
    diopiGetStream(ctx, &stream);
    // 1.1
    // name = "aclnnCos";
    std::string workSpaceName = name + "GetWorkspaceSize";
    static const auto getWorkspaceSizeFuncAddr = getOpApiFuncAddr(workSpaceName.c_str());
    ASCEND_CHECK(getWorkspaceSizeFuncAddr != nullptr, "can't get ", name, " workspace function.");

    // 1.2
    uint64_t workspaceSize = 0;
    uint64_t* workspaceSizeAddr = &workspaceSize;
    aclOpExecutor* executor = nullptr;
    aclOpExecutor** executorAddr = &executor;
    auto convertedParams = convertTypes(self, out, workspaceSizeAddr, executorAddr);
    static auto getWorkspaceSizeFunc = convertToOpApiFunc(convertedParams, getWorkspaceSizeFuncAddr);

    // 1.3 完成workspace调用逻辑
    auto workspaceStatus = call(getWorkspaceSizeFunc, convertedParams);
    ASCEND_CHECK(workspaceStatus != 0, "workspaceStatus= ", workspaceStatus, " , not equal 0.");
    // return 0;

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }

    // 2.1
    static const auto opApiFuncAddr = getOpApiFuncAddr(name.c_str());
    ASCEND_CHECK(opApiFuncAddr != nullptr, "can't get ", name, " op function.");

    // 2.2
    typedef int (*OpApiFunc)(void*, uint64_t, aclOpExecutor*, const aclrtStream);
    OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
    auto ret = opApiFunc(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCos failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return 0;
}

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ACLNN_ADAPTOR_HPP_
