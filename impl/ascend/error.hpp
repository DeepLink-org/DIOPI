#ifndef IMPL_ASCEND_ERROR_HPP_
#define IMPL_ASCEND_ERROR_HPP_

#include <acl/acl.h>

#include <mutex>
#include <utility>

#define TRACK_ACL(x)                                                    \
    do {                                                                \
        static bool enable = std::getenv("DIOPI_TRACK_ACL") != nullptr; \
        if (enable) {                                                   \
            printf("[%s: %d]:%s\n", __FILE__, __LINE__, x);             \
        }                                                               \
    } while (0);

#define CALL_ACLRT(Expr)                                                                          \
    do {                                                                                          \
        TRACK_ACL(#Expr);                                                                         \
        ::aclError ret = Expr;                                                                    \
        if (ret != ::ACL_SUCCESS) {                                                               \
            throw std::runtime_error(std::string("ascend device error:") + aclGetRecentErrMsg()); \
        }                                                                                         \
    } while (0);

#define ASCEND_CHECK_ABORT(condition, ...)               \
    do {                                                 \
        if (!(condition)) {                              \
            printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
            printf(__VA_ARGS__);                         \
            printf("\n");                                \
            std::abort();                                \
        }                                                \
    } while (0);

#define ASCEND_CHECK_NULLPTR_ABORT(ptr) ASCEND_CHECK_ABORT(ptr, "Variable is nullptr, pls check.")

#define error(...)                               \
    printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                         \
    printf("\n");                                \
    std::abort();

#define warning(...)                             \
    printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                         \
    printf("\n");

#define info(...)                                \
    printf("[%s:%d]: ", __FUNCTION__, __LINE__); \
    printf(__VA_ARGS__);                         \
    printf("\n");

namespace impl {

namespace ascend {

extern char strLastError[8192];
extern char strLastErrorOther[4096];
extern std::mutex mtxLastError;

template <typename... Types>
inline void set_last_error_string(const char* szFmt, Types&&... args) {
    std::lock_guard<std::mutex> lock(mtxLastError);
    sprintf(strLastErrorOther, szFmt, std::forward<Types>(args)...);
}

const char* ascend_get_last_error_string();

}  // namespace ascend

}  // namespace impl

#endif  // IMPL_ASCEND_ERROR_HPP_
