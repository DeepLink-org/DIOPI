/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_MACRO_HPP_
#define IMPL_ASCEND_MACRO_HPP_

#include <algorithm>
#include <sstream>
#include <string>

namespace impl {
namespace ascend {

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

#define CHECK_ASCENDRT(Expr)                                                \
    do {                                                                    \
        TRACK_ACL(#Expr);                                                   \
        ::diopiError_t ret = Expr;                                          \
        if (ret != ::diopiSuccess) {                                        \
            throw std::runtime_error(std::string("call function failed.")); \
        }                                                                   \
    } while (0);

#define ASCEND_CHECK_ABORT(condition, ...)                            \
    do {                                                              \
        if (!(condition)) {                                           \
            printf("[%s:%s:%d]: ", __FILE__, __FUNCTION__, __LINE__); \
            printf(__VA_ARGS__);                                      \
            printf("\n");                                             \
            std::abort();                                             \
        }                                                             \
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

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_MACRO_HPP_
