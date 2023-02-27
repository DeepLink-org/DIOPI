#ifndef CNNL_HELPER_HPP_
#define CNNL_HELPER_HPP_

#include <cnnl.h>
#include "diopi_helper.hpp"

#define DIOPI_CHECK(cond, str)                                             \
    do {                                                                   \
        if (!(cond)) {                                                     \
            set_last_error_string("%s at %s:%d", str, __FILE__, __LINE__); \
            return diopiErrorOccurred;                                     \
        }                                                                  \
    } while (false);

#define DIOPI_CALLCNNL(Expr)                                                 \
    do {                                                                     \
        ::cnnlStatus_t ret = Expr;                                           \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                  \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%s", \
                                              ret,                           \
                                              ::cnnlGetErrorString(ret),     \
                                              __FILE__,                      \
                                              __LINE__);                     \
            return diopiErrorOccurred;                                       \
        }                                                                    \
    } while (false);

#define DIOPI_CHECKCNNL(Expr)                                                \
    do {                                                                     \
        ::cnnlStatus_t ret = Expr;                                           \
        if (ret != ::CNNL_STATUS_SUCCESS) {                                  \
            impl::camb::set_last_error_string("cnnl error %d : %s at %s:%s", \
                                              ret,                           \
                                              ::cnnlGetErrorString(ret),     \
                                              __FILE__,                      \
                                              __LINE__);                     \
        }                                                                    \
    } while (false);

template<typename T, ::cnnlStatus_t(*fnCreate)(T*), ::cnnlStatus_t(*fnDestroy)(T)>
class CnnlResourceGuard final {
public:
    CnnlResourceGuard() {
        DIOPI_CHECKCNNL(fnCreate(&resource_));
    }

    ~CnnlResourceGuard() {
        DIOPI_CHECKCNNL(fnDestroy(resource_));
    }

    T& get() {
        return resource_;
    }

protected:
    T resource_ {0};
};

diopiError_t convertType(cnnlDataType_t *cnnlType, diopiDtype_t type);

#endif  // CNNL_HELPER_HPP_