/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2024, DeepLink.
 */

#ifndef IMPL_ASCEND_ACLNN_ACL_SCALAR_HPP_
#define IMPL_ASCEND_ACLNN_ACL_SCALAR_HPP_

#include "../common/acloprunner.hpp"
#include "aclnn/acl_meta.h"

namespace impl {
namespace ascend {

class AclScalar final {
public:
    explicit AclScalar(const diopiScalar_t* scalar, diopiDtype_t dtype) {
        int64_t nbytes = 0;
        uint64_t buff = 0;  // just for store the value with the different type.
        switch (dtype) {
            case diopi_dtype_bool: {
                nbytes = 1;
                *reinterpret_cast<bool*>(&buff) = getValue<bool>(scalar);
                break;
            }
            case diopi_dtype_int8: {
                nbytes = 1;
                *reinterpret_cast<int8_t*>(&buff) = getValue<int8_t>(scalar);
                break;
            }
            case diopi_dtype_uint8: {
                nbytes = 1;
                *reinterpret_cast<uint8_t*>(&buff) = getValue<uint8_t>(scalar);
                break;
            }
            case diopi_dtype_int16: {
                nbytes = 2;
                *reinterpret_cast<int16_t*>(&buff) = getValue<int16_t>(scalar);
                break;
            }
            case diopi_dtype_uint16: {
                nbytes = 2;
                *reinterpret_cast<uint16_t*>(&buff) = getValue<uint16_t>(scalar);
                break;
            }
            case diopi_dtype_int32: {
                nbytes = 4;
                *reinterpret_cast<int32_t*>(&buff) = getValue<int32_t>(scalar);
                break;
            }
            case diopi_dtype_uint32: {
                nbytes = 4;
                *reinterpret_cast<uint32_t*>(&buff) = getValue<uint32_t>(scalar);
                break;
            }
            case diopi_dtype_int64: {
                nbytes = 8;
                *reinterpret_cast<int64_t*>(&buff) = getValue<int64_t>(scalar);
                break;
            }
            case diopi_dtype_uint64: {
                nbytes = 8;
                *reinterpret_cast<uint64_t*>(&buff) = getValue<uint64_t>(scalar);
                break;
            }
            case diopi_dtype_float16: {
                nbytes = 2;
                *reinterpret_cast<half_float::half*>(&buff) = getValue<half_float::half>(scalar);
                break;
            }
            case diopi_dtype_float32: {
                nbytes = 4;
                *reinterpret_cast<float*>(&buff) = getValue<float>(scalar);
                break;
            }
            case diopi_dtype_float64: {
                nbytes = 8;
                *reinterpret_cast<double*>(&buff) = getValue<double>(scalar);
                break;
            }
            default: {
                error(__FILE__, __LINE__, __FUNCTION__, "the input tensor dtype %s is not allown", diopiDtypeToStr(dtype));
            }
        }
        // create aclScalar
        acl_ = aclCreateScalar(&buff, getAclDataType(dtype));
    }
    explicit AclScalar(const diopiScalar_t* scalar) : AclScalar(scalar, scalar->stype) {}

    explicit operator aclScalar*() { return acl_; }

    bool defined() const { return acl_; }

private:
    aclScalar* acl_ = nullptr;
};

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_ACLNN_ACL_SCALAR_HPP_
