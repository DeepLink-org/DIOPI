/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_ACLNN_ACLNN_HPP_
#define IMPL_ASCEND_ACLNN_ACLNN_HPP_

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../ascend_tensor.hpp"
#include "acl/acl.h"
#include "aclnnop/aclnn_add.h"  // TODO(zmz): add all
#include "aclnnop/aclnn_cos.h"  // TODO(zmz): add all
#include "aclnnop/aclnn_sin.h"  // TODO(zmz): add all
#include "impl_functions.hpp"

namespace impl {
namespace ascend {

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

int aclnnAddAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self, diopiConstTensorHandle_t other, const diopiScalar_t* alpha,
                    diopiTensorHandle_t out);

int aclnnSinAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self, diopiTensorHandle_t out);

int aclnnCosAdaptor(diopiContextHandle_t ctx, diopiConstTensorHandle_t self, diopiTensorHandle_t out);

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_ACLNN_ACLNN_HPP_
