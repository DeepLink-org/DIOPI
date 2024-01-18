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
#include "aclnnop/aclnn_add.h"  // TODO: add all
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

inline int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

inline int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
    // 固定写法，acl初始化
    // auto ret = aclrtSetDevice(deviceId);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    // ret = aclrtCreateContext(context, deviceId);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    // ret = aclrtSetCurrentContext(*context);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    auto ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

    // ret = aclInit(nullptr);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    return ret;
}

int aclnnAddTest(int32_t deviceId, aclrtContext& context, aclrtStream& stream, diopiConstTensorHandle_t self, diopiConstTensorHandle_t other,
                 const diopiScalar_t* alpha, diopiTensorHandle_t out);

}  // namespace ascend
}  // namespace impl

#endif  //  IMPL_ASCEND_ACLNN_ACLNN_HPP_
