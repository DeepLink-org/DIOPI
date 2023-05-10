/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_CAMB_MLU_HELPER_HPP_
#define IMPL_CAMB_MLU_HELPER_HPP_

#include <cnrt.h>

#include <iostream>

#include "cn_api.h"
#include "diopi_helper.hpp"
namespace impl {

namespace camb {

#define NFU_ALIGN_SIZE 128

#define PAD_UP(x, y) (((x) / (y) + (int)((x) % (y) > 0)) * (y))

#define PAD_DOWN(x, y) (((x) / (y)) * (y))

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

#define CEIL_ALIGN(x, y) (((x) + (y)-1) / (y) * (y))

constexpr uint32_t rem_for_stack = 128 * 1024;

inline uint32_t getDeviceAttr(cnrtDeviceAttr_t attr) {
  int dev_ordinal = 0;
  int device_attr = 1;
  cnrtGetDevice(&dev_ordinal);
  cnrtDeviceGetAttribute(&device_attr, attr, dev_ordinal);
  if (attr == cnrtAttrNramSizePerMcore) {
    device_attr -= rem_for_stack;
  }
  return device_attr;
}

inline int32_t getJobLimitCapability() {
  CNcontext drv_ctx;
  DIOPI_CHECK(CN_SUCCESS == cnCtxGetCurrent(&drv_ctx), "cnCtxGetCurrent fails");
  CNctxConfigParam ctx_conf_param;
  DIOPI_CHECK(CN_SUCCESS == cnGetCtxConfigParam(drv_ctx, CN_CTX_CONFIG_UNION_LIMIT, &ctx_conf_param), "cnGetCtxConfigParam fails.");
  return (int32_t)ctx_conf_param.unionLimit;
}

inline int32_t getCoreNumOfJobLimitCapability() {
  switch (getJobLimitCapability()) {
    default:
        return getDeviceAttr(cnrtAttrMcorePerCluster) * getJobLimitCapability();
    case CN_KERNEL_CLASS_BLOCK:
        return 1;
    case CN_KERNEL_CLASS_UNION:
        return getDeviceAttr(cnrtAttrMcorePerCluster);
    case CN_KERNEL_CLASS_UNION2:
        return getDeviceAttr(cnrtAttrMcorePerCluster) * 2;
    case CN_KERNEL_CLASS_UNION4:
        return getDeviceAttr(cnrtAttrMcorePerCluster) * 4;
    case CN_KERNEL_CLASS_UNION8:
        return getDeviceAttr(cnrtAttrMcorePerCluster) * 8;
    case CN_KERNEL_CLASS_UNION16:
        return getDeviceAttr(cnrtAttrMcorePerCluster) * 16;
  }
}

inline cnrtDataType_t dtype2CnrtDtype(const diopiDtype_t dt) {
  switch (dt) {
  case diopi_dtype_uint8:
    return CNRT_UINT8;
  case diopi_dtype_int8:
    return CNRT_INT8;
  case diopi_dtype_int16:
    return CNRT_INT16;
  case diopi_dtype_int32:
  case diopi_dtype_uint32:
    return CNRT_INT32;
  case diopi_dtype_int64:
  case diopi_dtype_uint64:
    return CNRT_INT64;
  case diopi_dtype_float16:
    return CNRT_FLOAT16;
  case diopi_dtype_float32:
    return CNRT_FLOAT32;
  case diopi_dtype_float64:
    return CNRT_FLOAT64;
  case diopi_dtype_bool:
    return CNRT_BOOL;
  default:
    std::cerr << "diopi dytpe not supported in pytorch+diopi scenario)";
  }
}

}  // namespace camb

}  // namespace impl

#endif  // IMPL_CAMB_MLU_HELPER_HPP_
