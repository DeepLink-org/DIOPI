#pragma once

#include <iostream>
#include <string>
#include <stdio.h>
#include "acl/acl_base.h"


#define NPUStatus std::string
#define SUCCESS "SUCCESS"
#define INTERNEL_ERROR "INTERNEL_ERROR"
#define PARAM_ERROR "PARAM_ERROR"
#define ALLOC_ERROR "ALLOC_ERROR"
#define FAILED "FAILED"

#define ASCEND_LOGE(fmt, ...) \
  aclAppLog(ACL_ERROR, __FILE__, __FUNCTION__, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__)
#define ASCEND_LOGW(fmt, ...) \
  aclAppLog(ACL_WARNING, __FILE__, __FUNCTION__, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__)
#define ASCEND_LOGI(fmt, ...) \
  aclAppLog(ACL_INFO, __FILE__, __FUNCTION__, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__)
#define ASCEND_LOGD(fmt, ...) \
  aclAppLog(ACL_DEBUG, __FILE__, __FUNCTION__, __LINE__, "[PTA]:"#fmt, ##__VA_ARGS__)
