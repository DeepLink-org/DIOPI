/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_NPU_DIOPI_IMPL_ERROR_HPP_
#define IMPL_ASCEND_NPU_DIOPI_IMPL_ERROR_HPP_

extern "C" {

void _set_last_error_string(const char* err);
const char* cuda_get_last_error_string();

}  // extern "C"

#endif  // IMPL_ASCEND_NPU_DIOPI_IMPL_ERROR_HPP_
