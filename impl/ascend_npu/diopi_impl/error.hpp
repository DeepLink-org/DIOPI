/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_NPU_DIOPI_IMPL_ERROR_HPP_
#define IMPL_ASCEND_NPU_DIOPI_IMPL_ERROR_HPP_

extern "C" {

void setLastErrorString(const char* err);
const char* cudaGetLastErrorString();

}  // extern "C"

#endif  // IMPL_ASCEND_NPU_DIOPI_IMPL_ERROR_HPP_
