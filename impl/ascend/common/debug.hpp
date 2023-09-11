/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef IMPL_ASCEND_COMMON_DEBUG_HPP_
#define IMPL_ASCEND_COMMON_DEBUG_HPP_

#include <diopi/diopirt.h>

#include <algorithm>
#include <sstream>

namespace impl {
namespace ascend {

class AscendTensor;

std::string dumpTensor(diopiConstTensorHandle_t th, const std::string& msg = "");

std::string dumpTensor(const AscendTensor& th, const std::string& msg = "");

}  // namespace ascend
}  // namespace impl

#endif  // IMPL_ASCEND_COMMON_DEBUG_HPP_
