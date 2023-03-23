/**
 * @file
 * @author OpenComputeLab
 * @copyright  (c) 2023, OpenComputeLab.
 */

#ifndef IMPL_TORCH_ERROR_HPP_
#define IMPL_TORCH_ERROR_HPP_

void _set_last_error_string(const char *err);
const char* cuda_get_last_error_string();

#endif  // IMPL_TORCH_ERROR_HPP_
