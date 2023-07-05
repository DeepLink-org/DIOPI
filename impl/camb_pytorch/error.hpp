/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#ifndef DIOPI_IMPL_CAMB_PYTORCH_ERROR_HPP_
#define DIOPI_IMPL_CAMB_PYTORCH_ERROR_HPP_

void _set_last_error_string(const char *err);
const char* camb_get_last_error_string();

#endif  // DIOPI_IMPL_CAMB_PYTORCH_ERROR_HPP_
