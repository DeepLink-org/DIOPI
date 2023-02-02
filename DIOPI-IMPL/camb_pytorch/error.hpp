#ifndef IMPL_CAMB_PYTORCH_ERROR_HPP_
#define IMPL_CAMB_PYTORCH_ERROR_HPP_

void _set_last_error_string(const char *err);
const char* camb_get_last_error_string();

#endif  // IMPL_CAMB_PYTORCH_ERROR_HPP_
