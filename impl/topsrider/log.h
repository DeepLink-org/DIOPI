#pragma once

#include <mutex>

namespace impl::tops {
extern char str_last_error[8192];
extern char str_last_error_other[4096];
extern std::mutex mtx_last_error;

}  // namespace impl::tops

inline const char* tops_get_last_error_string() {
  std::lock_guard<std::mutex> lock(impl::tops::mtx_last_error);
  sprintf(impl::tops::str_last_error,
          "tops error: %s; other error: %s",
          "topsGetErrorString(error)",
          impl::tops::str_last_error_other);
  return impl::tops::str_last_error;
}
