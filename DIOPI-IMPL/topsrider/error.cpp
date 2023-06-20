/*
* SPDX-FileCopyrightText: Copyright (c) 2022 Enflame. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
#include <tops_runtime.h>
#include <mutex>

#include "error.h"

static char strLastError[4096] = {0};
static char strLastErrorOther[2048] = {0};
static std::mutex mtxLastError;

const char* tops_get_last_error_string() {
  topsError_t error = topsGetLastError();
  std::lock_guard<std::mutex> lock(mtxLastError);
  sprintf(strLastError, "tops error: %s; other error: %s",
          topsGetErrorString(error), strLastErrorOther);
  return strLastError;
}

void _set_last_error_string(const char* err) {
  std::lock_guard<std::mutex> lock(mtxLastError);
  sprintf(strLastErrorOther, "%s", err);
}