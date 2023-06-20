/*
* SPDX-FileCopyrightText: Copyright (c) 2022 Enflame. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
#include <diopi/diopirt.h>
#include <diopi_register.h>
#include <tops_runtime.h>
#include <tops_runtime_api.h>

#include <cstdio>
#include <mutex>

#include "helper.hpp"

extern "C" {

#define CALL_TOPS(Expr)                                                   \
  {                                                                       \
    topsError_t ret = Expr;                                               \
    if (ret != topsSuccess) {                                             \
      printf("call a topsrt function (%s) failed. return code=%d", #Expr, \
             ret);                                                        \
    }                                                                     \
  }

void* tops_malloc(uint64_t bytes) {
  void* ptr;
  CALL_TOPS(topsMalloc(&ptr, bytes));
  return ptr;
}

void tops_free(void* ptr) { CALL_TOPS(topsFree(ptr)); }

int32_t tops_make_stream(diopiStreamHandle_t* stream_handle_ptr) {
  topsStream_t phStream;
  CALL_TOPS(topsStreamCreate(&phStream));
  *stream_handle_ptr = (diopiStreamHandle_t)phStream;
  return diopiSuccess;
}

int32_t tops_destroy_stream(diopiStreamHandle_t stream_handle) {
  topsStream_t phStream = (topsStream_t)stream_handle;
  CALL_TOPS(topsStreamDestroy(phStream));
  return diopiSuccess;
}

int32_t tops_synchronize_stream(diopiStreamHandle_t stream_handle) {
  topsStream_t phStream = (topsStream_t)stream_handle;
  CALL_TOPS(topsStreamSynchronize(phStream));
  return diopiSuccess;
}

int32_t tops_memcpy_h2d_async(diopiStreamHandle_t stream_handle, void* dst,
                              const void* src, uint64_t bytes) {
  topsStream_t phStream = (topsStream_t)stream_handle;
  CALL_TOPS(topsMemcpyAsync(dst, src, bytes, topsMemcpyHostToDevice, phStream));
  return diopiSuccess;
}

int32_t tops_memcpy_d2h_async(diopiStreamHandle_t stream_handle, void* dst,
                              const void* src, uint64_t bytes) {
  topsStream_t phStream = (topsStream_t)stream_handle;
  CALL_TOPS(topsMemcpyAsync(dst, src, bytes, topsMemcpyDeviceToHost, phStream));
  return diopiSuccess;
}

int32_t tops_memcpy_d2d_async(diopiStreamHandle_t stream_handle, void* dst,
                              const void* src, uint64_t bytes) {
  topsStream_t phStream = (topsStream_t)stream_handle;
  CALL_TOPS(
      topsMemcpyAsync(dst, src, bytes, topsMemcpyDeviceToDevice, phStream));
  return diopiSuccess;
}

int32_t initLibrary() {
  diopiRegisterDeviceMallocFunc(tops_malloc);
  diopiRegisterDevMemFreeFunc(tops_free);
  diopiRegisterStreamCreateFunc(tops_make_stream);
  diopiRegisterStreamDestroyFunc(tops_destroy_stream);
  diopiRegisterSynchronizeStreamFunc(tops_synchronize_stream);
  diopiRegisterMemcpyD2HAsyncFunc(tops_memcpy_d2h_async);
  diopiRegisterMemcpyD2DAsyncFunc(tops_memcpy_d2d_async);
  diopiRegisterMemcpyH2DAsyncFunc(tops_memcpy_h2d_async);
  diopiRegisterGetLastErrorFunc(tops_get_last_error_string);

  return diopiSuccess;
}

int32_t finalizeLibrary() { return diopiSuccess; }

}  // extern "C"