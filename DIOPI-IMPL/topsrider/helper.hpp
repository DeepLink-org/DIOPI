/*
* SPDX-FileCopyrightText: Copyright (c) 2022 Enflame. All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef _DIOPI_REFERENCE_IMPLTOPS_HELPER_HPP_
#define _DIOPI_REFERENCE_IMPLTOPS_HELPER_HPP_

#include <diopi/diopirt.h>
#include <tops_runtime.h>
#include <topsdnn.h>
#include <iostream>
#include <mutex>
#include "error.h"

#define DIOPI_CALL(Expr)       \
  {                            \
    diopiError_t ret = Expr;   \
    if (diopiSuccess != ret) { \
      return ret;              \
    }                          \
  }

namespace impl {

namespace tops {

template <typename TensorType>
struct DataType;

template <>
struct DataType<diopiTensorHandle_t> {
  using type = void*;

  static void* data(diopiTensorHandle_t& tensor) {
    void* data;
    diopiGetTensorData(&tensor, &data);
    return data;
  }
};

template <>
struct DataType<diopiConstTensorHandle_t> {
  using type = const void*;

  static const void* data(diopiConstTensorHandle_t& tensor) {
    const void* data;
    diopiGetTensorDataConst(&tensor, &data);
    return data;
  }
};

template <typename TensorType>
class DiopiTensor final {
 public:
  explicit DiopiTensor(TensorType& tensor) : tensor_(tensor) {}

  diopiDevice_t device() const {
    diopiDevice_t device;
    diopiGetTensorDevice(tensor_, &device);
    return device;
  }
  diopiDtype_t dtype() const {
    diopiDtype_t dtype;
    diopiGetTensorDtype(tensor_, &dtype);
    return dtype;
  }

  const diopiSize_t& shape() {
    diopiGetTensorShape(tensor_, &shape_);
    return shape_;
  }
  const diopiSize_t& stride() {
    diopiGetTensorStride(tensor_, &stride_);
    return stride_;
  }

  int64_t numel() const {
    int64_t numel;
    diopiGetTensorNumel(tensor_, &numel);
    return numel;
  }
  int64_t elemsize() const {
    int64_t elemsize;
    diopiGetTensorElemSize(tensor_, &elemsize);
    return elemsize;
  }

  typename DataType<TensorType>::type data() {
    return DataType<TensorType>::data(tensor_);
  }

 protected:
  TensorType tensor_;

  diopiSize_t shape_;
  diopiSize_t stride_;
};

template <typename TensorType>
auto makeTensor(TensorType& tensor) -> DiopiTensor<TensorType> {
  return DiopiTensor<TensorType>(tensor);
}

inline DiopiTensor<diopiTensorHandle_t> requiresTensor(diopiContextHandle_t ctx,
                                                       const diopiSize_t& size,
                                                       diopiDtype_t dtype) {
  diopiTensorHandle_t tensor;
  diopiRequireTensor(ctx, &tensor, &size, nullptr, dtype, diopi_device);
  return makeTensor(tensor);
}

inline DiopiTensor<diopiTensorHandle_t> requiresBuffer(diopiContextHandle_t ctx,
                                                       int64_t num_bytes) {
  diopiTensorHandle_t tensor;
  diopiRequireBuffer(ctx, &tensor, num_bytes, diopi_device);
  return makeTensor(tensor);
}

inline topsStream_t getStream(diopiContextHandle_t ctx) {
  diopiStreamHandle_t stream_handle;
  diopiGetStream(ctx, &stream_handle);
  return static_cast<topsStream_t>(stream_handle);
}

template <typename... Types>
void set_last_error_string(const char* szFmt, Types&&... args) {
  char szBuf[4096] = {0};
  sprintf(szBuf, szFmt, std::forward<Types>(args)...);
  _set_last_error_string(szBuf);
}

class SingleDNN;
class SafeDeletor {
 public:
  void operator()(SingleDNN* sf) { delete sf; }
};
class SingleDNN {
 private:
  SingleDNN() {
    auto status = topsdnnCreate(&handle);
    if (status != TOPSDNN_STATUS_SUCCESS) {
      throw std::runtime_error("topsdnnHandle_t create failed!");
    }
  }
  ~SingleDNN() {
    auto status = topsdnnDestroy(handle);
    if (status != TOPSDNN_STATUS_SUCCESS) {
      throw std::runtime_error("topsdnnHandle_t destroy failed!");
    }
  }

  SingleDNN(const SingleDNN&) = delete;
  SingleDNN& operator=(const SingleDNN&) = delete;
  friend class SafeDeletor;
  topsdnnHandle_t handle;

 public:
  static std::shared_ptr<SingleDNN> GetInst() {
    if (single != nullptr) {
      return single;
    }
    s_mutex.lock();
    if (single != nullptr) {
      s_mutex.unlock();
      return single;
    }
    single = std::shared_ptr<SingleDNN>(new SingleDNN, SafeDeletor());
    s_mutex.unlock();
    return single;
  }
  topsdnnHandle_t getHandle() { return handle; };

 private:
  static std::shared_ptr<SingleDNN> single;
  static std::mutex s_mutex;
};

}  // namespace tops

}  // namespace impl

#endif  // _DIOPI_REFERENCE_IMPLtops_HELPER_HPP_