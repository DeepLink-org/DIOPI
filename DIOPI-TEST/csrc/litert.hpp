/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink Inc.
 * @brief A reference implemention for DIOPI runtime, which is utilized to support conformance test suite of DIOPI
 */

#include <diopi/diopirt.h>
#include <conform_test.h>
#include <diopi/functions.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <set>
#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;
// #define inline
// #include <diopi_adaptors.hpp>
// #undef inline
template <class T>
class PtrWrapper
{
    public:
        PtrWrapper() : ptr(nullptr) {}
        PtrWrapper(py::none) : ptr(nullptr) {}
        PtrWrapper(T* ptr) : ptr(ptr) {}
        PtrWrapper(const PtrWrapper& other) : ptr(other.ptr) {}
        T& operator* () const { std::cout << ptr << std::endl; return *ptr; }
        T* operator->() const { return  ptr; }
        void set(T val) {
            std::cout << val << std::endl;
            *ptr = val;
            std::cout << *ptr << std::endl;
            std::cout << ptr << std::endl;
            std::cout << *ptr << std::endl;
        }
        T* get() const { 
            return ptr; 
        }
        void destroy() { delete ptr; }
        // T& operator[](std::size_t idx) const { return ptr[idx]; }
    private:
        T* ptr;
};


extern "C" {

class Storage final {
private:
    malloc_func_t malloc_fn_;
    free_func_t free_fn_;
    int64_t nbytes_;
    void* ptr_;

public:
    Storage(malloc_func_t malloc_fn, free_func_t free_fn, int64_t nbytes) : malloc_fn_(malloc_fn), free_fn_(free_fn), nbytes_(nbytes) {
        assert(free_fn_);
        assert(malloc_fn_);
        ptr_ = malloc_fn_(nbytes_);
    }

    ~Storage() {
        free_fn_(ptr_);
        ptr_ = nullptr;
        nbytes_ = 0;
    }

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    int64_t nbytes() const { return nbytes_; }
};

struct diopiTensor {
private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> stride_;
    diopiDtype_t dtype_;
    diopiDevice_t device_;
    int64_t numel_;
    std::shared_ptr<Storage> storage_ = nullptr;
    diopiContextHandle_t context_;

public:
    diopiTensor(const diopiSize_t* shape, const diopiSize_t* stride, diopiDtype_t dtype,
                         diopiDevice_t device, diopiContextHandle_t context, const void* src);
    diopiTensor() {};
    ~diopiTensor() {};

    diopiSize_t shape() const {
        diopiSize_t size(shape_.data(), static_cast<int64_t>(shape_.size()));
        return size;
    }

    diopiSize_t stride() const {
        diopiSize_t stride(stride_.data(), static_cast<int64_t>(stride_.size()));
        return stride;
    }

    bool reset_shape(const diopiSize_t* size);

    diopiDtype_t dtype() const { return dtype_; }
    diopiDevice_t device() const { return device_; }
    int64_t numel() const { return numel_; }

    void* data() { return storage_->data(); }
    const void* data() const { return storage_->data(); }
    int64_t nbytes() const { return storage_->nbytes(); }
    int64_t elemSize() const;    
    py::buffer_info buffer() const noexcept {
        if (storage_ == nullptr) {
            return py::buffer_info();
        }
        try {
            diopiStreamHandle_t stream;
            auto ptr = malloc(nbytes());;
            diopiGetStream(get_ctx(), &stream);
            device_memcpy_d2h_async(stream, ptr, data(), nbytes());
            device_synchronize_stream(stream);
            ssize_t esize = elemSize();
            std::vector<ssize_t> buffer_shape;
            std::vector<ssize_t> buffer_strides;
            for (int64_t i = 0; i < shape().len; ++i) {
                buffer_shape.push_back(shape().data[i]);
                buffer_strides.push_back(stride().data[i] * esize);
            }
            static const char fmt[] = "bBhHiIlLefd?";
            auto temp = reinterpret_cast<double*>(ptr);
            return py::buffer_info(
                ptr,                        /* Pointer to buffer */
                esize,                      /* Size of one scalar */
                std::string(1, fmt[static_cast<size_t>(dtype())]), /* Python struct format descriptor */
                shape().len,                  /* Number of dimensions */
                buffer_shape,               /* Buffer dimensions */
                buffer_strides              /* Strides (in bytes) for each index */
            );  // NOLINT
        } catch (const std::exception& e) {
            // XXX(xintian): return an invalid buffer to raise an exception.
            return py::buffer_info((char*){0}, -1);
        }
    }

    diopiContextHandle_t get_ctx() const { return context_; }
};

struct diopiContext {
private:
    diopiStreamHandle_t stream_{nullptr};
    std::set<diopiTensorHandle_t> setTensors_;

public:
    diopiContext() {}

    ~diopiContext() {
        if (nullptr != stream_) {
            device_destroy_stream(stream_);
        }
        for (auto it : setTensors_) {
            delete it;
        }
        setTensors_.clear();
    }

    diopiStreamHandle_t getStreamHandle() {
        if (stream_ == nullptr) {
            device_make_stream(&stream_);
        }
        return stream_;
    }

    diopiTensorHandle_t createTensor(const diopiSize_t* size, const diopiSize_t* stride, const diopiDtype_t dtype, const diopiDevice_t dev) {
        diopiTensorHandle_t tensor = new diopiTensor(size, stride, dtype, dev, this, nullptr);
        setTensors_.insert(tensor);
        return tensor;
    }

    void destroyTensor(diopiTensorHandle_t tensor) {
        auto it = setTensors_.find(tensor);
        if (setTensors_.end() != it) {
            setTensors_.erase(it);
            delete tensor;
        }
    }

    void clearTensors() {
        if (stream_ != nullptr) {
            for (auto it : setTensors_) {
                delete it;
            }
            setTensors_.clear();
        }
    }
};

DIOPI_RT_API diopiError_t _diopiTensorCopyToBuffer(diopiContextHandle_t      ctx,
                                                diopiConstTensorHandle_t tensor,
                                                void*                     dst);

DIOPI_RT_API diopiError_t diopiInit();
}  // extern "C"


