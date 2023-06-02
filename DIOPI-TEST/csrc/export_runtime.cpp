/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink Inc.
 * @brief A reference implemention for DIOPI runtime, which is utilized to support conformance test suite of DIOPI
 */

#include <conform_test.h>
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <litert.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(export_runtime, m) {
    py::options options;
    options.disable_function_signatures();
    py::class_<diopiTensor, std::shared_ptr<diopiTensor>>(m, "diopiTensor", py::buffer_protocol())
        .def(py::init([](diopiSize_t* shape, diopiSize_t* stride, diopiDtype_t dtype, diopiDevice_t device, diopiContextHandle_t context, const void* src) {
            auto tensor = diopiTensor(shape, stride, dtype, device, context, src);
            return tensor;
        }))
        .def(py::init([](diopiSize_t* shape, diopiSize_t* stride, diopiDtype_t dtype, diopiDevice_t device, diopiContextHandle_t context) {
            auto tensor = diopiTensor(shape, stride, dtype, device, context, nullptr);
            return tensor;
        }))
        .def(py::init([]() {
            auto tensor = diopiTensor();
            return tensor;
        }))
        .def("shape", &diopiTensor::shape)
        .def("get_stride", &diopiTensor::stride)
        .def("get_dtype", &diopiTensor::dtype)
        .def("get_device", &diopiTensor::device)
        .def("numel", &diopiTensor::numel)
        .def("reset_shape", &diopiTensor::resetShape)
        .def("itemsize", &diopiTensor::elemSize)
        .def("context", &diopiTensor::getCtx)
        .def_buffer(&diopiTensor::buffer);
    py::class_<diopiContext>(m, "Context", py::buffer_protocol()).def(py::init<>()).def("clear_tensors", &diopiContext::clearTensors);
    py::enum_<diopiDevice_t>(m, "Device").value("Host", diopiDevice_t::diopi_host).value("AIChip", diopiDevice_t::diopi_device);
    py::enum_<diopiDtype_t>(m, "Dtype")
        .value("int8", diopiDtype_t::diopi_dtype_int8)
        .value("uint8", diopiDtype_t::diopi_dtype_uint8)
        .value("int16", diopiDtype_t::diopi_dtype_int16)
        .value("uint16", diopiDtype_t::diopi_dtype_uint16)
        .value("int32", diopiDtype_t::diopi_dtype_int32)
        .value("uint32", diopiDtype_t::diopi_dtype_uint32)
        .value("int64", diopiDtype_t::diopi_dtype_int64)
        .value("uint64", diopiDtype_t::diopi_dtype_uint64)
        .value("float16", diopiDtype_t::diopi_dtype_float16)
        .value("float32", diopiDtype_t::diopi_dtype_float32)
        .value("float64", diopiDtype_t::diopi_dtype_float64)
        .value("bool", diopiDtype_t::diopi_dtype_bool)
        .value("bfloat16", diopiDtype_t::diopi_dtype_bfloat16)
        .value("tfloat32", diopiDtype_t::diopi_dtype_tfloat32)
        .value("complex64", diopiDtype_t::diopi_dtype_complex64)
        .value("complex128", diopiDtype_t::diopi_dtype_complex128);
    py::enum_<diopiError_t>(m, "diopiError")
        .value("diopi_success", diopiError_t::diopiSuccess)
        .value("diopi_error_occurred", diopiError_t::diopiErrorOccurred)
        .value("diopi_not_inited", diopiError_t::diopiNotInited)
        .value("diopi_no_registered_stream_create_function", diopiError_t::diopiNoRegisteredStreamCreateFunction)
        .value("diopi_no_registered_stream_destory_function", diopiError_t::diopiNoRegisteredStreamDestoryFunction)
        .value("diopi_no_registered_stream_sync_function", diopiError_t::diopiNoRegisteredStreamSyncFunction)
        .value("diopi_no_registered_device_memory_malloc_function", diopiError_t::diopiNoRegisteredDeviceMemoryMallocFunction)
        .value("diopi_no_registered_device_memory_free_function", diopiError_t::diopiNoRegisteredDeviceMemoryFreeFunction)
        .value("diopi_no_registered_device2device_memory_copy_function", diopiError_t::diopiNoRegisteredDevice2DdeviceMemoryCopyFunction)
        .value("diopi_no_registered_device2host_memory_copy_function", diopiError_t::diopiNoRegisteredDevice2HostMemoryCopyFunction)
        .value("diopi_no_registered_host2device_memory_copy_function", diopiError_t::diopiNoRegisteredHost2DeviceMemoryCopyFunction)
        .value("diopi_no_registered_get_last_error_function", diopiError_t::diopiNoRegisteredGetLastErrorFunction)
        .value("diopi_5d_not_supported", diopiError_t::diopi5DNotSupported)
        .value("diopi_no_implement", diopiError_t::diopiNoImplement)
        .value("diopi_dtype_not_supported", diopiError_t::diopiDtypeNotSupported);
    py::enum_<diopiReduction_t>(m, "diopiReduction")
        .value("ReductionNone", diopiReduction_t::ReductionNone)
        .value("ReductionMean", diopiReduction_t::ReductionMean)
        .value("ReductionSum", diopiReduction_t::ReductionSum)
        .value("ReductionEND", diopiReduction_t::ReductionEND);
    py::enum_<diopiRoundMode_t>(m, "diopiRoundMode")
        .value("RoundModeNone", diopiRoundMode_t::RoundModeNone)
        .value("RoundModeTrunc", diopiRoundMode_t::RoundModeTrunc)
        .value("RoundModeFloor", diopiRoundMode_t::RoundModeFloor)
        .value("RoundModeEND", diopiRoundMode_t::RoundModeEND);
    py::class_<diopiSize_t>(m, "diopiSize")
        .def(py::init<>())
        .def(py::init([](py::list& sizeList, int64_t nums) {
            int64_t* sizes = new int64_t[nums];
            for (int i = 0; i < nums; ++i) sizes[i] = sizeList[i].cast<int64_t>();
            auto self = diopiSize_t(sizes, nums);
            return self;
        }))
        .def(py::init<const int64_t*, int64_t>())
        .def_property_readonly("len", &diopiSize_t::getLen)
        .def_property_readonly("data", [](diopiSize_t& size) {
            std::vector<int64_t> data(size.len);
            for (int i = 0; i < size.len; i++) data[i] = size.data[i];
            return data;
        });
    py::class_<diopiScalar_t>(m, "diopiScalar")
        .def(py::init<>())
        .def(py::init([](diopiDtype_t dtype, double val) {
            auto scalar = diopiScalar_t();
            scalar.stype = dtype;
            scalar.fval = val;
            return scalar;
        }))
        .def(py::init([](diopiDtype_t dtype, int64_t val) {
            auto scalar = diopiScalar_t();
            scalar.stype = dtype;
            scalar.ival = val;
            return scalar;
        }))
        .def_property_readonly("type", &diopiScalar_t::type)
        .def_property_readonly("val", &diopiScalar_t::val);
    py::class_<PtrWrapper<diopiTensor>>(m, "TensorP").def(py::init<diopiTensor*>()).def(py::init<py::none>()).def("data", &PtrWrapper<diopiTensor>::operator*);
    m.def("diopi_tensor_copy_to_buffer", [](diopiContextHandle_t context, diopiConstTensorHandle_t tensor, py::array_t<double>& arr) {
        py::buffer_info buf = arr.request();
        diopiTensorCopyToBuffer(context, tensor, buf.ptr);
    });
    m.def("get_last_error_string", &diopiGetLastErrorString);
    m.def("diopi_init", &diopiInit);
    m.def("diopi_finalize", &diopiFinalize);
    m.def("finalize_library", &finalizeLibrary);
}
