from ctypes import cdll, byref, Structure, POINTER, cast
from .dtype import *
import numpy as np
import atexit


@unique
class Device(Enum):
    Host = 0
    AIChip = 1


def device(dev : str) -> Device:
    if dev == "cpu" or dev == "host":
        return Device(0)
    else:
        return Device(1)


def from_numpy_dtype(dtype : np.dtype) -> Dtype:
    if dtype == np.int8:
        return Dtype.int8
    elif dtype == np.int16:
        return Dtype.int16
    elif dtype == np.int32:
        return Dtype.int32
    elif dtype == np.int64:
        return Dtype.int64
    elif dtype == np.uint8:
        return Dtype.uint8
    elif dtype == np.uint16:
        return Dtype.uint16
    elif dtype == np.uint32:
        return Dtype.uint32
    elif dtype == np.uint64:
        return Dtype.uint64
    elif dtype == np.float16:
        return Dtype.float16
    elif dtype == np.float64:
        return Dtype.float64
    elif dtype == np.bool_:
        return Dtype.bool
    else:
        return Dtype.float32


def to_numpy_dtype(dtype : Dtype) -> np.dtype:
    if dtype == Dtype.int8:
        return np.int8
    elif dtype == Dtype.int16:
        return np.int16
    elif dtype == Dtype.int32:
        return np.int32
    elif dtype == Dtype.int64:
        return np.int64
    elif dtype == Dtype.uint8:
        return np.uint8
    elif dtype == Dtype.uint16:
        return np.uint16
    elif dtype == Dtype.uint32:
        return np.uint32
    elif dtype == Dtype.uint64:
        return np.uint64
    elif dtype == Dtype.float16:
        return np.float16
    elif dtype == Dtype.float64:
        return np.float64
    elif dtype == Dtype.bool:
        return np.bool_
    else:
        return np.float32


diopirt_lib = cdll.LoadLibrary("../lib/libdiopirt.so")
diopirt_lib.diopiInit()

device_impl_lib = cdll.LoadLibrary("../lib/libdevice_impl.so")
device_impl_lib.initLibrary()


def on_litert_exit():
    device_impl_lib.finalizeLibrary()
    diopirt_lib.diopiFinalize()


atexit.register(on_litert_exit)


TensorHandle = c_void_p
ContextHandle = c_void_p


class Context:
    _c_lib = diopirt_lib

    def __init__(self):
        self.context_handle = ContextHandle()
        self.__class__._c_lib.diopiCreateContext(byref(self.context_handle))

    def __del__(self):
        self.__class__._c_lib.diopiDestoryContext(self.context_handle)

    def get_handle(self):
        return self.context_handle


default_context = Context()


class Sizes(Structure):
    _fields_ = [("data", POINTER(c_int64)), ("len", c_int64)]

    def __init__(self, shape=()):
        self.carray = (c_int64 * len(shape))(*shape)
        carray = (c_int64 * len(shape))(*shape)
        super().__init__(self.carray, len(shape))


def get_last_error():
    last_error_str = c_char_p()
    diopirt_lib._getLastErrorString(byref(last_error_str))
    if last_error_str.value is not None:
        last_error_str = str(last_error_str.value, encoding="utf-8")
    return last_error_str


class Tensor:
    def __init__(
        self,
        size,
        dtype,
        device=device("host"),
        stride=None,
        context_handle=default_context.get_handle(),
    ):
        self.tensor_handle = TensorHandle()
        self.context_handle = context_handle

        assert isinstance(size, (tuple, list))
        assert isinstance(device, Device)
        assert isinstance(dtype, Dtype)

        diopirt_lib.diopiRequireTensor(
            self.context_handle,
            byref(self.tensor_handle),
            byref(Sizes(tuple(size))),
            None if stride is None else byref(Sizes(tuple(stride))),
            dtype.value,
            device.value,
        )

    def __del__(self):
        diopirt_lib.diopiDestoryTensor(self.context_handle, self.tensor_handle)

    def __str__(self):
        check_return_value(diopirt_lib.diopiDumpTensor(
            self.context_handle, self.tensor_handle), throw_exception=False)
        return ""

    def raw_like(self, device=None):
        size = self.size()
        stride = self.stride()
        dtype = self.get_dtype()
        target_device = self.get_device() if device is None else device
        return Tensor(size=size, dtype=dtype, device=target_device, stride=stride, context_handle=self.context_handle)

    def numel(self):
        numel = c_int64()
        diopirt_lib.diopiGetTensorNumel(self.tensor_handle, byref(numel))
        return numel.value

    def size(self):
        cshape = Sizes()
        diopirt_lib.diopiGetTensorShape(self.tensor_handle, byref(cshape))
        shape = []
        for i in range(cshape.len):
            shape.append(cshape.data[i])
        self.shape = tuple(shape)
        return self.shape

    def stride(self):
        cstride = Sizes()
        diopirt_lib.diopiGetTensorStride(self.tensor_handle, byref(cstride))
        stride = []
        for i in range(cstride.len):
            stride.append(cstride.data[i])
        self.strides = tuple(stride)
        return self.strides

    def itemsize(self):
        itemsize = c_int64()
        diopirt_lib.diopiGetTensorElemSize(self.tensor_handle, byref(itemsize))
        return itemsize.value

    def get_device(self):
        device = c_int32()
        diopirt_lib.diopiGetTensorDevice(self.tensor_handle, byref(device))
        self.device = Device(device.value)
        return self.device

    def get_dtype(self):
        dtype = c_int32()
        diopirt_lib.diopiGetTensorDtype(self.tensor_handle, byref(dtype))
        self.dtype = Dtype(dtype.value)
        return self.dtype

    def data_ptr(self):
        ptr = c_void_p()
        diopirt_lib.diopiGetTensorData(self.tensor_handle, byref(ptr))
        self.c_buf = cast(ptr, POINTER(dtype_to_ctype(self.dtype)))
        return ptr.value

    @staticmethod
    def from_numpy(darray, device=device("device")):
        if not isinstance(darray, (np.generic, np.ndarray)):
            raise TypeError("expected np.ndarray (got {})".format(type(darray)))

        dtype = from_numpy_dtype(darray.dtype)
        stride = [int(darray.strides[i]/darray.itemsize) for i in range(len(darray.strides))]
        tr = Tensor(size=darray.shape, dtype=dtype, device=device, stride=stride)
        diopirt_lib._diopiTensorCopyFromBuffer(tr.context_handle, c_void_p(darray.ctypes.data), tr.tensor_handle)
        return tr

    def numpy(self) -> np.ndarray:
        dtype = to_numpy_dtype(self.get_dtype())
        itemsize = self.itemsize()
        stride = self.stride()
        strides = [int(stride[i]*itemsize) for i in range(len(stride))]
        darray = np.ndarray(shape=self.size(), dtype=dtype, strides=strides)
        diopirt_lib._diopiTensorCopyToBuffer(self.context_handle, self.tensor_handle, c_void_p(darray.ctypes.data))
        return darray
