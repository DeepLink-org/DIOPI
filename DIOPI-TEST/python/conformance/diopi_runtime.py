import os
from enum import Enum, unique
from ctypes import (cdll, byref, Structure, Union, POINTER)
from ctypes import (c_void_p, c_char_p, c_int64, c_int32, c_double)
from .dtype import Dtype
import numpy as np
import atexit


@unique
class Device(Enum):
    Host = 0
    AIChip = 1


def device(dev: str) -> Device:
    if dev == "cpu" or dev == "host":
        return Device(0)
    else:
        return Device(1)


def from_numpy_dtype(dtype: np.dtype) -> Dtype:
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
    elif dtype == np.float32:
        return Dtype.float32
    elif dtype == np.float64:
        return Dtype.float64
    elif dtype == np.bool_:
        return Dtype.bool
    else:
        return None


def to_numpy_dtype(dtype: Dtype) -> np.dtype:
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
    elif dtype == Dtype.float32:
        return np.float32
    elif dtype == Dtype.float64:
        return np.float64
    elif dtype == Dtype.bool:
        return np.bool_
    else:
        return None


_cur_dir = os.path.dirname(os.path.abspath(__file__))
diopirt_lib = cdll.LoadLibrary(os.path.join(_cur_dir, "../../lib/libdiopirt.so"))
diopirt_lib.diopiInit()

device_impl_lib = cdll.LoadLibrary(os.path.join(_cur_dir, "../../lib/libdevice_impl.so"))
device_impl_lib.initLibrary()


def on_diopi_rt_exit():
    device_impl_lib.finalizeLibrary()
    diopirt_lib.diopiFinalize()


atexit.register(on_diopi_rt_exit)


def get_last_error():
    last_error_str = c_char_p()
    diopirt_lib._getLastErrorString(byref(last_error_str))
    if last_error_str.value is not None:
        last_error_str = str(last_error_str.value, encoding="utf-8")
    return last_error_str


ContextHandle = c_void_p
TensorHandle = c_void_p


class Context:
    _c_lib = diopirt_lib

    def __init__(self):
        self.context_handle = ContextHandle()
        self.__class__._c_lib._diopiCreateContext(byref(self.context_handle))

    def __del__(self):
        self.__class__._c_lib._diopiDestroyContext(self.context_handle)

    def get_handle(self):
        return self.context_handle


default_context = Context()


class Sizes(Structure):
    _fields_ = [("data", POINTER(c_int64)), ("len", c_int64)]

    def __init__(self, shape=()):
        self.carray = (c_int64 * len(shape))(*shape)
        super().__init__(self.carray, len(shape))


class ScalarUnion(Union):
    _fields_ = [("fval", c_double), ("ival", c_int64)]


class Scalar(Structure):
    _fields_ = [("stype", c_int32), ("val", ScalarUnion)]

    def __init__(self, dtype, value):
        self.stype = dtype.value
        if dtype in [Dtype.float16, Dtype.float32, Dtype.float64]:
            self.val.fval = value
        else:
            self.val.ival = value
        super().__init__(dtype.value, self.val)


class Tensor:
    def __init__(
        self,
        size,
        dtype,
        stride=None,
        context_handle=default_context.get_handle(),
        tensor_handle=None,
    ):
        if tensor_handle is not None and size is None:
            self.tensor_handle = tensor_handle
            self.context_handle = context_handle
        else:
            self.tensor_handle = TensorHandle()
            self.context_handle = context_handle

            assert isinstance(size, (tuple, list))
            assert isinstance(dtype, Dtype)

            diopirt_lib.diopiRequireTensor(
                self.context_handle,
                byref(self.tensor_handle),
                byref(Sizes(tuple(size))),
                None if stride is None else byref(Sizes(tuple(stride))),
                dtype.value,
                Device.AIChip.value
            )

    @classmethod
    def from_handle(cls, tensor_handle):
        ctx_handle = ContextHandle()
        diopirt_lib._diopiTensorGetCtxHandle(tensor_handle, byref(ctx_handle))
        return cls(size=None, dtype=None, context_handle=ctx_handle, tensor_handle=tensor_handle)

    def __del__(self):
        diopirt_lib._diopiDestoryTensor(self.context_handle,
                                        self.tensor_handle)

    def __str__(self):
        array = self.numpy()
        string = f"{array.__str__()}\n"
        string += f"{self.get_dtype()}, shape:{self.size()},\
                     stride:{self.get_stride()}, numel:{self.numel()}\n"
        return string

    def raw_like(self):
        size = self.size()
        stride = self.get_stride()
        dtype = self.get_dtype()
        return Tensor(size=size, dtype=dtype, stride=stride,
                      context_handle=self.context_handle)

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

    def shape(self):
        return self.size()

    def get_stride(self):
        cstride = Sizes()
        diopirt_lib.diopiGetTensorStride(self.tensor_handle, byref(cstride))
        stride = []
        for i in range(cstride.len):
            stride.append(cstride.data[i])
        self.stride = tuple(stride)
        return self.stride

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

    def reset_shape(self, shape):
        assert isinstance(shape, (tuple, list))
        diopirt_lib._diopiTensorResetShape(self.tensor_handle, byref(Sizes(tuple(shape))))

    @classmethod
    def from_numpy(cls, darray):
        if not isinstance(darray, (np.generic, np.ndarray)):
            raise TypeError(f"expected np.ndarray (got {type(darray)})")

        dtype = from_numpy_dtype(darray.dtype)
        stride = [int(darray.strides[i]/darray.itemsize)
                  for i in range(len(darray.strides))]
        tr = cls(size=darray.shape, dtype=dtype, stride=stride)
        diopirt_lib._diopiTensorCopyFromBuffer(tr.context_handle,
                                               c_void_p(darray.ctypes.data),
                                               tr.tensor_handle)
        return tr

    def numpy(self) -> np.ndarray:
        dtype = to_numpy_dtype(self.get_dtype())
        itemsize = self.itemsize()
        stride = self.get_stride()
        strides = [int(stride[i]*itemsize) for i in range(len(stride))]
        darray = np.ndarray(shape=self.size(), dtype=dtype, strides=strides)
        diopirt_lib._diopiTensorCopyToBuffer(self.context_handle,
                                             self.tensor_handle,
                                             c_void_p(darray.ctypes.data))
        return darray


def raw_like(tensor) -> Tensor:
    return tensor.raw_like()
