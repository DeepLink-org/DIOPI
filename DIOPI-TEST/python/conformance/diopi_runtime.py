# Copyright (c) 2023, DeepLink.
import os
from enum import Enum, unique
import ctypes
from ctypes import (CDLL, RTLD_GLOBAL, cdll, byref, Union, POINTER)
from ctypes import (c_void_p, c_char_p, c_int64, c_double)
from .dtype import Dtype
import numpy as np
import atexit
from diopi_runtime import diopiTensor, diopiSize, diopiScalar, Context, Device, diopi_tensor_copy_to_buffer, get_last_error_string

# @unique
# class Device(Enum):
#     Host = 0
#     AIChip = 1


def device(dev: str) -> Device:
    if dev == "cpu" or dev == "host":
        return Device.Host
    else:
        return Device.AIChip


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


def compute_nhwc_stride_2d(sizes, itemsize=1):
    dim = len(sizes)
    strides = [itemsize for i in range(dim)]
    assert dim == 3 or dim == 4, "not supported dim"
    if dim == 3:
        strides[0] = itemsize
        strides[2] = strides[0] * sizes[0]
        strides[1] = strides[2] * sizes[2]
    elif dim == 4:
        strides[1] = itemsize
        strides[3] = strides[0] * sizes[1]
        strides[2] = strides[3] * sizes[3]
        strides[0] = strides[2] * sizes[2]
    return strides


def compute_nhwc_stride_3d(sizes, itemsize=1):
    dim = len(sizes)
    strides = [itemsize for i in range(dim)]
    assert dim == 4 or dim == 5, "not supported dim"
    if dim == 4:
        strides[0] = itemsize
        strides[3] = strides[0] * sizes[0]
        strides[2] = strides[3] * sizes[3]
        strides[1] = strides[2] * sizes[2]
    elif dim == 5:
        strides[1] = itemsize
        strides[4] = strides[0] * sizes[1]
        strides[3] = strides[4] * sizes[4]
        strides[2] = strides[3] * sizes[3]
        strides[0] = strides[2] * sizes[2]
    return strides


def compute_nhwc_stride(size, itemsize=1, name=None):
    if name == '2d':
        return compute_nhwc_stride_2d(size, itemsize)
    if name == '3d':
        return compute_nhwc_stride_3d(size, itemsize)

    dim = len(size)
    if dim < 5:
        return compute_nhwc_stride_2d(size, itemsize)
    else:
        return compute_nhwc_stride_3d(size, itemsize)


_cur_dir = os.path.dirname(os.path.abspath(__file__))
diopirt_lib = CDLL(name=os.path.join(_cur_dir, "../../../DIOPI-IMPL/lib/libdiopirt.so"), mode=RTLD_GLOBAL)
diopirt_lib.diopiInit()

device_impl_lib = cdll.LoadLibrary(os.path.join(_cur_dir, "../../../DIOPI-IMPL/lib/libdiopi_impl.so"))


def on_diopi_rt_exit():
    diopirt_lib.finalizeLibrary()
    diopirt_lib.diopiFinalize()


atexit.register(on_diopi_rt_exit)


def get_last_error():
    last_error_str = get_last_error_string()
    return last_error_str


default_context = Context()


class Sizes(diopiSize):

    def __init__(self, shape=()):
        super(Sizes, self).__init__(list(shape), len(shape))
        self.shape = self.data


class Scalar(diopiScalar):

    def __init__(self, value, dtype=None):
        if dtype is None:
            dtype = Dtype.int64 if isinstance(value, int) else Dtype.float64
        diopiScalar.__init__(self, dtype, value)


class Tensor(diopiTensor):
    def __init__(
        self,
        size=None,
        dtype=None,
        stride=None,
        context=default_context,
        data_ptr=None
    ):
        if size is None:
            return diopiTensor.__init__(self)

        if isinstance(size, (tuple, list)):
            size = Sizes(list(size))

        if data_ptr is None:
            diopiTensor.__init__(self, size, stride, dtype,
                                 Device.AIChip, context)
        else:
            diopiTensor.__init__(self, size, stride, dtype,
                                 Device.AIChip, context, data_ptr)

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
                      context=self.context())

    def size(self):
        return self.shape()

    def reset_shape(self, shape):
        assert isinstance(shape, (tuple, list))
        self.reset_shape(Sizes(list(shape)))

    @classmethod
    def from_numpy(cls, darray, context=None):
        if not isinstance(darray, (np.generic, np.ndarray)):
            raise TypeError(f"expected np.ndarray (got {type(darray)})")
        dtype = from_numpy_dtype(darray.dtype)
        stride = [int(darray.strides[i] / darray.itemsize)
                  for i in range(len(darray.strides))]

        size = Sizes(list(darray.shape))
        stride = Sizes(list(stride))
        PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
        PyCapsule_New = ctypes.pythonapi.PyCapsule_New
        PyCapsule_New.restype = ctypes.py_object
        PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor)
        capsule = PyCapsule_New(c_void_p(darray.ctypes.data), None, PyCapsule_Destructor(0))
        if context:
            tr = cls(size=size, dtype=dtype, stride=stride, data_ptr=capsule, context=context)
        else:
            tr = cls(size=size, dtype=dtype, stride=stride, data_ptr=capsule, )
        return tr

    def numpy(self) -> np.ndarray:
        dtype = to_numpy_dtype(self.get_dtype())
        itemsize = self.itemsize()
        stride = self.get_stride()
        strides = [int(stride.data[i] * itemsize) for i in range(stride.len)]
        darray = np.ndarray(shape=self.size().data, dtype=dtype, strides=strides)
        darray = np.array(self, copy=False)
        diopi_tensor_copy_to_buffer(self.context(), self, darray)
        return darray


def raw_like(tensor) -> Tensor:
    return tensor.raw_like()
