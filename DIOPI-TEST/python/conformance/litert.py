from ctypes import cdll, byref, Structure, POINTER, cast
from .dtype import *
import atexit


@unique
class Device(Enum):
    Host = 0
    AIChip = 1


def device(dev):
    if dev == "cpu" or dev == "host":
        return Device(0)
    else:
        return Device(1)


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
        size=None,
        dtype=None,
        device=device("host"),
        stride=None,
        tensor_handle=None,
        context_handle=default_context.get_handle(),
    ):
        self.tensor_handle = TensorHandle()
        self.context_handle = context_handle
        if size is not None:
            assert size is not None
            assert device is not None
            assert dtype is not None
            assert isinstance(size, (tuple, list))
            assert isinstance(device, Device)
            assert isinstance(dtype, Dtype)

            diopirt_lib.diopiRequireTensor(
                self.context_handle,
                byref(self.tensor_handle),
                byref(Sizes(tuple(size))),
                None,
                dtype.value,
                device.value,
            )
        elif tensor_handle is not None:
            self.tensor_handle = tensor_handle

        self.update_member_property()

    def __del__(self):
        diopirt_lib.diopiDestoryTensor(self.context_handle, self.tensor_handle)

    def __str__(self):
        check_return_value(diopirt_lib.diopiDumpTensor(
            self.context_handle, self.tensor_handle), throw_exception=False)
        return ""

    def update_member_property(self):
        self.size()
        self.stride()
        self.numel()
        self.get_device()
        self.get_dtype()
        self.data_ptr()

    def numel(self):
        numel = c_uint64()
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
