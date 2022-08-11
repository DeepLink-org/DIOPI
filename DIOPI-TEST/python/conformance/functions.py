from .litert import Tensor, device_impl_lib
from .utils import raw_like
from .dtype import check_return_value
from ctypes import c_float, byref


def add(input, other, out=None) -> Tensor:
    if out is None:
        out = raw_like(input)
    else:
        assert isinstance(out, Tensor)

    ret = device_impl_lib.add(input.context_handle, out.tensor_handle,
                              input.tensor_handle, other.tensor_handle)
    check_return_value(ret)
    return out
