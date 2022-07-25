from . import *
from ctypes import c_float, byref


def tensor_to_host(self):
    if self.device == device("cpu"):
        return self
    else:
        tensor_handle = TensorHandle()
        sync = 1
        check_return_value(diopirt_lib.diopiTransferTensorToHost(
            self.context_handle, self.tensor_handle, byref(tensor_handle), sync))
        return Tensor(tensor_handle=tensor_handle)


Tensor.cpu = tensor_to_host
Tensor.to_host = tensor_to_host


def tensor_to_device(self):
    if self.device == device("device"):
        return self
    else:
        tensor_handle = TensorHandle()
        sync = 1
        check_return_value(
            diopirt_lib.diopiTransferTensorToDevice(
                self.context_handle, self.tensor_handle, byref(tensor_handle), sync
            )
        )
        return Tensor(tensor_handle=tensor_handle)


Tensor.to_device = tensor_to_device


def to(tensor, dev):
    if isinstance(dev, str):
        dev = device(dev)
    assert isinstance(dev, type(device("host")))
    if dev == device("host"):
        return tensor_to_host(tensor)
    else:
        return tensor_to_device(tensor)


Tensor.to = to


def fill(tensor, value):
    error_code = device_impl_lib.fill(tensor.context_handle, tensor.tensor_handle, c_float(value))
    check_return_value(error_code)
    return tensor


Tensor.fill_ = fill


def add(input, other, out=None):
    if out is None:
        out_tensor_handle = TensorHandle()
    else:
        assert isinstance(out, Tensor)
        out_tensor_handle = out.tensor_handle

    check_return_value(
        device_impl_lib.add(
            input.context_handle,
            byref(out_tensor_handle),
            input.tensor_handle,
            other.tensor_handle,
        )
    )
    if out is None:
        out = Tensor(tensor_handle=out_tensor_handle)
    else:
        out.update_member_property()
    return out
