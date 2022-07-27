from . import *
from ctypes import c_float, byref


def raw_like(tensor, device=None) -> Tensor:
    return tensor.raw_like(device)


def tensor_to_host(tensor) -> Tensor:
    if tensor.device == device("cpu"):
        return tensor
    else:
        cpu_tensor = raw_like(tensor, device("cpu"))
        check_return_value(
            diopirt_lib._diopiTensorCopyDeviceToHost(
                tensor.context_handle, tensor.tensor_handle, cpu_tensor.tensor_handle
            )
        )
        return cpu_tensor


def tensor_to_device(tensor) -> Tensor:
    if tensor.device == device("device"):
        return tensor
    else:
        device_tensor = raw_like(tensor, device("device"))
        check_return_value(
            diopirt_lib._diopiTensorCopyHostToDevice(
                tensor.context_handle, tensor.tensor_handle, device_tensor.tensor_handle
            )
        )
        return device_tensor


Tensor.cpu = tensor_to_host
Tensor.to_host = tensor_to_host
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
