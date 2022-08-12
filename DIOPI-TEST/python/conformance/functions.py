from .litert import Tensor, device_impl_lib
from .utils import raw_like
from .dtype import check_return_value
from ctypes import c_float, c_int64, c_int32, byref


def add(input, other, out=None) -> Tensor:
    if out is None:
        out = raw_like(input)
    else:
        assert isinstance(out, Tensor)

    ret = device_impl_lib.add(input.context_handle, out.tensor_handle,
                              input.tensor_handle, other.tensor_handle)
    check_return_value(ret)
    return out


def softmax(input, dim, dtype=None):
    r"""Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Arguments:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.

    """
    if dim == None: dim = 0
    if input.numel() == 0: return input
    out = raw_like(input)
    if dtype is None:
        dtype = input.get_dtype()

    ret = device_impl_lib.diopiSoftmax(input.context_handle, out.tensor_handle,
                              input.tensor_handle, c_int64(dim), c_int32(dtype.value))
    check_return_value(ret)
    return out


def relu(input, inplace=False):
    # type: (Tensor, bool) -> Tensor
    r"""relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    if inplace:
        out = input
    else:
        out = raw_like(input)

    ret = device_impl_lib.diopiRelu(input.context_handle, out.tensor_handle,
                                    input.tensor_handle)
    check_return_value(ret)
    return out
