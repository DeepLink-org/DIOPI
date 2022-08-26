from . import Dtype, raw_like, check_return_value
from .litert import Sizes, Scalar, Tensor, device_impl_lib
from .utils import FunctionNotImplementedError
from ctypes import c_float, c_int64, c_int32, byref


def check_funtions(fn_name):
    try:
        c_func = eval(f"device_impl_lib.{fn_name}")
    except AttributeError as e:
        raise FunctionNotImplementedError(e.args)
    return c_func


def broadcast_out_size(size1, size2):
    sizeO = size1 if len(size1) > len(size2) else size2
    length = len(size2) if len(size1) > len(size2) else len(size1)
    idx = -1
    while length > 0:
        assert size1[idx] == size2[idx] or size1[idx] == 1 or size2[idx] == 1,\
            "size1 and size2 must be broadcastable"
        sizeO[idx] = size1[idx] if size2[idx] == 1 else size2[idx]
        idx -= 1
        length -= 1

    return sizeO


def fill(tensor, value):
    r"""
    Fill a Tensor with a specific value
    """
    func = check_funtions("diopiFill")
    ret = func(tensor.context_handle, tensor.tensor_handle, c_float(value))
    check_return_value(ret)
    return tensor


def ones_like(tensor):
    new_tensor = raw_like(tensor)
    fill(new_tensor, 1)
    return new_tensor


def unary_op(input, inplace, call) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_funtions(call)
        ret = func(input.context_handle, input.tensor_handle)
    else:
        out = raw_like(input)
        func = check_funtions(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle)

    check_return_value(ret)
    return out


def binary_op(input, other, inplace, call) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_funtions(call)
        ret = func(input.context_handle, input.tensor_handle,
                   other.tensor_handle)
    else:
        out = raw_like(input)
        func = check_funtions(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, other.tensor_handle)

    check_return_value(ret)
    return out


def binary_op_scalar(input, other, inplace, call, alpha=None) -> Tensor:
    args = "input.context_handle, "
    if inplace:
        out = input
    else:
        if not isinstance(other, Tensor):
            out = raw_like(input)
        else:
            sizeI = input.size()
            sizeO = other.size()
            outsize = broadcast_out_size(list(sizeI), list(sizeO))
            out = Tensor(outsize, input.get_dtype())
        args = args + "out.tensor_handle, "

    if not isinstance(other, Tensor):
        call = call + "Scalar"
        other = Scalar(input.get_dtype(), other)
        args = args + "input.tensor_handle, byref(other)"
    else:
        args = args + "input.tensor_handle, other.tensor_handle"\

    if alpha is not None:
        alpha = Scalar(input.get_dtype(), alpha)
        args = args + ", byref(alpha)"

    func = check_funtions(call)
    ret = eval(f'func({args})')

    check_return_value(ret)
    return out


def softmax(input, dim, dtype=None):
    r"""

    """
    if dim is None:
        dim = 0
    if input.numel() == 0:
        return input
    if dtype is None:
        dtype = input.get_dtype()
    out = raw_like(input)

    func = check_funtions('diopiSoftmax')
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(dim), c_int32(dtype.value))
    check_return_value(ret)
    return out


def relu(input, inplace=False) -> Tensor:

    """Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    return unary_op(input, inplace, 'diopiRelu')


def abs(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiAbs')


def floor(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiFloor')


def sign(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSign')


def sigmoid(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSigmoid')


def sqrt(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSqrt')


def neg(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiNeg')


def sin(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSin')


def cos(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiCos')


def tanh(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiTanh')


def exp(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiExp')


def log(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiLog')


def log2(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiLog2')


def log10(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiLog10')


def erf(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiErf')


def add(input, other, alpha=1, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiAdd', alpha=alpha)


def sub(input, other, alpha=1.0, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiSub', alpha=alpha)


def eq(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiEq')


def ne(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiNe')


def ge(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiGe')


def le(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiLe')


def lt(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiLt')


def mul(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiMul')


def div(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiDiv')


def logical_and(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiBitwiseAnd')


def logical_or(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiBitwiseOr')


def leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor:
    negative_slope = byref(Scalar(Dtype.float64, negative_slope))
    if inplace:
        out = input
        func = check_funtions("diopiLeakyReLuInp")
        ret = func(input.context_handle,
                   out.tensor_handle, input.tensor_handle, negative_slope)
    else:
        out = raw_like(input)
        func = check_funtions("diopiLeakyReLu")
        ret = func(input.context_handle,
                   out.tensor_handle, input.tensor_handle, negative_slope)

    check_return_value(ret)
    return out


def bmm(input, mat2) -> Tensor:
    size1 = input.size()
    assert(len(size1) == 3), 'input must be 3d tensor'
    size2 = mat2.size()
    assert(len(size2) == 3), 'mat2 must be 3d tensor'
    assert(size1[0] == size2[0]), 'invalid args'
    assert(size1[2] == size2[1]), 'invalid args'

    size_out = size1
    size_out[2] = size2[2]
    out = Tensor(size_out, input.get_dtype())

    func = check_funtions("diopiBmm")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, mat2.tensor_handle)
    check_return_value(ret)
    return out


def addcmul(input, tensor1, tensor2, value=1) -> Tensor:
    types = Dtype.float_no_half_types
    assert(input.get_dtype() in types), 'input must be float/double tensor'
    assert(tensor1.get_dtype() in types), 'tensor1 must be float/double tensor'
    assert(tensor2.get_dtype() in types), 'tensor2 must be float/double tensor'

    out = raw_like(input)
    value = byref(Scalar(input.get_dtype(), value))

    func = check_funtions("diopiAddcmul")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               tensor1.tensor_handle, tensor1.tensor_handle, value)
    check_return_value(ret)
    return out


def matmul(input, other) -> Tensor:
    # tocheck: the shape of out tensor
    out = raw_like(input)
    sizeI = input.size()
    sizeO = other.size()

    # vector x vector
    if len(sizeI) == 1 and len(sizeO) == 1:
        out = Tensor((), input.get_dtype())
    # (batched) matrix x vector
    elif len(sizeO) == 1:
        sizeI[-1] = 1
        out = Tensor(sizeI,  input.get_dtype())
    # pretended matrix x (batched) matrix
    elif len(sizeI) == 1:
        sizeO[-2] = 1
        out = Tensor(sizeO, input.get_dtype())
    # (batched) matrix x (batched) matrix
    else:
        sizeI[-1] = sizeO[-1]
        if len(sizeI) > 3 and len(sizeO) > 2:
            assert sizeI[-3] == sizeO[-3] or sizeI[-3] == 1 or sizeO[-3] == 1,\
                'input and other should be broadcastable'
            sizeI[-3] = sizeI[-3] if sizeI[-3] == 1 else sizeO[-3]
        out = Tensor(sizeI, input.get_dtype())

    func = check_funtions("diopiMatmul")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, other.tensor_handle)
    check_return_value(ret)

    # out.squeeze()
    return out


def clamp(input, min, max, inplace=False) -> Tensor:
    call = "Clamp"
    args = "input.context_handle, "
    if inplace:
        out = input
        call = call + "Inp"
    else:
        out = raw_like(input)
        args = args + "out.tensor_handle, "

    if isinstance(min, Tensor):
        assert(isinstance(max, Tensor)), 'min and max must have same type'
        args += "input.tensor_handle, min.tensor_handle, max.tensor_handle"
    else:
        assert(~isinstance(max, Tensor)), 'min and max must have same type'
        call = call + 'Scalar'
        min = byref(Scalar(input.get_dtype(), min))
        max = byref(Scalar(input.get_dtype(), max))
        args = args + "input.tensor_handle, min, max"

    func = check_funtions(call)
    ret = func(eval(f'{args}'))
    check_return_value(ret)
    return out


def clamp_min(input, min, inplace=False) -> Tensor:
    call = "ClampMin"
    args = "input.context_handle, "
    if inplace:
        out = input
        call = call + "Inp"
    else:
        out = raw_like(input)
        args = args + "out.tensor_handle, "

    if isinstance(min, Tensor):
        args = args + "input.tensor_handle, min.tensor_handle"
    else:
        call = call + 'Scalar'
        min = byref(Scalar(input.get_dtype(), min))
        args = args + "input.tensor_handle, min"

    func = check_funtions(call)
    ret = func(eval(f'{args}'))
    check_return_value(ret)
    return out


def clamp_max(input, max, inplace=False) -> Tensor:
    call = "ClampMax"
    args = "input.context_handle, "
    if inplace:
        out = input
        call = call + "Inp"
    else:
        out = raw_like(input)
        args = args + "out.tensor_handle, "

    if isinstance(max, Tensor):
        args = args + "input.tensor_handle, max.tensor_handle"
    else:
        call = call + 'Scalar'
        max = byref(Scalar(input.get_dtype(), max))
        args = args + "input.tensor_handle, max"

    func = check_funtions(call)
    ret = func(eval(f'{args}'))
    check_return_value(ret)
    return out


def mean(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    """

    """
    assert(isinstance(dim, [None, int, list])), "dim should be int or list"

    size1 = input.size()
    if dim is None:
        for i in len(size1):
            size1[i] = 1
    elif isinstance(dim, list):
        for i in dim:
            size1[i] = 1
    else:
        size1[dim] = 1

    if dtype is None:
        dtype = input.get_dtype()

    dim1 = Sizes(tuple(dim))
    out = Tensor(size1, input.get_dtype())
    func = check_funtions("diopiMean")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim1, keepdim, c_int32(dtype.value))
    check_return_value(ret)

    # if ~keepdim:
    #     out.squeeze()
    return out


def std(input, unbiased=False, dim=None, keepdim=False) -> Tensor:
    """

    """
    assert(isinstance(dim, [None, int, list])), "dim should be int or list"

    size1 = input.size()
    if dim is None:
        for i in len(size1):
            size1[i] = 1
    elif isinstance(dim, list):
        for i in dim:
            size1[i] = 1
    else:
        size1[dim] = 1

    dim1 = Sizes(tuple(dim))
    out = Tensor(size1, input.get_dtype())
    func = check_funtions("diopiStd")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim1, unbiased, keepdim)
    check_return_value(ret)

    # if ~keepdim:
    #     out.squeeze()
    return out


def min(input, dim=None, keepdim=False) -> Tensor:
    """

    """
    assert(isinstance(dim, [None, int])), "dim should be int"

    size1 = input.size()
    if dim is None:
        for i in len(size1):
            size1[i] = 1
        dim = -1
    elif isinstance(dim, list):
        for i in dim:
            size1[i] = 1
    else:
        size1[dim] = 1

    out = Tensor(size1, input.get_dtype())
    indices = Tensor(size1, Dtype.int64)
    func = check_funtions("diopiMin")
    ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
               input.tensor_handle, dim, keepdim)
    check_return_value(ret)

    # if ~keepdim:
    #     out.squeeze()
    return out


def convert_reduction(name):
    if name == 'none':
        return 0
    if name == 'mean':
        return 1
    if name == "sum":
        return 2
    return 3


def binary_cross_entropy_with_logits(input, target, weight=None,
                                     reduction='mean', pos_weight=None):
    assert input.shape == target.shape, \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'
    if pos_weight is not None:
        assert isinstance(pos_weight, Tensor), \
            'pos_weigth must be a Tensor'
    else:
        # todo: how to represent pos_weight = None
        pos_weight = Tensor((), input.get_dtype())

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
    else:
        weight = Tensor((), input.get_dtype())

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_funtions("diopiBCEWithLogits")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight.tensor_handle,
               pos_weight.tensor_handle, reduction_mode)
    check_return_value(ret)
    return out


def cross_entropy(input, target, weight=None, ignore_index=- 100,
                  reduction='mean', label_smoothing=0.0):
    assert input.shape == target.shape, \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
    else:
        weight = Tensor((), input.get_dtype())

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_funtions("diopiCrossEntropyLoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight.tensor_handle, reduction_mode,
               ignore_index, label_smoothing)
    check_return_value(ret)
    return out


def mse_loss(input, target, reduction='mean'):
    assert input.shape == target.shape, \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_funtions("diopiMSELoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, reduction_mode)
    check_return_value(ret)
    return out


def conv2d(input, weight, bias=None, stride=1,
           padding=0, dilation=1, groups=1) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
    else:
        bias = Tensor((), input.get_dtype())

    sizeI = input.size()
    sizeW = weight.size()
    assert len(sizeI) == 4 and len(sizeW) == 4,\
        'input and weight must be 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    sizeO.append(sizeW[0])

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    for i in range(-2, 0):
        # equivalent kernel size
        sizeW[i] += (sizeW[i] - 1) * (dilation[i] - 1)
        sizeO.append((sizeI[i] - sizeW[i] + 2*padding[i])/stride[i] + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    dilation = Sizes(tuple(dilation))
    out = Tensor(sizeO, input.get_dtype())
    func = check_funtions("diopiConvolution2d")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               weight.tensor_handle, bias.tensor_handle, stride, padding,
               dilation, groups)
    check_return_value(ret)
    return out


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None) -> Tensor:
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    for i in range(-2, 0):
        sizeO.append((sizeI[i] - kernel_size[i] + 2*padding[i])/stride[i] + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    out = Tensor(sizeO, input.get_dtype())

    func = check_funtions("diopiAvgPool2d")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               kernel_size, stride, padding, ceil_mode, count_include_pad,
               byref(divisor_override))
    check_return_value(ret)
    return out


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False) -> Tensor:
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    for i in range(-2, 0):
        sizeO.append((sizeI[i] - kernel_size[i] + 2*padding[i])/stride[i] + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    dilation = Sizes(tuple(dilation))
    out = Tensor(sizeO, input.get_dtype())
    if not return_indices:
        func = check_funtions("diopiMaxPool2d")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, kernel_size,
                   stride, padding, dilation, ceil_mode)
    else:
        func = check_funtions("diopiMaxPool2dWithIndices")
        indices = Tensor(sizeO, Dtype.int64)
        ret = func(input.context_handle, out.tensor_handle,
                   indices.tensor_handle, input.tensor_handle,
                   kernel_size, stride, padding, dilation, ceil_mode)
    check_return_value(ret)
    return out


def adaptive_avg_pool2d(input, output_size):
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    for i in range(-2, 0):
        sizeO.append(output_size[i])

    out = Tensor(sizeO, input.get_dtype())
    output_size = Sizes(tuple(output_size))

    func = check_funtions("diopiAaptiveAvgPool2d")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, output_size)
    check_return_value(ret)
    return out


def adaptive_max_pool2d(input, output_size, return_indices=False):
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    for i in range(-2, 0):
        sizeO.append(output_size[i])

    out = Tensor(sizeO, input.get_dtype())
    output_size = Sizes(tuple(output_size))

    func = check_funtions("diopiAaptiveMaxPool2d")
    if return_indices:
        indices = Tensor(sizeO, Dtype.int64)
    else:
        indices = Tensor((), Dtype.int64)
    ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
               input.tensor_handle, output_size)
    check_return_value(ret)
    return out


def dropout(input, p=0.5, training=True, inplace=False):
    call = "Dropout"
    args = 'input.context_handle, '
    if inplace:
        out = input
        call = call + 'Inp'
    else:
        out = raw_like(input)
        args = args + 'out.tensor_handle, '

    args = args + "input.tensor_handle, p, train"
    func = check_funtions(call)
    ret = func(eval(f'{args}'))
    check_return_value(ret)
    return out


def index_select(input, dim, index) -> Tensor:
    sizeI = input.size()
    sizeI[dim] = index.numel()
    out = Tensor(sizeI, input.get_dtype())

    func = check_funtions("diopiIndexSelect")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, dim, index.tensor_handle)
    check_return_value(ret)
    return out


def select(input, dim, index) -> Tensor:
    sizeI = input.size()
    sizeI[dim] = 1
    out = Tensor(sizeI, input.get_dtype())

    func = check_funtions("diopiSelect")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, dim, index)
    check_return_value(ret)
    return out


def masked_scatter(input, mask, source) -> Tensor:
    assert mask.get_dtype() == Dtype.bool, \
        "mask must be bool tensor"
    out = raw_like(input)

    func = check_funtions("MaskedScatter")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               mask.tensor_handle, source.tensor_handle)
    check_return_value(ret)
    return out


def nonzero(input):
    # todo: pytorch(1.12) has argument 'as_tuple' to return multiple 1d tensor
    out = Tensor((), Dtype.int64)
    func = check_funtions("diopiNonzero")
    ret = func(input.context_handle, byref(out.tensor_handle),
               input.tensor_handle)
    check_return_value(ret)
    return out


def linear(input, weight, bias=None) -> Tensor:

    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
    else:
        bias = Tensor((), input.get_dtype())

    sizeI = input.size()
    sizeW = weight.size()
    sizeI[-1] = sizeW[-2] if len(sizeW) == 2 else 1
    out = Tensor(sizeI, input.get_dtype())
    func = check_funtions("diopiLinear")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               weight.tensor_handle, bias.tensor_handle)
    check_return_value(ret)
    return out


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    sizeI = list(input.size())
    sizeW = weight.size()
    sizeI.append(sizeW[-1])
    # todo: add max_norm and norm_type in function declaration
    out = Tensor(sizeI, weight.get_dtype())
    func = check_funtions("diopiEmbedding")
    ret = func(input.context_handle, out.tensor_handle, weight.tensor_handle,
               input.tensor_handle, padding_idx, scale_grad_by_freq, sparse)
    check_return_value(ret)
    return out


def tril(input, diagonal=0) -> Tensor:
    out = raw_like(input)
    func = check_funtions("diopiTril")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, diagonal)
    check_return_value(ret)
    return out


def cat(tensors, dim=0) -> Tensor:
    insNum = len(tensors)
    sum = 0
    c_tensors = []
    for tensor in tensors:
        sizeI = tensor.size()
        sum += sizeI[dim]
        c_tensors.append(tensor.tensor_handle)

    sizeI[dim] = sum
    out = Tensor(sizeI, tensors[0].get_dtype())
    func = check_funtions("diopiCat")
    ret = func(input.context_handle, out.tensor_handle,
               byref(c_tensors), insNum, dim)
    check_return_value(ret)
    return out


def stack(tensors, dim=0) -> Tensor:
    insNum = len(tensors)
    sizeI = tensors[0].size()
    sum = insNum * sizeI[dim]

    c_tensors = []
    for tensor in tensors:
        c_tensors.append(tensor.tensor_handle)

    sizeI[dim] = sum
    out = Tensor(sizeI, tensors[0].get_dtype())
    func = check_funtions("diopiStack")
    ret = func(input.context_handle, out.tensor_handle,
               byref(c_tensors), insNum, dim)
    check_return_value(ret)
    return out


def sort(input, dim=- 1, descending=False, stable=False):
    vals = raw_like(input)
    sizeI = input.size()
    indices = Tensor(sizeI, Dtype.int64)

    func = check_funtions("diopiSort")
    ret = func(input.context_handle, vals.tensor_handle, indices.tensor_handle,
               input.tensor_handle, dim, descending, byref(stable))
    check_return_value(ret)
    return vals, indices


def topk(input, k, dim=-1, largest=True, sorted=True):
    sizeI = input.size()
    sizeI[dim] = k
    values = Tensor(sizeI, input.get_dtype())
    indices = Tensor(sizeI, Dtype.int64)

    func = check_funtions("diopiTopk")
    ret = func(input.context_handle, values.tensor_handle,
               indices.tensor_handle, input.tensor_handle,
               k, dim, largest, sorted)
    check_return_value(ret)
    return values, indices


def transpose(input, dim0, dim1) -> Tensor:
    sizeI = input.size()
    tmp = sizeI[dim0]
    sizeI[dim0] = sizeI[dim1]
    sizeI[dim1] = tmp
    out = Tensor(sizeI, input.get_dtype())

    func = check_funtions("diopiTranspose")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, dim0, dim1)
    check_return_value(ret)
    return out


def one_hot(tensor, num_classes=- 1):
    sizeI = tensor.size()
    sizeI += (num_classes, )
    # todo: create out with case num_classes=-1
    out = Tensor(sizeI, Dtype.int64)
    func = check_funtions("diopiOneHot")
    ret = func(input.context_handle, out.tensor_handle,
               tensor.tensor_handle, num_classes)
    check_return_value(ret)
    return out


def split(tensor, split_size_or_sections, dim=0):
    assert isinstance(split_size_or_sections, [int, list]),\
        "split_size_or_sections must be int or list"
    sizeI = tensor.size()
    sum = sizeI[dim]
    outs = []
    idx = 0
    splitSizes = ()
    is_int = isinstance(split_size_or_sections, int)

    while sum > 0:
        sizeI[dim] = split_size_or_sections if is_int else\
                     split_size_or_sections[idx]
        sizeI[dim] = sizeI[dim] if sum > sizeI[dim] else sum
        idx += 1
        sum -= sizeI[dim]
        splitSizes += (sizeI[dim], )
        out = Tensor(sizeI, Dtype.int64)
        outs.append(out)

    splitSizes = Sizes(splitSizes)
    assert sum == 0,\
        "split_size_or_sections should be compatible with tensor shape"
    func = check_funtions("diopiSplitWithSizes")
    ret = func(input.context_handle, byref(outs), idx,
               tensor.tensor_handle, byref(splitSizes), dim)
    check_return_value(ret)
    return outs


def pow(input, exponent) -> Tensor:
    if not isinstance(input, Tensor):
        assert isinstance(exponent, Tensor),\
            "exponent must be tensor when input is scalar"
        func = check_funtions("diopiPowScalar")
        # todo: return type = input type or float ?
        out = raw_like(exponent)
        if isinstance(input, int):
            input = byref(Scalar(Dtype.int64, input))
        else:
            input = byref(Scalar(Dtype.float64, input))
        ret = func(exponent.context_handle, out.tensor_handle, input, exponent.tensor_handle)
    elif not isinstance(exponent, Tensor):
        assert isinstance(input, Tensor),\
            "input must be tensor when exponent is scalar"
        func = check_funtions("diopiPow")
        out = raw_like(input)
        if isinstance(exponent, int):
            exponent = byref(Scalar(Dtype.int64, exponent))
        else:
            exponent = byref(Scalar(Dtype.float64, exponent))
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, exponent)
    else:
        sizeI = input.size()
        sizeE = exponent.size()
        sizeO = broadcast_out_size(sizeI, sizeE)
        out = Tensor(sizeO, input.get_dtype())
        # todo : add function definition for this case
        func = check_funtions("diopiPowTensor")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, exponent.tensor_handle)
    check_return_value(ret)
    return out


# def where(condition, x, y) -> Tensor:
    # todo: add scalar version
