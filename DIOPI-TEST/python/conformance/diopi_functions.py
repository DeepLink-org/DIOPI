# Copyright (c) 2023, DeepLink.
# -*- coding: UTF-8 -*-
import math
import itertools

from ctypes import c_float, c_double, c_int64, c_bool, c_void_p, byref, pointer
from .diopi_runtime import Sizes, Scalar, Tensor, TensorHandle, compute_nhwc_stride, compute_nhwc_stride_2d, compute_nhwc_stride_3d
from .utils import check_returncode, check_function, glob_vars
from . import Dtype, raw_like
from collections import namedtuple
import numpy as np


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


def reduce_op_process(input, dim=None, keepdim=False, dtype=None):
    sizeI = input.size()
    size = len(sizeI)
    sizeO = []
    dim_list = []
    dim = list(dim) if isinstance(dim, tuple) else dim

    if dim is None and keepdim:
        sizeO = [1 for i in range(0, size)]
    elif dim is not None:
        dim_list = dim if isinstance(dim, list) else [dim]
        for i in range(0, len(dim_list)):
            if dim_list[i] < 0:
                dim_list[i] += size

        dim_list.sort()
        for i in range(0, size):
            if i not in dim_list:
                sizeO.append(sizeI[i])
            elif keepdim:
                sizeO.append(1)

    if dtype is None:
        dtype = input.get_dtype()

    out = Tensor(sizeO, dtype)
    return dim_list, out


def common_dtype(input, other) -> Dtype:
    if isinstance(input, Tensor):
        dtype1 = input.get_dtype()
    elif isinstance(input, int):
        dtype1 = glob_vars.int_type
    elif isinstance(input, float):
        dtype1 = Dtype.float32
    else:
        assert 0, "not supported type of input"

    if isinstance(other, Tensor):
        dtype2 = other.get_dtype()
    elif isinstance(other, int):
        dtype2 = glob_vars.int_type
    elif isinstance(other, float):
        dtype2 = Dtype.float32
    else:
        assert 0, "not supported type of other"

    float_types = [Dtype.float16, Dtype.float32, Dtype.float64]
    if dtype1 in float_types and dtype2 not in float_types:
        return dtype1
    if dtype1 not in float_types and dtype2 in float_types:
        return dtype2
    if dtype1 == Dtype.bool and dtype2 == Dtype.bool:
        return dtype1
    elif dtype1 == Dtype.bool:
        return dtype2
    elif dtype2 == Dtype.bool:
        return dtype1
    return dtype1 if dtype1.value >= dtype2.value else dtype2


def promote_type(input: Tensor, promoted_dtype: Dtype) -> Dtype:
    dtype1 = input.get_dtype()
    need_promote_types = [Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64,
                          Dtype.uint8, Dtype.uint16, Dtype.uint32, Dtype.uint64, Dtype.bool]
    return dtype1 if dtype1 not in need_promote_types else promoted_dtype


def fill_(input, value):
    func = check_function("diopiFill")
    value = byref(Scalar(value))
    ret = func(input.context_handle, input.tensor_handle, value)
    check_returncode(ret)
    return input


def ones_like(tensor):
    new_tensor = raw_like(tensor)
    fill_(new_tensor, 1)
    return new_tensor


def zeros_like(tensor):
    new_tensor = raw_like(tensor)
    fill_(new_tensor, 0)
    return new_tensor


def unary_op(input, inplace, call, dtype=None) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle)
    else:
        if dtype is not None:
            out = Tensor(input.size(), dtype)
        else:
            out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle)

    check_returncode(ret)
    return out


def binary_op(input, other, inplace, call) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle,
                   other.tensor_handle)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, other.tensor_handle)

    check_returncode(ret)
    return out


def binary_op_scalar(input, other, inplace, call, alpha=None, dtype=None) -> Tensor:
    args = "input.context_handle, "
    if dtype is None:
        dtype = common_dtype(input, other)

    if inplace:
        call = call + "Inp"
        out = input
    else:
        sizeI = input.size()
        if not isinstance(other, Tensor):
            out = Tensor(sizeI, dtype)
        else:
            sizeO = other.size()
            outsize = broadcast_out_size(list(sizeI), list(sizeO))
            out = Tensor(outsize, dtype)
        args = args + "out.tensor_handle, "

    if not isinstance(other, Tensor):
        call = call + "Scalar"
        other = Scalar(other)
        args = args + "input.tensor_handle, byref(other)"
    else:
        args = args + "input.tensor_handle, other.tensor_handle"\

    if alpha is not None:
        alpha = Scalar(alpha)
        args = args + ", byref(alpha)"

    func = check_function(call)
    ret = eval(f'func({args})')

    check_returncode(ret)
    return out


def softmax(input, dim, dtype=None):
    if dim is None:
        dim = 0
    if input.numel() == 0:
        return input
    out = raw_like(input) if dtype is None else Tensor(input.size(), dtype)

    func = check_function('diopiSoftmax')
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, c_int64(dim))
    check_returncode(ret)
    return out


def relu(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiRelu')


def abs(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiAbs')


def floor(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiFloor')


def sign(input) -> Tensor:
    return unary_op(input, False, 'diopiSign')


def sigmoid(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSigmoid')


def silu(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSilu')


def silu_backward(input, grad_outputs, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    func = check_function("diopiSiluBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def sqrt(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSqrt', promote_type(input, Dtype.float32))


def rsqrt(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiRsqrt', promote_type(input, Dtype.float32))


def neg(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiNeg')


def sin(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiSin', promote_type(input, Dtype.float32))


def cos(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiCos', promote_type(input, Dtype.float32))


def tanh(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiTanh', promote_type(input, Dtype.float32))


def exp(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiExp', promote_type(input, Dtype.float32))


def log(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiLog', promote_type(input, Dtype.float32))


def log2(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiLog2', promote_type(input, Dtype.float32))


def log10(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiLog10', promote_type(input, Dtype.float32))


def erf(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiErf', promote_type(input, Dtype.float32))


def add(input, other, inplace=False, alpha=1) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiAdd', alpha=alpha)


def sub(input, other, inplace=False, alpha=1) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiSub', alpha=alpha)


def eq(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiEq', dtype=Dtype.bool)


def ne(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiNe', dtype=Dtype.bool)


def ge(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiGe', dtype=Dtype.bool)


def gt(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiGt', dtype=Dtype.bool)


def le(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiLe', dtype=Dtype.bool)


def lt(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiLt', dtype=Dtype.bool)


def mul(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiMul', dtype=promote_type(input, Dtype.float32))


def div(input, other, inplace=False, rounding_mode=None) -> Tensor:
    call = "diopiDiv"
    args = "input.context_handle, "
    sizeI = input.size()
    rounding_mode = convert_round_mode(rounding_mode)
    if inplace:
        call = call + "Inp"
        out = input
    else:
        out_type = promote_type(input, Dtype.float32)
        if not isinstance(other, Tensor):
            out = Tensor(sizeI, out_type)
        else:
            sizeO = other.size()
            outsize = broadcast_out_size(list(sizeI), list(sizeO))
            out = Tensor(outsize, out_type)
        args = args + "out.tensor_handle, "

    if not isinstance(other, Tensor):
        call = call + "Scalar"
        other = Scalar(other)
        args = args + "input.tensor_handle, byref(other)"
    else:
        args = args + "input.tensor_handle, other.tensor_handle"

    func = check_function(call)
    ret = eval(f'func({args}, rounding_mode)')

    check_returncode(ret)
    return out


def logical_and(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiLogicalAnd', dtype=Dtype.bool)


def logical_or(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, 'diopiLogicalOr', dtype=Dtype.bool)


def logical_not(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiLogicalNot', dtype=Dtype.bool)


def leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor:
    negative_slope = byref(Scalar(negative_slope, Dtype.float64))
    if inplace:
        out = input
        func = check_function("diopiLeakyReluInp")
        ret = func(input.context_handle,
                   input.tensor_handle, negative_slope)
    else:
        out = raw_like(input)
        func = check_function("diopiLeakyRelu")
        ret = func(input.context_handle,
                   out.tensor_handle, input.tensor_handle, negative_slope)

    check_returncode(ret)
    return out


def bmm(input, mat2) -> Tensor:
    size1 = list(input.size())
    assert (len(size1) == 3), 'input must be 3d tensor'
    size2 = mat2.size()
    assert (len(size2) == 3), 'mat2 must be 3d tensor'
    assert (size1[0] == size2[0]), 'invalid args'
    assert (size1[2] == size2[1]), 'invalid args'

    size_out = size1
    size_out[2] = size2[2]
    out = Tensor(size_out, input.get_dtype())

    func = check_function("diopiBmm")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, mat2.tensor_handle)
    check_returncode(ret)
    return out


def baddbmm(input, batch1, batch2, beta, alpha, inplace=False) -> Tensor:
    size1 = list(input.size())
    size2 = list(batch1.size())
    assert (len(size2) == 3), 'batch1 must be 3d tensor'
    size3 = list(batch2.size())
    assert (len(size3) == 3), 'batch2 must be 3d tensor'
    input_len = len(input.size())
    out_shape = size1
    if input_len == 3:
        assert (size2[2] == size3[1] and size1[0] == size2[0] and size1[0] == size3[0] or size1[0] == 1), 'invalid args'
        assert (size1[2] == size3[2] or size1[2] == 1 or size3[2] == 1), 'invalid args'
    elif input_len == 2:
        assert (((size1[1] == size3[2] or size1[1] == 1) and (size1[0] == size2[1] or size1[0] == 1))), 'invalid args'
        out_shape = (size2[0], size1[0], size1[1])
    elif input_len == 1:
        assert (size1[0] == size3[2] or size1[0] == 1), 'invalid args'
        out_shape = (size2[0], size2[1], size1[0])
    if out_shape[0] != size2[0]:
        out_shape = (size2[0], size1[1], size1[2])
    if out_shape[1] != size2[1]:
        out_shape = (size1[0], size2[1], size1[2])
    if out_shape[2] != size3[2]:
        out_shape = (size1[0], size1[1], size3[2])
    if inplace:
        func = check_function("diopiBaddbmmInp")
        ret = func(input.context_handle, input.tensor_handle, batch1.tensor_handle, batch2.tensor_handle, c_double(beta), c_double(alpha))
        check_returncode(ret)
        return input
    else:
        out = Tensor(size=out_shape, dtype=input.get_dtype())
        func = check_function("diopiBaddbmm")
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, batch1.tensor_handle, batch2.tensor_handle, c_double(beta), c_double(alpha))
        check_returncode(ret)
        return out


def addcmul(input, tensor1, tensor2, value=1, inplace=False) -> Tensor:
    size1 = list(tensor1.size())
    size2 = list(tensor2.size())
    sizeI = list(input.size())
    sizeO = broadcast_out_size(size1, size2)
    sizeO = broadcast_out_size(sizeI, sizeO)
    value = byref(Scalar(value))

    if inplace:
        out = input
        assert list(sizeO) == sizeI, 'can not be inplaced'
        func = check_function("diopiAddcmulInp")
        ret = func(input.context_handle, input.tensor_handle,
                   tensor1.tensor_handle, tensor2.tensor_handle, value)
    else:
        out = Tensor(sizeO, input.get_dtype())
        func = check_function("diopiAddcmul")
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
                   tensor1.tensor_handle, tensor2.tensor_handle, value)
    check_returncode(ret)
    return out


def matmul(input, other) -> Tensor:
    out = raw_like(input)
    sizeI = list(input.size())
    sizeO = list(other.size())

    # vector x vector
    if len(sizeI) == 1 and len(sizeO) == 1:
        out = Tensor((), input.get_dtype())
    # (batched) matrix x vector
    elif len(sizeO) == 1:
        sizeI[-1] = 1
        out = Tensor(sizeI, input.get_dtype())
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

    func = check_function("diopiMatmul")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, other.tensor_handle)
    check_returncode(ret)
    return out


def clamp(input, min=None, max=None, inplace=False) -> Tensor:
    assert min is not None or max is not None,\
        "min and max can not be None in the meantime"
    if max is None:
        return clamp_min(input, min, inplace)
    if min is None:
        return clamp_max(input, max, inplace)

    call = "diopiClamp"
    args = "input.context_handle, "
    if inplace:
        out = input
        call = call + "Inp"
    else:
        out = raw_like(input)
        args = args + "out.tensor_handle, "

    if isinstance(min, Tensor):
        assert (isinstance(max, Tensor)), 'min and max must have same type'
        args += "input.tensor_handle, min.tensor_handle, max.tensor_handle"
    else:
        assert (~isinstance(max, Tensor)), 'min and max must have same type'
        call = call + 'Scalar'
        min = byref(Scalar(min))
        max = byref(Scalar(max))
        args = args + "input.tensor_handle, min, max"

    func = check_function(call)
    ret = eval(f'func({args})')
    check_returncode(ret)
    return out


def clamp_min(input, min, inplace=False) -> Tensor:
    call = "diopiClampMin"
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
        min = byref(Scalar(min))
        args = args + "input.tensor_handle, min"

    func = check_function(call)
    ret = eval(f'func({args})')
    check_returncode(ret)
    return out


def clamp_max(input, max, inplace=False) -> Tensor:
    call = "diopiClampMax"
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
        max = byref(Scalar(max))
        args = args + "input.tensor_handle, max"

    func = check_function(call)
    ret = eval(f'func({args})')
    check_returncode(ret)
    return out


def mean(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    assert isinstance(dim, (int, list)) or dim is None,\
        "dim should be int or list or None"

    dim, out = reduce_op_process(input, dim, keepdim, dtype)
    func = check_function("diopiMean")
    dim1 = Sizes(tuple(dim))
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dim1)
    check_returncode(ret)
    return out


def std(input, unbiased=True, dim=None, keepdim=False) -> Tensor:
    assert isinstance(dim, (int, list)) or dim is None,\
        "dim should be int or list or None"

    dim, out = reduce_op_process(input, dim, keepdim)
    dim1 = Sizes(tuple(dim))
    func = check_function("diopiStd")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim1, unbiased)
    check_returncode(ret)
    return out


def min(input, dim=None, keepdim=False) -> Tensor:
    if dim is None:
        out = Tensor([], input.get_dtype())
        func = check_function("diopiMinAll")
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle)
        check_returncode(ret)
        return out

    assert isinstance(dim, int), "dim should be int"

    sizeI = list(input.size())
    if keepdim:
        sizeI[dim] = 1
    else:
        del sizeI[dim]
    out = Tensor(sizeI, input.get_dtype())
    indices = Tensor(out.size(), glob_vars.int_type)
    func = check_function("diopiMin")

    ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
               input.tensor_handle, c_int64(dim))
    check_returncode(ret)
    Res = namedtuple('Res', ['values', 'indices'])
    output = Res(out, indices)
    return output


def convert_reduction(name):
    if name == 'none':
        return 0
    if name == 'mean':
        return 1
    if name == "sum":
        return 2
    return 3


def convert_round_mode(name):
    if name is None:
        return 0
    if name == 'trunc':
        return 1
    if name == "floor":
        return 2
    return 4


def binary_cross_entropy(input, target, weight=None, reduction='mean'):
    assert input.size() == target.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if weight is not None:
        assert isinstance(weight, Tensor), 'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiBCELoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, c_int64(reduction_mode))
    check_returncode(ret)
    return out


def binary_cross_entropy_with_logits(input, target, weight=None,
                                     reduction='mean', pos_weight=None):
    assert input.size() == target.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'
    if pos_weight is not None:
        assert isinstance(pos_weight, Tensor), \
            'pos_weigth must be a Tensor'
        pos_weight = pos_weight.tensor_handle
    else:
        # represent pos_weight = None by pass a nullptr
        pos_weight = c_void_p()

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiBCEWithLogits")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, pos_weight, c_int64(reduction_mode))
    check_returncode(ret)
    return out


def cross_entropy(input, target, weight=None, ignore_index=- 100,
                  reduction='mean', label_smoothing=0.0):
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    sizeI = list(input.size())
    sizeO = [sizeI[0]] + sizeI[2:]
    if reduction == 'none':
        out = Tensor(sizeO, input.get_dtype())
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiCrossEntropyLoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, c_int64(reduction_mode),
               c_int64(ignore_index), c_double(label_smoothing))
    check_returncode(ret)
    return out


def mse_loss(input, target, reduction='mean'):
    assert input.shape() == target.shape(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiMSELoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, c_int64(reduction_mode))
    check_returncode(ret)
    return out


def conv2d(input, weight, bias=None, stride=1,
           padding=0, dilation=1, groups=1) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
        bias = bias.tensor_handle
    else:
        bias = c_void_p()

    sizeI = input.size()
    sizeW = list(weight.size())
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
        sizeO.append(int((sizeI[i] - sizeW[i] + 2 * padding[i]) / stride[i]) + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    dilation = Sizes(tuple(dilation))

    nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)
    func = check_function("diopiConvolution2d")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               weight.tensor_handle, bias, stride, padding, dilation, groups)
    check_returncode(ret)
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
        if ceil_mode:
            sizeO.append(math.ceil((sizeI[i] - kernel_size[i] + 2 * padding[i]) / stride[i]) + 1)
        else:
            sizeO.append(math.floor((sizeI[i] - kernel_size[i] + 2 * padding[i]) / stride[i]) + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)

    if divisor_override is None:
        divisor_override = c_void_p()
    else:
        divisor_override = c_int64(divisor_override)
        divisor_override = byref(divisor_override)

    func = check_function("diopiAvgPool2d")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               kernel_size, stride, padding, c_bool(ceil_mode), c_bool(count_include_pad),
               divisor_override)
    check_returncode(ret)
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
        tmp_ker_size = kernel_size[i] + (kernel_size[i] - 1) * (dilation[i] - 1)
        tmp_size = (sizeI[i] - tmp_ker_size + 2 * padding[i]) / stride[i] + 1
        tmp_size = tmp_size if tmp_size > 1 else 1
        if ceil_mode:
            sizeO.append(math.ceil(tmp_size))
        else:
            sizeO.append(math.floor(tmp_size))

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    dilation = Sizes(tuple(dilation))
    nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)

    if not return_indices:
        func = check_function("diopiMaxPool2d")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, kernel_size,
                   stride, padding, dilation, ceil_mode)
        check_returncode(ret)
        return out
    else:
        func = check_function("diopiMaxPool2dWithIndices")
        nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
        indices = Tensor(sizeO, glob_vars.int_type, stride=nhwc_stride)
        ret = func(input.context_handle, out.tensor_handle,
                   indices.tensor_handle, input.tensor_handle,
                   kernel_size, stride, padding, dilation, c_bool(ceil_mode))
        check_returncode(ret)
        return out, indices


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
        if output_size[i] is None:
            sizeO.append(sizeI[i])
        else:
            sizeO.append(output_size[i])

    nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)
    output_size = Sizes((sizeO[-2], sizeO[-1]))

    func = check_function("diopiAdaptiveAvgPool2d")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, output_size)
    check_returncode(ret)
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
        if output_size[i] is None:
            sizeO.append(sizeI[i])
        else:
            sizeO.append(output_size[i])

    nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)
    output_size = Sizes(tuple(output_size))

    if return_indices:
        func = check_function("diopiAdaptiveMaxPool2dWithIndices")
        nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
        indices = Tensor(sizeO, glob_vars.int_type, stride=nhwc_stride)
        ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
                   input.tensor_handle, output_size)
        check_returncode(ret)
        return out, indices
    else:
        func = check_function("diopiAdaptiveMaxPool2d")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, output_size)
    check_returncode(ret)
    return out


def dropout_impl(input, size_mask, p=0.5, training=True, inplace=False):
    call = "diopiDropout"
    args = 'input.context_handle, out.tensor_handle, mask.tensor_handle, '

    if inplace:
        out = input
        call = call + 'Inp'
    else:
        out = raw_like(input)
        args = args + 'input.tensor_handle, '

    mask = Tensor(size_mask, Dtype.uint8)
    args = args + "c_double(p), c_bool(training)"

    func = check_function(call)
    ret = eval(f'func({args})')
    check_returncode(ret)
    return out, mask


def dropout(input, p=0.5, training=True, inplace=False):
    return dropout_impl(input, input.size(), p, training, inplace)


def dropout2d(input, p=0.5, training=True, inplace=False):
    sizeI = list(input.size())
    for i in range(2, len(sizeI)):
        sizeI[i] = 1
    return dropout_impl(input, sizeI, p, training, inplace)


def index_select(input, dim, index) -> Tensor:
    sizeI = list(input.size())
    sizeI[dim] = index.numel()
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiIndexSelect")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(dim), index.tensor_handle)
    check_returncode(ret)
    return out


def select(input, dim, index) -> Tensor:
    sizeI = list(input.size())
    del sizeI[dim]
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiSelect")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(dim), c_int64(index))
    check_returncode(ret)
    return out


def masked_scatter(input, mask, source) -> Tensor:
    assert mask.get_dtype() == Dtype.bool, \
        "mask must be bool tensor"
    out = raw_like(input)

    func = check_function("diopiMaskedScatter")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               mask.tensor_handle, source.tensor_handle)
    check_returncode(ret)
    return out


def nonzero(input):
    # note: pytorch(1.12) has argument 'as_tuple' to return multiple 1d tensor
    out_tensor_handle = TensorHandle()
    func = check_function("diopiNonzero")
    ret = func(input.context_handle, pointer(out_tensor_handle),
               input.tensor_handle)
    check_returncode(ret)
    out = Tensor.from_handle(out_tensor_handle)
    return out


def linear(input, weight, bias=None) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
        bias = bias.tensor_handle
    else:
        bias = c_void_p()

    sizeI = list(input.size())
    sizeW = list(weight.size())
    sizeI[-1] = sizeW[-2] if len(sizeW) == 2 else 1
    out = Tensor(sizeI, input.get_dtype())
    func = check_function("diopiLinear")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               weight.tensor_handle, bias)
    check_returncode(ret)
    return out


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    sizeI = list(input.size())
    sizeW = weight.size()
    sizeI.append(sizeW[-1])
    out = Tensor(sizeI, weight.get_dtype())
    padding_idx = -100 if padding_idx is None else padding_idx

    if max_norm is not None:
        func2 = check_function("diopiEmbeddingRenorm_")
        ret2 = func2(input.context_handle, weight.tensor_handle, input.tensor_handle, c_double(max_norm), c_double(norm_type))
        check_returncode(ret2)

    # note: scale_grad_by_freq and sparse are useless during forward phase
    func = check_function("diopiEmbedding")
    ret = func(input.context_handle, out.tensor_handle, weight.tensor_handle,
               input.tensor_handle, c_int64(padding_idx), c_bool(scale_grad_by_freq), c_bool(sparse))
    check_returncode(ret)

    return out


def tril(input, diagonal=0) -> Tensor:
    out = raw_like(input)
    func = check_function("diopiTril")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(diagonal))
    check_returncode(ret)
    return out


def cat(tensors, dim=0) -> Tensor:
    assert isinstance(tensors, (list, tuple)),\
        "tensors must be a list or tuple"
    insNum = len(tensors)
    sum = 0
    c_tensors = []
    for tensor in tensors:
        sizeI = list(tensor.size())
        sum += sizeI[dim]
        c_tensors.append(tensor.tensor_handle)
    c_tensors = (c_void_p * insNum)(*c_tensors)

    sizeI[dim] = sum
    out = Tensor(sizeI, tensors[0].get_dtype())
    func = check_function("diopiCat")
    ret = func(tensors[0].context_handle, out.tensor_handle,
               pointer(c_tensors), c_int64(insNum), c_int64(dim))
    check_returncode(ret)
    return out


def stack(tensors, dim=0) -> Tensor:
    assert isinstance(tensors, (list, tuple)),\
        "tensors must be a list or tuple"
    insNum = len(tensors)
    outNum = insNum + 1
    sizeI = list(tensors[0].size())
    size_dim = dim
    if dim < 0:
        size_dim = outNum + dim
    sizeI.insert(size_dim, insNum)

    c_tensors = [t.tensor_handle for t in tensors]
    c_tensors = (c_void_p * insNum)(*c_tensors)

    out = Tensor(sizeI, tensors[0].get_dtype())
    func = check_function("diopiStack")
    ret = func(tensors[0].context_handle, out.tensor_handle,
               pointer(c_tensors), c_int64(insNum), c_int64(dim))
    check_returncode(ret)
    return out


def sort(input, dim=- 1, descending=False, stable=False):
    vals = raw_like(input)
    sizeI = input.size()
    indices = Tensor(sizeI, glob_vars.int_type)
    stable_c = c_void_p() if stable is None else pointer(c_bool(stable))
    func = check_function("diopiSort")
    ret = func(input.context_handle, vals.tensor_handle, indices.tensor_handle,
               input.tensor_handle, c_int64(dim), c_bool(descending), stable_c)
    check_returncode(ret)
    # if not stable, need to reconstruct indices and use "input[indices]" to check
    if not stable:
        # reconstruct the indices
        lst = []
        for dim_size in input.shape:
            temp_lst = [i for i in range(dim_size)]
            lst.append(temp_lst)
        temp_indices = list(itertools.product(*lst))
        for i in range(len(temp_indices)):
            temp_indices[i] = list(temp_indices[i])
            temp_indices[i][dim] = indices.numpy().flatten()[i]

        # use input[indices] to check
        temp_vals = []
        input_np = input.numpy()
        for idx in temp_indices:
            res = input_np
            # use for loop to index since idx is a list
            for i in idx:
                res = res[i]
            temp_vals.append(res)
        return vals, temp_vals
    return vals, indices


def topk(input, k, dim=-1, largest=True, sorted=True):
    sizeI = list(input.size())
    sizeI[dim] = k
    values = Tensor(sizeI, input.get_dtype())
    indices = Tensor(sizeI, glob_vars.int_type)

    func = check_function("diopiTopk")
    ret = func(input.context_handle, values.tensor_handle,
               indices.tensor_handle, input.tensor_handle,
               c_int64(k), c_int64(dim), c_bool(largest), c_bool(sorted))
    check_returncode(ret)
    return values, indices


def transpose(input, dim0, dim1) -> Tensor:
    sizeI = list(input.size())
    sizeI[dim0], sizeI[dim1] = sizeI[dim1], sizeI[dim0]
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiTranspose")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(dim0), c_int64(dim1))
    check_returncode(ret)
    return out


def one_hot(input, num_classes=- 1):
    assert num_classes == -1 or num_classes > 0,\
        "num_classes must be -1 or >0"

    sizeI = input.size()
    if num_classes == -1:
        sizeI += (np.max(input.numpy()) + 1, )
        out = Tensor(sizeI, glob_vars.int_type)
    else:
        sizeI += (num_classes, )
        out = Tensor(sizeI, glob_vars.int_type)

    func = check_function("diopiOneHot")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(num_classes))
    check_returncode(ret)
    return out


def split(tensor, split_size_or_sections, dim=0):
    assert isinstance(split_size_or_sections, (int, list, tuple)),\
        "split_size_or_sections must be int or list"
    sizeI = list(tensor.size())
    sum = sizeI[dim]
    outs = []
    idx = 0
    splitSizes = ()
    is_int = isinstance(split_size_or_sections, int)

    while sum > 0:
        sizeI[dim] = split_size_or_sections if is_int else split_size_or_sections[idx]
        sizeI[dim] = sizeI[dim] if sum > sizeI[dim] else sum
        idx += 1
        sum -= sizeI[dim]
        splitSizes += (sizeI[dim], )
        out = Tensor(sizeI, tensor.get_dtype())
        outs.append(out)

    c_outs = []
    for i in range(idx):
        c_outs.append(outs[i].tensor_handle)

    c_outs = (c_void_p * idx)(*c_outs)
    splitSizes = Sizes(splitSizes)
    assert sum == 0,\
        "split_size_or_sections should be compatible with tensor shape"
    func = check_function("diopiSplitWithSizes")
    ret = func(tensor.context_handle, pointer(c_outs), c_int64(idx),
               tensor.tensor_handle, splitSizes, c_int64(dim))
    check_returncode(ret)
    return outs


def pow(input=None, self=None, exponent=None, inplace=False) -> Tensor:
    float_types = [Dtype.float16, Dtype.float32, Dtype.float64]
    if input is None and self is not None:
        assert isinstance(exponent, Tensor),\
            "exponent must be tensor when input is scalar"
        func = check_function("diopiPowScalar")
        # todo: return type = input type or float
        out_dtype = None
        exponent_dtype = exponent.get_dtype()
        if isinstance(self, float) or exponent_dtype in float_types:
            out_dtype = exponent_dtype if exponent_dtype in float_types else Dtype.float32
        else:
            out_dtype = exponent_dtype
        out = Tensor(exponent.size(), out_dtype)
        self = byref(Scalar(self))
        ret = func(exponent.context_handle, out.tensor_handle, self, exponent.tensor_handle)
    elif not isinstance(exponent, Tensor):
        assert isinstance(input, Tensor),\
            "input must be tensor when exponent is scalar"
        exponent = byref(Scalar(exponent))
        if inplace:
            func = check_function("diopiPowInp")
            ret = func(input.context_handle, input.tensor_handle, exponent)
        else:
            func = check_function("diopiPow")
            input_dtype = input.get_dtype()
            out_dtype = Dtype.float32 if input_dtype not in float_types else input_dtype
            out = Tensor(input.size(), out_dtype)
            ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, exponent)
    elif inplace:
        func = check_function("diopiPowInpTensor")
        ret = func(input.context_handle, input.tensor_handle, exponent.tensor_handle)
    else:
        sizeI = list(input.size())
        sizeE = list(exponent.size())
        sizeO = broadcast_out_size(sizeI, sizeE)
        out_dtype = common_dtype(input, exponent)
        out = Tensor(sizeO, out_dtype)
        func = check_function("diopiPowTensor")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, exponent.tensor_handle)
    if inplace:
        out = input

    check_returncode(ret)
    return out


def where(condition, input, other) -> Tensor:
    # todo: add scalar version for pytorch 1.12
    assert (condition.get_dtype() in (Dtype.bool, Dtype.uint8)),\
        "condition must be a bool tensor"
    sizeX = list(input.size())
    sizeY = list(other.size())
    sizeC = list(condition.size())
    sizeO = broadcast_out_size(sizeX, sizeY)
    sizeO = broadcast_out_size(sizeC, sizeO)
    assert (input.get_dtype() == other.get_dtype()),\
        " input and other shoule be the same type "
    out = Tensor(sizeO, input.get_dtype())

    func = check_function("diopiWhere")
    ret = func(input.context_handle, out.tensor_handle, condition.tensor_handle,
               input.tensor_handle, other.tensor_handle)
    check_returncode(ret)
    return out


def clip_grad_norm_(tensors, max_norm, norm_type=2.0, error_if_nonfinite=False):
    assert (isinstance(max_norm, (int, float))),\
        "max_norm must be a int or float"
    assert (isinstance(norm_type, (int, float))),\
        "norm_type must be a int or float"

    if isinstance(tensors, Tensor):
        ctx = tensors.context_handle
        grads = [tensors.tensor_handle]
        grads = (c_void_p * 1)(*grads)
        num_grads = 1
    else:
        ctx = tensors[0].context_handle
        num_grads = len(tensors)
        grads = [grad.tensor_handle for grad in tensors]
        grads = (c_void_p * num_grads)(*grads)

    out = c_double(0.0)

    func = check_function("diopiClipGradNorm")
    ret = func(ctx, pointer(out), pointer(grads), c_int64(num_grads), c_double(max_norm), c_double(norm_type),
               c_bool(error_if_nonfinite))
    check_returncode(ret)
    return out.value


def batch_norm(input, running_mean, running_var, weight, bias,
               training=False, momentum=0.1, eps=1e-05, backward=False) -> Tensor:
    dim = len(list(input.size()))
    dim = [0] + [i for i in range(2, dim)]
    dtype = Dtype.float32 if input.get_dtype() == Dtype.float16 else None
    _, save_mean = reduce_op_process(input, dim, dtype=dtype)
    save_invstd = raw_like(save_mean)

    if not training:
        assert (running_mean is not None and running_var is not None),\
            "if not trainging, running_mean and running_var must be defined"
    running_mean = c_void_p() if running_mean is None else running_mean.tensor_handle
    running_var = c_void_p() if running_var is None else running_var.tensor_handle

    out = raw_like(input)
    func = check_function("diopiBatchNorm")
    ret = func(input.context_handle, out.tensor_handle, save_mean.tensor_handle, save_invstd.tensor_handle,
               input.tensor_handle, weight.tensor_handle, bias.tensor_handle, running_mean, running_var, c_bool(training),
               c_double(momentum), c_double(eps))
    check_returncode(ret)
    if backward:
        return save_mean, save_invstd
    return out


def log_softmax(input, dim=None, dtype=None):
    if dim is None:
        dim = 0
    if input.numel() == 0:
        return input
    out = raw_like(input) if dtype is None else Tensor(input.size(), dtype)

    func = check_function('diopiLogSoftmax')
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(dim))
    check_returncode(ret)
    return out


def hardtanh(input, min_val=- 1.0, max_val=1.0, inplace=False) -> Tensor:
    call = "diopiHardtanh"
    min_val = byref(Scalar(min_val))
    max_val = byref(Scalar(max_val))
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle, min_val, max_val)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, min_val, max_val)

    check_returncode(ret)
    return out


def hardswish(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiHardswish')


def threshold(input, threshold, value, inplace=False) -> Tensor:
    call = "diopiThreshold"
    threshold = byref(Scalar(threshold))
    value = byref(Scalar(value))
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle, threshold, value)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, threshold, value)

    check_returncode(ret)
    return out


def gelu(input, approximate='none') -> Tensor:
    assert isinstance(approximate, str),\
        "approximate must be a string."
    out = raw_like(input)
    func = check_function("diopiGelu")

    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, approximate.encode('UTF-8'))

    check_returncode(ret)
    return out


def addcdiv(input, tensor1, tensor2, value=1, inplace=False) -> Tensor:
    size1 = list(tensor1.size())
    size2 = list(tensor2.size())
    sizeI = list(input.size())
    sizeO = broadcast_out_size(size1, size2)
    sizeO = broadcast_out_size(sizeI, sizeO)
    value = byref(Scalar(value))

    if inplace:
        out = input
        assert list(sizeO) == sizeI, 'can not be inplaced'
        func = check_function("diopiAddcdivInp")
        ret = func(input.context_handle, input.tensor_handle,
                   tensor1.tensor_handle, tensor2.tensor_handle, value)
    else:
        out = Tensor(sizeO, input.get_dtype())
        func = check_function("diopiAddcdiv")
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
                   tensor1.tensor_handle, tensor2.tensor_handle, value)
    check_returncode(ret)
    return out


def addmm(input, mat1, mat2, beta=1, alpha=1) -> Tensor:
    size1 = list(mat1.size())
    size2 = mat2.size()
    size1[-1] = size2[-1]
    sizeI = list(input.size())
    sizeO = broadcast_out_size(sizeI, size1)
    out = Tensor(sizeO, input.get_dtype())
    alpha = byref(Scalar(alpha))
    beta = byref(Scalar(beta))

    func = check_function("diopiAddmm")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               mat1.tensor_handle, mat2.tensor_handle, beta, alpha)
    check_returncode(ret)
    return out


def sum(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    assert isinstance(dim, (int, list, tuple)) or dim is None,\
        "dim should be int or list"
    func = check_function("diopiSum")
    out_dtype = dtype if dtype is not None else promote_type(input, Dtype.int64)
    dim, out = reduce_op_process(input, dim, keepdim, out_dtype)
    dim1 = Sizes(tuple(dim))
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dim1)
    check_returncode(ret)
    return out


def max(input, dim=None, keepdim=False):
    if dim is None:
        out = Tensor([], input.get_dtype())
        func = check_function("diopiMaxAll")
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle)
        check_returncode(ret)
        return out

    assert isinstance(dim, int), "dim should be int"
    sizeI = list(input.size())
    if keepdim:
        sizeI[dim] = 1
    else:
        del sizeI[dim]
    out = Tensor(sizeI, input.get_dtype())
    indices = Tensor(out.size(), glob_vars.int_type)

    func = check_function("diopiMax")
    ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
               input.tensor_handle, c_int64(dim))
    check_returncode(ret)
    Res = namedtuple('Res', ['values', 'indices'])
    output = Res(out, indices)
    return output


def any(input, dim=None, keepdim=False) -> Tensor:
    if dim is None:
        out = Tensor([], Dtype.bool)
        dim = c_void_p()
    else:
        assert isinstance(dim, int), "dim should be int"
        _, out = reduce_op_process(input, dim, keepdim, dtype=Dtype.bool)
        dim = byref(c_int64(dim))
    func = check_function("diopiAny")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dim)
    check_returncode(ret)
    return out


def all(input, dim=None, keepdim=False) -> Tensor:
    if dim is None:
        out = Tensor([], Dtype.bool)
        dim = c_void_p()
    else:
        assert isinstance(dim, int), "dim should be int"
        _, out = reduce_op_process(input, dim, keepdim, dtype=Dtype.bool)
        dim = byref(c_int64(dim))
    func = check_function("diopiAll")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dim)
    check_returncode(ret)
    return out


def nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    if reduction == 'none':
        out = Tensor(target.size(), input.get_dtype())
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiNLLLoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, c_int64(reduction_mode), c_int64(ignore_index))
    check_returncode(ret)
    return out


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction='none') -> Tensor:
    assert inputs.size() == targets.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if reduction == 'none':
        out = raw_like(inputs)
    else:
        out = Tensor((), inputs.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiSigmoidFocalLoss")
    ret = func(inputs.context_handle, out.tensor_handle, inputs.tensor_handle,
               targets.tensor_handle, c_float(alpha), c_float(gamma), c_int64(reduction_mode))
    check_returncode(ret)
    return out


def nms(boxes, scores, iou_threshold) -> Tensor:
    size_boxes = boxes.size()
    assert len(size_boxes) == 2 and size_boxes[1] == 4,\
        "boxes must be a tensor of shape (N,4)"

    size_scores = scores.size()
    assert len(size_scores) == 1 and size_scores[0] == size_boxes[0],\
        "boxes must be a tensor of shape (N)"

    out_tensor_handle = TensorHandle()
    func = check_function("diopiNms")
    ret = func(boxes.context_handle, pointer(out_tensor_handle), boxes.tensor_handle,
               scores.tensor_handle, c_double(iou_threshold))
    out = Tensor.from_handle(out_tensor_handle)
    check_returncode(ret)
    return out


def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False) -> Tensor:
    if isinstance(boxes, Tensor):
        size_boxes = boxes.size()
        assert len(size_boxes) == 2 and size_boxes[1] == 5,\
            "boxes should be a tensor of shape (N,5)"
    elif isinstance(boxes, list):
        size_boxes = boxes[0].size()
        assert len(size_boxes) == 2 and size_boxes[1] == 4,\
            "boxes should be a list of tensor of shape (N,4)"

    sizeI = list(input.size())
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    sizeI[-1] = output_size[-1]
    sizeI[-2] = output_size[-2]

    nhwc_stride = compute_nhwc_stride_2d(sizeI) if glob_vars.nhwc else None
    out = Tensor(sizeI, input.get_dtype(), stride=nhwc_stride)
    func = check_function("diopiRoiAlign")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               boxes.tensor_handle, c_double(spatial_scale), c_int64(output_size[-2]),
               c_int64(output_size[-1]), c_int64(sampling_ratio), c_bool(aligned))
    check_returncode(ret)
    return out


def slice_op(input, dim, index) -> Tensor:
    sizeI = list(input.size())
    num = int((index.stop - index.start + index.step - 1) / index.step)
    sizeI[dim] = num
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiSlice")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               c_int64(dim), c_int64(index.start), c_int64(index.stop), c_int64(index.step))

    check_returncode(ret)
    return out


def index(input, **kwargs) -> Tensor:
    new_args = []
    hasEllipsis = False
    once = True
    for ele in kwargs.values():
        if ele is None:
            hasEllipsis = True
        else:
            if hasEllipsis and once:
                once = False
                sizeI = input.size()
                sizeE = ele.size()
                length = len(sizeI) - len(sizeE) - len(new_args)
                for i in range(length):
                    tmp = c_void_p()
                    new_args.append(tmp.value)

            new_args.append(ele.tensor_handle)

    nums = len(new_args)
    c_indices = (c_void_p * nums)(*new_args)

    out_tensor_handle = TensorHandle()
    func = check_function("diopiIndex")
    ret = func(input.context_handle, pointer(out_tensor_handle), input.tensor_handle,
               pointer(c_indices), c_int64(nums))
    out = Tensor.from_handle(out_tensor_handle)
    check_returncode(ret)
    return out


def sgd(param, param_grad, lr, buf=None, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    # note: buf, param_grad are mutable
    func = check_function("diopiSgd")

    arg_buf = c_void_p() if buf is None else buf.tensor_handle
    ret = func(param.context_handle, param.tensor_handle, param_grad.tensor_handle, arg_buf,
               c_double(lr), c_double(momentum), c_double(dampening), c_double(weight_decay), c_bool(nesterov))
    check_returncode(ret)
    return param, buf


def adaptive_max_pool2d_backward(input, grad_outputs, output_size, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    _, indices = adaptive_max_pool2d(input, output_size, return_indices=True)

    func = check_function("diopiAdaptiveMaxPool2dBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, indices.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def slice_op_backward(input, grad_outputs, dim, index, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    sizeI = Sizes(input.size())

    func = check_function("diopiSliceBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               sizeI, c_int64(dim), c_int64(index.start), c_int64(index.stop), c_int64(index.step))
    check_returncode(ret)
    return {"input": grad_input}


def adaptive_avg_pool2d_backward(input, grad_outputs, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiAdaptiveAvgPool2dBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def index_backward(input, grad_outputs, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    zeros_like_input = zeros_like(input)
    new_args = []
    hasEllipsis = False
    once = True
    for ele in kwargs.values():
        if ele is None:
            hasEllipsis = True
        else:
            if hasEllipsis and once:
                once = False
                sizeI = input.size()
                sizeE = ele.size()
                length = len(sizeI) - len(sizeE) - len(new_args)
                for i in range(length):
                    tmp = c_void_p()
                    new_args.append(tmp.value)
            new_args.append(ele.tensor_handle)
    nums = len(new_args)
    c_indices = (c_void_p * nums)(*new_args)

    func = check_function("diopiIndexBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, zeros_like_input.tensor_handle,
               pointer(c_indices), c_int64(nums), grad_outputs[0].tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def leaky_relu_backward(input, grad_outputs, negative_slope=0.01, input_is_result=False, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    negative_slope = byref(Scalar(negative_slope, Dtype.float64))

    func = check_function("diopiLeakyReluBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, negative_slope, c_bool(input_is_result))
    check_returncode(ret)
    return {"input": grad_input}


def sigmoid_focal_loss_backward(inputs, grad_outputs, targets, alpha=0.25, gamma=2, reduction='none', **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    assert inputs.size() == targets.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    grad_input = raw_like(inputs)
    reduction = convert_reduction(reduction)
    func = check_function("diopiSigmoidFocalLossBackward")

    ret = func(inputs.context_handle, grad_outputs[0].tensor_handle, inputs.tensor_handle, targets.tensor_handle,
               grad_input.tensor_handle, c_float(gamma), c_float(alpha), c_int64(reduction))
    check_returncode(ret)
    return {"inputs": grad_input}


def roi_align_backward(input, grad_outputs, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    if isinstance(boxes, Tensor):
        size_boxes = boxes.size()
        assert len(size_boxes) == 2 and size_boxes[1] == 5,\
            "boxes should be a tensor of shape (N,5)"
    elif isinstance(boxes, list):
        size_boxes = boxes[0].size()
        assert len(size_boxes) == 2 and size_boxes[1] == 4,\
            "boxes should be a list of tensor of shape (N,4)"

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    out = raw_like(input)
    sizeI = input.size()

    func = check_function("diopiRoiAlignBackward")
    ret = func(input.context_handle, out.tensor_handle, grad_outputs[0].tensor_handle,
               boxes.tensor_handle, c_double(spatial_scale), c_int64(output_size[-2]),
               c_int64(output_size[-1]), c_int64(sizeI[0]), c_int64(sizeI[1]), c_int64(sizeI[2]),
               c_int64(sizeI[3]), c_int64(sampling_ratio), c_bool(aligned))
    check_returncode(ret)
    return {"input": out}


def conv2d_backward(input, grad_outputs, weight, bias=None, stride=1,
                    padding=0, dilation=1, groups=1, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    sizeI = input.size()
    sizeW = weight.size()
    assert len(sizeI) == 4 and len(sizeW) == 4,\
        'input and weight must be 4d tensors'

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    dilation = Sizes(tuple(dilation))

    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    out = {"input": grad_input, "weight": grad_weight}

    if bias is None:
        grad_bias = c_void_p()
        sizeBias = c_void_p()
    else:
        gradBias = raw_like(bias)
        grad_bias = gradBias.tensor_handle
        sizeBias = byref(Sizes(bias.size()))
        out.update({"bias": grad_bias})

    # todo: no transposed/output_padding in forward
    transposed = False
    output_padding = Sizes((0, 0))

    func = check_function("diopiConvolution2dBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_weight.tensor_handle, grad_bias,
               grad_outputs[0].tensor_handle, input.tensor_handle, weight.tensor_handle, sizeBias, stride,
               padding, dilation, c_bool(transposed), output_padding, c_int64(groups))
    check_returncode(ret)
    return out


def hardtanh_backward(input, grad_outputs, min_val=-1.0, max_val=1.0, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    min_val = byref(Scalar(min_val))
    max_val = byref(Scalar(max_val))

    func = check_function("diopiHardtanhBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, min_val, max_val)
    check_returncode(ret)
    return {"input": grad_input}


def hardswish_backward(input, grad_outputs, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    func = check_function("diopiHardswishBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def gelu_backward(input, grad_outputs, approximate='none', **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiGeluBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, approximate.encode('UTF-8'))
    check_returncode(ret)
    return {"input": grad_input}


def avg_pool2d_backward(input, grad_outputs, kernel_size, stride=None, padding=0, ceil_mode=False,
                        count_include_pad=True, divisor_override=None, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))

    if divisor_override is None:
        divisor_override = c_void_p()
    else:
        divisor_override = byref(c_int64(divisor_override))

    func = check_function("diopiAvgPool2dBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, kernel_size, stride, padding, c_bool(ceil_mode),
               c_bool(count_include_pad), divisor_override)
    check_returncode(ret)
    return {"input": grad_input}


def embedding_backward(input, grad_outputs, weight, padding_idx=None, scale_grad_by_freq=False, sparse=False, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_weight = raw_like(weight)
    num_weight = weight.size()[0]
    padding_idx = -100 if padding_idx is None else padding_idx

    # note: scale_grad_by_freq and sparse are useless during forward phase
    func = check_function("diopiEmbeddingBackward")
    ret = func(input.context_handle, grad_weight.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, c_int64(num_weight), c_int64(padding_idx), c_bool(scale_grad_by_freq), c_bool(sparse))
    check_returncode(ret)
    return {"weight": grad_weight}


def mse_loss_backward(input, grad_outputs, target, reduction='mean', **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    reduction_mode = convert_reduction(reduction)

    func = check_function("diopiMSELossBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, target.tensor_handle, c_int64(reduction_mode))
    check_returncode(ret)
    return {"input": grad_input}


def tanh_backward(input, grad_outputs, output, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiTanhBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               output.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def index_select_backward(input, grad_outputs, dim, index, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    inputSize = Sizes(input.size())

    func = check_function("diopiIndexSelectBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               inputSize, c_int64(dim), index.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def select_backward(input, grad_outputs, dim, index, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    inputSize = Sizes(input.size())

    func = check_function("diopiSelectBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               inputSize, c_int64(dim), c_int64(index))
    check_returncode(ret)
    return {"input": grad_input}


def softmax_backward(input, grad_outputs, output, dim, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiSoftmaxBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               output.tensor_handle, c_int64(dim))
    check_returncode(ret)
    return {"input": grad_input}


def log_softmax_backward(input, grad_outputs, output, dim, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiLogSoftmaxBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               output.tensor_handle, c_int64(dim))
    check_returncode(ret)
    return {"input": grad_input}


def sigmoid_backward(input, grad_outputs, output, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiSigmoidBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               output.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def threshold_backward(input, grad_outputs, threshold, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    threshold = byref(Scalar(threshold))

    func = check_function("diopiThresholdBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, threshold)
    check_returncode(ret)
    return {"input": grad_input}


def binary_cross_entropy_backward(input, grad_outputs, target, weight=None,
                                  reduction='mean', **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    assert input.size() == target.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    grad_input = raw_like(input)
    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiBCELossBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, target.tensor_handle, weight, c_int64(reduction_mode))
    check_returncode(ret)
    return {"input": grad_input}


def binary_cross_entropy_with_logits_backward(input, grad_outputs, target, weight=None,
                                              reduction='mean', pos_weight=None, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    assert input.size() == target.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if pos_weight is not None:
        assert isinstance(pos_weight, Tensor), \
            'pos_weigth must be a Tensor'
        pos_weight = pos_weight.tensor_handle
    else:
        # represent pos_weight = None by pass a nullptr
        pos_weight = c_void_p()

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    grad_input = raw_like(input)
    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiBCEWithLogitsBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, target.tensor_handle, weight, pos_weight, c_int64(reduction_mode))
    check_returncode(ret)
    return {"input": grad_input}


def nll_loss_backward(input, grad_outputs, target, weight=None, ignore_index=-100, reduction='mean', **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        # total_weight = sum(weight)
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    reduction_mode = convert_reduction(reduction)

    func = check_function("diopiNLLLossBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, target.tensor_handle, weight, c_int64(reduction_mode), c_int64(ignore_index))
    check_returncode(ret)
    return {"input": grad_input}


def max_pool2d_backward(input, grad_outputs, kernel_size, stride=None, padding=0, dilation=1,
                        ceil_mode=False, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3, 'input must be 3d or 4d tensors'

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    _, indices = max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode, True)
    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    dilation = Sizes(tuple(dilation))

    func = check_function("diopiMaxPool2dBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, kernel_size, stride, padding, dilation, c_bool(ceil_mode), indices.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def batch_norm_backward(input, grad_outputs, running_mean, running_var, weight, bias,
                        training=False, eps=1e-05, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    save_mean, save_invstd = batch_norm(input, running_mean, running_var, weight, bias, training, 0.1, eps, backward=True)

    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    grad_bias = raw_like(bias)

    if not training:
        assert (running_mean is not None and running_var is not None),\
            "if not trainging, running_mean and running_var must be defined"
    running_mean = c_void_p() if running_mean is None else running_mean.tensor_handle
    running_var = c_void_p() if running_var is None else running_var.tensor_handle
    out = {"input": grad_input, "weight": grad_weight, "bias": grad_bias}
    func = check_function("diopiBatchNormBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_weight.tensor_handle, grad_bias.tensor_handle,
               grad_outputs[0].tensor_handle, input.tensor_handle, weight.tensor_handle, running_mean, running_var, save_mean.tensor_handle,
               save_invstd.tensor_handle, c_bool(training), c_double(eps))
    check_returncode(ret)
    return out


def arange(end, start=0, step=1, dtype=None) -> Tensor:
    if dtype is None:
        if type(start) == float or type(end) == float or type(step) == float:
            dtype = Dtype.float32
        else:
            dtype = glob_vars.int_type

    numel = int((end - start) / step)
    out = Tensor((numel,), dtype)

    func = check_function("diopiArange")
    ret = func(out.context_handle, out.tensor_handle, byref(Scalar(start)), byref(Scalar(end)), byref(Scalar(step)))
    check_returncode(ret)
    return out


def randperm(n: int, dtype=None) -> Tensor:
    dtype = glob_vars.int_type if dtype is None else dtype
    numel = n
    out = Tensor((numel,), dtype)

    func = check_function("diopiRandperm")
    ret = func(out.context_handle, out.tensor_handle, c_int64(n), c_int64(0))
    check_returncode(ret)
    return out


def uniform(input, start=0, end=1) -> Tensor:
    func = check_function("diopiUniformInp")
    ret = func(input.context_handle, input.tensor_handle, c_double(start), c_double(end), c_int64(0))
    check_returncode(ret)
    return input


def random(input, start=0, end=None) -> Tensor:
    func = check_function("diopiRandomInp")
    end = c_void_p() if end is None else pointer(c_int64(end))
    ret = func(input.context_handle, input.tensor_handle, c_int64(start), end, c_int64(0))
    check_returncode(ret)
    return input


def bernoulli(input, inplace=False, p=None) -> Tensor:
    out = input

    if p is not None:
        func = check_function("diopiBernoulliScalar")
        ret = func(input.context_handle, input.tensor_handle, c_double(p), c_int64(0))
    elif inplace:
        func = check_function("diopiBernoulliInp")
        ret = func(input.context_handle, input.tensor_handle, c_int64(0))
    else:
        out = raw_like(input)
        func = check_function("diopiBernoulli")
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, c_int64(0))

    check_returncode(ret)
    return out


def masked_fill(input, mask, value, inplace=False) -> Tensor:
    assert mask.get_dtype() == Dtype.bool, "mask must be bool tensor"
    out = raw_like(input)

    call = "diopiMaskedFill"

    call_scalar = False
    if isinstance(value, Tensor):
        value = value.tensor_handle
    else:
        value = byref(Scalar(value))
        call_scalar = True

    if inplace:
        out = input
        call = call + "Inp"
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle, mask.tensor_handle, value)
    else:
        out = raw_like(input)
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, mask.tensor_handle, value)

    check_returncode(ret)
    return out


def adamw(param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr,
          beta1, beta2, eps, weight_decay, step, amsgrad=False):
    # note: buf, param_grad are mutable
    func = check_function("diopiAdamW")
    ret = func(param.context_handle, param.tensor_handle, param_grad.tensor_handle, exp_avg.tensor_handle,
               exp_avg_sq.tensor_handle, max_exp_avg_sq.tensor_handle, c_float(lr), c_float(beta1), c_float(beta2),
               c_float(eps), c_float(weight_decay), c_int64(step), c_bool(amsgrad))
    check_returncode(ret)
    return param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq


def adam(param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq, lr,
         beta1, beta2, eps, weight_decay, step, amsgrad=False):
    # note: buf, param_grad are mutable
    func = check_function("diopiAdam")
    ret = func(param.context_handle, param.tensor_handle, param_grad.tensor_handle, exp_avg.tensor_handle,
               exp_avg_sq.tensor_handle, max_exp_avg_sq.tensor_handle, c_float(lr), c_float(beta1), c_float(beta2),
               c_float(eps), c_float(weight_decay), c_int64(step), c_bool(amsgrad))
    check_returncode(ret)
    return param, param_grad, exp_avg, exp_avg_sq, max_exp_avg_sq


def adadelta(param, param_grad, square_avg, acc_delta, lr, rho, eps, weight_decay):
    # note: buf, param_grad are mutable
    func = check_function("diopiAdadelta")
    ret = func(param.context_handle, param.tensor_handle, param_grad.tensor_handle, square_avg.tensor_handle,
               acc_delta.tensor_handle, c_float(lr), c_float(rho), c_float(eps), c_float(weight_decay))
    check_returncode(ret)
    return param, param_grad, square_avg, acc_delta


def rmsprop(param, param_grad, square_avg, grad_avg, momentum_buffer, lr, alpha, eps, weight_decay, momentum, centered):
    func = check_function("diopiRmsprop")
    ret = func(param.context_handle, param.tensor_handle, param_grad.tensor_handle, square_avg.tensor_handle,
               grad_avg.tensor_handle, momentum_buffer.tensor_handle, c_float(lr), c_float(alpha), c_float(eps),
               c_float(weight_decay), c_float(momentum), c_bool(centered))
    check_returncode(ret)
    return param, param_grad, square_avg, grad_avg, momentum_buffer


def conv_transpose2d(input, weight, bias=None, stride=1,
                     padding=0, output_padding=0, groups=1, dilation=1) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
        bias = bias.tensor_handle
    else:
        bias = c_void_p()

    sizeI = input.size()
    sizeW = list(weight.size())
    assert len(sizeI) == 4 and len(sizeW) == 4,\
        'input and weight must be 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    sizeO.append(sizeW[1] * groups)

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(output_padding, int):
        output_padding = (output_padding, output_padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    for i in range(-2, 0):
        # equivalent kernel size
        sizeW[i] = (sizeW[i] - 1) * dilation[i]
        sizeO.append(int((sizeI[i] - 1) * stride[i] - 2 * padding[i] + sizeW[i] + output_padding[i]) + 1)
    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    output_padding = Sizes(tuple(output_padding))
    dilation = Sizes(tuple(dilation))

    out = Tensor(sizeO, input.get_dtype())
    func = check_function("diopiConvTranspose2d")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               weight.tensor_handle, bias, stride, padding, output_padding, c_int64(groups), dilation)
    check_returncode(ret)
    return out


def cumsum(input, dim, dtype=None):
    assert isinstance(dim, int), "dim should be int"

    sizeI = list(input.size())
    assert dim < len(sizeI), "dim out of index"

    out = Tensor(input.size(), promote_type(input, Dtype.int64)) if dtype is None else Tensor(input.size(), dtype)
    func = check_function("diopiCumsum")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, c_int64(dim))
    check_returncode(ret)
    return out


def infer_size(a, b):
    dimsA = len(a)
    dimsB = len(b)
    ndim = dimsA if dimsA >= dimsB else dimsB
    expanded_sizes = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1
        assert sizeA == sizeB or sizeA == 1 or sizeB == 1, \
            f"The size of tensor a ({sizeA}) must match the size of tensor b ({sizeB}) at non-singleton dimension {i}"
        expanded_sizes[i] = sizeA if sizeA != 1 else sizeB
    return expanded_sizes


def cdist(x1, x2, p, compute_mode=None):
    sizeX1 = list(x1.size())
    sizeX2 = list(x2.size())
    dim1 = len(sizeX1)
    dim2 = len(sizeX2)
    assert dim1 > 1 and dim2 > 1, "cdist only supports at least 2D tensors"
    assert sizeX1[-1] == sizeX2[-1], "X1 and X2 must have the same number of elements at the last dimension"
    row1 = sizeX1[-2]
    row2 = sizeX2[-2]
    batch_tensor1 = sizeX1[:-2]
    batch_tensor2 = sizeX2[:-2]
    expand_batch_portion = infer_size(batch_tensor1, batch_tensor2)
    out_shape = expand_batch_portion + [row1, row2]
    if compute_mode is not None:
        if compute_mode == 'use_mm_for_euclid_dist':
            compute_mode = 1
        else:
            compute_mode = 2
        compute_mode = byref(c_int64(compute_mode))
    else:
        compute_mode = c_void_p()
    out = Tensor(out_shape, x1.get_dtype())
    func = check_function("diopiCdist")
    ret = func(x1.context_handle, out.tensor_handle, x1.tensor_handle, x2.tensor_handle, c_double(p), compute_mode)
    check_returncode(ret)
    return out


def cdist_backward(x1, grad_outputs, output, x2, p, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    sizeX1 = list(x1.size())
    sizeX2 = list(x2.size())
    dim1 = len(sizeX1)
    dim2 = len(sizeX2)
    assert dim1 > 1 and dim2 > 1, "cdist only supports at least 2D tensors"
    assert sizeX1[-1] == sizeX2[-1], "X1 and X2 must have the same number of elements at the last dimension"
    column1 = sizeX1[-1]
    row1 = sizeX1[-2]
    batch_tensor1 = sizeX1[:-2]
    batch_tensor2 = sizeX2[:-2]
    expand_batch_portion = infer_size(batch_tensor1, batch_tensor2)
    grad_x1_shape = expand_batch_portion + [row1, column1]
    grad_x1 = Tensor(grad_x1_shape, x1.get_dtype())
    func = check_function("diopiCdistBackward")
    ret = func(x1.context_handle, grad_x1.tensor_handle, grad_outputs[0].tensor_handle, x1.tensor_handle,
               x2.tensor_handle, c_double(p), output.tensor_handle)
    grad_x1 = grad_x1.numpy()
    i = len(grad_x1.shape) - 1
    j = dim1 - 1
    while i >= 0 and j >= 0 and len(grad_x1.shape) != dim1:
        while i > 0 and j > 0 and grad_x1.shape[i] != sizeX1[j]:
            grad_x1 = np.sum(grad_x1, axis=i)
            i -= 1
        j = j - 1
        i = i - 1
    if i == 0 and j == -1:
        grad_x1 = np.sum(grad_x1, axis=i)
    for index in range(dim1):
        if sizeX1[index] != grad_x1.shape[index]:
            grad_x1 = np.sum(grad_x1, axis=index, keepdims=True)
    grad_x1 = Tensor.from_numpy(grad_x1)
    check_returncode(ret)
    return {'x1': grad_x1}


def reciprocal(input, inplace=False) -> Tensor:
    out = raw_like(input)
    call = "diopiReciprocal"

    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle)
    else:
        out = Tensor(input.size(), promote_type(input, Dtype.float32))
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle)

    check_returncode(ret)
    return out


def bitwise_not(input, inplace=False):
    assert input.get_dtype() in [Dtype.bool, Dtype.uint8, Dtype.int8, Dtype.int16, Dtype.int32, glob_vars.int_type], \
        "input tensor must be of integral or boolean"
    return unary_op(input, inplace, "diopiBitwiseNot")


def bitwise_and(input, other, inplace=False):
    assert input.get_dtype() in [Dtype.bool, Dtype.uint8, Dtype.int8, Dtype.int16, Dtype.int32, glob_vars.int_type], \
        "input tensor must be of integral or boolean"
    if isinstance(other, Tensor):
        assert other.get_dtype() in [Dtype.bool, Dtype.uint8, Dtype.int8, Dtype.int16, Dtype.int32, glob_vars.int_type], \
            "other tensor must be of integral or boolean"
    else:
        assert isinstance(other, int), "other must be of integral or boolean"
    out_dtype = common_dtype(input, other)
    return binary_op_scalar(input, other, inplace, "diopiBitwiseAnd", dtype=out_dtype)


def bitwise_or(input, other, inplace=False):
    assert input.get_dtype() in [Dtype.bool, Dtype.uint8, Dtype.int8, Dtype.int16, Dtype.int32, glob_vars.int_type], \
        "input tensor must be of integral or boolean"
    if isinstance(other, Tensor):
        assert other.get_dtype() in [Dtype.bool, Dtype.uint8, Dtype.int8, Dtype.int16, Dtype.int32, glob_vars.int_type], \
            "other tensor must be of integral or boolean"
    else:
        assert isinstance(other, int), "other must be of integral or boolean"
    out_dtype = common_dtype(input, other)
    return binary_op_scalar(input, other, inplace, "diopiBitwiseOr", dtype=out_dtype)


def argmax(input, dim=None, keepdim=False):
    sizeO = list(input.size())
    if dim is not None:
        assert dim < len(sizeO), "dim out of index"
        if keepdim:
            sizeO[dim] = 1
        else:
            sizeO = sizeO[:dim] + sizeO[dim + 1:]
        dim = byref(c_int64(dim))
    else:
        sizeO = [1]
        dim = c_void_p()

    out = Tensor(sizeO, glob_vars.int_type)
    func = check_function("diopiArgmax")
    # todo: check the reason of using keepdim
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dim, c_bool(keepdim))
    check_returncode(ret)

    return out


def smooth_l1_loss(input, target, reduction='mean', beta=1.0):
    assert input.shape() == target.shape(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiSmoothL1Loss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, c_int64(reduction_mode), c_double(beta))
    check_returncode(ret)
    return out


def smooth_l1_loss_backward(input, grad_outputs, target, reduction='mean', beta=1.0, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiSmoothL1LossBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, target.tensor_handle, c_int64(reduction_mode), c_double(beta))
    check_returncode(ret)
    return {"input": grad_input}


def maximum(input, other) -> Tensor:
    size = broadcast_out_size(list(input.size()), list(other.size()))
    out = Tensor(size, common_dtype(input, other))

    func = check_function("diopiMaximum")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, other.tensor_handle)
    check_returncode(ret)
    return out


def minimum(input, other) -> Tensor:
    size = broadcast_out_size(list(input.size()), list(other.size()))
    out = Tensor(size, common_dtype(input, other))

    func = check_function("diopiMinimum")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, other.tensor_handle)
    check_returncode(ret)
    return out


def mm(input, mat2) -> Tensor:
    size1 = list(input.size())
    assert (len(size1) == 2), 'input must be 2d tensor'
    size2 = mat2.size()
    assert (len(size2) == 2), 'mat2 must be 2d tensor'
    assert (size1[1] == size2[0]), 'invalid args'

    size_out = size1
    size_out[1] = size2[1]
    out = Tensor(size_out, input.get_dtype())

    func = check_function("diopiMm")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, mat2.tensor_handle)
    check_returncode(ret)
    return out


def conv3d(input, weight, bias=None, stride=1,
           padding=0, dilation=1, groups=1) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
        bias = bias.tensor_handle
    else:
        bias = c_void_p()

    sizeI = input.size()
    sizeW = list(weight.size())
    assert len(sizeI) == 5 and len(sizeW) == 5,\
        'input and weight must be 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    sizeO.append(sizeW[0])

    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    for i in range(-3, 0):
        # equivalent kernel size
        sizeW[i] += (sizeW[i] - 1) * (dilation[i] - 1)
        sizeO.append(int((sizeI[i] - sizeW[i] + 2 * padding[i]) / stride[i]) + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    dilation = Sizes(tuple(dilation))

    nhwc_stride = compute_nhwc_stride_3d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)
    func = check_function("diopiConvolution3d")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               weight.tensor_handle, bias, stride, padding, dilation, c_int64(groups))
    check_returncode(ret)
    return out


def conv3d_backward(input, grad_outputs, weight, bias=None, stride=1,
                    padding=0, dilation=1, groups=1, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    sizeI = input.size()
    sizeW = weight.size()
    assert len(sizeI) == 5 and len(sizeW) == 5,\
        'input and weight must be 5d tensors'

    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    dilation = Sizes(tuple(dilation))

    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    out = {"input": grad_input, "weight": grad_weight}

    if bias is None:
        grad_bias = c_void_p()
        sizeBias = c_void_p()
    else:
        gradBias = raw_like(bias)
        grad_bias = gradBias.tensor_handle
        sizeBias = byref(Sizes(bias.size()))
        out.update({"bias": grad_bias})

    # todo: no transposed/output_padding in forward
    transposed = False
    output_padding = Sizes((0, 0, 0))

    func = check_function("diopiConvolution3dBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_weight.tensor_handle, grad_bias,
               grad_outputs[0].tensor_handle, input.tensor_handle, weight.tensor_handle, sizeBias, stride,
               padding, dilation, c_bool(transposed), output_padding, c_int64(groups))
    check_returncode(ret)
    return out


def expand(input, size) -> Tensor:
    SizeI = input.size()
    size = list(size)
    for i in range(-1, -len(SizeI) - 1, -1):
        if size[i] == -1:
            size[i] = SizeI[i]
        else:
            assert size[i] == SizeI[i] or SizeI[i] == 1,\
                "size must be broadcastable with input"

    if len(size) > len(SizeI):
        assert size[0] >= 0, "the size of new dimension can't be negative"

    out = Tensor(size, input.get_dtype())

    func = check_function("diopiExpand")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle)
    check_returncode(ret)
    return out


def unfold(input, dimension, size, step):
    sizeO = list(input.size())
    sizeO[dimension] = int((sizeO[dimension] - size) / step + 1)
    sizeO.append(size)

    out = Tensor(sizeO, input.get_dtype())
    func = check_function("diopiUnfold")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, c_int64(dimension), c_int64(size), c_int64(step))
    check_returncode(ret)
    return out


def unfold_backward(input, grad_outputs, dimension, size, step, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    sizeI = Sizes(input.size())

    func = check_function("diopiUnfoldBackward")
    ret = func(grad_input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle, sizeI,
               c_int64(dimension), c_int64(size), c_int64(step))
    check_returncode(ret)
    return {"input": grad_input}


def masked_select(input, mask) -> Tensor:
    assert mask.get_dtype() == Dtype.bool, "mask must be bool tensor"
    out_tensor_handle = TensorHandle()

    func = check_function("diopiMaskedSelect")
    ret = func(input.context_handle, pointer(out_tensor_handle), input.tensor_handle,
               mask.tensor_handle)
    check_returncode(ret)
    out = Tensor.from_handle(out_tensor_handle)
    return out


def masked_select_backward(input, grad_outputs, mask) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiMaskedSelectBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, mask.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def index_fill(input, dim, index, value, inplace=False) -> Tensor:
    out = raw_like(input)

    call = "diopiIndexFill"
    call_scalar = False
    if isinstance(value, Tensor):
        value = value.tensor_handle
    else:
        value = byref(Scalar(value))
        call_scalar = True

    if inplace:
        out = input
        call = call + "Inp"
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle, c_int64(dim), index.tensor_handle, value)
    else:
        out = raw_like(input)
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, c_int64(dim), index.tensor_handle, value)

    check_returncode(ret)
    return out


def linspace(start, end, steps, dtype=None):
    dtype = Dtype.float32 if dtype is None else dtype

    out = Tensor((steps, ), dtype)

    start = byref(Scalar(start))
    end = byref(Scalar(end))
    func = check_function("diopiLinspace")

    ret = func(out.context_handle, out.tensor_handle, start, end, c_int64(steps))
    check_returncode(ret)
    return out


def roll(input, shifts, dims=None):
    if isinstance(shifts, int):
        shifts = (shifts, )
    shifts = Sizes(tuple(shifts))

    if dims is not None:
        dims = Sizes(tuple(dims))
    else:
        dims = Sizes(tuple(()))

    out = raw_like(input)
    func = check_function("diopiRoll")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, shifts, dims)
    check_returncode(ret)
    return out


def norm(input, p, dim=None, keepdim=False, dtype=None):
    p = byref(Scalar(p))
    dim, out = reduce_op_process(input, dim, keepdim, dtype)
    dim = Sizes(tuple(dim))

    func = check_function("diopiNorm")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, p, dim)
    check_returncode(ret)
    return out


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05, backward=False):
    dim = list(input.size())
    save_mean = Tensor((dim[0], num_groups), input.get_dtype())
    save_invstd = raw_like(save_mean)

    weight = c_void_p() if weight is None else weight.tensor_handle
    bias = c_void_p() if bias is None else bias.tensor_handle

    out = raw_like(input)
    func = check_function("diopiGroupNorm")
    ret = func(input.context_handle, out.tensor_handle, save_mean.tensor_handle, save_invstd.tensor_handle,
               input.tensor_handle, weight, bias, c_int64(num_groups), c_double(eps))
    check_returncode(ret)
    if backward:
        return save_mean, save_invstd
    return out


def group_norm_backward(input, grad_outputs, num_groups, weight=None, bias=None, eps=1e-05, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    save_mean, save_invstd = group_norm(input, num_groups, weight, bias, eps, backward=True)
    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    grad_bias = raw_like(bias)
    weight = c_void_p() if weight is None else weight.tensor_handle
    bias = c_void_p() if bias is None else bias.tensor_handle

    out = {"input": grad_input, "weight": grad_weight, "bias": grad_bias}
    func = check_function("diopiGroupNormBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_weight.tensor_handle, grad_bias.tensor_handle,
               grad_outputs[0].tensor_handle, input.tensor_handle, weight, save_mean.tensor_handle, save_invstd.tensor_handle,
               c_int64(num_groups))
    check_returncode(ret)
    return out


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05, backward=False):
    sizeI = input.size()
    dims = len(sizeI) - len(normalized_shape)
    size = [i for i in sizeI[0:dims]]
    save_mean = Tensor(size, input.get_dtype())
    save_invstd = raw_like(save_mean)

    weight = c_void_p() if weight is None else weight.tensor_handle
    bias = c_void_p() if bias is None else bias.tensor_handle

    out = raw_like(input)
    func = check_function("diopiLayerNorm")
    ret = func(input.context_handle, out.tensor_handle, save_mean.tensor_handle, save_invstd.tensor_handle,
               input.tensor_handle, weight, bias, Sizes(normalized_shape), c_double(eps))
    check_returncode(ret)
    if backward:
        return save_mean, save_invstd
    return out


def layer_norm_backward(input, grad_outputs, normalized_shape, weight=None, bias=None, eps=1e-05, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    save_mean, save_invstd = layer_norm(input, normalized_shape, weight, bias, eps, backward=True)
    grad_input = raw_like(input)
    out = {"input": grad_input}

    if weight is None:
        weight = c_void_p()
        grad_weight_handle = c_void_p()
    else:
        grad_weight = raw_like(weight)
        weight = weight.tensor_handle
        grad_weight_handle = grad_weight.tensor_handle
        out['weight'] = grad_weight

    if bias is None:
        bias = c_void_p()
        grad_bias_handle = c_void_p()
    else:
        grad_bias = raw_like(bias)
        bias = bias.tensor_handle
        grad_bias_handle = grad_bias.tensor_handle
        out['bias'] = grad_bias

    func = check_function("diopiLayerNormBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_weight_handle, grad_bias_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, weight, bias, save_mean.tensor_handle, save_invstd.tensor_handle, Sizes(normalized_shape))
    check_returncode(ret)
    return out


def adaptive_avg_pool3d(input, output_size):
    sizeI = input.size()
    assert len(sizeI) == 5 or len(sizeI) == 4,\
        'input must be 4d or 5d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 5:
        sizeO.append(sizeI[1])

    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    for i in range(-3, 0):
        if output_size[i] is None:
            sizeO.append(sizeI[i])
        else:
            sizeO.append(output_size[i])

    nhwc_stride = compute_nhwc_stride_3d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)
    output_size = Sizes((sizeO[-3], sizeO[-2], sizeO[-1]))

    func = check_function("diopiAdaptiveAvgPool3d")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, output_size)
    check_returncode(ret)
    return out


def adaptive_avg_pool3d_backward(input, grad_outputs, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiAdaptiveAvgPool3dBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def adaptive_max_pool3d(input, output_size, return_indices=False):
    sizeI = input.size()
    assert len(sizeI) == 5 or len(sizeI) == 4,\
        'input must be 4d or 5d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 5:
        sizeO.append(sizeI[1])

    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    for i in range(-3, 0):
        if output_size[i] is None:
            sizeO.append(sizeI[i])
        else:
            sizeO.append(output_size[i])

    nhwc_stride = compute_nhwc_stride_3d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)
    output_size = Sizes(tuple(output_size))

    if return_indices:
        func = check_function("diopiAdaptiveMaxPool3dWithIndices")
        nhwc_stride = compute_nhwc_stride_3d(sizeO) if glob_vars.nhwc else None
        indices = Tensor(sizeO, glob_vars.int_type, stride=nhwc_stride)
        ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
                   input.tensor_handle, output_size)
        check_returncode(ret)
        return out, indices
    else:
        func = check_function("diopiAdaptiveMaxPool3d")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, output_size)
    check_returncode(ret)
    return out


def adaptive_max_pool3d_backward(input, grad_outputs, output_size, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    _, indices = adaptive_max_pool3d(input, output_size, return_indices=True)

    func = check_function("diopiAdaptiveMaxPool3dBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, indices.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False) -> Tensor:
    sizeI = input.size()
    assert len(sizeI) == 5 or len(sizeI) == 4,\
        'input must be 5d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 5:
        sizeO.append(sizeI[1])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    for i in range(-3, 0):
        tmp_ker_size = kernel_size[i] + (kernel_size[i] - 1) * (dilation[i] - 1)
        tmp_size = (sizeI[i] - tmp_ker_size + 2 * padding[i]) / stride[i] + 1
        tmp_size = tmp_size if tmp_size > 1 else 1
        if ceil_mode:
            sizeO.append(math.ceil(tmp_size))
        else:
            sizeO.append(math.floor(tmp_size))

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    dilation = Sizes(tuple(dilation))
    out = Tensor(sizeO, input.get_dtype())

    if not return_indices:
        func = check_function("diopiMaxPool3d")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, kernel_size,
                   stride, padding, dilation, c_bool(ceil_mode))
        check_returncode(ret)
        return out
    else:
        func = check_function("diopiMaxPool3dWithIndices")
        indices = Tensor(sizeO, glob_vars.int_type)
        ret = func(input.context_handle, out.tensor_handle,
                   indices.tensor_handle, input.tensor_handle,
                   kernel_size, stride, padding, dilation, c_bool(ceil_mode))
        check_returncode(ret)
        return out, indices


def max_pool3d_backward(input, grad_outputs, kernel_size, stride=None, padding=0, dilation=1,
                        ceil_mode=False, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    sizeI = input.size()
    assert len(sizeI) == 5 or len(sizeI) == 4, 'input must be 5d or 4d tensors'

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    _, indices = max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode, True)
    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    dilation = Sizes(tuple(dilation))

    func = check_function("diopiMaxPool3dBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, kernel_size, stride, padding, dilation, c_bool(ceil_mode), indices.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def permute(input, dims=None) -> Tensor:
    assert isinstance(dims, (tuple, list)) or dims is None,\
        "dims should be tuple or list"

    sizeI = list(input.size())
    sizeO = list(input.size())
    for i in range(len(dims)):
        sizeO[i] = sizeI[dims[i]]
    out = Tensor(sizeO, input.get_dtype())
    dims = Sizes(tuple(dims))
    func = check_function("diopiPermute")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dims)
    check_returncode(ret)
    return out


def copy_(input, other) -> Tensor:
    func = check_function("diopiCopyInp")
    ret = func(input.context_handle, other.tensor_handle, input.tensor_handle)
    check_returncode(ret)
    return input


def gather(input, dim, index):
    assert isinstance(dim, int), "dim must be int"
    assert len(input.size()) == len(index.size()), "input and index must have the same number of dimensions"
    out = Tensor(index.size(), input.get_dtype())
    func = check_function("diopiGather")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, c_int64(dim), index.tensor_handle)
    check_returncode(ret)
    return out


def gather_backward(input, grad_outputs, dim, index, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    assert isinstance(dim, int), "dim must be int"
    grad_input = raw_like(input)
    func = check_function("diopiGatherBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, c_int64(dim), index.tensor_handle)
    check_returncode(ret)
    return {"input": grad_input}


def remainder(other, input=None, self=None):
    if self is not None:
        input = self
    call = "diopiRemainder"
    if isinstance(input, Tensor):
        context = input.context_handle
        if isinstance(other, Tensor):
            call += "Tensor"
            sizeO = list(input.size())
            sizeOther = list(other.size())
            for i in range(0, len(sizeOther)):
                if sizeO[i] != sizeOther[i]:
                    assert sizeO[i] == 1 or sizeOther[i] == 1, \
                        "input and other must Supports broadcasting to a common shape"
                    if sizeO[i] == 1:
                        sizeO[i] = sizeOther[i]
            out_dtype = common_dtype(input, other)
            out = Tensor(sizeO, out_dtype)
            input = input.tensor_handle
            other = other.tensor_handle
        else:
            call += "Scalar"
            out_dtype = common_dtype(input, other)
            out = Tensor(input.size(), out_dtype)
            other = byref(Scalar(other))
            input = input.tensor_handle
    else:
        assert isinstance(other, Tensor), "input or other must be tensor"
        context = other.context_handle
        out_dtype = common_dtype(input, other)
        out = Tensor(other.size(), out_dtype)
        input = byref(Scalar(input))
        other = other.tensor_handle
    func = check_function(call)
    ret = func(context, out.tensor_handle, input, other)
    check_returncode(ret)
    return out


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False, backward=False):
    sizeO = (1, )
    sizeI = list(log_probs.size())
    reduction_mode = convert_reduction(reduction)
    max_target_length = int(max(target_lengths, 0)[0].numpy())
    max_target_length = 2 * max_target_length + 1
    if reduction == 'none':
        sizeO = (sizeI[1], )
    neg_log_likelihood = Tensor((sizeI[1], ), log_probs.get_dtype())
    log_alpha = Tensor((sizeI[1], sizeI[0], max_target_length), log_probs.get_dtype())
    out = Tensor(sizeO, log_probs.get_dtype())

    func = check_function("diopiCTCLoss")
    ret = func(log_probs.context_handle, out.tensor_handle, neg_log_likelihood.tensor_handle,
               log_alpha.tensor_handle, log_probs.tensor_handle, targets.tensor_handle, input_lengths.tensor_handle,
               target_lengths.tensor_handle, c_int64(blank), c_int64(reduction_mode), c_bool(zero_infinity))
    check_returncode(ret)
    if backward:
        return neg_log_likelihood, log_alpha
    return out


def ctc_loss_backward(log_probs, grad_outputs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(log_probs)
    neg_log_likelihood, log_alpha = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction,
                                             zero_infinity, backward=True)

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiCTCLossBackward")
    ret = func(log_probs.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle, log_probs.tensor_handle,
               targets.tensor_handle, input_lengths.tensor_handle, target_lengths.tensor_handle, neg_log_likelihood.tensor_handle,
               log_alpha.tensor_handle, c_int64(blank), c_int64(reduction_mode), c_bool(zero_infinity))
    check_returncode(ret)
    return {"log_probs": grad_input}


def index_put(input, values, indices1, indices2=None, accumulate=False, inplace=False):
    if indices2 is not None:
        c_tensors = [indices1.tensor_handle, indices2.tensor_handle]
        indices_counts = 2
    else:
        c_tensors = [indices1.tensor_handle]
        indices_counts = 1
    c_tensors = (c_void_p * 2)(*c_tensors)
    call = "diopiIndexPut"
    out = raw_like(input)
    if inplace:
        call += "Inp"
        out = input
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle, values.tensor_handle,
                   pointer(c_tensors), c_int64(indices_counts), c_bool(accumulate))
    else:
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, values.tensor_handle,
                   pointer(c_tensors), c_int64(indices_counts), c_bool(accumulate))
    check_returncode(ret)
    return out


def scatter(input, dim, index, src=None, value=None, reduce=None, inplace=False):
    assert isinstance(dim, int), "dim must be int"
    assert len(input.size()) == len(index.size()), \
        "input and index must have the same number of dimensions"
    assert (src is not None) or (value is not None)
    if reduce is not None:
        assert reduce == 'add' or reduce == 'multiply', "reduce argument must be either add or multiply."
    else:
        reduce = ""
    if src is not None:
        assert len(input.size()) == len(src.size()), \
            "input and src must have the same number of dimensions"
    else:
        src = value
    out = raw_like(input)
    call = "diopiScatter"
    call_scalar = False
    if isinstance(src, Tensor):
        src = src.tensor_handle
    else:
        src = byref(Scalar(src))
        call_scalar = True

    if inplace:
        out = input
        call = call + "Inp"
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle, c_int64(dim),
                   src, index.tensor_handle, reduce.encode('UTF-8'))
    else:
        out = raw_like(input)
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
                   c_int64(dim), src, index.tensor_handle, reduce.encode('UTF-8'))

    check_returncode(ret)
    return out


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=False) -> Tensor:
    assert size is None or scale_factor is None, "only one of size or scale_factor should be defined"
    sizeI = list(input.size())
    if size is not None:
        if isinstance(size, int):
            size = [size for _ in range(len(sizeI) - 2)]
        for i in range(len(size)):
            sizeI[-i - 1] = size[-i - 1]
    else:
        dim = len(sizeI) - 2
        if not isinstance(scale_factor, tuple):
            scale_factor = [scale_factor for _ in range(dim)]
        for i in range(2, dim + 2):
            sizeI[i] = int(scale_factor[i - 2] * sizeI[i])

    nhwc_stride = compute_nhwc_stride(sizeI) if glob_vars.nhwc else None
    out = Tensor(sizeI, input.get_dtype(), stride=nhwc_stride)

    c_size = Sizes(tuple(sizeI[2:]))
    if mode == "nearest":
        func = check_function("diopiUpsampleNearest")
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, c_size)
    else:
        func = check_function("diopiUpsampleLinear")
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, c_size,
                   c_bool(align_corners), mode.encode('UTF-8'))
    check_returncode(ret)
    return out


def interpolate_backward(input, grad_outputs, size, mode="nearest", align_corners=None, **kwargs) -> Tensor:
    in_size = Sizes(input.size())
    out_size = Sizes(grad_outputs[0].size()[2:])
    grad_input = raw_like(input)

    if mode == "nearest":
        func = check_function("diopiUpsampleNearestBackward")
        ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle, out_size, in_size)
    else:
        func = check_function("diopiUpsampleLinearBackward")
        ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle, out_size, in_size,
                   c_bool(align_corners), mode.encode('UTF-8'))
    check_returncode(ret)
    return {'input': grad_input}


def pad(input, pad, mode='constant', value=None):
    assert mode in ['constant', 'reflect', 'replicate', 'circular'], \
        "mode must one of ""'constant', 'reflect', 'replicate', 'circular'"
    sizeO = list(input.size())
    assert len(pad) % 2 == 0, "Padding length must be divisible by 2"
    assert len(pad) // 2 <= len(sizeO), \
        "Padding length must be equal or more than length of input"
    paded_length = len(pad) // 2
    for i in range(paded_length):
        if len(pad) <= len(sizeO):
            pad_idx = paded_length - i
        else:
            pad_idx = i + 1
        sizeO[-pad_idx] += (pad[2 * i] + pad[2 * i + 1])
    pad = Sizes(pad)
    if value is None and mode == 'constant':
        value = 0
    if value is None:
        value = c_void_p()
    else:
        value = byref(c_double(value))

    nhwc_stride = compute_nhwc_stride(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)
    func = check_function("diopiPad")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, pad,
               mode.encode('UTF-8'), value)
    check_returncode(ret)
    return out


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    out_tensor_handle = TensorHandle()
    if return_inverse:
        sizeI = list(input.size())
        if dim is not None:
            sizeI = (sizeI[dim], )
        indices = Tensor(sizeI, glob_vars.int_type)
        indices_handle = indices.tensor_handle
    else:
        indices_handle = c_void_p()
    counts = TensorHandle()

    if dim is None:
        dim = c_void_p()
    else:
        dim = byref(c_int64(dim))

    func = check_function("diopiUnique")
    ret = func(input.context_handle, pointer(out_tensor_handle), input.tensor_handle, dim, c_bool(sorted),
               c_bool(return_counts), indices_handle, pointer(counts))
    check_returncode(ret)
    out = Tensor.from_handle(out_tensor_handle)
    if return_counts:
        counts = Tensor.from_handle(counts)
    if return_inverse and not return_counts:
        return out, indices
    elif not return_inverse and return_counts:
        return out, counts
    elif return_inverse and return_counts:
        return out, indices, counts
    else:
        return out


def prod(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    assert isinstance(dim, (int)) or dim is None,\
        "dim should be int"

    out_dtype = dtype if dtype is not None else promote_type(input, Dtype.int64)
    _, out = reduce_op_process(input, dim, keepdim, out_dtype)
    if dim is None:
        dim = c_void_p()
    else:
        dim = byref(c_int64(dim))

    func = check_function("diopiProd")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dim)
    check_returncode(ret)
    return out


def linear_backward(input, grad_outputs, weight, bias=None, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"

    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
        grad_bias = raw_like(bias)
        grad_bias_handle = grad_bias.tensor_handle
    else:
        grad_bias_handle = c_void_p()

    func = check_function("diopiLinearBackward")

    ret = func(input.context_handle, grad_input.tensor_handle, grad_weight.tensor_handle, grad_bias_handle, grad_outputs[0].tensor_handle,
               input.tensor_handle, weight.tensor_handle)
    check_returncode(ret)
    if bias is None:
        return {"input": grad_input, "weight": grad_weight}
    return {"input": grad_input, "weight": grad_weight, "bias": grad_bias}


def cross_entropy_backward(input, grad_outputs, target, weight=None, ignore_index=- 100,
                           reduction='mean', label_smoothing=0.0, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    grad_input = raw_like(input)
    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiCrossEntropyLossBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, c_int64(reduction_mode), c_int64(ignore_index), c_double(label_smoothing))
    check_returncode(ret)
    return {"input": grad_input}


def erfinv(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, 'diopiErfinv')


def im2col(input, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    sizeI = input.size()
    assert len(sizeI) == 4, "only support 4d tensor"
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    num_blocks = 1
    for i in range(2):
        num_blocks *= int((sizeI[i + 2] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
    channels = sizeI[1]
    for i in range(len(kernel_size)):
        channels *= kernel_size[i]
    sizeO = [sizeI[0], channels, num_blocks]

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    dilation = Sizes(tuple(dilation))

    out = Tensor(sizeO, input.get_dtype())
    func = check_function("diopiIm2Col")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, kernel_size,
               dilation, padding, stride)
    check_returncode(ret)
    return out


def col2im(input, output_size, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    sizeI = input.size()
    assert len(sizeI) == 3, "only support 3d tensor"
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    channels = sizeI[1]
    for i in range(len(kernel_size)):
        channels = channels // kernel_size[i]
    sizeO = [sizeI[0], channels, output_size[0], output_size[1]]

    output_size = Sizes(tuple(output_size))
    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    dilation = Sizes(tuple(dilation))

    out = Tensor(sizeO, input.get_dtype())
    func = check_function("diopiCol2Im")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, output_size, kernel_size,
               dilation, padding, stride)
    check_returncode(ret)
    return out


def flip(input, dims):
    out = raw_like(input)
    dims = Sizes(tuple(dims))
    func = check_function("diopiFlip")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dims)
    check_returncode(ret)
    return out


def cholesky_ex(input, upper=False, check_errors=False):
    out = raw_like(input)
    sizeI = input.size()
    nums = sizeI[0:-2] if len(sizeI) > 2 else ()
    info = Tensor(nums, Dtype.int32)
    func = check_function("diopiCholesky")
    ret = func(input.context_handle, out.tensor_handle, info.tensor_handle, input.tensor_handle, c_bool(upper), c_bool(check_errors))
    check_returncode(ret)
    return out, info


def cholesky_ex_backward(input, grad_outputs, output, upper=False, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    func = check_function("diopiCholeskyBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_outputs[0].tensor_handle, output.tensor_handle, c_bool(upper))
    check_returncode(ret)
    return {"input": grad_input}


def triangular_solve(input, A, upper=True, transpose=False, unitriangular=False):
    sizeA = list(A.size())
    sizeI = list(input.size())
    sizeO = sizeA if len(sizeA) > len(sizeI) else sizeI
    sizeO[-1] = sizeI[-1]
    out = Tensor(sizeO, A.get_dtype())
    sizeO[-1] = sizeA[-1]
    cloned_mat = Tensor(sizeO, A.get_dtype())
    func = check_function("diopiTriangularSolve")
    ret = func(input.context_handle, out.tensor_handle, cloned_mat.tensor_handle, input.tensor_handle,
               A.tensor_handle, c_bool(upper), c_bool(transpose), c_bool(unitriangular))
    check_returncode(ret)
    Res = namedtuple('Res', ['solution', 'cloned_coefficient'])
    output = Res(out, cloned_mat)
    return output


def triangular_solve_backward(input, grad_outputs, output, A, upper=True, transpose=False, unitriangular=False, **kwargs):
    assert len(grad_outputs) <= 2, "accept at most 2 gradient to do backward"
    grad_cloned_mat = c_void_p() if len(grad_outputs) == 1 else grad_outputs[1].tensor_handle
    grad_A = raw_like(A)
    grad_input = raw_like(input)
    func = check_function("diopiTriangularSolveBackward")
    ret = func(input.context_handle, grad_input.tensor_handle, grad_A.tensor_handle, grad_outputs[0].tensor_handle,
               grad_cloned_mat, output.tensor_handle, input.tensor_handle, A.tensor_handle, c_bool(upper), c_bool(transpose), c_bool(unitriangular))
    check_returncode(ret)
    return {"input": grad_input, "A": grad_A}


def repeat(input, repeats):
    sizeI = list(input.size())
    input_ndims = len(sizeI)
    repeats_size = list(repeats)
    out_ndims = len(repeats)
    assert input_ndims <= out_ndims, f'input_ndims ({input_ndims}) should <= out_ndims ({out_ndims})'

    output_size = []
    for i in range(out_ndims):
        idx = input_ndims + i - out_ndims
        k = repeats_size[i] * sizeI[idx] if idx >= 0 else repeats_size[i]
        output_size.append(k)

    sizeO = Sizes(output_size)
    repeats_size = Sizes(repeats)

    out = Tensor(output_size, input.get_dtype())
    func = check_function("diopiRepeat")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, repeats_size)
    check_returncode(ret)
    return out


def normal(mean, std, size=None):
    call = "diopiNormal"
    if isinstance(mean, Tensor) and isinstance(std, Tensor):
        sizeX1 = list(mean.size())
        sizeX2 = list(std.size())
        if mean.numel() <= std.numel():
            out_size = infer_size(sizeX1, sizeX2)
            out = Tensor(out_size, std.get_dtype())
        if mean.numel() > std.numel():
            out_size = infer_size(sizeX1, sizeX2)
            out = Tensor(out_size, mean.get_dtype())
        call += "Tensor"
    elif isinstance(mean, Tensor):
        out = Tensor(mean.size(), mean.get_dtype())
        call += "TensorScalar"
    elif isinstance(std, Tensor):
        out = Tensor(std.size(), std.get_dtype())
        call += "ScalarTensor"
    else:
        if size is not None:
            out = Tensor(size, Dtype.float32)
        else:
            out = Tensor((), Dtype.float32)
    arg_mean = mean.tensor_handle if isinstance(mean, Tensor) else c_double(mean)
    arg_std = std.tensor_handle if isinstance(std, Tensor) else c_double(std)
    func = check_function(call)
    ret = func(out.context_handle, out.tensor_handle, arg_mean, arg_std)
    check_returncode(ret)
    return out


def normal_(input, mean, std, shape=None) -> Tensor:
    call = "diopiNormalInp"
    func = check_function(call)
    ret = func(input.context_handle, input.tensor_handle, c_double(mean), c_double(std))
    check_returncode(ret)
    return input


def meshgrid(tensors, shape=None):
    assert isinstance(tensors, (list, tuple)),\
        "tensors must be a list or tuple"
    inputsNum = len(tensors)
    c_tensors = []
    co_tensors = []
    dims = []
    for tensor in tensors:
        c_tensors.append(tensor.tensor_handle)
        if len(tensor.size()) > 0:
            dims.append(tensor.size()[0])
        else:
            dims.append(1)
    c_tensors = (c_void_p * inputsNum)(*c_tensors)
    out = [Tensor(dims, tensors[0].get_dtype()) for i in range(inputsNum)]
    for tensor in out:
        co_tensors.append(tensor.tensor_handle)
    co_tensors = (c_void_p * inputsNum)(*co_tensors)
    func = check_function("diopiMeshGrid")
    ret = func(tensors[0].context_handle, pointer(co_tensors), pointer(c_tensors), c_int64(inputsNum))
    check_returncode(ret)
    return out


def cast_dtype(input, out) -> Tensor:
    call = "diopiCastDtype"
    func = check_function(call)
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle)
    check_returncode(ret)
    return out


def multinomial(input, num_samples, replacement) -> Tensor:
    call = "diopiMultinomial"
    func = check_function(call)
    if len(input.size()) == 2:
        out = Tensor(size=(input.size()[0], num_samples), dtype=Dtype.int64)
    if len(input.size()) == 1:
        out = Tensor(size=(num_samples,), dtype=Dtype.int64)
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, c_int64(num_samples), c_bool(replacement))
    check_returncode(ret)
    return out
