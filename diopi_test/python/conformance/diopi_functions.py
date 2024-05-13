# Copyright (c) 2023, DeepLink.
# -*- coding: UTF-8 -*-
import math
import ctypes
import itertools
import numpy as np
import diopilib

from collections import namedtuple
from ctypes import c_double, byref
from .diopi_runtime import (
    Sizes,
    Scalar,
    Tensor,
    TensorP,
    Dtype,
    diopiReduction,
    diopiRoundMode,
    diopiError,
    compute_nhwc_stride,
    compute_nhwc_stride_2d,
    compute_nhwc_stride_3d,
    to_numpy_dtype,
    from_numpy_dtype,
    Generator,
)
from .diopi_runtime import raw_like, int_types, float_types, get_last_error
from .utils import logger
from conformance.global_settings import glob_vars
from typing import List, Optional, Union
from diopilib import build_generator_state


GLOBAL_STATE = {}


def get_capsule(src):
    PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = (
        ctypes.c_void_p,
        ctypes.c_char_p,
        PyCapsule_Destructor,
    )
    capsule = PyCapsule_New(src, None, PyCapsule_Destructor(0))
    return capsule


class DiopiException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FunctionNotImplementedError(DiopiException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FunctionNotDefinedError(DiopiException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def check_returncode(returncode, throw_exception=True):
    if 0 != returncode:
        if returncode == diopiError.diopi_no_implement:
            glob_vars.func_status[glob_vars.cur_test_func] = "skipped"
            raise FunctionNotImplementedError(
                glob_vars.cur_test_func + " not implement"
            )
        glob_vars.func_status[glob_vars.cur_test_func] = "failed"
        error_info = f"Returncode: {returncode}"
        error_detail = get_last_error()
        error_info += ", Details: " + error_detail
        if throw_exception:
            raise DiopiException(error_info)
        else:
            logger.info(error_info)


def check_function(fn_name):
    glob_vars.cur_test_func = fn_name
    if hasattr(diopilib, f"{fn_name}"):
        glob_vars.func_status[glob_vars.cur_test_func] = "passed"
        func = eval(f"diopilib.{fn_name}")
        return func
    else:
        glob_vars.func_status[glob_vars.cur_test_func] = "skipped"
        raise FunctionNotDefinedError(f"[diopilib] {fn_name} not defined.")


def broadcast_out_size(size1, size2):
    sizeO = size1 if len(size1) > len(size2) else size2
    length = len(size2) if len(size1) > len(size2) else len(size1)
    idx = -1
    while length > 0:
        assert (
            size1[idx] == size2[idx] or size1[idx] == 1 or size2[idx] == 1
        ), "size1 and size2 must be broadcastable"
        sizeO[idx] = size1[idx] if size2[idx] == 1 else size2[idx]
        idx -= 1
        length -= 1

    return sizeO


def reduce_op_process(input, dim=None, keepdim=False, dtype=None):
    sizeI = list(input.size().data)
    size = len(sizeI)
    sizeO = []
    dim_list = []
    dim = list(dim) if isinstance(dim, tuple) else dim

    if dim is None and keepdim:
        sizeO = [1 for i in range(0, size)]
    elif dim is not None:
        dim_list = dim[:] if isinstance(dim, list) else [dim]
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


def get_dtype(input: Union[Tensor, int, float]) -> Dtype:
    if isinstance(input, Tensor):
        return input.get_dtype()
    elif isinstance(input, int):
        return from_numpy_dtype(glob_vars.int_type)
    elif isinstance(input, float):
        return from_numpy_dtype(glob_vars.float_type)
    else:
        assert 0, "not supported type of input"


def infer_tensor_scalar_common_dtype(input: Tensor, other: Union[float, int]) -> Dtype:
    dtype1, dtype2 = get_dtype(input), get_dtype(other)
    if dtype1 in float_types:
        return dtype1
    if dtype1 not in float_types and dtype2 in float_types:
        return Dtype.float32
    if dtype1 == Dtype.bool and dtype2 != Dtype.bool:
        return Dtype.float32 if dtype2 in float_types else Dtype.int64
    return dtype1


def common_dtype(input, other) -> Dtype:
    dtype1, dtype2 = get_dtype(input), get_dtype(other)
    if isinstance(input, Tensor) and (
        isinstance(other, int) or isinstance(other, float)
    ):
        return infer_tensor_scalar_common_dtype(input, other)
    if isinstance(other, Tensor) and (
        isinstance(input, int) or isinstance(input, float)
    ):
        return infer_tensor_scalar_common_dtype(other, input)
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


def pow_dtype(input, other) -> Dtype:
    dtype1, dtype2 = get_dtype(input), get_dtype(other)
    if (
        dtype1 not in int_types
        and dtype2 not in int_types
        and dtype1 != Dtype.bool
        and dtype2 != Dtype.bool
    ):
        return dtype1
    else:
        if dtype1 in int_types and dtype2 == Dtype.uint8:
            return dtype1 if dtype1 != Dtype.int8 else Dtype.int16
        if dtype1 == Dtype.bool:
            return Dtype.float32 if dtype2 in float_types else Dtype.int64
        if dtype2 not in int_types:
            return dtype1 if dtype1 == Dtype.float16 else Dtype.float32
    return dtype1


def remainder_dtype(
    input: Union[Tensor, float, int], other: Union[Tensor, float, int]
) -> Dtype:
    dtype1, dtype2 = get_dtype(input), get_dtype(other)
    if dtype1 == Dtype.int8 and dtype2 == Dtype.uint8:
        return Dtype.int16
    if dtype1 == Dtype.uint8 and dtype2 == Dtype.int8:
        return Dtype.int16
    return common_dtype(input, other)


def promote_type(input: Tensor, promoted_dtype: Dtype) -> Dtype:
    dtype1 = input.get_dtype()
    need_promote_types = [
        Dtype.int8,
        Dtype.int16,
        Dtype.int32,
        Dtype.int64,
        Dtype.uint8,
        Dtype.uint16,
        Dtype.uint32,
        Dtype.uint64,
        Dtype.bool,
    ]
    return dtype1 if dtype1 not in need_promote_types else promoted_dtype


def fill_(input, value):
    func = check_function("diopiFill")
    value = Scalar(value)
    ret = func(input.context(), input, value)
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


def ones(default_context, size):
    func = check_function("diopiOnes")
    size = Sizes(list(size))
    out = Tensor(size=size, dtype=Dtype.float32)
    ret = func(default_context, out, size)
    check_returncode(ret)
    return out


def zeros(default_context, size):
    func = check_function("diopiZeros")
    size = Sizes(list(size))
    out = Tensor(size=size, dtype=Dtype.float32)
    ret = func(default_context, out, size)
    check_returncode(ret)
    return out


def zero_(input):
    func = check_function("diopiZeroInp")
    ret = func(input.context(), input)
    check_returncode(ret)
    return input


def unary_op(input, inplace, call, dtype=None) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context(), input)
    else:
        if dtype is not None:
            out = Tensor(input.size().data, dtype)
        else:
            out = raw_like(input)
        func = check_function(call)

        ret = func(input.context(), out, input)

    check_returncode(ret)
    return out


def binary_op(input, other, inplace, call) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context(), input, other)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context(), out, input, other)

    check_returncode(ret)
    return out


def binary_op_scalar(input, other, inplace, call, alpha=None, dtype=None) -> Tensor:
    args = "input.context(), "
    if dtype is None:
        dtype = common_dtype(input, other)

    if inplace:
        call = call + "Inp"
        out = input
    else:
        sizeI = input.size().data
        if not isinstance(other, Tensor):
            out = Tensor(sizeI, dtype)
        else:
            sizeO = other.size().data
            outsize = broadcast_out_size(list(sizeI), list(sizeO))
            out = Tensor(outsize, dtype)
        args = args + "out, "

    if not isinstance(other, Tensor):
        call = call + "Scalar"
        other = Scalar(other)
        args = args + "input, other"
    else:
        args = args + "input, other"
    if alpha is not None:
        alpha = Scalar(alpha)
        args = args + ", alpha"

    func = check_function(call)
    ret = eval(f"func({args})")

    check_returncode(ret)
    return out


def softmax(input, dim, dtype=None):
    if dim is None:
        dim = 0
    if input.numel() == 0:
        return input
    out = raw_like(input) if dtype is None else Tensor(input.size().data, dtype)

    func = check_function("diopiSoftmax")
    ret = func(input.context(), out, input, dim)
    check_returncode(ret)
    return out


def relu(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiRelu")


def abs(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiAbs")


def floor(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiFloor")


def sign(input) -> Tensor:
    return unary_op(input, False, "diopiSign")


def sigmoid(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiSigmoid")


def silu(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiSilu")


def silu_backward(input, grad_outputs, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    func = check_function("diopiSiluBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], input)
    check_returncode(ret)
    return {"input": grad_input}


def sqrt(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiSqrt", promote_type(input, Dtype.float32))


def rsqrt(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiRsqrt", promote_type(input, Dtype.float32))


def neg(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiNeg")


def sin(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiSin", promote_type(input, Dtype.float32))


def cos(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiCos", promote_type(input, Dtype.float32))


def tanh(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiTanh", promote_type(input, Dtype.float32))


def atan(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiAtan", promote_type(input, Dtype.float32))


def exp(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiExp", promote_type(input, Dtype.float32))


def log(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiLog", promote_type(input, Dtype.float32))


def log2(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiLog2", promote_type(input, Dtype.float32))


def log10(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiLog10", promote_type(input, Dtype.float32))


def erf(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiErf", promote_type(input, Dtype.float32))


def add(input, other, inplace=False, alpha=1) -> Tensor:
    return binary_op_scalar(input, other, inplace, "diopiAdd", alpha=alpha)


def sub(input, other, inplace=False, alpha=1) -> Tensor:
    return binary_op_scalar(input, other, inplace, "diopiSub", alpha=alpha)


def eq(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, "diopiEq", dtype=Dtype.bool)


def ne(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, "diopiNe", dtype=Dtype.bool)


def ge(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, "diopiGe", dtype=Dtype.bool)


def gt(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, "diopiGt", dtype=Dtype.bool)


def le(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, "diopiLe", dtype=Dtype.bool)


def lt(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, "diopiLt", dtype=Dtype.bool)


def equal(input, other) -> bool:
    call = "diopiEqual"
    func = check_function(call)

    out = ctypes.c_bool(True)
    PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = (
        ctypes.c_void_p,
        ctypes.c_char_p,
        PyCapsule_Destructor,
    )
    capsule = PyCapsule_New(
        ctypes.c_void_p(ctypes.addressof(out)), None, PyCapsule_Destructor(0)
    )
    ret = func(input.context(), capsule, input, other)
    check_returncode(ret)
    return out.value


def mul(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(
        input,
        other,
        inplace,
        "diopiMul",
        dtype=promote_type(input, Dtype.float32),
    )


def div(input, other, inplace=False, rounding_mode=None) -> Tensor:
    call = "diopiDiv"
    args = "input.context(), "
    sizeI = input.size().data
    rounding_mode = convert_round_mode(rounding_mode)
    if inplace:
        call = call + "Inp"
        out = input
    else:
        out_type = promote_type(input, Dtype.float32)
        if not isinstance(other, Tensor):
            out = Tensor(sizeI, out_type)
        else:
            sizeO = other.size().data
            outsize = broadcast_out_size(list(sizeI), list(sizeO))
            out = Tensor(outsize, out_type)
        args = args + "out, "

    if not isinstance(other, Tensor):
        call = call + "Scalar"
        other = Scalar(other)
        args = args + "input, other"
    else:
        args = args + "input, other"

    func = check_function(call)
    ret = eval(f"func({args}, rounding_mode)")

    check_returncode(ret)
    return out


def logical_and(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, "diopiLogicalAnd", dtype=Dtype.bool)


def logical_or(input, other, inplace=False) -> Tensor:
    return binary_op_scalar(input, other, inplace, "diopiLogicalOr", dtype=Dtype.bool)


def logical_not(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiLogicalNot", dtype=Dtype.bool)


def leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor:
    dtype = Dtype.int64 if isinstance(negative_slope, int) else Dtype.float64
    negative_slope = Scalar(negative_slope, dtype)
    if inplace:
        out = input
        func = check_function("diopiLeakyReluInp")
        ret = func(input.context(), input, negative_slope)
    else:
        out = raw_like(input)
        func = check_function("diopiLeakyRelu")
        ret = func(input.context(), out, input, negative_slope)

    check_returncode(ret)
    return out


def bmm(input, mat2) -> Tensor:
    size1 = list(input.size().data)
    assert len(size1) == 3, "input must be 3d tensor"
    size2 = mat2.size().data
    assert len(size2) == 3, "mat2 must be 3d tensor"
    assert size1[0] == size2[0], "invalid args"
    assert size1[2] == size2[1], "invalid args"

    size_out = size1
    size_out[2] = size2[2]
    out = Tensor(size_out, input.get_dtype())

    func = check_function("diopiBmm")
    ret = func(input.context(), out, input, mat2)
    check_returncode(ret)
    return out


def baddbmm(input, batch1, batch2, beta, alpha, inplace=False) -> Tensor:
    size1 = list(input.size().data)
    size2 = list(batch1.size().data)
    assert len(size2) == 3, "batch1 must be 3d tensor"
    size3 = list(batch2.size().data)
    assert len(size3) == 3, "batch2 must be 3d tensor"
    input_len = input.size().len
    out_shape = size1
    if input_len == 3:
        assert (
            size2[2] == size3[1]
            and size1[0] == size2[0]
            and size1[0] == size3[0]
            or size1[0] == 1
        ), "invalid args"
        assert size1[2] == size3[2] or size1[2] == 1 or size3[2] == 1, "invalid args"
    elif input_len == 2:
        assert (size1[1] == size3[2] or size1[1] == 1) and (
            size1[0] == size2[1] or size1[0] == 1
        ), "invalid args"
        out_shape = (size2[0], size1[0], size1[1])
    elif input_len == 1:
        assert size1[0] == size3[2] or size1[0] == 1, "invalid args"
        out_shape = (size2[0], size2[1], size1[0])
    if out_shape[0] != size2[0]:
        out_shape = (size2[0], size1[1], size1[2])
    if out_shape[1] != size2[1]:
        out_shape = (size1[0], size2[1], size1[2])
    if out_shape[2] != size3[2]:
        out_shape = (size1[0], size1[1], size3[2])
    if inplace:
        func = check_function("diopiBaddbmmInp")
        ret = func(input.context(), input, batch1, batch2, beta, alpha)
        check_returncode(ret)
        return input
    else:
        out = Tensor(size=out_shape, dtype=input.get_dtype())
        func = check_function("diopiBaddbmm")
        ret = func(input.context(), out, input, batch1, batch2, beta, alpha)
        check_returncode(ret)
        return out


def addcmul(input, tensor1, tensor2, value=1, inplace=False) -> Tensor:
    size1 = tensor1.size().data
    size2 = tensor2.size().data
    sizeI = input.size().data
    sizeO = broadcast_out_size(size1, size2)
    sizeO = broadcast_out_size(sizeI, sizeO)
    value = Scalar(value)

    if inplace:
        out = input
        assert list(sizeO) == sizeI, "can not be inplaced"
        func = check_function("diopiAddcmulInp")
        ret = func(input.context(), input, tensor1, tensor2, value)
    else:
        out = Tensor(sizeO, input.get_dtype())
        func = check_function("diopiAddcmul")
        ret = func(input.context(), out, input, tensor1, tensor2, value)
    check_returncode(ret)
    return out


def matmul(input, other) -> Tensor:
    out = raw_like(input)
    sizeI = input.size().data
    sizeO = other.size().data
    # vector x vector
    if len(sizeI) == 1 and len(sizeO) == 1:
        out = Tensor((), input.get_dtype())
    # matrix x vector
    elif len(sizeI) > 1 and len(sizeO) == 1:
        out = Tensor(sizeI[:-1], input.get_dtype())
    # vector x matrix
    elif len(sizeI) == 1 and len(sizeO) > 1:
        out = Tensor(sizeO[:-2] + [sizeO[-1]], input.get_dtype())
    # (batched) matrix x (batched)matrix
    else:
        assert (
            len(sizeI) > 1 and len(sizeO) > 1
        ), "Inputs must be at least 2-dimensional for matrix multiplication"
        assert (
            sizeI[-1] == sizeO[-2]
        ), "Last dimension of the first input must match second to last dimension of the second input"
        max_dim = len(sizeI) if len(sizeI) > len(sizeO) else len(sizeO)
        sizeI = [1] * (max_dim - len(sizeI)) + sizeI
        sizeO = [1] * (max_dim - len(sizeO)) + sizeO
        out_size = [
            sizeI[i] if sizeI[i] > sizeO[i] else sizeO[i] for i in range(max_dim - 2)
        ] + [sizeI[-2], sizeO[-1]]
        out = Tensor(out_size, input.get_dtype())
    func = check_function("diopiMatmul")
    ret = func(input.context(), out, input, other)
    check_returncode(ret)
    return out


def clamp(input, min=None, max=None, inplace=False) -> Tensor:
    assert (
        min is not None or max is not None
    ), "At least one of 'min' or 'max' must not be None"
    call = "diopiClamp"
    args = "input.context(), "
    if inplace:
        call = call + "Inp"
    else:
        args = args + "out, "

    convert_float = False
    if max is not None and min is not None:
        if isinstance(min, Tensor):
            assert isinstance(max, Tensor), "min and max must have same type"
            args += "input, min, max"
        else:
            assert isinstance(min, (int, float)), "min must be Tensor or Value"
            assert isinstance(max, (int, float)), "min and max must have same type"
            if isinstance(min, float) and input.get_dtype() in int_types:
                convert_float = True
            call = call + "Scalar"
            min = Scalar(min)
            max = Scalar(max)
            args = args + "input, min, max"
    elif max is not None and min is None:
        if isinstance(max, Tensor):
            args += "input, None, max"
        else:
            assert isinstance(max, (int, float)), "max must be Tensor or Value"
            if isinstance(max, float) and input.get_dtype() in int_types:
                convert_float = True
            call = call + "Scalar"
            max = Scalar(max)
            args = args + "input, None, max"
    elif min is not None and max is None:
        if isinstance(min, Tensor):
            args += "input, min, None"
        else:
            assert isinstance(min, (int, float)), "min must be Tensor or Value"
            if isinstance(min, float) and input.get_dtype() in int_types:
                convert_float = True
            call = call + "Scalar"
            min = Scalar(min)
            args = args + "input, min, None"

    if inplace:
        if convert_float:
            input_np = input.numpy()
            input = Tensor.from_numpy(input_np.astype(to_numpy_dtype(Dtype.float32)))
        out = input
    else:
        if convert_float:
            out = Tensor(input.size(), Dtype.float32)
        else:
            out = raw_like(input)

    func = check_function(call)
    ret = eval(f"func({args})")
    check_returncode(ret)
    return out


def clamp_min(input, min, inplace=False) -> Tensor:
    call = "diopiClampMin"
    args = "input.context(), "
    if inplace:
        if isinstance(min, float) and input.get_dtype() in int_types:
            input_np = input.numpy()
            input = Tensor.from_numpy(input_np.astype(to_numpy_dtype(Dtype.float32)))
        out = input
        call = call + "Inp"
    else:
        if isinstance(min, float) and input.get_dtype() in int_types:
            out_dtype = Dtype.float32
        else:
            out_dtype = input.get_dtype()
        out = Tensor(input.size(), out_dtype)
        args = args + "out, "

    if isinstance(min, Tensor):
        args = args + "input, min"
    else:
        call = call + "Scalar"
        min = Scalar(min)
        args = args + "input, min"

    func = check_function(call)
    ret = eval(f"func({args})")
    check_returncode(ret)
    return out


def clamp_max(input, max, inplace=False) -> Tensor:
    call = "diopiClampMax"
    args = "input.context(), "
    if inplace:
        if isinstance(max, float) and input.get_dtype() in int_types:
            input_np = input.numpy()
            input = Tensor.from_numpy(input_np.astype(to_numpy_dtype(Dtype.float32)))
        out = input
        call = call + "Inp"
    else:
        if isinstance(max, float) and input.get_dtype() in int_types:
            out_dtype = Dtype.float32
        else:
            out_dtype = input.get_dtype()
        out = Tensor(input.size(), out_dtype)
        args = args + "out, "

    if isinstance(max, Tensor):
        args = args + "input, max"
    else:
        call = call + "Scalar"
        max = Scalar(max)
        args = args + "input, max"

    func = check_function(call)
    ret = eval(f"func({args})")
    check_returncode(ret)
    return out


def mean(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    assert (
        isinstance(dim, (int, list)) or dim is None
    ), "dim should be int or list or None"

    dim, out = reduce_op_process(input, dim, keepdim, dtype)
    func = check_function("diopiMean")
    dim1 = Sizes(list(dim))
    ret = func(input.context(), out, input, dim1)
    check_returncode(ret)
    return out


def std(input, unbiased=True, dim=None, keepdim=False) -> Tensor:
    assert (
        isinstance(dim, (int, list)) or dim is None
    ), "dim should be int or list or None"

    dim, out = reduce_op_process(input, dim, keepdim)
    dim1 = Sizes(list(dim))
    func = check_function("diopiStd")
    ret = func(input.context(), out, input, dim1, unbiased)
    check_returncode(ret)
    return out


def min(input, dim=None, keepdim=False) -> Tensor:
    if dim is None:
        out = Tensor([], input.get_dtype())
        func = check_function("diopiMinAll")
        ret = func(input.context(), out, input)
        check_returncode(ret)
        return out

    assert isinstance(dim, int), "dim should be int"

    sizeI = input.size().data
    if len(sizeI) > 0:
        if keepdim:
            sizeI[dim] = 1
        else:
            del sizeI[dim]
    out = Tensor(sizeI, input.get_dtype())
    indices = Tensor(out.size().data, from_numpy_dtype(glob_vars.int_type))
    func = check_function("diopiMin")
    ret = func(input.context(), out, indices, input, dim)
    check_returncode(ret)
    Res = namedtuple("Res", ["values", "indices"])
    output = Res(out, indices)
    return output


def convert_reduction(name):
    if name == "none":
        return diopiReduction.ReductionNone
    if name == "mean":
        return diopiReduction.ReductionMean
    if name == "sum":
        return diopiReduction.ReductionSum
    return 3


def convert_round_mode(name):
    if name is None:
        return diopiRoundMode.RoundModeNone
    if name == "trunc":
        return diopiRoundMode.RoundModeTrunc
    if name == "floor":
        return diopiRoundMode.RoundModeFloor
    return diopiRoundMode.RoundModeEND


def binary_cross_entropy(input, target, weight=None, reduction="mean"):
    assert (
        input.size().data == target.size().data
    ), "target shape must be the same as input shape"
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"

    if weight is not None:
        assert isinstance(weight, Tensor), "weigth must be a Tensor"
        weight = weight
    else:
        weight = None

    if reduction == "none":
        out = raw_like(input)
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiBCELoss")
    ret = func(input.context(), out, input, target, weight, reduction_mode)
    check_returncode(ret)
    return out


def binary_cross_entropy_with_logits(
    input, target, weight=None, reduction="mean", pos_weight=None
):
    assert (
        input.size().data == target.size().data
    ), "target shape must be the same as input shape"
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"
    if pos_weight is not None:
        assert isinstance(pos_weight, Tensor), "pos_weigth must be a Tensor"
        pos_weight = pos_weight
    else:
        # represent pos_weight = None by pass a nullptr
        pos_weight = None

    if weight is not None:
        assert isinstance(weight, Tensor), "weigth must be a Tensor"

    if reduction == "none":
        out = raw_like(input)
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiBCEWithLogits")
    ret = func(input.context(), out, input, target, weight, pos_weight, reduction_mode)
    check_returncode(ret)
    return out


def cross_entropy(
    input,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
):
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"

    if weight is not None:
        assert isinstance(weight, Tensor), "weigth must be a Tensor"
        weight = weight
    else:
        weight = None

    sizeI = input.size().data
    sizeO = [sizeI[0]] + sizeI[2:]
    if reduction == "none":
        out = Tensor(sizeO, input.get_dtype())
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiCrossEntropyLoss")
    ret = func(
        input.context(),
        out,
        input,
        target,
        weight,
        reduction_mode,
        ignore_index,
        label_smoothing,
    )
    check_returncode(ret)
    return out


def mse_loss(input, target, reduction="mean"):
    assert list(input.shape().data) == list(
        target.shape().data
    ), "target shape must be the same as input shape"
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"

    if reduction == "none":
        out = raw_like(input)
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiMSELoss")
    ret = func(input.context(), out, input, target, reduction_mode)
    check_returncode(ret)
    return out


def conv2d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), "bias must be a Tensor"

    sizeI = input.size().data
    sizeW = weight.size().data
    assert len(sizeI) == 4 and len(sizeW) == 4, "input and weight must be 4d tensors"

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

    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    dilation = Sizes(list(dilation))

    nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)
    func = check_function("diopiConvolution2d")
    ret = func(
        input.context(),
        out,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )
    check_returncode(ret)
    return out


def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
) -> Tensor:
    sizeI = input.size().data
    assert len(sizeI) == 4 or len(sizeI) == 3, "input must be 3d or 4d tensors"

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    for i in range(-2, 0):
        if ceil_mode:
            sizeO.append(
                math.ceil((sizeI[i] - kernel_size[i] + 2 * padding[i]) / stride[i]) + 1
            )
        else:
            sizeO.append(
                math.floor((sizeI[i] - kernel_size[i] + 2 * padding[i]) / stride[i]) + 1
            )

    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    kernel_size = Sizes(list(kernel_size))
    nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)

    func = check_function("diopiAvgPool2d")
    if divisor_override:
        ret = func(
            input.context(),
            out,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
    else:
        ret = func(
            input.context(),
            out,
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        )
    check_returncode(ret)
    return out


def max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
) -> Tensor:
    sizeI = input.size().data
    assert len(sizeI) == 4 or len(sizeI) == 3, "input must be 3d or 4d tensors"

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
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

    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    kernel_size = Sizes(list(kernel_size))
    dilation = Sizes(list(dilation))
    nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)

    if not return_indices:
        func = check_function("diopiMaxPool2d")
        ret = func(
            input.context(),
            out,
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        check_returncode(ret)
        return out
    else:
        func = check_function("diopiMaxPool2dWithIndices")
        nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
        indices = Tensor(
            sizeO, from_numpy_dtype(glob_vars.int_type), stride=nhwc_stride
        )
        ret = func(
            input.context(),
            out,
            indices,
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        check_returncode(ret)
        return out, indices


def adaptive_avg_pool2d(input, output_size):
    sizeI = input.size().data
    assert len(sizeI) == 4 or len(sizeI) == 3, "input must be 3d or 4d tensors"

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
    output_size = Sizes(list([sizeO[-2], sizeO[-1]]))

    func = check_function("diopiAdaptiveAvgPool2d")
    ret = func(input.context(), out, input, output_size)
    check_returncode(ret)
    return out


def adaptive_max_pool2d(input, output_size, return_indices=False):
    sizeI = input.size().data
    assert len(sizeI) == 4 or len(sizeI) == 3, "input must be 3d or 4d tensors"

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
    output_size = Sizes(list([sizeO[-2], sizeO[-1]]))

    if return_indices:
        func = check_function("diopiAdaptiveMaxPool2dWithIndices")
        nhwc_stride = compute_nhwc_stride_2d(sizeO) if glob_vars.nhwc else None
        indices = Tensor(
            sizeO, from_numpy_dtype(glob_vars.int_type), stride=nhwc_stride
        )
        ret = func(input.context(), out, indices, input, output_size)
        check_returncode(ret)
        return out, indices
    else:
        func = check_function("diopiAdaptiveMaxPool2d")
        ret = func(input.context(), out, input, output_size)
    check_returncode(ret)
    return out


def dropout_impl(
    input: Tensor,
    size_mask: list,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    generator: Generator = None,
):
    mask = Tensor(size_mask, Dtype.uint8)
    if inplace:
        out = input
        func = check_function("diopiDropoutInp")
        ret = func(input.context(), out, mask, p, training, generator)
    else:
        out = raw_like(input)
        func = check_function("diopiDropout")
        ret = func(input.context(), out, mask, input, p, training, generator)

    check_returncode(ret)
    return out, mask


def dropout(input, p=0.5, training=True, inplace=False, generator=None):
    return dropout_impl(input, input.size().data, p, training, inplace, generator)


def dropout2d(input, p=0.5, training=True, inplace=False, generator=None):
    sizeI = input.size().data
    for i in range(2, len(sizeI)):
        sizeI[i] = 1
    return dropout_impl(input, sizeI, p, training, inplace, generator)


def index_select(input, dim, index) -> Tensor:
    sizeI = input.size().data
    sizeI[dim] = index.numel()
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiIndexSelect")
    ret = func(input.context(), out, input, dim, index)
    check_returncode(ret)
    return out


def select(input, dim, index) -> Tensor:
    sizeI = input.size().data
    del sizeI[dim]
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiSelect")
    ret = func(input.context(), out, input, dim, index)
    check_returncode(ret)
    return out


def masked_scatter(input, mask, source) -> Tensor:
    assert mask.get_dtype() == Dtype.bool, "mask must be bool tensor"
    out = raw_like(input)

    func = check_function("diopiMaskedScatter")
    ret = func(input.context(), out, input, mask, source)
    check_returncode(ret)
    return out


def nonzero(input):
    # note: pytorch(1.12) has argument 'as_tuple' to return multiple 1d tensor
    out = Tensor()
    func = check_function("diopiNonzero")
    out_ptr = TensorP(out)
    ret = func(input.context(), out_ptr, input)
    check_returncode(ret)
    return out_ptr.data()


def linear(input, weight, bias=None) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), "bias must be a Tensor"
        bias = bias
    else:
        bias = None

    sizeI = input.size().data
    sizeW = weight.size().data
    sizeI[-1] = sizeW[0]
    out = Tensor(sizeI, input.get_dtype())
    func = check_function("diopiLinear")
    ret = func(input.context(), out, input, weight, bias)
    check_returncode(ret)
    return out


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    sizeW = weight.size().data
    assert len(sizeW) == 2, "Shape of Weight should be 2"
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < sizeW[0], "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -sizeW[0], "Padding_idx must be within num_embeddings"
            padding_idx = sizeW[0] + padding_idx
    else:
        padding_idx = -1

    if max_norm is not None:
        func2 = check_function("diopiEmbeddingRenorm_")
        ret2 = func2(input.context(), weight, input, max_norm, norm_type)
        check_returncode(ret2)

    sizeI = input.size().data
    sizeI.append(sizeW[1])
    # shape: [input_shape, ..., embedding_dim]
    out = Tensor(sizeI, weight.get_dtype())
    # note: scale_grad_by_freq and sparse are useless during forward phase
    func = check_function("diopiEmbedding")
    ret = func(
        input.context(),
        out,
        weight,
        input,
        padding_idx,
        scale_grad_by_freq,
        sparse,
    )
    check_returncode(ret)
    return out


def tril(input, diagonal=0) -> Tensor:
    out = raw_like(input)
    func = check_function("diopiTril")
    ret = func(input.context(), out, input, diagonal)
    check_returncode(ret)
    return out


def cat(tensors, dim=0) -> Tensor:
    assert isinstance(tensors, (list, tuple)), "tensors must be a list or tuple"
    insNum = len(tensors)
    sum = 0
    c_tensors = []
    for tensor in tensors:
        sizeI = tensor.size().data
        sum += sizeI[dim]
        c_tensors.append(TensorP(tensor))

    sizeI[dim] = sum
    out = Tensor(sizeI, tensors[0].get_dtype())
    func = check_function("diopiCat")
    ret = func(tensors[0].context(), out, list(c_tensors), insNum, dim)
    check_returncode(ret)
    return out


def stack(tensors, dim=0) -> Tensor:
    assert isinstance(tensors, (list, tuple)), "tensors must be a list or tuple"
    insNum = len(tensors)
    outNum = insNum + 1
    sizeI = tensors[0].size().data
    size_dim = dim
    if dim < 0:
        size_dim = outNum + dim
    sizeI.insert(size_dim, insNum)

    c_tensors = [TensorP(t) for t in tensors]

    out = Tensor(sizeI, tensors[0].get_dtype())
    func = check_function("diopiStack")
    ret = func(tensors[0].context(), out, list(c_tensors), insNum, dim)
    check_returncode(ret)
    return out


def sort(input, dim=-1, descending=False, stable=None):
    vals = raw_like(input)
    sizeI = input.size().data
    indices = Tensor(sizeI, from_numpy_dtype(glob_vars.int_type))
    func = check_function("diopiSort")
    ret = (
        func(input.context(), vals, indices, input, dim, descending)
        if stable is None
        else func(input.context(), vals, indices, input, dim, descending, stable)
    )
    check_returncode(ret)
    # if not stable, need to reconstruct indices and use "input[indices]" to check

    if len(sizeI) > 0 and not stable:
        # reconstruct the indices
        lst = []
        for dim_size in input.shape().data:
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
        return vals
    return vals, indices


def topk(input, k, dim=-1, largest=True, sorted=True):
    sizeI = input.size().data
    if len(sizeI) > 0:
        sizeI[dim] = k
    values = Tensor(sizeI, input.get_dtype())
    indices = Tensor(sizeI, from_numpy_dtype(glob_vars.int_type))

    func = check_function("diopiTopk")
    ret = func(input.context(), values, indices, input, k, dim, largest, sorted)
    check_returncode(ret)
    return values, indices


def transpose(input, dim0, dim1) -> Tensor:
    sizeI = input.size().data
    if len(sizeI) > 0:
        sizeI[dim0], sizeI[dim1] = sizeI[dim1], sizeI[dim0]
    out = Tensor(sizeI, input.get_dtype())
    func = check_function("diopiTranspose")
    ret = func(input.context(), out, input, dim0, dim1)
    check_returncode(ret)
    return out


def one_hot(input, num_classes=-1):
    assert num_classes == -1 or num_classes > 0, "num_classes must be -1 or >0"

    sizeI = input.size().data
    if num_classes == -1:
        sizeI += (np.max(input.numpy()) + 1,)
        out = Tensor(sizeI, from_numpy_dtype(glob_vars.int_type))
    else:
        sizeI += (num_classes,)
        out = Tensor(sizeI, from_numpy_dtype(glob_vars.int_type))

    func = check_function("diopiOneHot")
    ret = func(input.context(), out, input, num_classes)
    check_returncode(ret)
    return out


def split(tensor, split_size_or_sections, dim=0):
    assert isinstance(
        split_size_or_sections, (int, list, tuple)
    ), "split_size_or_sections must be int or list"
    sizeI = tensor.size().data
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
        splitSizes += (sizeI[dim],)
        out = Tensor(sizeI, tensor.get_dtype())
        outs.append(out)

    c_outs = []
    for i in range(idx):
        c_outs.append(TensorP(outs[i]))

    splitSizes = Sizes(list(splitSizes))
    assert sum == 0, "split_size_or_sections should be compatible with tensor shape"
    func = check_function("diopiSplitWithSizes")
    ret = func(tensor.context(), list(c_outs), idx, tensor, splitSizes, dim)
    check_returncode(ret)
    return outs


def pow(input=None, self=None, exponent=None, inplace=False) -> Tensor:
    if input is not None:
        out_dtype = pow_dtype(input, exponent)
    else:
        out_dtype = pow_dtype(self, exponent)
        if Scalar(self).stype == Dtype.int64:
            exponent_dtype = get_dtype(exponent)
            if exponent_dtype in int_types or exponent_dtype in float_types:
                out_dtype = exponent_dtype
    if input is None and self is not None:
        assert isinstance(
            exponent, Tensor
        ), "exponent must be tensor when input is scalar"
        func = check_function("diopiPowScalar")
        out = Tensor(exponent.size().data, out_dtype)
        self = Scalar(self)
        ret = func(exponent.context(), out, self, exponent)
    elif not isinstance(exponent, Tensor):
        assert isinstance(input, Tensor), "input must be tensor when exponent is scalar"
        temp_exponent = Scalar(exponent)
        if inplace:
            out_dtype = pow_dtype(input, exponent)
            input_np = input.numpy()
            input_np = input_np.astype(to_numpy_dtype(out_dtype))
            input = Tensor.from_numpy(input_np)
            func = check_function("diopiPowInp")
            ret = func(input.context(), input, temp_exponent)
        else:
            func = check_function("diopiPow")
            out = Tensor(input.size().data, out_dtype)
            ret = func(input.context(), out, input, temp_exponent)
    elif inplace:
        func = check_function("diopiPowInpTensor")
        ret = func(input.context(), input, exponent)
    else:
        sizeI = input.size().data
        sizeE = exponent.size().data
        sizeO = broadcast_out_size(sizeI, sizeE)
        out = Tensor(sizeO, out_dtype)
        func = check_function("diopiPowTensor")
        ret = func(input.context(), out, input, exponent)
    if inplace:
        out = input

    check_returncode(ret)
    return out


def where(condition, input, other) -> Tensor:
    # todo: add scalar version for pytorch 1.12
    assert condition.get_dtype() in (
        Dtype.bool,
        Dtype.uint8,
    ), "condition must be a bool tensor"
    sizeX = input.size().data
    sizeY = other.size().data
    sizeC = condition.size().data
    sizeO = broadcast_out_size(sizeX, sizeY)
    sizeO = broadcast_out_size(sizeC, sizeO)
    # assert (input.get_dtype() == other.get_dtype()),\
    #     " input and other shoule be the same type "
    out = Tensor(sizeO, input.get_dtype())

    func = check_function("diopiWhere")
    ret = func(input.context(), out, condition, input, other)
    check_returncode(ret)
    return out


def clip_grad_norm_(tensors, max_norm, norm_type=2.0, error_if_nonfinite=False):
    assert isinstance(max_norm, (int, float)), "max_norm must be a int or float"
    assert isinstance(norm_type, (int, float)), "norm_type must be a int or float"

    if isinstance(tensors, Tensor):
        ctx = tensors.context()
        grads = list([TensorP(grad) for grad in tensors])
        num_grads = 1
    else:
        ctx = tensors[0].context()
        num_grads = len(tensors)
        grads = list([TensorP(grad) for grad in tensors])

    func = check_function("diopiClipGradNorm")
    out = c_double(0.0)
    ret = func(
        ctx,
        get_capsule(byref(out)),
        grads,
        num_grads,
        max_norm,
        norm_type,
        error_if_nonfinite,
    )
    check_returncode(ret)

    return out.value


def batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training=False,
    momentum=0.1,
    eps=1e-05,
) -> Tensor:
    dim = input.size().len
    dim = [0] + [i for i in range(2, dim)]
    dtype = Dtype.float32 if input.get_dtype() == Dtype.float16 else None
    _, save_mean = reduce_op_process(input, dim, dtype=dtype)
    save_invstd = raw_like(save_mean)

    if not training:
        assert (
            running_mean is not None and running_var is not None
        ), "if not trainging, running_mean and running_var must be defined"

    out = raw_like(input)
    func = check_function("diopiBatchNorm")
    ret = func(
        input.context(),
        out,
        save_mean,
        save_invstd,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    )

    check_returncode(ret)
    GLOBAL_STATE["batch_norm_save_mean"] = save_mean
    GLOBAL_STATE["batch_norm_save_invstd"] = save_invstd
    return out


def batch_norm_stats(input, eps):
    func = check_function("diopiBatchNormStats")
    # cuda accumulate dtype mapping
    dtype = Dtype.float32 if input.get_dtype() == Dtype.float16 else input.get_dtype()
    mean = Tensor(Sizes(list([input.size().data[1]])), dtype)
    invstd = Tensor(Sizes(list([input.size().data[1]])), dtype)
    ret = func(input.context(), mean, invstd, input, eps)
    check_returncode(ret)
    out = (mean, invstd)
    return out


def batch_norm_gather_stats_with_counts(
    input,
    mean_all,
    invstd_all,
    running_mean,
    running_var,
    momentum,
    eps,
    count_all,
):
    func = check_function("diopiBatchNormGatherStatsWithCounts")
    mean = Tensor(Sizes(list([input.size().data[1]])), input.get_dtype())
    invstd = Tensor(Sizes(list([input.size().data[1]])), input.get_dtype())
    ret = func(
        input.context(),
        mean,
        invstd,
        input,
        mean_all,
        invstd_all,
        running_mean,
        running_var,
        momentum,
        eps,
        count_all,
    )
    check_returncode(ret)
    out = (mean, invstd)
    return out


def batch_norm_backward_reduce(
    grad_output, input, mean, invstd, weight, input_g, weight_g, bias_g
):
    func = check_function("diopiBatchNormBackwardReduce")
    if input_g:
        sum_dy = Tensor(Sizes(list([input.size().data[1]])), input.get_dtype())
        sum_dy_xmu = Tensor(Sizes(list([input.size().data[1]])), input.get_dtype())
    else:
        sum_dy = None
        sum_dy_xmu = None
    if weight_g:
        grad_weight = Tensor(Sizes(list([input.size().data[1]])), input.get_dtype())
    else:
        grad_weight = None
    if weight_g:
        grad_bias = Tensor(Sizes(list([input.size().data[1]])), input.get_dtype())
    else:
        grad_bias = None
    ret = func(
        input.context(),
        sum_dy,
        sum_dy_xmu,
        grad_weight,
        grad_bias,
        grad_output,
        input,
        mean,
        invstd,
        weight,
        input_g,
        weight_g,
        bias_g,
    )
    check_returncode(ret)
    out = (sum_dy, sum_dy_xmu, grad_weight, grad_bias)
    return out


def batch_norm_backward_elemt(
    grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count
):
    func = check_function("diopiBatchNormBackwardElemt")
    grad_input = Tensor(Sizes(list(grad_out.size().data)), input.get_dtype())
    ret = func(
        input.context(),
        grad_input,
        grad_out,
        input,
        mean,
        invstd,
        weight,
        sum_dy,
        sum_dy_xmu,
        count,
    )
    check_returncode(ret)
    out = grad_input
    return out


def batch_norm_elemt(input, weight, bias, mean, invstd, eps):
    func = check_function("diopiBatchNormElemt")
    out = Tensor(Sizes(list(input.size().data)), input.get_dtype())
    ret = func(input.context(), out, input, weight, bias, mean, invstd, eps)
    check_returncode(ret)
    return out


def log_softmax(input, dim=None, dtype=None):
    if dim is None:
        dim = 0
    if input.numel() == 0:
        return input
    out = raw_like(input) if dtype is None else Tensor(input.size().data, dtype)

    func = check_function("diopiLogSoftmax")
    ret = func(input.context(), out, input, dim)
    check_returncode(ret)
    return out


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False) -> Tensor:
    call = "diopiHardtanh"
    min_val = Scalar(min_val)
    max_val = Scalar(max_val)
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context(), input, min_val, max_val)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context(), out, input, min_val, max_val)

    check_returncode(ret)
    return out


def hardswish(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiHardswish")


def threshold(input, threshold, value, inplace=False) -> Tensor:
    call = "diopiThreshold"
    threshold = Scalar(threshold)
    value = Scalar(value)
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context(), input, threshold, value)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context(), out, input, threshold, value)

    check_returncode(ret)
    return out


def gelu(input, approximate="none") -> Tensor:
    assert isinstance(approximate, str), "approximate must be a string."
    out = raw_like(input)
    func = check_function("diopiGelu")

    ret = func(input.context(), out, input, approximate.encode("UTF-8"))

    check_returncode(ret)
    return out


def addcdiv(input, tensor1, tensor2, value=1, inplace=False) -> Tensor:
    size1 = tensor1.size().data
    size2 = tensor2.size().data
    sizeI = input.size().data
    sizeO = broadcast_out_size(size1, size2)
    sizeO = broadcast_out_size(sizeI, sizeO)
    value = Scalar(value)

    if inplace:
        out = input
        assert list(sizeO) == sizeI, "can not be inplaced"
        func = check_function("diopiAddcdivInp")
        ret = func(input.context(), input, tensor1, tensor2, value)
    else:
        out = Tensor(sizeO, input.get_dtype())
        func = check_function("diopiAddcdiv")
        ret = func(input.context(), out, input, tensor1, tensor2, value)
    check_returncode(ret)
    return out


def addmm(input, mat1, mat2, beta=1, alpha=1) -> Tensor:
    size1 = mat1.size().data
    size2 = mat2.size().data
    size1[-1] = size2[-1]
    sizeI = input.size().data
    sizeO = broadcast_out_size(sizeI, size1)
    out = Tensor(sizeO, input.get_dtype())
    alpha = Scalar(alpha)
    beta = Scalar(beta)

    func = check_function("diopiAddmm")
    ret = func(input.context(), out, input, mat1, mat2, beta, alpha)
    check_returncode(ret)
    return out


def sum(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    assert (
        isinstance(dim, (int, list, tuple)) or dim is None
    ), "dim should be int or list"
    func = check_function("diopiSum")
    out_dtype = dtype if dtype is not None else promote_type(input, Dtype.int64)
    dim, out = reduce_op_process(input, dim, keepdim, out_dtype)
    dim1 = Sizes(list(dim))
    ret = func(input.context(), out, input, dim1)
    check_returncode(ret)
    return out


def max(input, dim=None, keepdim=False):
    if dim is None:
        out = Tensor([], input.get_dtype())
        func = check_function("diopiMaxAll")
        ret = func(input.context(), out, input)
        check_returncode(ret)
        return out

    assert isinstance(dim, int), "dim should be int"
    sizeI = input.size().data
    if len(sizeI) > 0:
        if keepdim:
            sizeI[dim] = 1
        else:
            del sizeI[dim]
    out = Tensor(sizeI, input.get_dtype())
    indices = Tensor(out.size().data, from_numpy_dtype(glob_vars.int_type))

    func = check_function("diopiMax")
    ret = func(input.context(), out, indices, input, dim)
    check_returncode(ret)
    Res = namedtuple("Res", ["values", "indices"])
    output = Res(out, indices)
    return output


def any(input, dim=None, keepdim=False) -> Tensor:
    if dim is None:
        out = Tensor([], Dtype.bool)
    else:
        assert isinstance(dim, int), "dim should be int"
        _, out = reduce_op_process(input, dim, keepdim, dtype=Dtype.bool)
    func = check_function("diopiAny")

    ret = (
        func(input.context(), out, input)
        if dim is None
        else func(input.context(), out, input, dim)
    )
    check_returncode(ret)
    return out


def all(input, dim=None, keepdim=False) -> Tensor:
    if dim is None:
        out = Tensor([], Dtype.bool)
    else:
        assert isinstance(dim, int), "dim should be int"
        _, out = reduce_op_process(input, dim, keepdim, dtype=Dtype.bool)

    func = check_function("diopiAll")
    ret = (
        func(input.context(), out, input)
        if dim is None
        else func(input.context(), out, input, dim)
    )
    check_returncode(ret)
    return out


# todo: impl for diopiNLLLossV2
def nll_loss(input, target, weight=None, ignore_index=-100, reduction="mean"):
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"

    if weight is not None:
        assert isinstance(weight, Tensor), "weigth must be a Tensor"

    if reduction == "none":
        out = Tensor(target.size().data, input.get_dtype())
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiNLLLoss")
    ret = func(
        input.context(),
        out,
        input,
        target,
        weight,
        reduction_mode,
        ignore_index,
    )
    check_returncode(ret)
    return out


def sigmoid_focal_loss(
    inputs, targets, alpha=0.25, gamma=2, reduction="none"
) -> Tensor:
    assert (
        inputs.size().data == targets.size().data
    ), "target shape must be the same as input shape"
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"

    if reduction == "none":
        out = raw_like(inputs)
    else:
        out = Tensor((), inputs.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiSigmoidFocalLoss")
    ret = func(inputs.context(), out, inputs, targets, alpha, gamma, reduction_mode)
    check_returncode(ret)
    return out


def nms(boxes, scores, iou_threshold) -> Tensor:
    size_boxes = boxes.size().data
    assert (
        len(size_boxes) == 2 and size_boxes[1] == 4
    ), "boxes must be a tensor of shape (N,4)"

    size_scores = scores.size().data
    assert (
        len(size_scores) == 1 and size_scores[0] == size_boxes[0]
    ), "boxes must be a tensor of shape (N)"

    out = Tensor()
    func = check_function("diopiNms")
    out_ptr = TensorP(out)
    ret = func(boxes.context(), out_ptr, boxes, scores, iou_threshold)
    out = out_ptr.data()
    check_returncode(ret)
    return out


def roi_align(
    input,
    boxes,
    output_size,
    spatial_scale=1.0,
    sampling_ratio=-1,
    aligned=False,
) -> Tensor:
    if isinstance(boxes, Tensor):
        size_boxes = boxes.size().data
        assert (
            len(size_boxes) == 2 and size_boxes[1] == 5
        ), "boxes should be a tensor of shape (N,5)"
    elif isinstance(boxes, list):
        size_boxes = boxes[0].size().data
        assert (
            len(size_boxes) == 2 and size_boxes[1] == 4
        ), "boxes should be a list of tensor of shape (N,4)"

    sizeI = input.size().data
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    sizeI[-1] = output_size[-1]
    sizeI[-2] = output_size[-2]

    nhwc_stride = compute_nhwc_stride_2d(sizeI) if glob_vars.nhwc else None
    out = Tensor(sizeI, input.get_dtype(), stride=nhwc_stride)
    func = check_function("diopiRoiAlign")
    ret = func(
        input.context(),
        out,
        input,
        boxes,
        spatial_scale,
        output_size[-2],
        output_size[-1],
        sampling_ratio,
        aligned,
    )
    check_returncode(ret)
    return out


def slice_op(input, dim, index) -> Tensor:
    sizeI = list(input.size().data)
    num = int((index.stop - index.start + index.step - 1) / index.step)
    sizeI[dim] = num
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiSlice")
    ret = func(input.context(), out, input, dim, index.start, index.stop, index.step)

    check_returncode(ret)
    return out


def index(input, **kwargs) -> Tensor:
    new_args = list()
    hasEllipsis = False
    once = True
    for ele in kwargs.values():
        if ele is None:
            hasEllipsis = True
        else:
            if hasEllipsis and once:
                once = False
                sizeI = input.size().data
                sizeE = ele.size().data
                length = len(sizeI) - len(sizeE) - len(new_args)
                for i in range(length):
                    tmp_p = TensorP(None)
                    new_args.append(tmp_p)

            new_args.append(TensorP(ele))

    nums = len(new_args)

    out_tensor = Tensor()
    out_ptr = TensorP(out_tensor)
    func = check_function("diopiIndex")
    ret = func(input.context(), out_ptr, input, new_args, nums)
    out = out_ptr.data()
    check_returncode(ret)
    return out


def sgd(
    param,
    param_grad,
    lr,
    buf=None,
    momentum=0,
    dampening=0,
    weight_decay=0,
    nesterov=False,
):
    func = check_function("diopiSgd")

    arg_buf = buf if buf is None else buf
    ret = func(
        param.context(),
        param,
        param_grad,
        arg_buf,
        lr,
        momentum,
        dampening,
        weight_decay,
        nesterov,
    )
    check_returncode(ret)
    return param, buf


def adaptive_max_pool2d_backward(input, grad_outputs, output_size, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    _, indices = adaptive_max_pool2d(input, output_size, return_indices=True)

    func = check_function("diopiAdaptiveMaxPool2dBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], input, indices)
    check_returncode(ret)
    return {"input": grad_input}


def slice_op_backward(input, grad_outputs, dim, index, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    sizeI = input.size()

    func = check_function("diopiSliceBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        sizeI,
        dim,
        index.start,
        index.stop,
        index.step,
    )
    check_returncode(ret)
    return {"input": grad_input}


def adaptive_avg_pool2d_backward(input, grad_outputs, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    func = check_function("diopiAdaptiveAvgPool2dBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], input)
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
                sizeI = input.size().data
                sizeE = ele.size().data
                length = len(sizeI) - len(sizeE) - len(new_args)
                for i in range(length):
                    tmp_p = TensorP(None)
                    new_args.append(tmp_p)

            new_args.append(TensorP(ele))
    nums = len(new_args)

    func = check_function("diopiIndexBackward")
    ret = func(
        input.context(),
        grad_input,
        zeros_like_input,
        new_args,
        nums,
        grad_outputs[0],
    )
    check_returncode(ret)
    return {"input": grad_input}


def leaky_relu_backward(
    input, grad_outputs, negative_slope=0.01, input_is_result=False, **kwargs
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    dtype = Dtype.int64 if isinstance(negative_slope, int) else Dtype.float64
    negative_slope = Scalar(negative_slope, dtype)

    func = check_function("diopiLeakyReluBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        input,
        negative_slope,
        input_is_result,
    )
    check_returncode(ret)
    return {"input": grad_input}


def sigmoid_focal_loss_backward(
    inputs,
    grad_outputs,
    targets,
    alpha=0.25,
    gamma=2,
    reduction="none",
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    assert (
        inputs.size().data == targets.size().data
    ), "target shape must be the same as input shape"
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"

    grad_input = raw_like(inputs)
    reduction = convert_reduction(reduction)
    func = check_function("diopiSigmoidFocalLossBackward")

    ret = func(
        inputs.context(),
        grad_outputs[0],
        inputs,
        targets,
        grad_input,
        gamma,
        alpha,
        reduction,
    )
    check_returncode(ret)
    return {"inputs": grad_input}


def roi_align_backward(
    input,
    grad_outputs,
    boxes,
    output_size,
    spatial_scale=1.0,
    sampling_ratio=-1,
    aligned=False,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    if isinstance(boxes, Tensor):
        size_boxes = boxes.size().data
        assert (
            len(size_boxes) == 2 and size_boxes[1] == 5
        ), "boxes should be a tensor of shape (N,5)"
    elif isinstance(boxes, list):
        size_boxes = boxes[0].size().data
        assert (
            len(size_boxes) == 2 and size_boxes[1] == 4
        ), "boxes should be a list of tensor of shape (N,4)"

    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    out = raw_like(input)
    sizeI = input.size().data

    func = check_function("diopiRoiAlignBackward")
    ret = func(
        input.context(),
        out,
        grad_outputs[0],
        boxes,
        spatial_scale,
        output_size[-2],
        output_size[-1],
        sizeI[0],
        sizeI[1],
        sizeI[2],
        sizeI[3],
        sampling_ratio,
        aligned,
    )
    check_returncode(ret)
    return {"input": out}


def conv2d_backward(
    input,
    grad_outputs,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    sizeI = input.size().data
    sizeW = weight.size().data
    assert len(sizeI) == 4 and len(sizeW) == 4, "input and weight must be 4d tensors"

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    dilation = Sizes(list(dilation))

    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    out = {"input": grad_input, "weight": grad_weight}

    if bias is None:
        grad_bias = None
        sizeBias = None
    else:
        gradBias = raw_like(bias)
        grad_bias = gradBias
        sizeBias = bias.size()
        out.update({"bias": grad_bias})

    func = check_function("diopiConvolution2dBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_weight,
        grad_bias,
        grad_outputs[0],
        input,
        weight,
        sizeBias,
        stride,
        padding,
        dilation,
        groups,
    )
    check_returncode(ret)
    return out


def conv_transpose2d_backward(
    input,
    grad_outputs,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    sizeI = input.size().data
    sizeW = weight.size().data
    assert len(sizeI) == 4 and len(sizeW) == 4, "input and weight must be 4d tensors"

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    dilation = Sizes(list(dilation))

    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    out = {"input": grad_input, "weight": grad_weight}

    if bias is None:
        grad_bias = None
        sizeBias = None
    else:
        gradBias = raw_like(bias)
        grad_bias = gradBias
        sizeBias = bias.size()
        out.update({"bias": grad_bias})

    output_padding = Sizes(list([0, 0]))

    func = check_function("diopiConvTranspose2dBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_weight,
        grad_bias,
        grad_outputs[0],
        input,
        weight,
        sizeBias,
        stride,
        padding,
        dilation,
        output_padding,
        groups,
    )
    check_returncode(ret)
    return out


def hardtanh_backward(
    input, grad_outputs, min_val=-1.0, max_val=1.0, **kwargs
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    min_val = Scalar(min_val)
    max_val = Scalar(max_val)

    func = check_function("diopiHardtanhBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], input, min_val, max_val)
    check_returncode(ret)
    return {"input": grad_input}


def hardswish_backward(input, grad_outputs, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    func = check_function("diopiHardswishBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], input)
    check_returncode(ret)
    return {"input": grad_input}


def gelu_backward(input, grad_outputs, approximate="none", **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiGeluBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        input,
        approximate.encode("UTF-8"),
    )
    check_returncode(ret)
    return {"input": grad_input}


def avg_pool2d_backward(
    input,
    grad_outputs,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    kernel_size = Sizes(list(kernel_size))

    func = check_function("diopiAvgPool2dBackward")
    ret = (
        func(
            input.context(),
            grad_input,
            grad_outputs[0],
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )
        if divisor_override
        else func(
            input.context(),
            grad_input,
            grad_outputs[0],
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
        )
    )
    check_returncode(ret)
    return {"input": grad_input}


def embedding_backward(
    input: Tensor,
    grad_outputs: List[Tensor],
    weight: Tensor,
    padding_idx: Optional[int] = None,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
    **kwargs,
) -> dict:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_output = grad_outputs[0]
    num_weight = weight.size().data[0]
    grad_weight_shape = [num_weight, grad_output.size().data[-1]]
    grad_weight = Tensor(grad_weight_shape, grad_output.get_dtype())
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < num_weight, "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert (
                padding_idx >= -num_weight
            ), "Padding_idx must be within num_embeddings"
            padding_idx = num_weight + padding_idx
    else:
        padding_idx = -1

    func = check_function("diopiEmbeddingBackward")
    ret = func(
        input.context(),
        grad_weight,
        grad_output,
        input,
        num_weight,
        padding_idx,
        scale_grad_by_freq,
        sparse,
    )
    check_returncode(ret)
    return {"weight": grad_weight}


def mse_loss_backward(
    input, grad_outputs, target, reduction="mean", **kwargs
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    reduction_mode = convert_reduction(reduction)

    func = check_function("diopiMSELossBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        input,
        target,
        reduction_mode,
    )
    check_returncode(ret)
    return {"input": grad_input}


def tanh_backward(input, grad_outputs, output, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiTanhBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], output)
    check_returncode(ret)
    return {"input": grad_input}


def index_select_backward(input, grad_outputs, dim, index, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    inputSize = input.size()

    func = check_function("diopiIndexSelectBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], inputSize, dim, index)
    check_returncode(ret)
    return {"input": grad_input}


def select_backward(input, grad_outputs, dim, index, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    inputSize = input.size()

    func = check_function("diopiSelectBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], inputSize, dim, index)
    check_returncode(ret)
    return {"input": grad_input}


def softmax_backward(input, grad_outputs, output, dim, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiSoftmaxBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], output, dim)
    check_returncode(ret)
    return {"input": grad_input}


def log_softmax_backward(input, grad_outputs, output, dim, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiLogSoftmaxBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], output, dim)
    check_returncode(ret)
    return {"input": grad_input}


def sigmoid_backward(input, grad_outputs, output, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiSigmoidBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], output)
    check_returncode(ret)
    return {"input": grad_input}


def threshold_backward(input, grad_outputs, threshold, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    threshold = Scalar(threshold)

    func = check_function("diopiThresholdBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], input, threshold)
    check_returncode(ret)
    return {"input": grad_input}


def binary_cross_entropy_backward(
    input, grad_outputs, target, weight=None, reduction="mean", **kwargs
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    assert (
        input.size().data == target.size().data
    ), "target shape must be the same as input shape"
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"

    if weight is not None:
        assert isinstance(weight, Tensor), "weigth must be a Tensor"

    grad_input = raw_like(input)
    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiBCELossBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        input,
        target,
        weight,
        reduction_mode,
    )
    check_returncode(ret)
    return {"input": grad_input}


def binary_cross_entropy_with_logits_backward(
    input,
    grad_outputs,
    target,
    weight=None,
    reduction="mean",
    pos_weight=None,
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    assert (
        input.size().data == target.size().data
    ), "target shape must be the same as input shape"
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"

    if pos_weight is not None:
        assert isinstance(pos_weight, Tensor), "pos_weigth must be a Tensor"

    if weight is not None:
        assert isinstance(weight, Tensor), "weigth must be a Tensor"

    grad_input = raw_like(input)
    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiBCEWithLogitsBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        input,
        target,
        weight,
        pos_weight,
        reduction_mode,
    )
    check_returncode(ret)
    return {"input": grad_input}


# todo: impl for diopiNLLLossV2Backward
def nll_loss_backward(
    input,
    grad_outputs,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    if weight is not None:
        assert isinstance(weight, Tensor), "weigth must be a Tensor"

    reduction_mode = convert_reduction(reduction)

    func = check_function("diopiNLLLossBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        input,
        target,
        weight,
        reduction_mode,
        ignore_index,
    )
    check_returncode(ret)
    return {"input": grad_input}


def max_pool2d_backward(
    input,
    grad_outputs,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    sizeI = input.size().data
    assert len(sizeI) == 4 or len(sizeI) == 3, "input must be 3d or 4d tensors"

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    _, indices = max_pool2d(
        input, kernel_size, stride, padding, dilation, ceil_mode, True
    )
    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    kernel_size = Sizes(list(kernel_size))
    dilation = Sizes(list(dilation))

    func = check_function("diopiMaxPool2dBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices,
    )
    check_returncode(ret)
    return {"input": grad_input}


def batch_norm_backward(
    input,
    grad_outputs,
    running_mean,
    running_var,
    weight,
    bias,
    training=False,
    eps=1e-05,
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    save_mean = GLOBAL_STATE.pop("batch_norm_save_mean")
    save_invstd = GLOBAL_STATE.pop("batch_norm_save_invstd")

    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    grad_bias = raw_like(bias)

    if not training:
        assert (
            running_mean is not None and running_var is not None
        ), "if not trainging, running_mean and running_var must be defined"
    # running_mean = running_mean if running_mean is None else running_mean
    # running_var = running_var if running_var is None else running_var
    out = {"input": grad_input, "weight": grad_weight, "bias": grad_bias}
    func = check_function("diopiBatchNormBackward")
    grad_output = grad_outputs[0]
    ret = func(
        input.context(),
        grad_input,
        grad_weight,
        grad_bias,
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        training,
        eps,
    )
    check_returncode(ret)
    return out


def arange(end, start=0, step=1, dtype=None) -> Tensor:
    if dtype is None:
        if (
            isinstance(start, float)
            or isinstance(end, float)
            or isinstance(step, float)
        ):
            dtype = Dtype.float32
        else:
            dtype = from_numpy_dtype(glob_vars.int_type)

    numel = int(
        (end - start) / step + (((end - start) / step) > int((end - start) / step))
    )
    out = Tensor((numel,), dtype)

    func = check_function("diopiArange")
    ret = func(out.context(), out, Scalar(start), Scalar(end), Scalar(step))
    check_returncode(ret)
    return out


def randperm(n: int, dtype=None, generator=None) -> Tensor:
    dtype = from_numpy_dtype(glob_vars.int_type) if dtype is None else dtype
    numel = n
    out = Tensor((numel,), dtype)

    func = check_function("diopiRandperm")
    ret = func(out.context(), out, n, generator)
    check_returncode(ret)
    return out


def uniform(input, start=0, end=1, generator=None, inplace=True) -> Tensor:
    assert inplace, "DIOPI now doesn't support uniform without inplace"
    func = check_function("diopiUniformInp")
    ret = func(input.context(), input, start, end, generator)
    check_returncode(ret)
    return input


def random(input, start=0, end=None, generator=None) -> Tensor:
    func = check_function("diopiRandomInp")
    ret = (
        func(input.context(), input, start, end, generator)
        if end
        else func(input.context(), input, start, generator)
    )
    check_returncode(ret)
    return input


def bernoulli(input, inplace=False, p=None, generator=None) -> Tensor:
    out = input

    if p is not None:
        func = check_function("diopiBernoulliScalar")
        ret = func(input.context(), input, p, generator)
    elif inplace:
        func = check_function("diopiBernoulliInp")
        ret = func(input.context(), input, generator)
    else:
        out = raw_like(input)
        func = check_function("diopiBernoulli")
        ret = func(input.context(), out, input, generator)

    check_returncode(ret)
    return out


def masked_fill(input, mask, value, inplace=False) -> Tensor:
    assert mask.get_dtype() == Dtype.bool, "mask must be bool tensor"
    out = raw_like(input)
    call = "diopiMaskedFill"
    call_scalar = False
    if isinstance(value, Tensor):
        value_res = value
    else:
        value_res = Scalar(value)
        call_scalar = True
    out_size = infer_size(input.size().data, mask.size().data)
    if inplace:
        input_np = input.numpy()
        input_np = np.resize(input_np, out_size)
        input = Tensor.from_numpy(input_np)
        out = input
        call = call + "Inp"
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context(), input, mask, value_res)
    else:
        out = Tensor(out_size, input.get_dtype())
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context(), out, input, mask, value_res)

    check_returncode(ret)
    return out


def adamw(
    param,
    param_grad,
    exp_avg,
    exp_avg_sq,
    max_exp_avg_sq,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    amsgrad=False,
):
    func = check_function("diopiAdamW")
    ret = func(
        param.context(),
        param,
        param_grad,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step,
        amsgrad,
    )
    check_returncode(ret)
    return param, exp_avg, exp_avg_sq, max_exp_avg_sq


def adam(
    param,
    param_grad,
    exp_avg,
    exp_avg_sq,
    max_exp_avg_sq,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    amsgrad=False,
):
    func = check_function("diopiAdam")
    ret = func(
        param.context(),
        param,
        param_grad,
        exp_avg,
        exp_avg_sq,
        max_exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step,
        amsgrad,
    )
    check_returncode(ret)
    return param, exp_avg, exp_avg_sq, max_exp_avg_sq


def adadelta(param, param_grad, square_avg, acc_delta, lr, rho, eps, weight_decay):
    func = check_function("diopiAdadelta")
    ret = func(
        param.context(),
        param,
        param_grad,
        square_avg,
        acc_delta,
        lr,
        rho,
        eps,
        weight_decay,
    )
    check_returncode(ret)
    return param, square_avg, acc_delta


def rmsprop(
    param,
    param_grad,
    square_avg,
    grad_avg,
    momentum_buffer,
    lr,
    alpha,
    eps,
    weight_decay,
    momentum,
    centered,
):
    func = check_function("diopiRmsprop")
    ret = func(
        param.context(),
        param,
        param_grad,
        square_avg,
        grad_avg,
        momentum_buffer,
        lr,
        alpha,
        eps,
        weight_decay,
        momentum,
        centered,
    )
    check_returncode(ret)
    return param, square_avg, grad_avg, momentum_buffer


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), "bias must be a Tensor"

    sizeI = input.size().data
    sizeW = list(weight.size().data)
    assert len(sizeI) == 4 and len(sizeW) == 4, "input and weight must be 4d tensors"

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
        sizeO.append(
            int(
                (sizeI[i] - 1) * stride[i]
                - 2 * padding[i]
                + sizeW[i]
                + output_padding[i]
            )
            + 1
        )
    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    output_padding = Sizes(list(output_padding))
    dilation = Sizes(list(dilation))

    out = Tensor(sizeO, input.get_dtype())
    func = check_function("diopiConvTranspose2d")
    ret = func(
        input.context(),
        out,
        input,
        weight,
        bias,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
    )
    check_returncode(ret)
    return out


def cumsum(input, dim, dtype=None):
    assert isinstance(dim, int), "dim should be int"

    sizeI = list(input.size().data)
    if len(sizeI) == 0:
        assert dim in (0, -1), "dim out of index"
    else:
        assert dim < len(sizeI), "dim out of index"

    out = (
        Tensor(input.size().data, promote_type(input, Dtype.int64))
        if dtype is None
        else Tensor(input.size().data, dtype)
    )
    func = check_function("diopiCumsum")
    ret = func(input.context(), out, input, dim)
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
        assert (
            sizeA == sizeB or sizeA == 1 or sizeB == 1
        ), f"The size of tensor a ({sizeA}) must match the size of tensor b ({sizeB}) at non-singleton dimension {i}"
        expanded_sizes[i] = sizeA if sizeA != 1 else sizeB
    return expanded_sizes


def cdist(x1, x2, p, compute_mode=None):
    sizeX1 = list(x1.size().data)
    sizeX2 = list(x2.size().data)
    dim1 = len(sizeX1)
    dim2 = len(sizeX2)
    assert dim1 > 1 and dim2 > 1, "cdist only supports at least 2D tensors"
    assert (
        sizeX1[-1] == sizeX2[-1]
    ), "X1 and X2 must have the same number of elements at the last dimension"
    row1 = sizeX1[-2]
    row2 = sizeX2[-2]
    batch_tensor1 = sizeX1[:-2]
    batch_tensor2 = sizeX2[:-2]
    expand_batch_portion = infer_size(batch_tensor1, batch_tensor2)
    out_shape = expand_batch_portion + [row1, row2]
    if compute_mode is not None:
        if compute_mode == "use_mm_for_euclid_dist":
            compute_mode = 1
        else:
            compute_mode = 2
    else:
        compute_mode = None
    out = Tensor(out_shape, x1.get_dtype())
    func = check_function("diopiCdist")

    ret = (
        func(x1.context(), out, x1, x2, p, compute_mode)
        if compute_mode
        else func(x1.context(), out, x1, x2, p)
    )
    check_returncode(ret)
    return out


def cdist_backward(x1, grad_outputs, output, x2, p, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    sizeX1 = list(x1.size().data)
    sizeX2 = list(x2.size().data)
    dim1 = len(sizeX1)
    dim2 = len(sizeX2)
    assert dim1 > 1 and dim2 > 1, "cdist only supports at least 2D tensors"
    assert (
        sizeX1[-1] == sizeX2[-1]
    ), "X1 and X2 must have the same number of elements at the last dimension"
    column1 = sizeX1[-1]
    row1 = sizeX1[-2]
    batch_tensor1 = sizeX1[:-2]
    batch_tensor2 = sizeX2[:-2]
    expand_batch_portion = infer_size(batch_tensor1, batch_tensor2)
    grad_x1_shape = expand_batch_portion + [row1, column1]
    grad_x1 = Tensor(grad_x1_shape, x1.get_dtype())
    func = check_function("diopiCdistBackward")
    ret = func(x1.context(), grad_x1, grad_outputs[0], x1, x2, p, output)
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
    return {"x1": grad_x1}


def reciprocal(input, inplace=False) -> Tensor:
    out = raw_like(input)
    call = "diopiReciprocal"

    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context(), input)
    else:
        out = Tensor(input.size().data, promote_type(input, Dtype.float32))
        func = check_function(call)
        ret = func(input.context(), out, input)

    check_returncode(ret)
    return out


def bitwise_not(input, inplace=False):
    assert input.get_dtype() in [
        Dtype.bool,
        Dtype.uint8,
        Dtype.int8,
        Dtype.int16,
        Dtype.int32,
        from_numpy_dtype(glob_vars.int_type),
    ], "input tensor must be of integral or boolean"
    return unary_op(input, inplace, "diopiBitwiseNot")


def bitwise_and(input, other, inplace=False):
    assert input.get_dtype() in [
        Dtype.bool,
        Dtype.uint8,
        Dtype.int8,
        Dtype.int16,
        Dtype.int32,
        from_numpy_dtype(glob_vars.int_type),
    ], "input tensor must be of integral or boolean"
    if isinstance(other, Tensor):
        assert other.get_dtype() in [
            Dtype.bool,
            Dtype.uint8,
            Dtype.int8,
            Dtype.int16,
            Dtype.int32,
            from_numpy_dtype(glob_vars.int_type),
        ], "other tensor must be of integral or boolean"
    else:
        assert isinstance(other, int), "other must be of integral or boolean"
    out_dtype = common_dtype(input, other)
    return binary_op_scalar(input, other, inplace, "diopiBitwiseAnd", dtype=out_dtype)


def bitwise_or(input, other, inplace=False):
    assert input.get_dtype() in [
        Dtype.bool,
        Dtype.uint8,
        Dtype.int8,
        Dtype.int16,
        Dtype.int32,
        from_numpy_dtype(glob_vars.int_type),
    ], "input tensor must be of integral or boolean"
    if isinstance(other, Tensor):
        assert other.get_dtype() in [
            Dtype.bool,
            Dtype.uint8,
            Dtype.int8,
            Dtype.int16,
            Dtype.int32,
            from_numpy_dtype(glob_vars.int_type),
        ], "other tensor must be of integral or boolean"
    else:
        assert isinstance(other, int), "other must be of integral or boolean"
    out_dtype = common_dtype(input, other)
    return binary_op_scalar(input, other, inplace, "diopiBitwiseOr", dtype=out_dtype)


def argmax(input, dim=None, keepdim=False):
    sizeO = list(input.size().data)
    if len(sizeO) > 0 and dim is not None:
        assert dim < len(sizeO), "dim out of index"
        if keepdim:
            sizeO[dim] = 1
        else:
            sizeO = sizeO[:dim] + sizeO[dim + 1 :]
    else:
        sizeO = [1]

    out = Tensor(sizeO, from_numpy_dtype(glob_vars.int_type))
    func = check_function("diopiArgmax")
    # todo: check the reason of using keepdim
    ret = (
        func(input.context(), out, input, keepdim)
        if dim is None
        else func(input.context(), out, input, dim, keepdim)
    )
    check_returncode(ret)

    return out


def smooth_l1_loss(input, target, reduction="mean", beta=1.0):
    assert (
        input.shape().data == target.shape().data
    ), "target shape must be the same as input shape"
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"

    if reduction == "none":
        out = raw_like(input)
    else:
        out = Tensor((), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiSmoothL1Loss")
    ret = func(input.context(), out, input, target, reduction_mode, beta)
    check_returncode(ret)
    return out


def smooth_l1_loss_backward(
    input, grad_outputs, target, reduction="mean", beta=1.0, **kwargs
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiSmoothL1LossBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        input,
        target,
        reduction_mode,
        beta,
    )
    check_returncode(ret)
    return {"input": grad_input}


def maximum(input, other) -> Tensor:
    size = broadcast_out_size(list(input.size().data), list(other.size().data))
    out = Tensor(size, common_dtype(input, other))

    func = check_function("diopiMaximum")
    ret = func(input.context(), out, input, other)
    check_returncode(ret)
    return out


def minimum(input, other) -> Tensor:
    size = broadcast_out_size(list(input.size().data), list(other.size().data))
    out = Tensor(size, common_dtype(input, other))

    func = check_function("diopiMinimum")
    ret = func(input.context(), out, input, other)
    check_returncode(ret)
    return out


def mm(input, mat2) -> Tensor:
    size1 = list(input.size().data)
    assert len(size1) == 2, "input must be 2d tensor"
    size2 = mat2.size().data
    assert len(size2) == 2, "mat2 must be 2d tensor"
    assert size1[1] == size2[0], "invalid args"

    size_out = size1
    size_out[1] = size2[1]
    out = Tensor(size_out, input.get_dtype())

    func = check_function("diopiMm")
    ret = func(input.context(), out, input, mat2)
    check_returncode(ret)
    return out


def conv3d(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
) -> Tensor:
    if bias is not None:
        assert isinstance(bias, Tensor), "bias must be a Tensor"

    sizeI = input.size().data
    sizeW = list(weight.size().data)
    assert len(sizeI) == 5 and len(sizeW) == 5, "input and weight must be 4d tensors"

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

    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    dilation = Sizes(list(dilation))

    nhwc_stride = compute_nhwc_stride_3d(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)
    func = check_function("diopiConvolution3d")
    ret = func(
        input.context(),
        out,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )
    check_returncode(ret)
    return out


def conv3d_backward(
    input,
    grad_outputs,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    sizeI = input.size().data
    sizeW = weight.size().data
    assert len(sizeI) == 5 and len(sizeW) == 5, "input and weight must be 5d tensors"

    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    dilation = Sizes(list(dilation))

    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    out = {"input": grad_input, "weight": grad_weight}

    if bias is None:
        grad_bias = None
        sizeBias = None
    else:
        gradBias = raw_like(bias)
        grad_bias = gradBias
        sizeBias = bias.size()
        out.update({"bias": grad_bias})

    func = check_function("diopiConvolution3dBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_weight,
        grad_bias,
        grad_outputs[0],
        input,
        weight,
        sizeBias,
        stride,
        padding,
        dilation,
        groups,
    )
    check_returncode(ret)
    return out


def expand(input, size) -> Tensor:
    SizeI = input.size().data
    size = list(size)
    for i in range(-1, -len(SizeI) - 1, -1):
        if size[i] == -1:
            size[i] = SizeI[i]
        else:
            assert (
                size[i] == SizeI[i] or SizeI[i] == 1
            ), "size must be broadcastable with input"

    if len(size) > len(SizeI):
        assert size[0] >= 0, "the size of new dimension can't be negative"

    out = Tensor(size, input.get_dtype())

    func = check_function("diopiExpand")
    ret = func(input.context(), out, input)
    check_returncode(ret)
    return out


def unfold(input, dimension, size, step):
    sizeO = list(input.size().data)

    if len(sizeO) == 0:
        sizeO = [1]
    else:
        sizeO[dimension] = int((sizeO[dimension] - size) / step + 1)
        sizeO.append(size)

    out = Tensor(sizeO, input.get_dtype())
    func = check_function("diopiUnfold")
    ret = func(input.context(), out, input, dimension, size, step)
    check_returncode(ret)
    return out


def unfold_backward(input, grad_outputs, dimension, size, step, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    sizeI = input.size()

    func = check_function("diopiUnfoldBackward")
    ret = func(
        grad_input.context(),
        grad_input,
        grad_outputs[0],
        sizeI,
        dimension,
        size,
        step,
    )
    check_returncode(ret)
    return {"input": grad_input}


def masked_select(input, mask) -> Tensor:
    assert mask.get_dtype() == Dtype.bool, "mask must be bool tensor"
    out_tensor = Tensor()
    out_ptr = TensorP(out_tensor)

    func = check_function("diopiMaskedSelect")
    ret = func(input.context(), out_ptr, input, mask)
    check_returncode(ret)
    return out_ptr.data()


def masked_select_backward(input, grad_outputs, mask) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiMaskedSelectBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], input, mask)
    check_returncode(ret)
    return {"input": grad_input}


def index_fill(input, dim, index, value, inplace=False) -> Tensor:
    out = raw_like(input)

    call = "diopiIndexFill"
    call_scalar = False
    if isinstance(value, Tensor):
        value = value
    else:
        value = Scalar(value)
        call_scalar = True

    if inplace:
        out = input
        call = call + "Inp"
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context(), input, dim, index, value)
    else:
        out = raw_like(input)
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context(), out, input, dim, index, value)

    check_returncode(ret)
    return out


def linspace(start, end, steps, dtype=None):
    dtype = Dtype.float32 if dtype is None else dtype

    out = Tensor((steps,), dtype)

    start = Scalar(start)
    end = Scalar(end)
    func = check_function("diopiLinspace")

    ret = func(out.context(), out, start, end, steps)
    check_returncode(ret)
    return out


def roll(input, shifts, dims=None):
    if isinstance(shifts, int):
        shifts = (shifts,)
    shifts = Sizes(list(shifts))

    if isinstance(dims, int):
        dims = (dims,)
    if dims is not None:
        dims = Sizes(list(dims))
    else:
        dims = Sizes(list(()))

    out = raw_like(input)
    func = check_function("diopiRoll")
    ret = func(input.context(), out, input, shifts, dims)
    check_returncode(ret)
    return out


def norm(input, p, dim=None, keepdim=False, dtype=None):
    p = Scalar(p)
    dim, out = reduce_op_process(input, dim, keepdim, dtype)
    dim = Sizes(list(dim))

    func = check_function("diopiNorm")
    ret = func(input.context(), out, input, p, dim)
    check_returncode(ret)
    return out


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    dim = list(input.size().data)
    save_mean = Tensor((dim[0], num_groups), input.get_dtype())
    save_invstd = raw_like(save_mean)

    weight = None if weight is None else weight
    bias = None if bias is None else bias

    out = raw_like(input)
    func = check_function("diopiGroupNorm")
    ret = func(
        input.context(),
        out,
        save_mean,
        save_invstd,
        input,
        weight,
        bias,
        num_groups,
        eps,
    )
    check_returncode(ret)
    GLOBAL_STATE["group_norm_save_mean"] = save_mean
    GLOBAL_STATE["group_norm_save_invstd"] = save_invstd
    return out


def group_norm_backward(
    input,
    grad_outputs,
    num_groups,
    weight=None,
    bias=None,
    eps=1e-05,
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    save_mean = GLOBAL_STATE.pop("group_norm_save_mean")
    save_invstd = GLOBAL_STATE.pop("group_norm_save_invstd")
    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    grad_bias = raw_like(bias)
    weight = None if weight is None else weight
    bias = None if bias is None else bias

    out = {"input": grad_input, "weight": grad_weight, "bias": grad_bias}
    func = check_function("diopiGroupNormBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_weight,
        grad_bias,
        grad_outputs[0],
        input,
        weight,
        save_mean,
        save_invstd,
        num_groups,
    )
    check_returncode(ret)
    return out


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    sizeI = input.size().data
    dims = len(sizeI) - len(normalized_shape)
    size = [i for i in sizeI[0:dims]]
    save_mean = Tensor(size, input.get_dtype())
    save_invstd = raw_like(save_mean)

    weight = None if weight is None else weight
    bias = None if bias is None else bias

    out = raw_like(input)
    func = check_function("diopiLayerNorm")
    ret = func(
        input.context(),
        out,
        save_mean,
        save_invstd,
        input,
        weight,
        bias,
        Sizes(normalized_shape),
        eps,
    )
    check_returncode(ret)
    GLOBAL_STATE["layer_norm_save_mean"] = save_mean
    GLOBAL_STATE["layer_norm_save_invstd"] = save_invstd
    return out


def layer_norm_backward(
    input,
    grad_outputs,
    normalized_shape,
    weight=None,
    bias=None,
    eps=1e-05,
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    save_mean = GLOBAL_STATE.pop("layer_norm_save_mean")
    save_invstd = GLOBAL_STATE.pop("layer_norm_save_invstd")
    grad_input = raw_like(input)
    out = {"input": grad_input}

    if weight is None:
        weight = None
        grad_weight_capsule = None
    else:
        grad_weight = raw_like(weight)
        weight = weight
        grad_weight_capsule = grad_weight
        out["weight"] = grad_weight

    if bias is None:
        bias = None
        grad_bias_capsule = None
    else:
        grad_bias = raw_like(bias)
        bias = bias
        grad_bias_capsule = grad_bias
        out["bias"] = grad_bias

    func = check_function("diopiLayerNormBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_weight_capsule,
        grad_bias_capsule,
        grad_outputs[0],
        input,
        weight,
        bias,
        save_mean,
        save_invstd,
        Sizes(normalized_shape),
    )
    check_returncode(ret)
    return out


def adaptive_avg_pool3d(input, output_size):
    sizeI = input.size().data
    assert len(sizeI) == 5 or len(sizeI) == 4, "input must be 4d or 5d tensors"

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
    output_size = Sizes(list([sizeO[-3], sizeO[-2], sizeO[-1]]))

    func = check_function("diopiAdaptiveAvgPool3d")
    ret = func(input.context(), out, input, output_size)
    check_returncode(ret)
    return out


def adaptive_avg_pool3d_backward(input, grad_outputs, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)

    func = check_function("diopiAdaptiveAvgPool3dBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], input)
    check_returncode(ret)
    return {"input": grad_input}


def adaptive_max_pool3d(input, output_size, return_indices=False):
    sizeI = input.size().data
    assert len(sizeI) == 5 or len(sizeI) == 4, "input must be 4d or 5d tensors"

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
    output_size = Sizes(list([sizeO[-3], sizeO[-2], sizeO[-1]]))

    if return_indices:
        func = check_function("diopiAdaptiveMaxPool3dWithIndices")
        nhwc_stride = compute_nhwc_stride_3d(sizeO) if glob_vars.nhwc else None
        indices = Tensor(
            sizeO, from_numpy_dtype(glob_vars.int_type), stride=nhwc_stride
        )
        ret = func(input.context(), out, indices, input, output_size)
        check_returncode(ret)
        return out, indices
    else:
        func = check_function("diopiAdaptiveMaxPool3d")
        ret = func(input.context(), out, input, output_size)
    check_returncode(ret)
    return out


def adaptive_max_pool3d_backward(input, grad_outputs, output_size, **kwargs) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    _, indices = adaptive_max_pool3d(input, output_size, return_indices=True)

    func = check_function("diopiAdaptiveMaxPool3dBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], input, indices)
    check_returncode(ret)
    return {"input": grad_input}


def max_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
) -> Tensor:
    sizeI = input.size().data
    assert len(sizeI) == 5 or len(sizeI) == 4, "input must be 5d or 4d tensors"

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 5:
        sizeO.append(sizeI[1])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
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

    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    kernel_size = Sizes(list(kernel_size))
    dilation = Sizes(list(dilation))
    out = Tensor(sizeO, input.get_dtype())

    if not return_indices:
        func = check_function("diopiMaxPool3d")
        ret = func(
            input.context(),
            out,
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        check_returncode(ret)
        return out
    else:
        func = check_function("diopiMaxPool3dWithIndices")
        indices = Tensor(sizeO, from_numpy_dtype(glob_vars.int_type))
        ret = func(
            input.context(),
            out,
            indices,
            input,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        )
        check_returncode(ret)
        return out, indices


def max_pool3d_backward(
    input,
    grad_outputs,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    **kwargs,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    sizeI = input.size().data
    assert len(sizeI) == 5 or len(sizeI) == 4, "input must be 5d or 4d tensors"

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    _, indices = max_pool3d(
        input, kernel_size, stride, padding, dilation, ceil_mode, True
    )
    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    kernel_size = Sizes(list(kernel_size))
    dilation = Sizes(list(dilation))

    func = check_function("diopiMaxPool3dBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices,
    )
    check_returncode(ret)
    return {"input": grad_input}


def permute(input, dims=None) -> Tensor:
    assert (
        isinstance(dims, (tuple, list)) or dims is None
    ), "dims should be tuple or list"

    sizeI = list(input.size().data)
    sizeO = list(input.size().data)
    for i in range(len(dims)):
        sizeO[i] = sizeI[dims[i]]
    out = Tensor(sizeO, input.get_dtype())
    dims = Sizes(list(dims))
    func = check_function("diopiPermute")
    ret = func(input.context(), out, input, dims)
    check_returncode(ret)
    return out


def copy_(input, other) -> Tensor:
    func = check_function("diopiCopyInp")
    ret = func(input.context(), other, input)
    check_returncode(ret)
    return input


def gather(input, dim, index):
    assert isinstance(dim, int), "dim must be int"
    if len(input.size().data) > 1 and len(index.size().data) > 1:
        assert len(input.size().data) == len(
            index.size().data
        ), "input and index must have the same number of dimensions"
    out = Tensor(index.size().data, input.get_dtype())
    func = check_function("diopiGather")
    ret = func(input.context(), out, input, dim, index)
    check_returncode(ret)
    return out


def gather_backward(input, grad_outputs, dim, index, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    assert isinstance(dim, int), "dim must be int"
    grad_input = raw_like(input)
    func = check_function("diopiGatherBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], input, dim, index)
    check_returncode(ret)
    return {"input": grad_input}


def remainder(other, input=None, self=None):
    if self is not None:
        input = self
    call = "diopiRemainder"
    if isinstance(input, Tensor):
        context = input.context()
        if isinstance(other, Tensor):
            call += "Tensor"
            sizeO = list(input.size().data)
            sizeOther = list(other.size().data)
            sizeO = infer_size(sizeO, sizeOther)
            out_dtype = remainder_dtype(input, other)
            out = Tensor(sizeO, out_dtype)
            input = input
            other = other
        else:
            call += "Scalar"
            out_dtype = remainder_dtype(input, other)
            out = Tensor(input.size().data, out_dtype)
            other = Scalar(other)
            input = input
    else:
        assert isinstance(other, Tensor), "input or other must be tensor"
        context = other.context()
        if isinstance(input, int):
            out_dtype = get_dtype(other)
            if out_dtype == Dtype.bool:
                out_dtype = Dtype.int64
        else:
            out_dtype = remainder_dtype(input, other)
        out = Tensor(other.size().data, out_dtype)
        input = Scalar(input)
        other = other
    func = check_function(call)
    ret = func(context, out, input, other)
    check_returncode(ret)
    return out


def ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    log_probs_ = log_softmax(log_probs, 2)
    sizeO = (1,)
    sizeI = list(log_probs_.size().data)
    reduction_mode = convert_reduction(reduction)
    max_target_length = int(max(target_lengths, 0)[0].numpy())
    max_target_length = 2 * max_target_length + 1
    if reduction == "none":
        sizeO = (sizeI[1],)
    neg_log_likelihood = Tensor((sizeI[1],), log_probs.get_dtype())
    log_alpha = Tensor((sizeI[1], sizeI[0], max_target_length), log_probs.get_dtype())
    out = Tensor(sizeO, log_probs.get_dtype())

    func = check_function("diopiCTCLoss")
    ret = func(
        log_probs.context(),
        out,
        neg_log_likelihood,
        log_alpha,
        log_probs_,
        targets,
        input_lengths,
        target_lengths,
        blank,
        reduction_mode,
        zero_infinity,
    )
    check_returncode(ret)
    GLOBAL_STATE["ctc_loss_neg_log_likelihood"] = neg_log_likelihood
    GLOBAL_STATE["ctc_loss_log_alpha"] = log_alpha
    return out


def ctc_loss_backward(
    log_probs,
    grad_outputs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
) -> Tensor:
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    log_probs_ = log_softmax(log_probs, 2)
    grad_input = raw_like(log_probs_)
    neg_log_likelihood = GLOBAL_STATE.pop("ctc_loss_neg_log_likelihood")
    log_alpha = GLOBAL_STATE.pop("ctc_loss_log_alpha")

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiCTCLossBackward")
    ret = func(
        log_probs_.context(),
        grad_input,
        grad_outputs[0],
        log_probs_,
        targets,
        input_lengths,
        target_lengths,
        neg_log_likelihood,
        log_alpha,
        blank,
        reduction_mode,
        zero_infinity,
    )
    check_returncode(ret)
    return {
        "log_probs": log_softmax_backward(log_probs, [grad_input], log_probs_, 2)[
            "input"
        ]
    }


def index_put(
    input,
    values,
    indices1,
    indices2=None,
    indices3=None,
    accumulate=False,
    inplace=False,
):
    c_tensors = [TensorP(indices1)]
    if indices2 is not None:
        c_tensors.append(TensorP(indices2))
    if indices3 is not None:
        c_tensors.append(TensorP(indices3))
    indices_counts = len(c_tensors)
    call = "diopiIndexPut"
    out = raw_like(input)
    if inplace:
        call += "Inp"
        out = input
        func = check_function(call)
        ret = func(
            input.context(),
            input,
            values,
            c_tensors,
            indices_counts,
            accumulate,
        )
    else:
        copy_(out, input)  # out should be the same with original tensor
        func = check_function(call)
        ret = func(
            input.context(),
            out,
            input,
            values,
            c_tensors,
            indices_counts,
            accumulate,
        )
    check_returncode(ret)
    return out


def scatter(input, dim, index, src=None, value=None, reduce=None, inplace=False):
    assert isinstance(dim, int), "dim must be int"
    # if input.size().len != 0 and index.size().len != 0:
    #     assert input.size().len == index.size().len, \
    #         "input and index must have the same number of dimensions"
    assert (src is not None) or (value is not None)
    if reduce is not None:
        assert (
            reduce == "add" or reduce == "multiply"
        ), "reduce argument must be either add or multiply."
    else:
        reduce = ""
    # if src is not None:
    #     if input.size().len != 0 and src.size().len != 0:
    #         assert input.size().len == src.size().len, \
    #             "input and src must have the same number of dimensions"
    if src is None:
        src = value
    out = raw_like(input)
    call = "diopiScatter"
    call_scalar = False
    if isinstance(src, Tensor):
        src = src
    else:
        src = Scalar(src)
        call_scalar = True

    if inplace:
        out = input
        call = call + "Inp"
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(input.context(), input, dim, src, index, reduce.encode("UTF-8"))
    else:
        out = raw_like(input)
        if call_scalar:
            call = call + "Scalar"
        func = check_function(call)
        ret = func(
            input.context(),
            out,
            input,
            dim,
            src,
            index,
            reduce.encode("UTF-8"),
        )

    check_returncode(ret)
    return out


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=False
) -> Tensor:
    assert (
        size is None or scale_factor is None
    ), "only one of size or scale_factor should be defined"
    sizeI = list(input.size().data)
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

    c_size = Sizes(list(sizeI[2:]))
    if mode == "nearest":
        func = check_function("diopiUpsampleNearest")
        ret = func(input.context(), out, input, c_size)
    else:
        func = check_function("diopiUpsampleLinear")
        ret = func(
            input.context(),
            out,
            input,
            c_size,
            align_corners,
            mode.encode("UTF-8"),
        )
    check_returncode(ret)
    return out


def interpolate_backward(
    input, grad_outputs, size, mode="nearest", align_corners=None, **kwargs
) -> Tensor:
    in_size = input.size()
    out_size = grad_outputs[0].size().data[2:]
    out_size = Sizes(list(out_size))
    grad_input = raw_like(input)

    if mode == "nearest":
        func = check_function("diopiUpsampleNearestBackward")
        ret = func(input.context(), grad_input, grad_outputs[0], out_size, in_size)
    else:
        func = check_function("diopiUpsampleLinearBackward")
        ret = func(
            input.context(),
            grad_input,
            grad_outputs[0],
            out_size,
            in_size,
            align_corners,
            mode.encode("UTF-8"),
        )
    check_returncode(ret)
    return {"input": grad_input}


def pad(input, pad, mode="constant", value=None):
    assert mode in ["constant", "reflect", "replicate", "circular"], (
        "mode must one of " "'constant', 'reflect', 'replicate', 'circular'"
    )
    sizeO = list(input.size().data)
    assert len(pad) % 2 == 0, "Padding length must be divisible by 2"
    assert len(pad) // 2 <= len(
        sizeO
    ), "Padding length must be equal or more than length of input"
    paded_length = len(pad) // 2
    for i in range(paded_length):
        if len(pad) <= len(sizeO):
            pad_idx = paded_length - i
        else:
            pad_idx = i + 1
        sizeO[-pad_idx] += pad[2 * i] + pad[2 * i + 1]
    pad = Sizes(pad)
    if value is None and mode == "constant":
        value = 0

    nhwc_stride = compute_nhwc_stride(sizeO) if glob_vars.nhwc else None
    out = Tensor(sizeO, input.get_dtype(), stride=nhwc_stride)
    func = check_function("diopiPad")

    ret = (
        func(input.context(), out, input, pad, mode)
        if value is None
        else func(input.context(), out, input, pad, mode, value)
    )
    check_returncode(ret)
    return out


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    out_tensor = Tensor()
    out_ptr = TensorP(out_tensor)
    if return_inverse:
        sizeI = list(input.size().data)
        if dim is not None:
            sizeI = (sizeI[dim],)
        indices = Tensor(sizeI, from_numpy_dtype(glob_vars.int_type))
    else:
        indices = None

    if return_counts:
        counts_tensor = Tensor()
    else:
        counts_tensor = None
    counts_ptr = TensorP(counts_tensor)

    func = check_function("diopiUnique")
    ret = (
        func(
            input.context(),
            out_ptr,
            input,
            sorted,
            return_counts,
            indices,
            counts_ptr,
        )
        if dim is None
        else func(
            input.context(),
            out_ptr,
            input,
            dim,
            sorted,
            return_counts,
            indices,
            counts_ptr,
        )
    )
    check_returncode(ret)

    out = out_ptr.data()
    if return_counts:
        counts = counts_ptr.data()
    if return_inverse and not return_counts:
        return out, indices
    elif not return_inverse and return_counts:
        return out, counts
    elif return_inverse and return_counts:
        return out, indices, counts
    else:
        return out


def prod(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    assert isinstance(dim, (int)) or dim is None, "dim should be int"
    out_dtype = dtype if dtype is not None else promote_type(input, Dtype.int64)

    _, out = reduce_op_process(input, dim, keepdim, out_dtype)

    func = check_function("diopiProd")
    ret = (
        func(input.context(), out, input)
        if dim is None
        else func(input.context(), out, input, dim)
    )
    check_returncode(ret)
    return out


def linear_backward(input, grad_outputs, weight, bias=None, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"

    grad_input = raw_like(input)
    grad_weight = raw_like(weight)
    if bias is not None:
        assert isinstance(bias, Tensor), "bias must be a Tensor"
        grad_bias = raw_like(bias)
        grad_bias_capsule = grad_bias
    else:
        grad_bias_capsule = None

    func = check_function("diopiLinearBackward")

    ret = func(
        input.context(),
        grad_input,
        grad_weight,
        grad_bias_capsule,
        grad_outputs[0],
        input,
        weight,
    )
    check_returncode(ret)
    if bias is None:
        return {"input": grad_input, "weight": grad_weight}
    return {"input": grad_input, "weight": grad_weight, "bias": grad_bias}


def cross_entropy_backward(
    input,
    grad_outputs,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    label_smoothing=0.0,
    **kwargs,
):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    assert reduction in [
        "mean",
        "sum",
        "none",
    ], "reduction must be one of (mean, sum, none)"

    grad_input = raw_like(input)
    if weight is not None:
        assert isinstance(weight, Tensor), "weigth must be a Tensor"
        weight = weight
    else:
        weight = None

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiCrossEntropyLossBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_outputs[0],
        input,
        target,
        weight,
        reduction_mode,
        ignore_index,
        label_smoothing,
    )
    check_returncode(ret)
    return {"input": grad_input}


def erfinv(input, inplace=False) -> Tensor:
    return unary_op(input, inplace, "diopiErfinv")


def im2col(input, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    sizeI = input.size().data
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
        num_blocks *= (
            int(
                (sizeI[i + 2] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1)
                / stride[i]
            )
            + 1
        )
    channels = sizeI[1]
    for i in range(len(kernel_size)):
        channels *= kernel_size[i]
    sizeO = [sizeI[0], channels, num_blocks]

    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    kernel_size = Sizes(list(kernel_size))
    dilation = Sizes(list(dilation))

    out = Tensor(sizeO, input.get_dtype())
    func = check_function("diopiIm2Col")
    ret = func(input.context(), out, input, kernel_size, dilation, padding, stride)
    check_returncode(ret)
    return out


def col2im(input, output_size, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    sizeI = input.size().data
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

    output_size = Sizes(list(output_size))
    stride = Sizes(list(stride))
    padding = Sizes(list(padding))
    kernel_size = Sizes(list(kernel_size))
    dilation = Sizes(list(dilation))

    out = Tensor(sizeO, input.get_dtype())
    func = check_function("diopiCol2Im")
    ret = func(
        input.context(),
        out,
        input,
        output_size,
        kernel_size,
        dilation,
        padding,
        stride,
    )
    check_returncode(ret)
    return out


def flip(input, dims):
    out = raw_like(input)
    dims = Sizes(list(dims))
    func = check_function("diopiFlip")
    ret = func(input.context(), out, input, dims)
    check_returncode(ret)
    return out


def cholesky_ex(input, upper=False, check_errors=False):
    out = raw_like(input)
    sizeI = input.size().data
    nums = sizeI[0:-2] if len(sizeI) > 2 else ()
    info = Tensor(nums, Dtype.int32)
    func = check_function("diopiCholesky")
    ret = func(input.context(), out, info, input, upper, check_errors)
    check_returncode(ret)
    return out, info


def cholesky_ex_backward(input, grad_outputs, output, upper=False, **kwargs):
    assert len(grad_outputs) == 1, "only accept 1 gradient to do backward"
    grad_input = raw_like(input)
    func = check_function("diopiCholeskyBackward")
    ret = func(input.context(), grad_input, grad_outputs[0], output, upper)
    check_returncode(ret)
    return {"input": grad_input}


def triangular_solve(input, A, upper=True, transpose=False, unitriangular=False):
    sizeA = list(A.size().data)
    sizeI = list(input.size().data)
    sizeO = sizeA if len(sizeA) > len(sizeI) else sizeI
    sizeO[-1] = sizeI[-1]
    out = Tensor(sizeO, A.get_dtype())
    sizeO[-1] = sizeA[-1]
    cloned_mat = Tensor(sizeO, A.get_dtype())
    func = check_function("diopiTriangularSolve")
    ret = func(
        input.context(),
        out,
        cloned_mat,
        input,
        A,
        upper,
        transpose,
        unitriangular,
    )
    check_returncode(ret)
    Res = namedtuple("Res", ["solution", "cloned_coefficient"])
    output = Res(out, cloned_mat)
    return output


def triangular_solve_backward(
    input,
    grad_outputs,
    output,
    A,
    upper=True,
    transpose=False,
    unitriangular=False,
    **kwargs,
):
    assert len(grad_outputs) <= 2, "accept at most 2 gradient to do backward"
    grad_cloned_mat = None if len(grad_outputs) == 1 else grad_outputs[1]
    grad_A = raw_like(A)
    grad_input = raw_like(input)
    func = check_function("diopiTriangularSolveBackward")
    ret = func(
        input.context(),
        grad_input,
        grad_A,
        grad_outputs[0],
        grad_cloned_mat,
        output,
        input,
        A,
        upper,
        transpose,
        unitriangular,
    )
    check_returncode(ret)
    return {"input": grad_input, "A": grad_A}


def repeat(input, repeats):
    sizeI = list(input.size().data)
    input_ndims = len(sizeI)
    repeats_size = list(repeats)
    out_ndims = len(repeats)
    assert (
        input_ndims <= out_ndims
    ), f"input_ndims ({input_ndims}) should <= out_ndims ({out_ndims})"

    output_size = []
    for i in range(out_ndims):
        idx = input_ndims + i - out_ndims
        k = repeats_size[i] * sizeI[idx] if idx >= 0 else repeats_size[i]
        output_size.append(k)

    repeats_size = Sizes(list(repeats))

    out = Tensor(output_size, input.get_dtype())
    func = check_function("diopiRepeat")
    ret = func(input.context(), out, input, repeats_size)
    check_returncode(ret)
    return out


def normal(mean, std, size=None, generator=None):
    call = "diopiNormal"
    if isinstance(mean, Tensor) and isinstance(std, Tensor):
        sizeX1 = list(mean.size().data)
        sizeX2 = list(std.size().data)
        if mean.numel() <= std.numel():
            out_size = infer_size(sizeX1, sizeX2)
            out = Tensor(out_size, std.get_dtype())
        if mean.numel() > std.numel():
            out_size = infer_size(sizeX1, sizeX2)
            out = Tensor(out_size, mean.get_dtype())

        call += "Tensor"
    elif isinstance(mean, Tensor):
        out = Tensor(mean.size().data, mean.get_dtype())
        call += "TensorScalar"
    elif isinstance(std, Tensor):
        out = Tensor(std.size().data, std.get_dtype())
        call += "ScalarTensor"
    else:
        if size is not None:
            out = Tensor(size, Dtype.float32)
        else:
            out = Tensor((), Dtype.float32)
    arg_mean = mean if isinstance(mean, Tensor) else mean
    arg_std = std if isinstance(std, Tensor) else std
    func = check_function(call)
    ret = func(out.context(), out, arg_mean, arg_std, generator)
    check_returncode(ret)
    return out


def normal_(input, mean, std, shape=None, generator=None) -> Tensor:
    call = "diopiNormalInp"
    func = check_function(call)
    ret = func(input.context(), input, mean, std, generator)
    check_returncode(ret)
    return input


def meshgrid(tensors, shape=None):
    assert isinstance(tensors, (list, tuple)), "tensors must be a list or tuple"
    inputsNum = len(tensors)
    c_tensors = []
    co_tensors = []
    dims = []
    for tensor in tensors:
        c_tensors.append(TensorP(tensor))
        if tensor.size().len > 0:
            dims.append(tensor.size().data[0])
        else:
            dims.append(1)
    out = [Tensor(dims, tensors[0].get_dtype()) for i in range(inputsNum)]
    for tensor in out:
        co_tensors.append(TensorP(tensor))
    func = check_function("diopiMeshGrid")
    ret = func(tensors[0].context(), co_tensors, c_tensors, inputsNum)
    check_returncode(ret)
    return out


def cast_dtype(input, out) -> Tensor:
    call = "diopiCastDtype"
    func = check_function(call)
    ret = func(input.context(), out, input)
    check_returncode(ret)
    return out


def multinomial(input, num_samples, replacement, generator=None) -> Tensor:
    call = "diopiMultinomial"
    func = check_function(call)
    if len(input.size().data) == 2:
        out = Tensor(size=(input.size().data[0], num_samples), dtype=Dtype.int64)
    if len(input.size().data) == 1:
        out = Tensor(size=(num_samples,), dtype=Dtype.int64)
    ret = func(input.context(), out, input, num_samples, replacement, generator)
    check_returncode(ret)
    return out


def ceil(input, inplace=False) -> Tensor:
    call = "diopiCeil"
    if inplace:
        call += "Inp"
        func = check_function(call)
        ret = func(input.context(), input)
        check_returncode(ret)
        return input
    else:
        out = Tensor(input.size(), input.get_dtype())
        func = check_function(call)
        ret = func(input.context(), out, input)
        check_returncode(ret)
        return out


def polar(abs, angle) -> Tensor:
    call = "diopiPolar"
    out_shape = infer_size(abs.size().data, angle.size().data)
    if abs.get_dtype() == Dtype.float64:
        out = Tensor(out_shape, Dtype.complex128)
    elif abs.get_dtype() == Dtype.float32:
        out = Tensor(out_shape, Dtype.complex64)
    func = check_function(call)
    ret = func(abs.context(), out, abs, angle)

    check_returncode(ret)
    return out


def asin(input, inplace=False) -> Tensor:
    call = "diopiAsin"
    if inplace:
        call += "Inp"
        func = check_function(call)
        ret = func(input.context(), input)
        check_returncode(ret)
        return input
    else:
        dtype = input.get_dtype()
        if dtype != 8 and dtype != 9 and dtype != 10:
            out = Tensor(input.size(), Dtype.float32)
        else:
            out = Tensor(input.size(), input.get_dtype())
        func = check_function(call)
        ret = func(input.context(), out, input)
        check_returncode(ret)
        return out


def lerp(input, end, weight) -> Tensor:
    call = "diopiLerp"
    out_shape = input.size()
    if isinstance(weight, Tensor):
        out_shape = infer_size(list(input.size().data), list(end.size().data))
        out_shape = infer_size(out_shape, list(weight.size().data))
        func = check_function(call + "Tensor")
    else:
        weight = Scalar(weight)
        out_shape = infer_size(list(input.size().data), list(end.size().data))
        func = check_function(call + "Scalar")
    out = Tensor(out_shape, input.get_dtype())
    ret = func(input.context(), out, input, end, weight)
    check_returncode(ret)
    return out


def sgn(input, inplace=False) -> Tensor:
    call = "diopiSgn"
    if inplace:
        call += "Inp"
        func = check_function(call)
        ret = func(input.context(), input)
        check_returncode(ret)
        return input
    else:
        out = Tensor(input.size(), input.get_dtype())
        func = check_function(call)
        ret = func(input.context(), out, input)
        check_returncode(ret)
        return out


def triu(input, diagonal=0, inplace=False) -> Tensor:
    call = "diopiTriu"
    if inplace:
        call += "Inp"
        func = check_function(call)
        ret = func(input.context(), input, diagonal)
        check_returncode(ret)
        return input
    else:
        out = Tensor(input.size(), input.get_dtype())
        func = check_function(call)
        ret = func(input.context(), out, input, diagonal)
        check_returncode(ret)
        return out


def isnan(input) -> Tensor:
    call = "diopiIsNan"
    out = Tensor(input.size(), Dtype.bool)
    func = check_function(call)
    ret = func(input.context(), out, input)
    check_returncode(ret)
    return out


def amax(input, dim, keepdim) -> Tensor:
    call = "diopiAmax"
    func = check_function(call)
    assert (
        isinstance(dim, (int, list, tuple)) or dim is None
    ), "dim should be int or list or tuple or None"
    dim, out = reduce_op_process(input, dim, keepdim)
    dim1 = Sizes(list(dim))
    ret = func(input.context(), out, input, dim1, keepdim)
    check_returncode(ret)
    return out


def vector_norm(input, ord=2, dim=None, keepdim=False, dtype=None):
    call = "diopiLinalgVecNorm"
    func = check_function(call)

    dim, out = reduce_op_process(input, dim, keepdim)
    dimout = Sizes(list(dim))
    ord = Scalar(ord)
    ret = func(input.context(), out, input, ord, dimout, keepdim)
    check_returncode(ret)
    return out


def linalgqr(input, mode):
    call = "diopiLinalgQR"
    func = check_function(call)
    sizeI = list(input.size().data)
    sizeq = []
    sizer = []
    if mode == "reduced":
        if sizeI[-1] <= sizeI[-2]:
            sizeq = sizeI[:]
            sizer = sizeI[:-2] + [sizeI[-1], sizeI[-1]]
        else:
            sizer = sizeI[:]
            sizeq = sizeI[:-2] + [sizeI[-2], sizeI[-2]]
    elif mode == "complete":
        sizeq = sizeI[:-1]
        sizeq.append(sizeI[-2])
        sizer = sizeI[:]
    elif mode == "r":
        sizeq = [0]
        if sizeI[-1] <= sizeI[-2]:
            sizer = sizeI[:-2] + [sizeI[-1], sizeI[-1]]
        else:
            sizer = sizeI[:]
    else:
        raise ValueError("mode do not support.")
    q = Tensor(sizeq, input.get_dtype())
    r = Tensor(sizer, input.get_dtype())
    ret = func(input.context(), input, mode, q, r)
    out = [q, r]
    check_returncode(ret)
    return out


def rotary_emb(input, cos, sin, conj, interleaved):
    call = "diopiRotaryEmbedding"
    func = check_function(call)
    size = list(input.size().data)
    out = Tensor(size, input.get_dtype())
    ret = func(input.context(), out, input, cos, sin, conj, interleaved)
    check_returncode(ret)
    return out


def rms_norm(input, normalized_shape, weight, bias, eps):
    if bias is not None:
        assert isinstance(bias, Tensor), "bias must be a Tensor"
    call = "diopiRMSNorm"
    func = check_function(call)
    out = Tensor(input.size(), input.get_dtype())
    size = list(input.size().data)
    inv_rms_size = size.copy()
    n = len(normalized_shape)
    inv_rms_size = size[:-n]
    inv_dtype = input.get_dtype()
    # when input_dtype is bfloat16 or float16, inv_dtype is float32
    if inv_dtype == Dtype.float16:
        inv_dtype = Dtype.float32
    inv_rms = Tensor(inv_rms_size, inv_dtype)
    # When weight and bias exist, its shape is equal to normalized_shape.
    # If not specified, normalized_shape generally defaults to the size of the last dimension of the input tensor.
    normalized_shape = Sizes(normalized_shape)
    ret = func(
        input.context(),
        out,
        inv_rms,
        input,
        normalized_shape,
        weight,
        bias,
        eps,
    )
    check_returncode(ret)
    GLOBAL_STATE["rms_norm_inv_rms"] = inv_rms
    return out


def rms_norm_backward(grad_outputs, input, weight, bias, normalized_shape, eps):
    call = "diopiRMSNormBackward"
    func = check_function(call)
    grad_input = Tensor(list(input.size().data), input.get_dtype())
    grad_weight = Tensor(list(weight.size().data), weight.get_dtype())
    if bias is not None:
        assert isinstance(bias, Tensor), "bias must be a Tensor"
        grad_bias = raw_like(bias)
    else:
        grad_bias = None
    # When weight and bias exist, its shape is equal to normalized_shape.
    # If not specified, normalized_shape generally defaults to the size of the last dimension of the input tensor.
    normalized_shape = Sizes(normalized_shape)

    inv_rms = GLOBAL_STATE.pop('rms_norm_inv_rms')
    ret = func(input.context(), grad_input, grad_weight, grad_bias, grad_outputs[0], input, weight, bias, inv_rms,
               normalized_shape, eps)
    check_returncode(ret)
    if bias is None:
        return {"input": grad_input, "weight": grad_weight}
    return {"input": grad_input, "weight": grad_weight, "bias": grad_bias}


def multihead_attention(
    q, k, v, dropout_p, is_causal, return_debug_mask, scale, generator=None
):
    call = "diopiMultiHeadAttention"
    func = check_function(call)
    q_size = list(q.size().data)
    out = Tensor(q_size, q.get_dtype())
    softmax_lse = Tensor([q_size[0], q_size[2], q_size[1]], dtype=Dtype.float32)
    debug_attn_mask = Tensor([0], q.get_dtype())
    softmax_scale = 1.0 / math.sqrt(q.shape().data[-1]) if not scale else scale
    try:
        ret = func(
            q.context(),
            q,
            k,
            v,
            dropout_p,
            is_causal,
            return_debug_mask,
            softmax_scale,
            out,
            softmax_lse,
            generator,
            debug_attn_mask,
        )
    except RuntimeError as rte:
        if (
            "unable to call flash mha_fwd: DIOPI is built without flash-attention"
            == rte.args[0]
        ):
            raise FunctionNotImplementedError
        else:
            raise RuntimeError("execute diopiMultiHeadAttention failed!")
    else:
        check_returncode(ret)
        GLOBAL_STATE["multihead_attention_softmax_lse"] = softmax_lse
        GLOBAL_STATE["multihead_attention_generator"] = generator
        return out


def multihead_attention_backward(
    q, k, v, out, grad_outputs, dropout_p, is_causal, scale, **kwargs
):
    call = "diopiMultiHeadAttentionBackward"
    func = check_function(call)
    grad_q = raw_like(q)
    grad_k = raw_like(k)
    grad_v = raw_like(v)
    softmax_lse = GLOBAL_STATE.pop("multihead_attention_softmax_lse")
    generator = GLOBAL_STATE.pop("multihead_attention_generator")
    softmax_scale = 1.0 / math.sqrt(q.shape().data[-1]) if not scale else scale
    try:
        ret = func(
            q.context(),
            grad_outputs[0],
            q,
            k,
            v,
            out,
            softmax_lse,
            dropout_p,
            is_causal,
            generator,
            softmax_scale,
            grad_q,
            grad_k,
            grad_v,
        )
    except RuntimeError as rte:
        if (
            "unable to call flash mha_bwd: DIOPI is built without flash-attention"
            == rte.args[0]
        ):
            raise FunctionNotImplementedError
        else:
            raise RuntimeError("execute diopiMultiHeadAttentionBackward failed!")
    else:
        check_returncode(ret)
        return {"q": grad_q, "k": grad_k, "v": grad_v}


def multihead_attention_varlen(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    dropout_p,
    is_causal,
    return_debug_mask,
    scale,
    generator=None,
):
    call = "diopiMultiHeadAttentionVarLen"
    func = check_function(call)
    q_size = list(q.size().data)
    out = Tensor(q_size, q.get_dtype())
    softmax_lse = Tensor(
        [len(cu_seqlens) - 1, q_size[1], max_seqlen], dtype=Dtype.float32
    )
    cu_seqlens = Tensor.from_numpy(np.array(cu_seqlens, dtype=np.int32))
    debug_attn_mask = Tensor([0], q.get_dtype())
    softmax_scale = 1.0 / math.sqrt(q.shape().data[-1]) if not scale else scale

    try:
        ret = func(
            q.context(),
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            is_causal,
            return_debug_mask,
            softmax_scale,
            out,
            softmax_lse,
            generator,
            debug_attn_mask,
        )
    except RuntimeError as rte:
        if (
            "unable to call flash mha_varlen_fwd: DIOPI is built without flash-attention"
            == rte.args[0]
        ):
            raise FunctionNotImplementedError
        else:
            raise RuntimeError("execute diopiMultiHeadAttentionVarLen failed!")
    else:
        check_returncode(ret)
        GLOBAL_STATE["multihead_attention_varlen_softmax_lse"] = softmax_lse
        GLOBAL_STATE["multihead_attention_varlen_generator"] = generator
        return out


def multihead_attention_varlen_backward(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    out,
    grad_outputs,
    dropout_p,
    is_causal,
    scale,
    **kwargs,
):
    call = "diopiMultiHeadAttentionVarLenBackward"
    func = check_function(call)
    grad_q = raw_like(q)
    grad_k = raw_like(k)
    grad_v = raw_like(v)
    softmax_lse = GLOBAL_STATE.pop("multihead_attention_varlen_softmax_lse")
    cu_seqlens = Tensor.from_numpy(np.array(cu_seqlens, dtype=np.int32))
    generator = GLOBAL_STATE.pop("multihead_attention_varlen_generator")
    softmax_scale = 1.0 / math.sqrt(q.shape().data[-1]) if not scale else scale
    try:
        ret = func(
            q.context(),
            grad_outputs[0],
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            is_causal,
            generator,
            softmax_scale,
            grad_q,
            grad_k,
            grad_v,
        )
    except RuntimeError as rte:
        if (
            "unable to call flash mha_varlen_bwd: DIOPI is built without flash-attention"
            == rte.args[0]
        ):
            raise FunctionNotImplementedError
        else:
            raise RuntimeError("execute diopiMultiHeadAttentionVarLenBackward failed!")
    else:
        check_returncode(ret)
        return {"q": grad_q, "k": grad_k, "v": grad_v}


# todo: impl for diopiFlashAttentionV2
def flash_attention_v1(
    q, k, v, p_dropout, softmax_scale, is_causal, head_num, input_layout
):
    call = "diopiFlashAttention"
    func = check_function(call)
    out = raw_like(q)
    if is_causal:
        attention_mask = Tensor()
    else:
        attention_mask = None
    if p_dropout > 0 and p_dropout <= 1:
        dropout_mask = Tensor()
        state = build_generator_state(q.context())
        generator = Generator(state)
    elif p_dropout == 0:
        dropout_mask = None
        generator = None
    else:
        assert 0, "The p_dropout value must be in range of [0, 1]"
    softmax_max = Tensor()
    softmax_sum = Tensor()
    softmax_out = Tensor()
    attention_mask_ptr = TensorP(attention_mask)
    dropout_mask_ptr = TensorP(dropout_mask)
    softmax_max_ptr = TensorP(softmax_max)
    softmax_sum_ptr = TensorP(softmax_sum)
    softmax_out_ptr = TensorP(softmax_out)
    if input_layout == "SBH":
        head_dim = q.shape().data[-1] / head_num
    elif input_layout == "BSH":
        head_dim = q.shape().data[-1] / head_num
    elif input_layout == "BNSD":
        head_dim = q.shape().data[-1]
    elif input_layout == "BSND":
        head_dim = q.shape().data[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim) if not softmax_scale else softmax_scale
    ret = func(
        q.context(),
        out,
        attention_mask_ptr,
        dropout_mask_ptr,
        softmax_max_ptr,
        softmax_sum_ptr,
        softmax_out_ptr,
        generator,
        q,
        k,
        v,
        p_dropout,
        softmax_scale,
        is_causal,
        head_num,
        input_layout,
    )
    check_returncode(ret)
    GLOBAL_STATE["flash_attention_attention_mask"] = attention_mask
    GLOBAL_STATE["flash_attention_dropout_mask"] = dropout_mask
    GLOBAL_STATE["flash_attention_softmax_max"] = softmax_max
    GLOBAL_STATE["flash_attention_softmax_sum"] = softmax_sum
    GLOBAL_STATE["flash_attention_softmax_out"] = softmax_out
    return out


def flash_attention_v1_backward(
    q,
    k,
    v,
    out,
    grad_outputs,
    p_dropout,
    softmax_scale,
    is_causal,
    head_num,
    input_layout,
):
    call = "diopiFlashAttentionBackward"
    func = check_function(call)
    assert (
        p_dropout >= 0 and p_dropout <= 1
    ), "The p_dropout value must be in range of [0, 1]"
    grad_q = raw_like(q)
    grad_k = raw_like(k)
    grad_v = raw_like(v)
    attention_mask = GLOBAL_STATE.pop("flash_attention_attention_mask")
    dropout_mask = GLOBAL_STATE.pop("flash_attention_dropout_mask")
    softmax_max = GLOBAL_STATE.pop("flash_attention_softmax_max")
    softmax_sum = GLOBAL_STATE.pop("flash_attention_softmax_sum")
    softmax_out = GLOBAL_STATE.pop("flash_attention_softmax_out")
    if input_layout == "SBH":
        head_dim = q.shape().data[-1] / head_num
    elif input_layout == "BSH":
        head_dim = q.shape().data[-1] / head_num
    elif input_layout == "BNSD":
        head_dim = q.shape().data[-1]
    elif input_layout == "BSND":
        head_dim = q.shape().data[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim) if not softmax_scale else softmax_scale
    ret = func(
        q.context(),
        grad_q,
        grad_k,
        grad_v,
        grad_outputs[0],
        q,
        k,
        v,
        out,
        attention_mask,
        dropout_mask,
        softmax_max,
        softmax_sum,
        softmax_out,
        p_dropout,
        softmax_scale,
        head_num,
        input_layout,
    )
    check_returncode(ret)
    return {"q": grad_q, "k": grad_k, "v": grad_v}


def flash_attention_v3(q, k, v, p_dropout, softmax_scale, is_causal):
    call = "diopiFlashAttentionV3"
    func = check_function(call)
    out = raw_like(q)
    if p_dropout >= 0 and p_dropout <= 1:
        state = build_generator_state(q.context())
        generator = Generator(state)
    else:
        assert 0, "The p_dropout value must be in range of [0, 1]"
    q_size = list(q.size().data)
    head_dim = q_size[-1]
    softmax_lse = Tensor([q_size[0], q_size[2], q_size[1]], dtype=Dtype.float32)
    softmax_scale = 1.0 / math.sqrt(head_dim) if not softmax_scale else softmax_scale
    ret = func(
        q.context(),
        out,
        softmax_lse,
        generator,
        q,
        k,
        v,
        p_dropout,
        softmax_scale,
        is_causal,
    )
    check_returncode(ret)
    GLOBAL_STATE["flash_attention_v3_softmax_lse"] = softmax_lse
    GLOBAL_STATE["flash_attention_v3_generator"] = generator
    return out

def flash_attention_v3_backward(q, k, v, out, grad_outputs, p_dropout, softmax_scale, is_causal):
    call = "diopiFlashAttentionV3Backward"
    func = check_function(call)
    assert p_dropout >=0 and p_dropout <=1, "The p_dropout value must be in range of [0, 1]"
    grad_q = raw_like(q)
    grad_k = raw_like(k)
    grad_v = raw_like(v)
    q_size = list(q.size().data)
    head_dim = q_size[-1]
    softmax_lse = GLOBAL_STATE.pop('flash_attention_v3_softmax_lse')
    generator = GLOBAL_STATE.pop('flash_attention_v3_generator')
    softmax_scale = 1.0 / math.sqrt(head_dim) if not softmax_scale else softmax_scale
    ret = func(q.context(), grad_q, grad_k, grad_v, grad_outputs[0], generator, q, k, v, out, softmax_lse, p_dropout, softmax_scale, is_causal)
    check_returncode(ret)
    return {'q': grad_q, 'k': grad_k, 'v': grad_v}

def flash_attention_varlen(q, k, v, max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv, p_dropout, softmax_scale, is_causal):
    call = "diopiFlashAttentionVarLen"
    func = check_function(call)
    q_size = list(q.size().data)
    out = Tensor(q_size, q.get_dtype())
    cu_seqlens_q = Sizes(cu_seqlens_q[1:])
    cu_seqlens_kv = Sizes(cu_seqlens_kv[1:])
    if is_causal:
        attention_mask = Tensor()
    else:
        attention_mask = None
    if p_dropout > 0 and p_dropout <= 1:
        dropout_mask = Tensor()
        state = build_generator_state(q.context())
        generator = Generator(state)
    elif p_dropout == 0:
        dropout_mask = None
        generator = None
    else:
        assert 0, "The p_dropout value must be in range of [0, 1]"
    softmax_max = Tensor()
    softmax_sum = Tensor()
    softmax_out = Tensor()
    attention_mask_ptr = TensorP(attention_mask)
    dropout_mask_ptr = TensorP(dropout_mask)
    softmax_max_ptr = TensorP(softmax_max)
    softmax_sum_ptr = TensorP(softmax_sum)
    softmax_out_ptr = TensorP(softmax_out)
    softmax_scale = 1.0 / math.sqrt(q.shape().data[-1]) if not softmax_scale else softmax_scale
    ret = func(
        q.context(),
        out,
        attention_mask_ptr,
        dropout_mask_ptr,
        softmax_max_ptr,
        softmax_sum_ptr,
        softmax_out_ptr,
        generator,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        p_dropout,
        softmax_scale,
        is_causal,
    )
    check_returncode(ret)
    GLOBAL_STATE["flash_attention_varlen_attention_mask"] = attention_mask
    GLOBAL_STATE["flash_attention_varlen_dropout_mask"] = dropout_mask
    GLOBAL_STATE["flash_attention_varlen_softmax_max"] = softmax_max
    GLOBAL_STATE["flash_attention_varlen_softmax_sum"] = softmax_sum
    GLOBAL_STATE["flash_attention_varlen_softmax_out"] = softmax_out
    return out

def flash_attention_varlen_backward(q, k, v, out, grad_outputs, max_seqlen_q, max_seqlen_kv, cu_seqlens_q, cu_seqlens_kv, p_dropout, softmax_scale, is_causal):
    call = "diopiFlashAttentionVarLenBackward"
    func = check_function(call)
    assert p_dropout >=0 and p_dropout <=1, "The p_dropout value must be in range of [0, 1]"
    head_dim = q.shape().data[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim) if not softmax_scale else softmax_scale
    cu_seqlens_q = Sizes(cu_seqlens_q[1:])
    cu_seqlens_kv = Sizes(cu_seqlens_kv[1:])
    grad_q = raw_like(q)
    grad_k = raw_like(k)
    grad_v = raw_like(v)
    attention_mask = GLOBAL_STATE.pop("flash_attention_varlen_attention_mask")
    dropout_mask = GLOBAL_STATE.pop("flash_attention_varlen_dropout_mask")
    softmax_max = GLOBAL_STATE.pop("flash_attention_varlen_softmax_max")
    softmax_sum = GLOBAL_STATE.pop("flash_attention_varlen_softmax_sum")
    softmax_out = GLOBAL_STATE.pop("flash_attention_varlen_softmax_out")
    ret = func(
        q.context(),
        grad_q,
        grad_k,
        grad_v,
        grad_outputs[0],
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        out,
        attention_mask,
        dropout_mask,
        softmax_max,
        softmax_sum,
        softmax_out,
        max_seqlen_q,
        max_seqlen_kv,
        p_dropout,
        softmax_scale,
    )
    check_returncode(ret)
    return out

def attention(query, key, value, attn_mask=None, attn_bias=None, dropout_p=0.0, is_causal=False, scale=None):
    func = check_function("diopiAttention")
    attn_out = raw_like(query)
    max_tensor_num_for_backward = 16
    save_for_backward_array = [Tensor() for i in range(max_tensor_num_for_backward)]
    save_for_backward = [TensorP(tensor) for tensor in save_for_backward_array]
    save_tensor_num = ctypes.c_long(max_tensor_num_for_backward)
    generator = Generator(build_generator_state(query.context()))
    if scale is None:
        scale = 1.0 / math.sqrt(query.size().data[-1])
    ret = func(query.context(), attn_out, save_for_backward, get_capsule(byref(save_tensor_num)),
               query, key, value, attn_mask, attn_bias, dropout_p, generator, scale, is_causal)
    check_returncode(ret)
    save_for_backward_tensor_list = []
    for i in range(save_tensor_num.value):
        save_for_backward_tensor_list.append(save_for_backward[i])

    GLOBAL_STATE["attn_bias"] = attn_bias
    GLOBAL_STATE["attention_save_tensor_num"] = save_tensor_num.value
    GLOBAL_STATE["save_for_backward_tensor_list"] = save_for_backward_tensor_list
    return attn_out


def attention_backward(
    query,
    key,
    value,
    out,
    grad_outputs,
    dropout_p,
    scale,
    is_causal,
    attn_bias
):
    call = "diopiAttentionBackward"
    func = check_function(call)
    assert (
        dropout_p >= 0 and dropout_p <= 1
    ), "The p_dropout value must be in range of [0, 1]"
    grad_q = raw_like(query)
    grad_k = raw_like(key)
    grad_v = raw_like(value)
    grad_attn_bias = None
    if attn_bias is not None:
        grad_attn_bias = raw_like(attn_bias)
    save_tensor_num = GLOBAL_STATE.pop("attention_save_tensor_num")
    save_for_backward_tensor_list = GLOBAL_STATE.pop("save_for_backward_tensor_list")

    save_for_backward = [TensorP(tensor) for tensor in save_for_backward_tensor_list]
    generator = Generator(build_generator_state(query.context()))
    head_dim = query.shape().data[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim) if not scale else scale
    ret = func(
        query.context(),
        grad_q,
        grad_k,
        grad_v,
        grad_attn_bias,
        grad_outputs[0],
        query,
        key,
        value,
        out,
        save_for_backward,
        save_tensor_num,
        dropout_p,
        generator,
        softmax_scale
    )
    check_returncode(ret)
    return {"query": grad_q, "key": grad_k, "value": grad_v}


def attention_varlen(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    attn_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    func = check_function("diopiAttentionVarLen")
    attn_out = raw_like(query)
    max_tensor_num_for_backward = 16
    save_for_backward_array = [Tensor() for i in range(max_tensor_num_for_backward)]
    save_for_backward = [TensorP(tensor) for tensor in save_for_backward_array]
    save_tensor_num = ctypes.c_long(max_tensor_num_for_backward)
    generator = Generator(build_generator_state(query.context()))
    if scale is None:
        scale = 1.0 / math.sqrt(query.size().data[-1])
    ret = func(
        query.context(),
        attn_out,
        save_for_backward,
        get_capsule(byref(save_tensor_num)),
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        attn_mask,
        attn_bias,
        dropout_p,
        generator,
        scale,
        is_causal,
    )
    check_returncode(ret)
    save_for_backward_tensor_list = []
    for i in range(save_tensor_num.value):
        save_for_backward_tensor_list.append(save_for_backward[i])

    GLOBAL_STATE["attn_bias"] = attn_bias
    GLOBAL_STATE["attention_save_tensor_num"] = save_tensor_num.value
    GLOBAL_STATE["save_for_backward_tensor_list"] = save_for_backward_tensor_list
    return attn_out


def attention_varlen_backward(
    query,
    key,
    value,
    out,
    cu_seqlens_q,
    cu_seqlens_kv,
    max_seqlen_q,
    max_seqlen_kv,
    grad_outputs,
    attn_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
):
    call = "diopiAttentionVarLenBackward"
    func = check_function(call)
    assert (
        dropout_p >= 0 and dropout_p <= 1
    ), "The p_dropout value must be in range of [0, 1]"
    grad_q = raw_like(query)
    grad_k = raw_like(key)
    grad_v = raw_like(value)
    grad_attn_bias = None
    if attn_bias is not None:
        #grad_attn_bias = raw_like(attn_bias)
        pass
    save_tensor_num = GLOBAL_STATE.pop("attention_save_tensor_num")
    save_for_backward_tensor_list = GLOBAL_STATE.pop("save_for_backward_tensor_list")

    save_for_backward = [TensorP(tensor) for tensor in save_for_backward_tensor_list]
    generator = Generator(build_generator_state(query.context()))
    head_dim = query.shape().data[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim) if not scale else scale
    ret = func(
        query.context(),
        grad_q,
        grad_k,
        grad_v,
        grad_attn_bias,
        grad_outputs[0],
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_kv,
        out,
        save_for_backward,
        save_tensor_num,
        dropout_p,
        generator,
        softmax_scale
    )
    check_returncode(ret)
    return {"query": grad_q, "key": grad_k, "value": grad_v}


def scaled_masked_softmax(input, mask, scale, fixed_triu_mask):
    call = "diopiScaledMaskedSoftmax"
    func = check_function(call)
    size = list(input.size().data)
    out = Tensor(size, input.get_dtype())
    ret = func(
        input.context(),
        out,
        input,
        mask,
        scale,
        fixed_triu_mask,
    )
    check_returncode(ret)
    return out


def scaled_masked_softmax_backward(
    grad_outputs, out, input, mask, scale, fixed_triu_mask
):
    call = "diopiScaledMaskedSoftmaxBackward"
    func = check_function(call)
    size = list(out.size().data)
    grad_input = Tensor(size, out.get_dtype())
    ret = func(
        out.context(),
        grad_input,
        grad_outputs[0],
        out,
        mask,
        scale,
        fixed_triu_mask,
    )
    check_returncode(ret)
    return {"input": grad_input}


def apply_penalty(
    logits,
    presence_penalty,
    frequency_penalty,
    p_token_ids,
    p_token_counts,
    p_cumsum_seq_len,
    p_max_len_in_batch,
):
    call = "diopiApplyPenalty"
    func = check_function(call)
    # some checks
    logits_shape = list(logits.size().data)
    presence_penalty_shape = list(presence_penalty.size().data)
    frequency_penalty_shape = list(frequency_penalty.size().data)
    p_cumsum_seq_len_shape = list(p_cumsum_seq_len.size().data)
    p_token_ids_shape = list(p_token_ids.size().data)
    p_token_counts_shape = list(p_token_counts.size().data)

    assert (
        logits_shape[0] == presence_penalty_shape[0]
        and logits_shape[0] == frequency_penalty_shape[0]
    ), "The first dimensions of logits, presence penalty, and frequency penalty must be equal."
    assert (
        logits_shape[0] + 1 == p_cumsum_seq_len_shape[0]
    ), "The first dimension of logits_shape plus one must equal the first dimension of p_cumsum_seq_len_shape."
    assert (
        p_token_ids_shape == p_token_counts_shape
    ), "The shape of p_token_ids must be equal to the shape of p_token_counts."

    ret = func(
        logits.context(),
        logits,
        presence_penalty,
        frequency_penalty,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        p_max_len_in_batch,
    )
    out = logits
    check_returncode(ret)
    return out


def destindex_copy_kv(k, dest_loc, out):
    call = "diopiDestIndexCopyKV"
    func = check_function(call)

    ret = func(k.context(), out, k, dest_loc)
    check_returncode(ret)
    return out


def token_attention(q, k, out, b_loc, b_start_loc, b_seq_len, max_input_len):
    call = "diopiTokenAttentionInference"
    func = check_function(call)

    ret = func(q.context(), out, q, k, b_loc, b_start_loc, b_seq_len, max_input_len)
    check_returncode(ret)
    return out


def token_softmax_reducev(
    logics,
    v,
    out,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_input_len,
    other_kv_index,
):
    call = "diopiTokenSoftmaxReduceVInference"
    func = check_function(call)

    ret = func(
        logics.context(),
        out,
        logics,
        v,
        b_loc,
        b_start_loc,
        b_seq_len,
        max_input_len,
        other_kv_index,
    )
    check_returncode(ret)
    return out


def context_attention(q, k, v, out, b_start_loc, b_seq_len, max_input_len):
    call = "diopiContextAttentionInference"
    func = check_function(call)

    ret = func(q.context(), out, q, k, v, b_start_loc, b_seq_len, max_input_len)
    check_returncode(ret)
    return out
