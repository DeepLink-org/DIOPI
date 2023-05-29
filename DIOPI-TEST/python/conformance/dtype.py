# Copyright (c) 2023, DeepLink.
from ctypes import (
    c_int64,
    c_uint64,
    c_int32,
    c_uint32,
    c_int16,
    c_uint16,
    c_int8,
    c_uint8,
    c_float,
    c_double,
)
from diopi_runtime import Dtype
from enum import Enum


all_types = [Dtype.float16, Dtype.float32, Dtype.float64, Dtype.int32, Dtype.int64]
float_types = [Dtype.float16, Dtype.float32, Dtype.float64]
float_no_half_types = [Dtype.float32, Dtype.float64]
int_types = [Dtype.int32, Dtype.int64]
complex_types = [Dtype.complex64, Dtype.complex128]
default = all_types


def from_dtype_str(dtype_str: str) -> Dtype:
    if dtype_str == 'int8':
        return Dtype.int8
    elif dtype_str == 'uint8':
        return Dtype.uint8
    elif dtype_str == 'int16':
        return Dtype.int16
    elif dtype_str == 'uint16':
        return Dtype.uint16
    elif dtype_str == 'int32':
        return Dtype.int32
    elif dtype_str == 'uint32':
        return Dtype.uint32
    elif dtype_str == 'int64':
        return Dtype.int64
    elif dtype_str == 'uint64':
        return Dtype.uint64
    elif dtype_str == 'float16':
        return Dtype.float16
    elif dtype_str == 'float32':
        return Dtype.float32
    elif dtype_str == 'float64':
        return Dtype.float64
    elif dtype_str == 'bool':
        return Dtype.bool
    else:
        return None
