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
from enum import Enum


class Dtype(Enum):
    int8 = 0
    uint8 = 1
    int16 = 2
    uint16 = 3
    int32 = 4
    uint32 = 5
    int64 = 6
    uint64 = 7
    float16 = 8
    float32 = 9
    float64 = 10
    bool = 11
    bfloat16 = 12
    tfloat32 = 13

    all_types = [float16, float32, float64, int32, int64]
    float_types = [float16, float32, float64]
    float_no_half_types = [float32, float64]
    int_types = [int32, int64]
    default = all_types


def dtype_to_ctype(dtype):
    if dtype == Dtype.float32:
        return c_float
    if dtype == Dtype.float64:
        return c_double
    if dtype == Dtype.int8:
        return c_int8
    if dtype == Dtype.uint8:
        return c_uint8
    if dtype == Dtype.int16:
        return c_int16
    if dtype == Dtype.uint16:
        return c_uint16
    if dtype == Dtype.int32:
        return c_int32
    if dtype == Dtype.uint32:
        return c_uint32
    if dtype == Dtype.int64:
        return c_int64
    if dtype == Dtype.uint64:
        return c_uint64


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
