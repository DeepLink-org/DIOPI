from ctypes import (
    c_void_p,
    c_char_p,
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
from enum import Enum, unique


@unique
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


def check_return_value(returncode, throw_exception=True):
    if returncode == 0:
        return
    elif returncode == 1000:
        error_info = f"returncode {returncode}: dtype is not supported."
        if throw_exception:
            assert returncode == 0, error_info
        else:
            print(error_info)
            return

    assert returncode == 0, f"returncode :{returncode}"
