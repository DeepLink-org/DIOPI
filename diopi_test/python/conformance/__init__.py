# Copyright (c) 2023, DeepLink.
import os
import sys
from diopilib import diopiTensor, Context, Device, Dtype
from .diopi_runtime import Tensor, raw_like
from .diopi_runtime import get_last_error, device
from .diopi_runtime import from_numpy_dtype, to_numpy_dtype, float_types, int_types
# from .diopi_configs import diopi_configs, ops_with_states
# from .config import Config
# from .conformance_test import ConformanceTest
from .skip import Skip 

sys.path.append(os.path.dirname(__file__))

int8 = Dtype.int8
uint8 = Dtype.uint8
int16 = Dtype.int16
uint16 = Dtype.uint16
int32 = Dtype.int32
uint32 = Dtype.uint32
int64 = Dtype.int64
uint64 = Dtype.uint64
float16 = Dtype.float16
float32 = Dtype.float32
float64 = Dtype.float64
bfloat16 = Dtype.bfloat16
tfloat32 = Dtype.tfloat32
complex64 = Dtype.complex64
complex128 = Dtype.complex128
bool = Dtype.bool


__all__ = [
    'Config',
    'diopi_configs',
    'ops_with_states',
    'Tensor',
    'device',
    'Dtype',
    'Device',
    'Context',
    'raw_like',
    'get_last_error',
    'from_numpy_dtype',
    'to_numpy_dtype',
    'ConformanceTest',
    'diopiTensor',
    'float_types',
    'int_types',
    'db_operation'
]
