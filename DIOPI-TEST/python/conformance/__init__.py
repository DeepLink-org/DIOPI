from .diopi_runtime import Tensor, Context, Dtype, Device, raw_like
from .diopi_runtime import get_last_error, device
from .diopi_runtime import from_numpy_dtype, to_numpy_dtype
from .diopi_runtime import diopirt_lib, device_impl_lib
from .diopi_configs import diopi_configs
from .config import Config
from .gen_data import GenInputData, GenOutputData
from .conformance_test import ConformanceTest


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

bool = Dtype.bool


__all__ = [
    'diopirt_lib',
    'device_impl_lib',
    'Config',
    'GenInputData',
    'GenOutputData',
    'diopi_configs',
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
]

