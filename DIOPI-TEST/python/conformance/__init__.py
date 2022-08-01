from .litert import Tensor, device, TensorHandle, Context, Dtype
from .litert import check_return_value, get_last_error
from .litert import diopirt_lib, device_impl_lib
from .testcase_configs import configs
from .testcase_parse import CaseCollection
from .gen_inputs import GenData


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
    'CaseCollection',
    'GenData',
    'configs',
    'Tensor',
    'device',
    'Dtype',
    'Context',
    'TensorHandle',
    'get_last_error',
    'check_return_value',
]
