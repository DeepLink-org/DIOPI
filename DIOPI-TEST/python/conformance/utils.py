import logging
from . import diopi_runtime
from .diopi_runtime import get_last_error
from .dtype import Dtype
import os
import numpy as np
import csv


default_cfg_dict = dict(
    default_option=dict(
        atol=1e-5,
        rtol=1e-5,
        atol_half=1e-2,
        rtol_half=5e-2,
        memory_format="NCHW",
        fp16_exact_match=False,
        train=True,
    ),
    # set log_level = "DEBUG" for debug infos
    log_level="INFO"  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICA
)
np.set_printoptions(precision=5,
                    threshold=100,
                    edgeitems=3,
                    linewidth=275,
                    suppress=True,
                    formatter=None)

error_counter = [0]
error_content = []
error_content_other = []

# Note : 1. aten's cuda implementation doesn't support 3-dim nhwc Tensor
#        adaptive_max_pool2d（3d), max_pool3d,
#        adaptive_avg_pool3d, interpolate doesn't support nhwc memory format
#        avg_pool2d backward can't compute right along the edges
#        2. For camb test, adaptive_max_pool2d/max_pool2d need indices being int32
#        Only conv2d, bn, adaptive_avg_pool2d, adaptive_max_pool2d can be tested, because
#        the rest have't been implemented.
nhwc_op = {'conv2d': ["2d", "input", 'weight'],
           'conv3d': ["3d", "input", 'weight'],
           'batch_norm': ['input'],
           'adaptive_avg_pool2d': ["2d", 'input'],
           'adaptive_max_pool2d': ["2d", 'input'],
           'adaptive_avg_pool3d': ["3d", 'input'],
           'adaptive_max_pool3d': ["3d", 'input'],
           'avg_pool2d': ["2d", 'input'],
           'max_pool2d': ["2d", 'input'],
           # 'avg_pool3d': ["3d", 'input'], diopi doesn't hava avg_pool3d test
           'max_pool3d': ["3d", 'input'],
           # both embedding
           'interpolate': ['input'],
           'pad': ['input'],
           'roi_align': ['input']}

# Note : 1. camb test: all ops implemented is passed.
#        2. nv test: most of ops is not implemented for 'Int'.
#           Tests of index_select, bce, embedding passed for 'Int'.

dtype_op = {'nll_loss': ['target'],  # input using int32/float32 type
            'cross_entropy': ['target'],
            'index_select': ['index'],
            'index_put': ['indices1', 'indices2'],
            'binary_cross_entropy_with_logits': ['pos_weight'],
            'gather': ['index'],
            'scatter': ['index'],
            'embedding': ['input'],
            'index': ['idx1', 'idx2'],
            'ctc_loss': ['targets', 'input_lengths', 'target_lengths'],
            'index_fill': ['index'],
            'one_hot': ['input']}

# Note : 1. camb test: all ops implemented is passed.
#        2. nv test: most of ops is not implemented for 'Int'.
#           Tests of unique, arange, randperm, argmax passed for 'Int'.
dtype_out_op = {'max_pool2d': ['indices'],  # out using int32/float32 type
                'max_pool3d': ['indices'],
                'adaptive_max_pool2d': ['indices'],
                'adaptive_max_pool3d': ['indices'],
                'max': ['indices'],
                'min': ['indices'],
                'sort': ['indices'],
                'topk': ['indices'],
                'unique': ['indices'],
                'one_hot': ['out'],
                'arange': ['out'],
                'randperm': ['out'],
                'argmax': ['out']}


class glob_var(object):
    def __init__(self, nhwc=False, nhwc_min_dim=3, four_bytes=False):
        self.nhwc = nhwc
        self.nhwc_min_dim = nhwc_min_dim
        self.four_bytes = four_bytes
        self.int_type = Dtype.int64
        self.float_type = Dtype.float64

    def set_nhwc(self):
        self.nhwc = True

    def set_nhwc_min_dim(self, dim):
        self.nhwc_min_dim = dim

    def get_nhwc(self):
        return self.nhwc

    def set_four_bytes(self):
        self.four_bytes = True
        self.int_type = Dtype.int32
        self.float_type = Dtype.float32

    def get_four_bytes(self):
        return self.four_bytes


glob_vars = glob_var()


class Log(object):
    def __init__(self, level):
        self.logger = logging.getLogger("ConformanceTest")
        self.logger.setLevel(level)
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger


def wrap_logger_error(func):
    def inner(*args, tag=[], info=[], **kwargs):
        if args[0].startswith("NotImplemented") or \
                args[0].startswith("Skipped") or \
                args[0].startswith("AttributeError"):
            error_content_other.append(args[0] + "\n")
            return func(*args, **kwargs)
        global error_counter
        error_counter[0] += 1
        tag = str(tag).replace("'", "")
        info = str(info).replace("'", "")
        error_content.append(f"{error_counter[0]}--{args[0]}.   TestTag: {tag}  TensorInfo : {info}\n")
        error_content.append("---------------------------------\n")
        func(*args, **kwargs)
        if default_cfg_dict['log_level'] == "DEBUG":
            write_report()
            exit()
    return inner


def wrap_logger_debug(func):
    def inner(*args, **kwargs):
        if default_cfg_dict['log_level'] == "DEBUG":
            error_content.append(args[0])
        return func(*args, **kwargs)
    return inner


logger = Log(default_cfg_dict['log_level']).get_logger()
is_ci = os.getenv('CI', 'null')
logger.error = wrap_logger_error(logger.error)
logger.debug = wrap_logger_debug(logger.debug)


def write_report():
    if is_ci != 'null':
        return
    os.system("rm -f error_report.csv")
    with open("error_report.csv", "a") as f:
        f.write("Conformance-Test Error Report\n")
        f.write("---------------------------------\n")
        f.write(f"{error_counter[0]} Tests failed:\n")
        for ele in error_content:
            f.write(ele)
        f.write("Test skipped or op not implemented: \n")
        for ele in error_content_other:
            f.write(ele)


class DiopiException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FunctionNotImplementedError(DiopiException):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def check_returncode(returncode, throw_exception=True):
    if 0 != returncode:
        error_info = f"Returncode: {returncode}"
        error_detail = get_last_error()
        error_info += ", Details: " + error_detail

        if throw_exception:
            raise DiopiException(error_info)
        else:
            logger.info(error_info)


def check_function(fn_name):
    try:
        func = eval(f"diopi_runtime.device_impl_lib.{fn_name}")
    except AttributeError as e:
        raise FunctionNotImplementedError(e.args)
    return func


def squeeze(input: diopi_runtime.Tensor, dim=None):
    size = input.size()
    new_size = []
    if dim < 0:
        dim += len(size)

    for i in range(0, len(size)):
        if size[i] != 1:
            new_size.append(size[i])
        elif dim is not None and i != dim:
            new_size.append(size[i])

    input.reset_shape(new_size)


real_op_list = []


def need_process_func(cfg_func_name, func_name, model_name):
    if model_name != '':
        if cfg_func_name not in real_op_list:
            real_op_list.append(cfg_func_name)
    if func_name not in ['all_ops', cfg_func_name]:
        return False
    return True


record_env = os.getenv("RECORD_PRECISION", "OFF")
record = True if record_env != "OFF" else False
sigle_func_record = []
call_once = True
path = "record_all_precision.csv"
failed_path = "record_failed_precision.csv"
os.system('rm -f record_all_precision.csv record_failed_precision.csv')


def write_csv(w_path, content_list):
    with open(w_path, 'a', newline="") as w:
        writer = csv.writer(w)
        if content_list:
            writer.writerow(content_list)


def save_precision(cfg, output, output_reference, passed, var_name):
    rtol = cfg.get('rtol_half', 1e-5) if output.dtype == np.float16 else cfg.get('rtol', 1e-5)
    atol = cfg.get('atol_half', 1e-5) if output.dtype == np.float16 else cfg.get('atol', 1e-5)
    max_atol = 'none'
    need_atol = 'none'
    max_rtol = 'none'
    need_rtol = 'none'
    diff = np.abs(output - output_reference)
    nan_mask = ~np.isnan(diff)
    if nan_mask.sum() != 0 and output_reference.dtype != np.bool:
        diff = diff[nan_mask]
        output_reference = output_reference[nan_mask]
        # fixing rtol，compute atol needed to pass test
        # diff <= atol + rtol * np.abs(output_reference)
        max_atol = np.max(diff)
        need_atol = np.max(diff - rtol * np.abs(output_reference))
        # fixing atol，compute rtol needed to pass test
        # diff <= atol + rtol * np.abs(output_reference)
        zero_mask = ~(output_reference == 0)
        if zero_mask.sum() != 0:
            diff = diff[zero_mask]
            output_reference = output_reference[zero_mask]
            max_rtol = np.max(diff / output_reference)
            need_rtol = np.max((diff - atol) / np.abs(output_reference))

    global sigle_func_record
    sigle_func_record += [var_name, str(output.dtype), str(output.shape),
                          str(passed), str(need_atol), str(max_atol), str(need_rtol), str(max_rtol)]


def write_precision(cfg, func_name, passed=True):
    if not record:
        return

    global path, failed_path, sigle_func_record, call_once
    if call_once:
        call_once = False
        if record_env == "ALL":
            write_csv(path, ['func', "rtol_half", "atol_half", 'rtol', 'atol',
                             'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol', 'need_rtol', 'max_rtol',
                             'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol', 'need_rtol', 'max_rtol',
                             'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol', 'need_rtol', 'max_rtol'])
        write_csv(failed_path, ['failed_func', "rtol_half", "atol_half", 'rtol', 'atol',
                                'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol', 'need_rtol', 'max_rtol',
                                'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol', 'need_rtol', 'max_rtol',
                                'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol', 'need_rtol', 'max_rtol'])

    rtol_half = cfg.get('rtol_half', 1e-5)
    atol_half = cfg.get('atol_half', 1e-8)
    rtol = cfg.get('rtol', 1e-5)
    atol = cfg.get('atol', 1e-8)

    content = [func_name, str(rtol_half), str(atol_half), str(rtol), str(atol)] + sigle_func_record
    if not passed:
        write_csv(failed_path, content)
    if record_env == "ALL":
        write_csv(path, content)
    sigle_func_record = []
