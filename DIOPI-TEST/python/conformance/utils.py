import logging
from . import diopi_runtime
from .diopi_runtime import device_impl_lib, get_last_error
from .model_list import model_list, model_op_list
import os
import numpy as np
import csv


default_cfg_dict = dict(
    default_option=dict(
        atol=1e-8,
        rtol=1e-5,
        atol_half=1e-4,
        rtol_half=5e-3,
        memory_format="NCHW",
        fp16_exact_match=False,
        train=True,
    ),
    log_level="DEBUG"  # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICA
)
default_cfg_dict['log_level'] = 1
error_counter = [0]


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
    def inner(*args, **kwargs):
        if args[0].startswith("NotImplemented") or \
             args[0].startswith("AttributeError"):
            return func(*args, **kwargs)
        global error_counter
        error_counter[0] += 1
        return func(*args, **kwargs)
    return inner


logger = Log(default_cfg_dict['log_level']).get_logger()
is_ci = os.getenv('CI', 'null')
if is_ci != 'null':
    logger.error = wrap_logger_error(logger.error)


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


def need_process_func(cfg_func_name, func_name, model_name):
    if model_name != '':
        op_list = model_op_list[model_name]
        if cfg_func_name not in op_list:
            return False
    elif func_name not in ['all', cfg_func_name]:
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
    if output_reference.shape == ():
        diff = np.abs(output - output_reference)
        if np.isnan(diff):
            diff = 0
            output_reference = 1
        max_atol = np.max(diff)
        need_atol = np.max(diff - rtol * np.abs(output_reference))
    elif output_reference.dtype == np.bool:
        max_atol = 'none'
        need_atol = 'none'
    else:
        diff = np.abs(output - output_reference)
        nan_mask = ~np.isnan(diff)
        diff = diff[nan_mask]
        output_reference = output_reference[nan_mask]

        # fixing rtol，compute atol needed to pass test
        # diff <= atol + rtol * np.abs(output_reference)
        max_atol = np.max(diff)
        need_atol = np.max(diff - rtol * np.abs(output_reference))

    global sigle_func_record
    sigle_func_record += [var_name, str(output.dtype), str(output.shape),\
                          str(passed), str(need_atol), str(max_atol)]


def write_precision(cfg, func_name, passed=True):
    if not record:
        return

    global path, failed_path, sigle_func_record, call_once
    if call_once:
        call_once = False
        if record_env == "ALL":
            write_csv(path, ['func', "rtol_half", "atol_half", 'rtol', 'atol',\
                      'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol',\
                      'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol',\
                      'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol'])
        write_csv(failed_path, ['failed_func', "rtol_half", "atol_half", 'rtol', 'atol',\
                 'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol',\
                 'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol',\
                 'var_name', 'var_dtype', 'var_shape', 'passed', 'need_atol', 'max_atol'])

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
