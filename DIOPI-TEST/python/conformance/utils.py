import logging
from . import diopi_runtime
from .diopi_runtime import device_impl_lib, get_last_error
from .model_list import model_list, model_op_list
import os


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

