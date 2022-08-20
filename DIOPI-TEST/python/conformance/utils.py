from . import *
from ctypes import c_float, byref
import logging


default_vals = dict()
default_vals['log_level'] = 1
default_vals['test_case_paras'] = dict()


class Logger(object):
    def __init__(self, level):
        self.logger = logging.getLogger("conformance test suite")
        self.logger.setLevel(level)
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # create formatter
        formatter = logging.Formatter(
            '%(asctime)s-%(name)s-%(levelname)s- %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

    def get_loger(self):
        return self.logger


logger = Logger(default_vals['log_level']).get_loger()


def raw_like(tensor) -> Tensor:
    return tensor.raw_like()


def fill(tensor, value):
    error_code = device_impl_lib.fill(tensor.context_handle, tensor.tensor_handle, c_float(value))
    check_return_value(error_code)
    return tensor


Tensor.fill_ = fill
