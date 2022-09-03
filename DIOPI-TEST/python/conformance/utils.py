import logging
from .litert import Tensor, device_impl_lib


default_vals = dict(
    test_case_paras=dict(
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
default_vals['log_level'] = 1


class Log(object):
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

    def get_logger(self):
        return self.logger


logger = Log(default_vals['log_level']).get_logger()


class DiopiException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FunctionNotImplementedError(DiopiException):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def check_returncode(returncode, throw_exception=True):
    if 0 != returncode:
        error_info = f"returncode {returncode}"
        if throw_exception:
            raise DiopiException(errcode=returncode, info=error_info)
        else:
            logger.info(error_info)


def check_function(fn_name):
    try:
        func = eval(f"device_impl_lib.{fn_name}")
    except AttributeError as e:
        raise FunctionNotImplementedError(e.args)
    return func


def squeeze(input: Tensor):
    size = input.size()
    new_size = []
    for i in len(size):
        if size[i] != 1:
            new_size.append(size[i])
    input.reset_shape(new_size)
