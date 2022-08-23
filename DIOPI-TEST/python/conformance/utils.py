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


class DiopiException(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class FunctionNotImplementedError(DiopiException):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def check_return_value(returncode, throw_exception=True):
    if 0 != returncode:
        error_info = f"returncode {returncode}"
        if throw_exception:
            raise DiopiException(errcode=returncode, info=error_info)
        else:
            logger.info(error_info)
