# Copyright (c) 2023, DeepLink.
from diopilib import Dtype

class glob_var(object):
    def __init__(self, nhwc=False, nhwc_min_dim=3, four_bytes=False):
        self.nhwc = nhwc
        self.nhwc_min_dim = nhwc_min_dim
        self.four_bytes = four_bytes
        self.int_type = Dtype.int64
        self.float_type = Dtype.float64
        self._cur_test_func = ''

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

    @property
    def cur_test_func(self):
        return self._cur_test_func

    @cur_test_func.setter
    def cur_test_func(self, func):
        self._cur_test_func = func

glob_vars = glob_var()
