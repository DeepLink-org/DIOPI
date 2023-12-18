# Copyright (c) 2023, DeepLink.
import numpy as np


class glob_var(object):
    def __init__(self, nhwc=False, nhwc_min_dim=3, four_bytes=False):
        self.nhwc = nhwc
        self.nhwc_min_dim = nhwc_min_dim
        self.four_bytes = four_bytes
        self.int_type = np.int64
        self.float_type = np.float64
        self.input_mismatch_ratio_threshold = 1e-3
        self._cur_test_func = ''
        self._func_status = {}
        self._debug_level = 0
        self._use_db = None
        self.case_item = {
            "atol": 1e-5,
            "rtol": 1e-5,
            "atol_half": 1e-2,
            "rtol_half": 5e-2,
            "mismatch_ratio_threshold": 1e-3,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "gen_policy": "dafault",
        }

    def set_nhwc(self):
        self.nhwc = True

    def set_nhwc_min_dim(self, dim):
        self.nhwc_min_dim = dim

    def get_nhwc(self):
        return self.nhwc

    def set_four_bytes(self):
        self.four_bytes = True
        self.int_type = np.int32
        self.float_type = np.float32

    def get_four_bytes(self):
        return self.four_bytes

    @property
    def cur_test_func(self):
        return self._cur_test_func

    @cur_test_func.setter
    def cur_test_func(self, func):
        self._cur_test_func = func

    @property
    def func_status(self):
        return self._func_status

    @func_status.setter
    def func_status(self, func, status):
        self._func_status[func] = status

    @property
    def debug_level(self):
        return self._debug_level

    @debug_level.setter
    def debug_level(self, debug_level):
        self._debug_level = debug_level

    @property
    def use_db(self):
        return self._use_db

    @use_db.setter
    def use_db(self, use_db):
        self._use_db = use_db


glob_vars = glob_var()
