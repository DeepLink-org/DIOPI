# Copyright (c) 2023, DeepLink.
import logging
import os
import numpy as np
import csv
import pickle
import ctypes

cfg_file_name = '../cache/diopi_case_items.cfg'

default_cfg_dict = dict(
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


class Log(object):
    def __init__(self, level):
        self.logger = logging.getLogger("ConformanceTest")
        self.logger.setLevel(level)
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s')

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
# logger.error = wrap_logger_error(logger.error)
# logger.debug = wrap_logger_debug(logger.debug)


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


# def squeeze(input: diopi_runtime.Tensor, dim=None):
def squeeze(input, dim=None):
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
    if output.dtype == np.bool:
        output = output.astype(np.int32)
    if output_reference.dtype == np.bool:
        output_reference = output_reference.astype(np.int32)

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


def get_saved_pth_list(inputs_dir_path, cfg_file_name) -> list:
    with open(os.path.join(inputs_dir_path, cfg_file_name), "rb") as f:
        cfg_dict = pickle.load(f)

    return [k for k in cfg_dict]


def get_data_from_file(data_path, test_path, name=""):
    if not os.path.exists(data_path):
        logger.error(f"FileNotFound: No benchmark {name} data '{test_path}' was generated"
                     f" (No such file or directory: {data_path})")
        return None
    try:
        f = open(data_path, "rb")
        data = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed: {e}")
        return None
    else:
        f.close()
    return data


def gen_pytest_case_nodeid(dir, file, class_, func):
    """e.g.
    dir: ./gencases/diopi_case
    file: test_diopi_adadelta_adadelta.py
    class_: TestMdiopiSadadeltaFadadelta
    func: test_adadelta_0
    ->
    gencases/diopi_case/test_diopi_adadelta_adadelta.py::TestMdiopiSadadeltaFadadelta::test_adadelta_0"""
    return f'{os.path.join(os.path.normpath(dir), file)}::{class_}::{func}'
