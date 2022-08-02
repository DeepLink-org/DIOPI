import os
import pickle
import torch

from .utils import logger
from .dtype import Dtype


inputs_dir_path = "data/inputs"
outputs_dir_path = "data/outputs"


def load_testcases() -> list:
    testcases = []
    with open(os.path.join(inputs_dir_path, "cfgs.pth"), "rb") as file_cfgs:
        try:
            cfgs = pickle.load(file_cfgs)
        finally:
            for k, _ in cfgs.items():
                testcases.append(k)
    if len(testcases) == 0:
        logger.info("No test cases found")

    return testcases


def convert_cuda_and_float16(function_paras: dict, cfg: dict, half_cast_double: bool):
    for para in function_paras["kwargs"].keys():
        if isinstance(function_paras['kwargs'][para], torch.Tensor):
            function_paras['kwargs'][para] = function_paras['kwargs'][para].cuda()
    for i_para in range(len(function_paras["kargs"])):
        if isinstance(function_paras["kargs"][i_para], torch.Tensor):
            function_paras['kargs'][i_para] = function_paras['kargs'][i_para].cuda()

    i_kargs = 0
    for arg in cfg["call_para"]["args"]:
        if "dtype" in arg.keys():
            if arg["dtype"] == Dtype.float16:
                dtype = Dtype.float64 if half_cast_double else Dtype.float16
                if len(arg['gen_num_range']) != 2:
                    name = arg["ins"]
                    value = function_paras["kwargs"][name]
                    function_paras["kwargs"][name] = \
                        value.to(Dtype.float16).to(dtype) if value is not None else None
                else:
                    # not rename
                    if 'seq_name' not in cfg['call_para']:
                        for i_kargs in range(len(function_paras['kargs'])):
                            value = function_paras['kargs'][i_kargs]
                            function_paras['kargs'][i_kargs] = \
                                value.to(Dtype.float16).to(dtype) if value is not None else None
                            i_kargs += 1
                    else:
                        # has been renamed
                        name = cfg['call_para']['seq_name']
                        list_v = function_paras['kwargs'][name]
                        for i in range(len(list_v)):
                            list_v[i] = list_v[i].to(Dtype.float16).to(dtype) \
                                if list_v[i] is not None else None


def generate():
    testcases = load_testcases()
    for fname in iter(testcases):
        outputs = None
        with open(os.path.join(inputs_dir_path, fname), "rb") as file_inputs:
            data = pickle.load(file_inputs)

            op_name  = data["cfg"]["name"]
            fn_paras = data["function_paras"]
            convert_cuda_and_float16(fn_paras, data["cfg"], False)
            kargs    = fn_paras['kargs']
            kwargs   = fn_paras['kwargs']

            op_call  = f"torch.nn.functional.{op_name}(*kargs, **kwargs)"
            outputs  = eval(op_call)

        if outputs is not None:
            with open(os.path.join(outputs_dir_path, fname), "wb") as file_outputs:
                pickle.dump(outputs, file_outputs)

            logger.info(f"generate outputs for {op_name}")
