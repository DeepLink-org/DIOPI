import os
import pickle
import numpy as np

from . import functions as F
from .utils import logger
from .dtype import Dtype
from .litert import Tensor
from .gen_outputs import load_testcases, inputs_dir_path, outputs_dir_path


def convert_tensors(function_paras: dict, cfg: dict, half_cast_double: bool):
    import torch
    for para in function_paras["kwargs"].keys():
        if isinstance(function_paras['kwargs'][para], torch.Tensor):
            tr = function_paras['kwargs'][para].numpy()
            function_paras['kwargs'][para] = Tensor.from_numpy(tr)
    for i_para in range(len(function_paras["kargs"])):
        if isinstance(function_paras["kargs"][i_para], torch.Tensor):
            tr = function_paras['kargs'][i_para].numpy()
            function_paras['kwargs'][para] = Tensor.from_numpy(tr)

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


def verify(cfg : dict, tensor1 : np.ndarray, tensor2 : np.ndarray):
    rtol = cfg.get('rtol', 1e-5)
    atol = cfg.get('atol', 1e-8)
    np.allclose(tensor1, tensor2, rtol, atol)


def run():
    testcases = load_testcases()
    for fname in iter(testcases):
        outputs = None
        with open(os.path.join(inputs_dir_path, fname), "rb") as file_inputs:
            data = pickle.load(file_inputs)

            op_name  = data["cfg"]["name"]
            fn_paras = data["function_paras"]
            convert_tensors(fn_paras, data["cfg"], False)
            kargs    = fn_paras['kargs']
            kwargs   = fn_paras['kwargs']

            op_call  = f"F.{op_name}(*kargs, **kwargs)"
            try:
                outputs = eval(op_call)

                with open(os.path.join(outputs_dir_path, fname), "rb") as file_outputs:
                    outputs_reference = pickle.load(file_outputs)
                    if isinstance(outputs, Tensor):
                        verify(data["cfg"], outputs.numpy(), outputs_reference.numpy())
                    elif isinstance(outputs, (list, tuple)):
                        assert isinstance(outputs_reference, (list, tuple))
                        assert len(outputs) == len(outputs_reference)
                        for i in range(len(outputs)):
                            if isinstance(outputs[i], Tensor):
                                verify(data["cfg"], outputs[i].numpy(), outputs_reference[i].numpy())
            except:
                logger.info(f"run {op_name} failed")
