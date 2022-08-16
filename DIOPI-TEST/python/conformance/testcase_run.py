import os
import pickle
import numpy as np

from . import functions as F
from .utils import logger
from .litert import Tensor
from .gen_inputs import inputs_dir_path
from .gen_outputs import load_testcases, outputs_dir_path


def convert_tensors(function_paras: dict):
    for para in function_paras["kwargs"].keys():
        if isinstance(function_paras['kwargs'][para], np.ndarray):
            function_paras['kwargs'][para] = Tensor.from_numpy(function_paras['kwargs'][para])
    for i_para in range(len(function_paras["kargs"])):
        if isinstance(function_paras["kargs"][i_para], np.ndarray):
            function_paras['kwargs'][para] = Tensor.from_numpy(function_paras['kargs'][i_para])


def allclose(cfg : dict, tensor1 : np.ndarray, tensor2 : np.ndarray) -> bool:
    rtol = cfg.get('rtol', 1e-5)
    atol = cfg.get('atol', 1e-8)
    return np.allclose(tensor1, tensor2, rtol, atol, True)


def run(opname):
    testcases = load_testcases()
    for fname in iter(testcases):
        outputs = None
        with open(os.path.join(inputs_dir_path, fname), "rb") as file_inputs:
            data = pickle.load(file_inputs)
            op_name  = data["cfg"]["name"]
            if opname not in ['all', op_name]: continue

            fn_paras = data["function_paras"]
            convert_tensors(fn_paras)
            kargs    = fn_paras['kargs']
            kwargs   = fn_paras['kwargs']

            op_call  = f"F.{op_name}(*kargs, **kwargs)"
            try:
                outputs = eval(op_call)

                with open(os.path.join(outputs_dir_path, fname), "rb") as file_outputs:
                    outputs_reference = pickle.load(file_outputs)

                    passed = True
                    if isinstance(outputs, Tensor):
                        passed = allclose(data["cfg"], outputs.numpy(), outputs_reference)
                    elif isinstance(outputs, (list, tuple)):
                        assert isinstance(outputs_reference, (list, tuple))
                        assert len(outputs) == len(outputs_reference)
                        for i in range(len(outputs)):
                            if isinstance(outputs[i], Tensor):
                                passed = passed and allclose(data["cfg"], outputs[i].numpy(), outputs_reference[i])
                    if passed:
                        logger.info(f"run {op_name} succeed")
                    else:
                        logger.info(f"run {op_name} failed")
            except:
                logger.info(f"run {op_name} failed")
