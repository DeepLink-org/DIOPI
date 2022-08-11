import os
import pickle
import torch

from .utils import logger
from .dtype import Dtype
from .gen_inputs import inputs_dir_path


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


def transfer_tensor_to_device(function_paras: dict):
    for para in function_paras["kwargs"].keys():
        if isinstance(function_paras['kwargs'][para], torch.Tensor):
            function_paras['kwargs'][para] = function_paras['kwargs'][para].cuda()
    for i_para in range(len(function_paras["kargs"])):
        if isinstance(function_paras["kargs"][i_para], torch.Tensor):
            function_paras['kargs'][i_para] = function_paras['kargs'][i_para].cuda()


def generate():
    testcases = load_testcases()
    for fname in iter(testcases):
        outputs = None
        with open(os.path.join(inputs_dir_path, fname), "rb") as file_inputs:
            data = pickle.load(file_inputs)

            module = "torch.nn.functional"
            if "interface" in data["cfg"].keys():
                module = data["cfg"]["interface"][0]
            if module == 'tensor': continue

            fn_paras = data["function_paras"]
            transfer_tensor_to_device(fn_paras)
            kargs    = fn_paras['kargs']
            kwargs   = fn_paras['kwargs']

            op_name = data["cfg"]["name"]
            op_call = f"{module}.{op_name}(*kargs, **kwargs)"
            outputs = eval(op_call)

        if outputs is not None:
            with open(os.path.join(outputs_dir_path, fname), "wb") as file_outputs:
                pickle.dump(outputs, file_outputs)

            logger.info(f"generate outputs for {op_name}")
