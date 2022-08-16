import os
import pickle
import numpy as np
import torch

from .utils import logger
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
        if isinstance(function_paras['kwargs'][para], np.ndarray):
            tensor = torch.from_numpy(function_paras['kwargs'][para])
            function_paras['kwargs'][para] = tensor.cuda()
    for i_para in range(len(function_paras["kargs"])):
        if isinstance(function_paras["kargs"][i_para], np.ndarray):
            tensor = torch.from_numpy(function_paras['kargs'][i_para])
            function_paras['kargs'][i_para] = tensor.cuda()


def generate(opname):
    testcases = load_testcases()
    for fname in iter(testcases):
        outputs = None
        with open(os.path.join(inputs_dir_path, fname), "rb") as file_inputs:
            data = pickle.load(file_inputs)

            op_name = data["cfg"]["name"]
            if opname not in ['all', op_name]: continue

            module = "torch.nn.functional"
            if "interface" in data["cfg"].keys():
                module = data["cfg"]["interface"][0]
            if module == 'tensor': continue

            fn_paras = data["function_paras"]
            transfer_tensor_to_device(fn_paras)
            kargs    = fn_paras['kargs']
            kwargs   = fn_paras['kwargs']

            op_call = f"{module}.{op_name}(*kargs, **kwargs)"
            outputs = eval(op_call)

        if outputs is not None:
            with open(os.path.join(outputs_dir_path, fname), "wb") as file_outputs:
                if isinstance(outputs, torch.Tensor):
                    outputs_numpy = outputs.cpu().numpy()
                elif isinstance(outputs, (list, tuple)):
                    outputs_numpy = []
                    for i in range(len(outputs)):
                        if isinstance(outputs[i], torch.Tensor):
                            outputs_numpy.append(outputs[i].cpu().numpy())
                        else:
                            outputs_numpy.append(outputs[i])
                pickle.dump(outputs_numpy, file_outputs)

            logger.info(f"generate outputs for {op_name}")
