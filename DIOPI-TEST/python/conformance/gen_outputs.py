import os
import sys
import pickle
import numpy as np

from .utils import logger
from .gen_inputs import inputs_dir_path, outputs_dir_path, load_testcases


def transfer_tensor_to_device(function_paras: dict):
    import torch
    for para in function_paras["kwargs"].keys():
        if isinstance(function_paras['kwargs'][para], np.ndarray):
            tensor = torch.from_numpy(function_paras['kwargs'][para])
            function_paras['kwargs'][para] = tensor.cuda()
        if para == "tensors":
            tensors = function_paras['kwargs'][para]
            for idx, ele in enumerate(tensors):
                 tensors[idx] = torch.from_numpy(ele).cuda()
            function_paras['kwargs'][para] = tensors

    for i_para in range(len(function_paras["kargs"])):
        if isinstance(function_paras["kargs"][i_para], np.ndarray):
            tensor = torch.from_numpy(function_paras['kargs'][i_para])
            function_paras['kargs'][i_para] = tensor.cuda()


class GenOutputData(object):
    r'''
    Generate output data for all functions by using torch and input data
    '''

    def run(self, opname):
        import torch
        if not os.path.exists(inputs_dir_path):
            logger.error("input data is not generated!")
            sys.exit(0)

        if not os.path.exists(outputs_dir_path):
            os.makedirs(outputs_dir_path)

        num_total = 0
        op_name_last = ""
        testcases = load_testcases()
        for fname in iter(testcases):
            outputs = None
            with open(os.path.join(inputs_dir_path, fname), "rb") as file_inputs:
                data = pickle.load(file_inputs)

                op_name = data["cfg"]["name"]
                if opname not in ['all', op_name]: continue
                if op_name_last != op_name:
                    op_name_last = op_name
                    logger.debug(f"generating output(s) for {op_name} ...")

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
                    num_total += 1

        logger.info(f"gen_num: {num_total}")
        logger.info(f"generate output data done!")
