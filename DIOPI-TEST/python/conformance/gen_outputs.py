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
            if function_paras["requires_grad"][para] == [True]:
                tensor.requires_grad = True
            function_paras['kwargs'][para] = tensor.cuda()

        if para == "tensors":
            tensors = function_paras['kwargs'][para]
            for idx, ele in enumerate(tensors):
                tensors[idx] = torch.from_numpy(ele).cuda()
            function_paras['kwargs'][para] = tensors


def grad_kv(fn_args):
    inputs_for_grad_value = []
    inputs_for_grad_key = []
    for k, v in fn_args["kwargs"].items():
        if k in fn_args["requires_grad"].keys() \
                and fn_args["requires_grad"][k] == [True]:
            inputs_for_grad_value.append(v)
            inputs_for_grad_key.append(k)

    return inputs_for_grad_key, inputs_for_grad_value


def to_numpy(tensors):
    import torch
    if isinstance(tensors, torch.Tensor):
        ndarrays = tensors.detach().cpu().numpy()
    elif isinstance(tensors, (list, tuple)):
        ndarrays = []
        for i in range(len(tensors)):
            if isinstance(tensors[i], torch.Tensor):
                ndarrays.append(tensors[i].detach().cpu().numpy())
            else:
                ndarrays.append(tensors[i])
    elif isinstance(tensors, dict):
        ndarrays = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                ndarrays.update(k=v.detach().cpu().numpy())
            else:
                ndarrays.update(k=v)
    else:
        ndarrays = None

    return ndarrays


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
            try:
                file_inputs = open(os.path.join(inputs_dir_path, fname), "rb")
                data = pickle.load(file_inputs)
            except Exception as e:
                if 'file_inputs' in vars():
                    file_inputs.close()
                logger.error(f"failed to load input data for test {fname}, caused by {e}")
                continue

            op_name = data["cfg"]["name"]
            if opname not in ['all', op_name]:
                continue
            if op_name_last != op_name:
                op_name_last = op_name
                logger.debug(f"generating output(s) for {op_name} ...")

            module = "torch.nn.functional"
            if "interface" in data["cfg"].keys():
                module = data["cfg"]["interface"][0]

            fn_args = data["function_paras"]
            transfer_tensor_to_device(fn_args)
            kwargs = fn_args['kwargs']
            op_call = f"{module}.{op_name}(**kwargs)"
            try:
                outputs = eval(op_call)
            except Exception as e:
                logger.error(f"failed to execute function {op_call}, caused by {e}")
                continue

            if "do_backward" in data["cfg"].keys():
                fname = fname.split(".pth")[0] + "_backward.pth"
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]

                requires_backward = data["cfg"]["requires_backward"]
                if len(requires_backward) == 0:
                    outputs_for_backward = outputs
                else:
                    outputs_for_backward = [outputs[i] for i in requires_backward]

                input_names_for_grad, inputs_for_grad = grad_kv(fn_args)
                saved_grads = None
                if len(inputs_for_grad) != 0:
                    grad_outputs = [torch.ones_like(i) for i in outputs_for_backward]
                    grads = torch.autograd.grad(outputs_for_backward, inputs_for_grad, grad_outputs, allow_unused=True)
                    saved_grads = {k: v for k, v in zip(input_names_for_grad, grads)}

                saved_grads_numpy = to_numpy(saved_grads)

                with open(os.path.join(outputs_dir_path, fname), "wb") as file_outputs:
                    pickle.dump(saved_grads_numpy, file_outputs)
                    logger.info(f"generate backward outputs for {op_name} done")

            if outputs is not None:
                with open(os.path.join(outputs_dir_path, fname), "wb") as file_outputs:
                    outputs_numpy = to_numpy(outputs)
                    pickle.dump(outputs_numpy, file_outputs)
                    num_total += 1

        logger.info(f"gen_num: {num_total}")
        logger.info("generate output data done!")
