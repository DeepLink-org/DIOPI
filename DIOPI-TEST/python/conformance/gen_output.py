import os
import sys
import pickle
import numpy as np

from .utils import logger
from .gen_input import inputs_dir_path, outputs_dir_path, get_saved_pth_list


class CustomizedTest(object):
    def slice_op(input, dim, index):
        import torch
        size = len(input.size())
        slice_args = [True for i in range(size)]
        slice_args[dim] = index
        return torch.Tensor.__getitem__(input, slice_args)

    def index(input, **kwargs):
        import torch
        new_args = []
        for ele in kwargs.values():
            if ele is None:
                hasEllipsis = True
                if hasEllipsis and Ellipsis not in new_args:
                    new_args.append(...)
            else:
                new_args.append(ele)
        return torch.Tensor.__getitem__(input, new_args)

    def sgd(param, param_grad, buf, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        import torch
        param.requires_grad = True
        param.grad = param_grad
        optimizer = torch.optim.SGD([param, ], lr, momentum, dampening, weight_decay, nesterov)
        optimizer.state[param]['momentum_buffer'] = buf
        optimizer.step()
        return param, buf


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


def get_name_and_data_for_grad(function_paras):
    inputs_for_grad_value = []
    inputs_for_grad_key = []
    for k, v in function_paras["kwargs"].items():
        if k in function_paras["requires_grad"].keys() \
                and function_paras["requires_grad"][k] == [True]:
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

    @staticmethod
    def run(func_name):
        import torch
        import torchvision
        if not os.path.exists(inputs_dir_path):
            logger.error("input data is not generated!")
            sys.exit(0)

        if not os.path.exists(outputs_dir_path):
            os.makedirs(outputs_dir_path)

        gen_counter = 0
        func_name_flag = ""  # make the info log once
        saved_pth_list = get_saved_pth_list()
        for saved_pth in saved_pth_list:
            try:
                f = open(os.path.join(inputs_dir_path, saved_pth), "rb")
                data = pickle.load(f)
            except Exception as e:
                logger.error(f"failed to load data for {saved_pth}, caused by {e}")
                continue
            else:
                f.close()

            cfg_func_name = data["cfg"]["name"]
            if func_name not in ['all', cfg_func_name]:
                continue
            if func_name_flag != cfg_func_name:
                func_name_flag = cfg_func_name
                logger.info(f"generating output(s) for {cfg_func_name} ...")

            module = "torch.nn.functional"
            if "interface" in data["cfg"].keys():
                module = data["cfg"]["interface"][0]

            function_paras = data["function_paras"]
            transfer_tensor_to_device(function_paras)
            kwargs = function_paras['kwargs']
            func_call = f"{module}.{cfg_func_name}(**kwargs)"

            try:
                outputs = eval(func_call)
            except Exception as e:
                logger.error(f"failed to execute function {func_call}, caused by {e}")
                continue

            if "do_backward" in data["cfg"].keys():
                saved_pth = saved_pth.split(".pth")[0] + "_backward.pth"
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]

                requires_backward = data["cfg"]["requires_backward"]
                outputs_for_backward = outputs if len(requires_backward) == 0 \
                    else [outputs[i] for i in requires_backward]

                inputs_name_for_grad, inputs_for_grad = get_name_and_data_for_grad(function_paras)
                saved_grads = None
                if len(inputs_for_grad) != 0:
                    grad_outputs = [torch.ones_like(i) for i in outputs_for_backward]
                    grads = torch.autograd.grad(
                        outputs_for_backward, inputs_for_grad, grad_outputs, allow_unused=True)
                    saved_grads = {k: v for k, v in zip(inputs_name_for_grad, grads)}

                with open(os.path.join(outputs_dir_path, saved_pth), "wb") as f:
                    pickle.dump(to_numpy(saved_grads), f)
                    logger.info(f"generate backward outputs for {cfg_func_name} done")

            if outputs is not None:
                with open(os.path.join(outputs_dir_path, saved_pth), "wb") as f:
                    pickle.dump(to_numpy(outputs), f)
                    gen_counter += 1

        logger.info(f"generate test cases number: {gen_counter}")
        logger.info("generate output data done!")
