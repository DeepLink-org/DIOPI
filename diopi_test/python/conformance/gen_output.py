import pickle
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
import math
import torchvision

from conformance.utils import GenPolicy
from conformance.utils import logger, get_data_from_file
from conformance.db_operation import db_conn
from einops import rearrange
from customized_test import CustomizedTest

    def _amp_foreach_non_finite_check_and_unscale_(scaled_grads, found_inf, inv_scale):
        torch._amp_foreach_non_finite_check_and_unscale_(scaled_grads, found_inf, inv_scale)
        return scaled_grads, found_inf

    def _amp_update_scale_(scale, growth_tracker, found_inf, growth_factor, backoff_factor, growth_interval):
        torch._amp_update_scale_(scale, growth_tracker, found_inf, growth_factor, backoff_factor, growth_interval)
        return scale, growth_tracker


class GenOutputData(object):
    r"""
    Generate output data for all functions by using numpy and input data
    """
    db_case_items = {}
    err_case_counter = 0

    @staticmethod
    def run(
        diopi_item_config_path="diopi_case_items.cfg",
        input_path="data/inputs/",
        output_path="data/outputs/",
        fname="all_ops",
        model_name="diopi",
    ):
        if not os.path.exists(input_path):
            logger.error("Input data is not generated!")
            sys.exit(0)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(diopi_item_config_path, "rb") as f:
            all_cfg_dict = pickle.load(f)

        # XXX save case number in glob_var
        case_counter = 0
        func_name_list = []  # make the info log once

        for case_name in all_cfg_dict:
            each_cfg_dict = all_cfg_dict[case_name]
            func_name = each_cfg_dict["name"]
            item = {"case_name": case_name, "model_name": model_name}
            if "all_ops" not in fname and func_name not in fname:
                continue
            data_path = os.path.join(input_path, case_name)
            input_ = get_data_from_file(data_path, case_name, "input")
            if "no_output_ref" in each_cfg_dict:
                logger.info(
                    f"diopi_functions.{func_name} [{case_name}] is set to no_output_ref, skip generate output"
                )
                continue

            gen_tensor_obj = GenTensor(case_name, each_cfg_dict)

            try:
                output, saved_grads = gen_tensor_obj.gen_data(input_)
                item["result"] = "passed"
            except Exception as err_msg:
                logger.error(
                    f"Generate output data for diopi_functions.{func_name} [{case_name}] failed, cause by \n{err_msg}"
                )
                item.update({"result": "failed", "err_msg": err_msg})
                GenOutputData.err_case_counter += 1
                continue
            finally:
                GenOutputData.db_case_items[case_name] = item
            if output is not None:
                with open(os.path.join(output_path, case_name), "wb") as f:
                    pickle.dump(GenOutputData.to_numpy(output), f, protocol=4)
                    logger_str = "output"
                    case_counter += 1
                if saved_grads is not None:
                    saved_backward_pth = case_name.split(".pth")[0] + "_backward.pth"
                    with open(os.path.join(output_path, saved_backward_pth), "wb") as f:
                        pickle.dump(GenOutputData.to_numpy(saved_grads), f, protocol=4)
                    logger_str = f"{logger_str} and backward"

                if func_name not in func_name_list:
                    func_signature = f"diopi_functions.{func_name}"
                    logger.info(
                        f"Generate benchmark {logger_str} data for {func_signature}"
                    )
                    func_name_list.append(func_name)

        logger.info(f"Generate test cases number for output data: {case_counter}")
        if case_counter == 0:
            logger.info("No benchmark output data is generated")
        else:
            logger.info("Generate benchmark output and backward data done!")

    @staticmethod
    def to_numpy(tensors):
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
                    tmp = {k: v.detach().cpu().numpy()}
                else:
                    tmp = {k: v}
                ndarrays.update(tmp)
        elif isinstance(tensors, (int, float)):
            ndarrays = np.array(tensors)
        else:
            ndarrays = None

        return ndarrays


class GenTensor(object):
    def __init__(self, case_name, case_cfg) -> None:
        self.case_name = case_name
        self.case_cfg = case_cfg
        self.func_name = case_cfg["name"]
        self.module = "torch.nn.functional"
        self.input = None
        self.output = None
        self.if_forward_success = False

    def gen_data(self, input_data):
        output = self.gen_forward_data(input_data)
        saved_grads = self.gen_backward_data(input_data)
        return output, saved_grads

    def gen_forward_data(self, input_data):
        if self.case_cfg["interface"]:
            self.module = self.case_cfg["interface"][0]
        function_paras = input_data["function_paras"]
        self.transfer_tensor_to_device(function_paras)
        kwargs = function_paras["kwargs"]
        if self.module == "torch.Tensor":
            input = kwargs["input"]
            self.input = input
            self.module = "input"
            del kwargs["input"]
        if "dtype" in kwargs.keys():
            kwargs["dtype"] = self.change_np_dtype_to_torch(kwargs["dtype"])
        func_call = f"{self.module}.{self.func_name}(**kwargs)"
        self.output = eval(func_call)
        self.if_forward_success = True
        return self.output

    def gen_backward_data(self, input_data):
        if not self.if_forward_success:
            return None
        function_paras = input_data["function_paras"]
        kwargs = function_paras["kwargs"]
        saved_grads = None
        if function_paras["requires_grad"]:
            if self.module == "input":
                kwargs["input"] = self.input
            outputs = self.output
            if not isinstance(self.output, (list, tuple)):
                outputs = [self.output]

            requires_backward = self.case_cfg["requires_backward"]
            outputs_for_backward = (
                outputs
                if len(requires_backward) == 0
                else [outputs[i] for i in requires_backward]
            )

            inputs_name_for_grad, inputs_for_grad = self.get_name_and_data_for_grad(
                function_paras
            )
            if len(inputs_for_grad) != 0:
                grad_outputs = [torch.ones_like(i) for i in outputs_for_backward]
                grads = torch.autograd.grad(
                    outputs_for_backward,
                    inputs_for_grad,
                    grad_outputs,
                    allow_unused=True,
                )
                saved_grads = {k: v for k, v in zip(inputs_name_for_grad, grads)}
        return saved_grads

    def transfer_tensor_to_device(self, function_paras: dict):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for para in function_paras["kwargs"].keys():
            if isinstance(function_paras["kwargs"][para], np.ndarray):
                tensor = torch.from_numpy(function_paras["kwargs"][para])
                if function_paras["requires_grad"].get(para, []) == [True]:
                    tensor.requires_grad = True
                function_paras["kwargs"][para] = tensor.to(device=device)

            gen_policy = [
                i.get("gen_policy", None)
                for i in self.case_cfg["tensor_para"]["args"]
                if i["ins"] == para
            ]
            if_gen_list = (
                len(gen_policy) > 0 and gen_policy[0] in GenPolicy.gen_list_policy
            )
            if if_gen_list:
                if isinstance(function_paras["kwargs"][para], (list, tuple)):
                    tensors = function_paras["kwargs"][para]
                    for idx, ele in enumerate(tensors):
                        tensors[idx] = torch.from_numpy(ele).to(device=device)
                        if function_paras["requires_grad"].get(para, []) == [True]:
                            tensors[idx].requires_grad = True
                    function_paras["kwargs"][para] = tensors

    def get_name_and_data_for_grad(self, function_paras):
        inputs_for_grad_value = []
        inputs_for_grad_key = []
        for k, v in function_paras["kwargs"].items():
            if function_paras["requires_grad"].get(k, []) == [True]:
                inputs_for_grad_key.append(k)
                if isinstance(v, (list, tuple)):
                    inputs_for_grad_value.extend(v)
                else:
                    inputs_for_grad_value.append(v)
        return inputs_for_grad_key, inputs_for_grad_value

    def change_np_dtype_to_torch(self, dtype):
        if dtype == np.bool_:
            return torch.bool
        return eval(str(dtype).replace("<class 'numpy.", "torch.").replace("'>", ""))


if __name__ == "__main__":
    GenOutputData.run(
        os.path.join(os.path.dirname(__file__), "../cache/diopi_case_items.cfg"),
        os.path.join(os.path.dirname(__file__), "../cache/data/inputs/"),
        os.path.join(os.path.dirname(__file__), "../cache/data/outputs/"),
    )
