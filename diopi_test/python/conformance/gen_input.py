import pickle
import numpy as np
import os
from functools import partial

from generator import Genfunc
from conformance.utils import logger
from conformance.db_operation import db_conn


class GenPolicy:
    default = "default"
    gen_tensor_by_value = "gen_tensor_by_value"
    gen_tensor_list = "gen_tensor_list"
    gen_tensor_list_diff_shape = "gen_tensor_list_diff_shape"

    gen_list_policy = [gen_tensor_list, gen_tensor_list_diff_shape]


class GenInputData(object):
    r"""
    Generate input data for all functions by using numpy
    """
    db_case_items = []

    @staticmethod
    def run(
        diopi_item_config_path="diopi_case_items.cfg",
        input_path="data/inputs/",
        fname="all_ops",
        model_name="diopi",
    ):
        if not os.path.exists(input_path):
            os.makedirs(input_path)

        with open(diopi_item_config_path, "rb") as f:
            all_cfg_dict = pickle.load(f)

        # XXX save case number in glob_var
        case_counter = 0

        for case_name in all_cfg_dict:
            each_cfg_dict = all_cfg_dict[case_name]
            func_name = each_cfg_dict["name"]
            item = {
                "case_name": case_name,
                "model_name": model_name,
                "inplace_flag": 0,
                "backward_flag": 0,
                "func_name": func_name,
                "case_config": each_cfg_dict,
                "result": "skipped",
            }
            for tensor in each_cfg_dict["tensor_para"]["args"]:
                if True in tensor["requires_grad"]:
                    item["backward_flag"] = 1
                    break
            if (
                "is_inplace" in each_cfg_dict.keys() and each_cfg_dict["is_inplace"] is True
            ):
                item["inplace_flag"] = 1
            if fname not in [func_name, "all_ops"]:
                # GenInputData.db_case_items.append(item)
                continue
            # logger.info(f"diopi_functions.{func_name} [config] {each_cfg_dict}")
            try:
                case_dict = GenParas(each_cfg_dict).gen_data()
                with open(os.path.join(input_path, case_name), "wb") as f:
                    pickle.dump(case_dict, f)
                case_counter += 1
                logger.info(
                    f"Generate benchmark input data for diopi_functions.{func_name} [{case_name}]"
                )
                item["result"] = "passed"
            except Exception as err_msg:
                logger.error(
                    f"Generate input data for diopi_functions.{func_name} [{case_name}] failed, cause by \n{err_msg}"
                )
                item.update({"result": "failed", "err_msg": err_msg})
            finally:
                GenInputData.db_case_items.append(item)

        logger.info(f"Generate test cases number for input data: {case_counter}")
        if case_counter == 0:
            logger.warn(
                f'No benchmark input data is generated, "{fname}" may not be in the diopi-config'
            )
        else:
            logger.info("Generate benchmark input data done!")


class GenParas(object):
    def __init__(self, case_cfg: dict = None) -> None:
        self.case_cfg = case_cfg

    def gen_data(self):
        function_paras = self.gen_tensor_para()
        construct_paras = self.gen_para()
        function_paras["kwargs"].update(construct_paras)

        # HACK cfg will be remove
        cfg_info = {"function_paras": function_paras, "cfg": self.case_cfg}

        return cfg_info

    def gen_para(self):
        para_dict = self.case_cfg["para"]
        construct_paras = {k: para_dict[k] for k in para_dict}
        return construct_paras

    def gen_tensor_para(self):
        function_paras = {"kwargs": {}, "requires_grad": {}}
        tensor_para_args_list = self.case_cfg["tensor_para"]["args"]

        for arg in tensor_para_args_list:
            name = arg["ins"]
            shape = arg.get("shape", None)
            value = arg.get("value", None)
            gen_tensor_obj = GenTensor(arg)
            gen_policy = arg["gen_policy"]

            if ("shape" in arg and shape is None) or ("value" in arg and value is None):
                function_paras["kwargs"][name] = None
                continue

            if gen_policy == GenPolicy.default:
                value, requires_grad = gen_tensor_obj.gen_single_tensor()
            elif gen_policy == GenPolicy.gen_tensor_by_value:
                value, requires_grad = gen_tensor_obj.gen_value()
            elif gen_policy == GenPolicy.gen_tensor_list:
                value, requires_grad = gen_tensor_obj.gen_tensor_list()
            elif gen_policy == GenPolicy.gen_tensor_list_diff_shape:
                value, requires_grad = gen_tensor_obj.gen_tensor_list_diff_shape()
            else:
                raise Exception(
                    f"gen_policy {gen_policy} do not exist, only support default,gen_tensor_by_value,gen_tensor_list,gen_tensor_diff_shape"
                )
            function_paras["kwargs"][name] = value
            if requires_grad == [True]:
                function_paras["requires_grad"][name] = requires_grad

        return function_paras


class GenTensor(object):
    def __init__(self, arg) -> None:
        self.function_paras = {"kwargs": {}, "requires_grad": {}}
        self.arg = arg
        self.gen_policy = arg["gen_policy"]
        self.name = arg["ins"]
        self.shape = arg.get("shape", None)
        self.dtype = arg.get("dtype", None)
        self.value = arg.get("value", None)
        self.requires_grad = arg.get("requires_grad", [False])
        self.no_contiguous = arg.get("no_contiguous", False)
        self.gen_fn = self.parse_gen_fn(arg["gen_fn"])

        self._check_item()

    def _check_item(self):
        if self.gen_policy == GenPolicy.gen_tensor_by_value and "value" not in self.arg:
            raise Exception(
                f"when {GenPolicy.gen_tensor_by_value}, value must be provided, but got arg: {self.arg}"
            )
        if self.gen_policy != GenPolicy.gen_tensor_by_value and "shape" not in self.arg:
            raise Exception(f"shape must be provided, but got arg: {self.arg}")
        if (
            self.gen_policy == GenPolicy.gen_tensor_list and len(self.arg["gen_num_range"]) != 2
        ):
            raise Exception(
                f'when gen_policy is gen_tensor_list, gen_num_range length must be 2, but got {len(self.arg["gen_num_range"])}'
            )

        if isinstance(self.arg["gen_fn"], dict):
            gen_fn = eval(self.arg["gen_fn"]["fn"])
            assert (
                gen_fn == Genfunc.randint or gen_fn == Genfunc.uniform or Genfunc.randn_int
            ), "only randint & uniform & randn_int needs args"

    def gen_tensor_list(self):
        tensor_list = []
        # XXX gen tensors_num in parser?
        tensors_num = np.random.randint(
            self.arg["gen_num_range"][0], self.arg["gen_num_range"][1]
        )
        self.arg.setdefault("tensors_num", tensors_num)
        for _ in range(tensors_num):
            value = self.gen_tensor(self.shape, self.dtype)
            tensor_list.append(value)
        return tensor_list, self.requires_grad

    def gen_tensor_list_diff_shape(self):
        tensor_list = []
        for each_shape in self.shape:
            value = self.gen_tensor(each_shape, self.dtype)
            tensor_list.append(value)
        return tensor_list, self.requires_grad

    def gen_value(self):
        value = np.array(self.value, dtype=self.dtype)
        return value, self.requires_grad

    def gen_single_tensor(self):
        value = self.gen_tensor(self.shape, self.dtype)
        return value, self.requires_grad

    def gen_tensor(self, shape, dtype):
        value = self.gen_fn(shape=shape, dtype=dtype)
        if self.no_contiguous:
            value = value.transpose()
        return value

    def parse_gen_fn(self, gen_fn):
        if isinstance(gen_fn, dict):
            gen_fn = eval(self.arg["gen_fn"]["fn"])
            low = self.arg["gen_fn"].get("low", 0)
            high = self.arg["gen_fn"].get("high", 10)
            gen_fn = partial(gen_fn, low=low, high=high)
        else:
            gen_fn = eval(self.arg["gen_fn"])
            if (
                gen_fn == Genfunc.randint or gen_fn == Genfunc.uniform or gen_fn == Genfunc.randn_int
            ):
                gen_fn = partial(gen_fn, low=0, high=10)
        return gen_fn


# test for generate functions
if __name__ == "__main__":
    shape = (3, 4)
    dtype = np.float16
    func = f"Genfunc.randn({str(shape)}, np.{np.dtype(dtype).name})"
    t = eval(func)
    print(type(t), "\n", t.dtype)

    # GenInputData.run(os.path.join(os.path.dirname(__file__), '../cache/diopi_case_items.cfg'),
    #                  os.path.join(os.path.dirname(__file__), '../cache/data/inputs/'))

    cfg = {
        "atol": 1e-05,
        "rtol": 1e-05,
        "atol_half": 0.01,
        "rtol_half": 0.05,
        "mismatch_ratio_threshold": 0.001,
        "memory_format": "NCHW",
        "fp16_exact_match": False,
        "train": True,
        "gen_policy": "dafault",
        "name": "clip_grad_norm_",
        "interface": ["CustomizedTest"],
        "para": {"max_norm": 1.0, "norm_type": 0, "error_if_nonfinite": True},
        "tensor_para": {
            "args": [
                {
                    "ins": "tensors",
                    "shape": ((),),
                    "gen_fn": "Genfunc.randn",
                    "dtype": np.float32,
                    "gen_num_range": [1, 5],
                    "gen_policy": "gen_tensor_list_diff_shape",
                    "requires_grad": [False],
                }
            ],
            "seq_name": "tensors",
        },
        "requires_backward": [],
        "tag": [],
        "saved_args": {},
    }

    gc = GenParas(cfg)
    d = gc.gen_data()
    print(d)
