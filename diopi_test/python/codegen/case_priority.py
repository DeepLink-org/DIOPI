# Copyright (c) 2023, DeepLink.

from codegen.priority_config import P0, P1, P2
from typing import DefaultDict


class CasePriority(object):
    def __init__(self, case_name, case_cfg, priority_config) -> None:
        self._case_name = case_name
        self._case_cfg = case_cfg
        self._priority_config = priority_config

    def get_case_priority(self):
        priority = self._case_cfg.get("priority")
        if priority is not None and priority != '':
            return priority
        else:
            priority = self._gen_priority_by_priority_cfg()
            return self._gen_priority_by_case_cfg() if priority is None else priority

    def _gen_priority_by_priority_cfg(self):
        if self._case_cfg['name'] in self._priority_config.get(self._case_name, {}).get('name', []):
            op_priority_config = self._priority_config[self._case_name]

            if "tensor_para" in op_priority_config.keys() and "args" in op_priority_config["tensor_para"].keys() and 'tensor_para' in self._case_cfg:
                priority_args_dict = DefaultDict(dict)
                for each_args in op_priority_config["tensor_para"]["args"]:
                    if 'dtype' in each_args.keys():
                        for each_dtype in each_args['dtype']:
                            if isinstance(each_dtype, (P0, P1, P2)):
                                priority_args_dict[each_args['ins']
                                                   [0]]['dtype'] = each_dtype
                    if 'shape' in each_args.keys():
                        for each_shape in each_args['shape']:
                            if isinstance(each_shape, (P0, P1, P2)):
                                priority_args_dict[each_args['ins']
                                                   [0]]['shape'] = each_shape

                for tensor_arg in self._case_cfg['tensor_para']['args']:
                    if tensor_arg['ins'] in priority_args_dict.keys():
                        if 'dtype' in priority_args_dict[tensor_arg['ins']].keys():
                            if tensor_arg['dtype'] == priority_args_dict[tensor_arg['ins']]['dtype'].value():
                                return priority_args_dict[tensor_arg['ins']]['dtype'].priority()
                        if 'shape' in priority_args_dict[tensor_arg['ins']].keys():
                            if tensor_arg['shape'] == priority_args_dict[tensor_arg['ins']]['shape'].value():
                                return priority_args_dict[tensor_arg['ins']]['shape'].priority()

            if 'para' in op_priority_config and 'para' in self._case_cfg:
                for para_key, para_value in self._case_cfg['para'].items():
                    if para_key in op_priority_config['para']:
                        for priority_value in op_priority_config['para'][para_key]:
                            if isinstance(priority_value, (P0, P1, P2)) and para_value == priority_value.value():
                                return priority_value.priority()

    def _gen_priority_by_case_cfg(self):
        priority_methods = [
            self._priority_strategy_zero_size_tensor,
            self._priority_strategy_no_continuous,
            self._priority_strategy_stride,
            # Add more priority strategies here...
        ]
        for method in priority_methods:
            priority = method()
            if priority:
                return priority
        return 'P0'

    def _priority_strategy_no_continuous(self):
        # Check if input is a no_continuous tensor
        tensor_args = self._case_cfg.get("tensor_para", {}).get("args", [])
        for tensor_arg in tensor_args:
            if tensor_arg.get('no_contiguous'):
                return 'P1'

    def _priority_strategy_stride(self):
        # Check if input has stride
        tensor_args = self._case_cfg.get("tensor_para", {}).get("args", [])
        for tensor_arg in tensor_args:
            if tensor_arg.get('stride'):
                return 'P1'

    def _priority_strategy_zero_size_tensor(self):
        # Check if input is a zero-size tensor
        tensor_args = self._case_cfg.get("tensor_para", {}).get("args", [])
        for tensor_arg in tensor_args:
            if tensor_arg.get("shape") and 0 in tensor_arg.get("shape"):
                return 'P2'


if __name__ == "__main__":
    import numpy as np

    case_name = "threshold"
    priority_config = {
        case_name: dict(
            name=["threshold"],
            para=dict(
                threshold=[P2(False)],
                value=[P1(True)],
            ),
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "shape": [P0((64,))],
                        "dtype": [P1(np.float16)],
                    },
                ]
            ),
        ),
    }

    case_cfg = dict(
        name="threshold",
        is_inplace=True,
        para=dict(
            threshold=False,
            value=0,
        ),
        tensor_para=dict(
            genfunc='Genfunc.randn',
            args=[
                {
                    "ins": 'input',
                    "requires_grad": True,
                    "shape": (64, ),
                    "dtype": np.float16,
                },
            ]
        ),
    )
    case_priority = CasePriority(case_name, case_cfg, priority_config)
    assert case_priority.get_case_priority() == 'P1'
