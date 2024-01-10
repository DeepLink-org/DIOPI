# Copyright (c) 2023, DeepLink.
import os
import re
import pickle

import numpy as np
from collections import defaultdict

from torch import is_floating_point
from codegen.filemanager import FileManager
from codegen.case_template import CaseTemplate
from conformance.db_operation import db_conn
from conformance.utils import gen_pytest_case_nodeid
from conformance.global_settings import glob_vars
from conformance.global_op_list import dtype_op, dtype_out_op, nhwc_op, ops_with_states


class GenConfigTestCase(object):
    def __init__(
        self,
        module="diopi",
        config_path="./cache/device_case_items.cfg",
        tests_path="./gencases/diopi_case",
    ) -> None:
        self._config_path = config_path
        self._tests_path = tests_path
        self._module = module
        self.db_case_items = []

        self.__case_items = None
        # d = dict(
        #     batch_norm::batch_norm: dict(
        #         batch_norm::batch_norm_0.pth: dict(),
        #         batch_norm::batch_norm_1.pth: dict(),
        #         ...
        #     )
        # )
        self.__function_set = defaultdict(dict)

        self._load_items()
        self._gen_function_set()

    def _load_items(self):
        if not os.path.isfile(self._config_path):
            raise FileNotFoundError(
                f"[GenTestCase][load_items] config file {self._config_path} not found."
            )
        with open(self._config_path, "rb") as f:
            self.__case_items = pickle.load(f)

    def _gen_function_set(self):
        for key, value in self.__case_items.items():
            prefix_key, _ = re.split(r"_[0-9]+\.", key)
            self.__function_set[prefix_key][key] = value

    def get_function_set(self):
        return dict(self.__function_set)

    def gen_test_cases(self, fname="all_ops"):
        for tk, tv in self.__function_set.items():
            gc = GenTestCase(self._module, tk, tv, module_path=self._tests_path)
            gc.gen_test_module(fname)
            self.db_case_items.extend(gc.db_case_items)


####################################################################################################
# d = dict(
#     batch_norm::batch_norm : dict(
#         batch_norm::batch_norm_0.pth: dict(),
#         batch_norm::batch_norm_1.pth: dict(),
#         ...
#     )
# )
####################################################################################################
class GenTestCase(object):
    def __init__(
        self, module="diopi", suite_key="", case_set=dict(), module_path="."
    ) -> None:
        self._module = module
        self._suite_name, self._func_name = suite_key.split("::")
        self._case_set = case_set
        self._fm = FileManager(module_path)
        self.db_case_items = []

    def _set_fm_write(self):
        mt_name = f"test_{self._module}_{self._suite_name}_{self._func_name}.py"
        self._fm.will_write(mt_name)

    def _gen_test_priority(self, case_cfg):
        priority = 'P0'
        if len(case_cfg["tensor_para"]["args"]) > 0:
                for t in case_cfg["tensor_para"]["args"]:
                    if t['ins'] == 'input':
                        if not np.issubdtype(t['dtype'], np.floating):
                            priority = 'P1'
                        if  t.get('shape') and (len(t['shape']) == 0 or 0 in t['shape']):
                            priority = 'P2'
                        break
        return priority

    def gen_test_module(self, fname):
        test_diopi_head_import = ""
        test_case_items = []
        if fname not in [self._func_name, "all_ops"]:
            return
        for ck, cv in self._case_set.items():
            # test_diopi_function_module = 'diopi_functions'
            test_diopi_func_name = self._func_name

            # forward process
            case_config_name = ck.split("::")[1]
            func_case_name = case_config_name.split(".pth")[0]
            input_data_path = ck
            output_data_path = ck

            # get tol
            test_compare_tol = dict(atol=cv["atol"], rtol=cv["rtol"], mismatch_ratio_threshold=cv["mismatch_ratio_threshold"])
            for tensor in cv["tensor_para"]["args"]:
                if tensor["dtype"] in [
                    np.int16,
                    np.float16,
                    np.uint16,
                ]:
                    test_compare_tol["atol"] = cv["atol_half"]
                    test_compare_tol["rtol"] = cv["rtol_half"]
                    break

            # no_output_ref
            if "no_output_ref" in cv.keys() and cv["no_output_ref"] is True:
                test_function_forward_call = (
                    CaseTemplate.test_manual_function_forward_call.substitute(
                        env=dict(test_diopi_func_name=test_diopi_func_name)
                    )
                )
                test_diopi_head_import = (
                    CaseTemplate.test_diopi_manual_import.substitute(env={})
                )
                test_function_ref_data_path = ""
            else:
                test_function_forward_call = (
                    CaseTemplate.test_diopi_function_forward_call.substitute(
                        env=dict(
                            test_atol=cv["atol"],
                            test_rtol=cv["rtol"],
                            test_atol_half=cv["atol_half"],
                            test_rtol_half=cv["atol_half"],
                            test_compare_tol=test_compare_tol,
                            test_diopi_func_name=test_diopi_func_name,
                        )
                    )
                )
                test_function_ref_data_path = f"f_out = os.path.join(data_path, '{self._module}', 'outputs', '{output_data_path}')"
                # test_diopi_head_import = CaseTemplate.test_diopi_function_import

            test_set_four_bytes = ""
            if glob_vars.four_bytes and self._func_name in dtype_op:
                test_set_four_bytes = CaseTemplate.test_set_four_bytes.substitute(
                    env=dict(dtype_list=str(dtype_op[self._func_name]))
                )

            test_set_nhwc, test_diopi_nhwc_import = "", ""
            if glob_vars.nhwc and self._func_name in nhwc_op:
                test_set_nhwc = CaseTemplate.test_set_nhwc.substitute(
                    env=dict(
                        nhwc_list=str(nhwc_op[self._func_name]),
                        nhwc_min_dim=glob_vars.nhwc_min_dim,
                    )
                )
                test_diopi_nhwc_import = CaseTemplate.test_diopi_nhwc_import.substitute(env={})
            test_set_stride = ""
            has_stride = {
                i["ins"] + "stride": i[i["ins"] + "stride"]
                for i in cv["tensor_para"]["args"]
                if i.get(i["ins"] + "stride")
            }
            if len(has_stride) > 0:
                test_set_stride = CaseTemplate.test_set_stride.substitute(
                    env=dict(
                        has_stride=str(list(has_stride.keys())),
                        args_stride=str(has_stride),
                    )
                )

            gen_policy_args = [
                i["ins"]
                for i in cv["tensor_para"]["args"]
                if i.get("gen_policy")
                in ["gen_tensor_list", "gen_tensor_list_diff_shape"]
            ]
            if len(gen_policy_args) > 0:
                test_to_tensor = CaseTemplate.test_to_tensor_list.substitute(
                    env=dict(gen_policy_args=str(gen_policy_args))
                )
            else:
                test_to_tensor = CaseTemplate.test_to_tensor.substitute({})

            test_preprocess_parameters = (
                CaseTemplate.test_preprocess_parameters.substitute(
                    env=dict(
                        set_four_bytes=test_set_four_bytes,
                        set_nhwc=test_set_nhwc,
                        set_stride=test_set_stride,
                        to_tensor=test_to_tensor,
                    )
                )
            )

            # compare_input
            ignore_paras_for_input_check = ops_with_states.get(self._func_name, set())

            forward = CaseTemplate.test_function_body_forward.substitute(
                env=dict(
                    test_module_name=self._module,
                    input_data_path=input_data_path,
                    # output_data_path = output_data_path,
                    test_function_ref_data_path=test_function_ref_data_path,
                    test_function_forward_call=test_function_forward_call,
                    ignore_paras_for_input_check=ignore_paras_for_input_check,
                    preprocess_parameters=test_preprocess_parameters,
                )
            )

            # process backward if needed
            requires_grad = False
            test_import_diopi_bp_func = ""
            for tensor in cv["tensor_para"]["args"]:
                if True in tensor["requires_grad"]:
                    requires_grad = True
                    break
            nodeid = gen_pytest_case_nodeid(
                self._fm.output_dir,
                f"test_{self._module}_{self._suite_name}_{self._func_name}.py",
                f"TestM{self._module}S{self._suite_name}F{self._func_name}",
                f"test_{func_case_name}",
            )
            item = {
                "case_name": ck,
                "model_name": self._module,
                "pytest_nodeid": nodeid,
                "inplace_flag": 0,
                "backward_flag": 0,
                "func_name": self._func_name,
                "case_config": cv,
                "result": "skipped",
            }

            backward = ""
            if requires_grad:
                test_import_diopi_bp_func = f"from conformance.diopi_functions import {self._func_name}_backward"
                bp_output_data_path = (
                    output_data_path.split(".pth")[0] + "_backward.pth"
                )
                test_diopi_bp_func_name = self._func_name + "_backward"
                backward = CaseTemplate.test_function_body_backward.substitute(
                    env=dict(
                        test_module_name=self._module,
                        bp_output_data_path=bp_output_data_path,
                        test_diopi_bp_func_name=test_diopi_bp_func_name,
                    )
                )
                item["backward_flag"] = 1

            forward_inp = ""
            if "is_inplace" in cv.keys() and cv["is_inplace"] is True:
                item["inplace_flag"] = 1
                if requires_grad is True:
                    test_diopi_func_inp_remove_grad_args = (
                        CaseTemplate.test_diopi_func_inp_remove_grad_args.substitute({})
                    )
                else:
                    test_diopi_func_inp_remove_grad_args = ""
                if "no_output_ref" in cv.keys() and cv["no_output_ref"] is True:
                    forward_inp = CaseTemplate.test_manual_function_inp_forward_call.substitute(
                        env=dict(
                            test_diopi_func_inp_remove_grad_args=test_diopi_func_inp_remove_grad_args,
                            test_diopi_func_name=test_diopi_func_name,
                        )
                    )
                else:
                    forward_inp = CaseTemplate.test_diopi_function_inp_forward_call.substitute(
                        env=dict(
                            test_diopi_func_inp_remove_grad_args=test_diopi_func_inp_remove_grad_args,
                            test_diopi_func_name=test_diopi_func_name,
                        )
                    )

            test_dtype_marks = []
            if len(cv["tensor_para"]["args"]) > 0:
                for t in cv["tensor_para"]["args"]:
                    mark = CaseTemplate.test_function_case_marks.substitute(
                        env=dict(mark=t["dtype"].__name__)
                    ).strip()
                    if mark not in test_dtype_marks:
                        test_dtype_marks.append(mark)

            test_priority_mark = CaseTemplate.test_function_case_marks.substitute(
                        env=dict(mark=self._gen_test_priority(cv))
                    ).strip()

            test_function_templ = CaseTemplate.test_function_templ.substitute(
                env=dict(
                    test_priority_mark=test_priority_mark,
                    test_dtype_marks=test_dtype_marks,
                    func_case_name=func_case_name,
                    forward=forward,
                    backward=backward,
                    forward_inp=forward_inp,
                )
            )

            test_case_items.append(test_function_templ)
            self.db_case_items.append(item)
            # one case
        if test_diopi_head_import == "":
            test_diopi_head_import = CaseTemplate.test_diopi_function_import.substitute(
                env=dict(
                    test_diopi_func_name=test_diopi_func_name,
                    test_import_diopi_bp_func=test_import_diopi_bp_func,
                    test_diopi_nhwc_import=test_diopi_nhwc_import
                )
            )

        test_class_name = f"TestM{self._module}S{self._suite_name}F{self._func_name}"
        test_class_templ = CaseTemplate.test_class_templ.substitute(
            env=dict(
                test_diopi_head_import=test_diopi_head_import,
                test_class_name=test_class_name,
                test_case_items=test_case_items,
            )
        )

        file_name = f"test_{self._module}_{self._suite_name}_{self._func_name}.py"
        self._fm.write(file_name, test_class_templ)


if __name__ == "__main__":
    gctc = GenConfigTestCase()
    gctc.gen_test_cases()
