# Copyright (c) 2023, DeepLink.
import os
import re
import pickle

import numpy as np
from collections import defaultdict
from codegen.filemanager import FileManager
from codegen.case_template import CaseTemplate
from conformance.db_operation import db_conn
from conformance.utils import gen_pytest_case_nodeid


class GenConfigTestCase(object):
    def __init__(self, module='diopi', config_path='./cache/device_case_items.cfg', tests_path='./gencases/diopi_case') -> None:
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
            raise FileNotFoundError(f'[GenTestCase][load_items] config file {self._config_path} not found.')
        with open(self._config_path, 'rb') as f:
            self.__case_items = pickle.load(f)

    def _gen_function_set(self):
        for key, value in self.__case_items.items():
            prefix_key, _ = re.split('_[0-9]+\.', key)
            self.__function_set[prefix_key][key] = value

    def get_function_set(self):
        return dict(self.__function_set)

    def gen_test_cases(self):
        for tk, tv in self.__function_set.items():
            gc = GenTestCase(self._module, tk, tv, module_path=self._tests_path)
            gc.gen_test_module()
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
    def __init__(self, module='diopi', suite_key='', case_set=dict(), module_path='.') -> None:
        self._module = module
        self._suite_name, self._func_name = suite_key.split('::')
        self._case_set = case_set
        self._fm = FileManager(module_path)
        self.db_case_items = []

    def _set_fm_write(self):
        mt_name = f'test_{self._module}_{self._suite_name}_{self._func_name}.py'
        self._fm.will_write(mt_name)

    def gen_test_module(self):
        test_diopi_head_import = ''
        test_case_items = []
        for ck, cv in self._case_set.items():
            # test_diopi_function_module = 'diopi_functions'
            test_diopi_func_name = self._func_name

            # forward process
            case_config_name = ck.split('::')[1]
            func_case_name = case_config_name.split('.pth')[0]
            input_data_path = ck
            output_data_path = ck

            # get tol
            test_compare_tol = dict(atol = cv['atol'], rtol=cv['rtol'])
            for tensor in cv['tensor_para']['args']:
                if tensor['ins'] == 'input' and tensor['dtype'] in [np.int16, np.float16, np.uint16]:
                    test_compare_tol['atol'] = cv['atol_half']
                    test_compare_tol['rtol'] = cv['rtol_half']
                    break

            # no_output_ref
            if 'no_output_ref' in cv.keys() and cv['no_output_ref'] is True:
                test_function_forward_call = CaseTemplate.test_manual_function_forward_call.substitute(
                    env=dict(test_diopi_func_name = test_diopi_func_name)
                )
                test_diopi_head_import = CaseTemplate.test_diopi_manual_import.substitute(env={})
                test_function_ref_data_path = ''
            else:
                test_function_forward_call = CaseTemplate.test_diopi_function_forward_call.substitute(
                    env=dict(
                        test_caompare_tol = test_compare_tol,
                        test_diopi_func_name = test_diopi_func_name
                    )
                )
                test_function_ref_data_path = f"f_out = os.path.join(data_path, '{self._module}', 'outputs', '{output_data_path}')"
                # test_diopi_head_import = CaseTemplate.test_diopi_function_import

            forward = CaseTemplate.test_function_body_forward.substitute(
                env=dict(
                    test_module_name = self._module,
                    input_data_path = input_data_path,
                    # output_data_path = output_data_path,
                    test_function_ref_data_path = test_function_ref_data_path,
                    test_function_forward_call = test_function_forward_call
                )
            )

            # process backward if needed
            requires_grad = False
            test_import_diopi_bp_func = ''
            for tensor in cv['tensor_para']['args']:
                if True in tensor['requires_grad']:
                    requires_grad = True
                    break
            nodeid = gen_pytest_case_nodeid(self._fm.output_dir,
                                             f'test_{self._module}_{self._suite_name}_{self._func_name}.py',
                                             f'TestM{self._module}S{self._suite_name}F{self._func_name}',
                                             f'test_{func_case_name}')
            item = {'case_name': ck, 'model_name': self._module, 'pytest_nodeid': nodeid, 'inplace_flag': 0, 'backward_flag': 0,
                    'func_name': self._func_name, 'case_config': cv, 'result': 'skipped'}

            backward = ''
            if requires_grad:
                test_import_diopi_bp_func = f'from conformance.diopi_functions import {self._func_name}_backward'
                bp_output_data_path = output_data_path.split('.pth')[0] + '_backward.pth'
                test_diopi_bp_func_name = self._func_name + '_backward'
                backward = CaseTemplate.test_function_body_backward.substitute(env=dict(
                    test_module_name = self._module,
                    bp_output_data_path = bp_output_data_path,
                    test_diopi_bp_func_name = test_diopi_bp_func_name)
                )
                item['backward_flag'] = 1

            forward_inp = ''
            if 'is_inplace' in cv.keys() and cv['is_inplace'] is True:
                item['inplace_flag'] = 1
                if requires_grad is True:
                    test_diopi_func_inp_remove_grad_args = CaseTemplate.test_diopi_func_inp_remove_grad_args.substitute({})
                else:
                    test_diopi_func_inp_remove_grad_args = ''
                if 'no_output_ref' in cv.keys() and cv['no_output_ref'] is True:
                    forward_inp = CaseTemplate.test_manual_function_inp_forward_call.substitute(
                        env=dict(
                            test_diopi_func_inp_remove_grad_args = test_diopi_func_inp_remove_grad_args,
                            test_diopi_func_name = test_diopi_func_name
                        )
                    )
                else:
                    forward_inp = CaseTemplate.test_diopi_function_inp_forward_call.substitute(
                        env=dict(
                            test_diopi_func_inp_remove_grad_args = test_diopi_func_inp_remove_grad_args,
                            test_diopi_func_name = test_diopi_func_name
                        )
                    )

            # test function
            # import pdb
            # pdb.set_trace()
            if len(cv['tensor_para']['args']) > 0:
                input_dtype =  cv['tensor_para']['args'][0]['dtype'].__name__;
            else:
                input_dtype = 'float32'
            test_function_templ = CaseTemplate.test_function_templ.substitute(env=dict(
                input_dtype = input_dtype,
                func_case_name = func_case_name,
                forward = forward,
                backward = backward,
                forward_inp = forward_inp
            ))

            test_case_items.append(test_function_templ)
            self.db_case_items.append(item)
            # one case
        if test_diopi_head_import == '':
            test_diopi_head_import = CaseTemplate.test_diopi_function_import.substitute(
                env=dict(
                    test_diopi_func_name = test_diopi_func_name,
                    test_import_diopi_bp_func = test_import_diopi_bp_func
                )
            )

        test_class_name = f'TestM{self._module}S{self._suite_name}F{self._func_name}'
        test_class_templ = CaseTemplate.test_class_templ.substitute(env=dict(
            test_diopi_head_import = test_diopi_head_import,
            test_class_name = test_class_name,
            test_case_items = test_case_items
        ))

        file_name = f'test_{self._module}_{self._suite_name}_{self._func_name}.py'
        self._fm.write(file_name, test_class_templ)

if __name__ == '__main__':
    gctc = GenConfigTestCase()
    gctc.gen_test_cases()
