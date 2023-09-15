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

    def _set_fm_write(self):
        mt_name = f'test_{self._module}_{self._suite_name}_{self._func_name}.py'
        self._fm.will_write(mt_name)

    def gen_test_module(self):
        test_diopi_function_module = 'diopi_functions'
        test_diopi_func_import = self._func_name
        test_diopi_func_name = self._func_name

        test_forward_func_call_prefix = 'dev_out = '

        test_case_items = []
        all_case_items = []
        for ck, cv in self._case_set.items():
            # forward process
            case_config_name = ck.split('::')[1]
            func_case_name = case_config_name.split('.pth')[0]
            input_data_path = ck
            output_data_path = ck

            test_caompare_tol = dict(atol = cv['atol'], rtol=cv['rtol'])
            for tensor in cv['tensor_para']['args']:
                if tensor['ins'] == 'input' and tensor['dtype'] in [np.int16, np.float16, np.uint16]:
                    test_caompare_tol['atol'] = cv['atol_half']
                    test_caompare_tol['rtol'] = cv['rtol_half']
                    break
            if 'no_output_ref' in cv.keys() and cv['no_output_ref'] == True:
                test_function_forward_ref_compare = ''
                if test_diopi_function_module != 'diopi_manual_functions':
                    test_diopi_function_module = 'diopi_manual_functions'
                    test_diopi_func_import = 'ManualTest'
                    test_diopi_func_name = f'ManualTest.test_{test_diopi_func_name}'
                    test_forward_func_call_prefix = ''
            else:
                test_function_forward_ref_compare = CaseTemplate.test_function_forward_ref_compare.substitute(env={})

            forward = CaseTemplate.test_function_body_forward.substitute(env=dict(
                test_module_name = self._module,
                input_data_path = input_data_path,
                output_data_path = output_data_path,
                test_caompare_tol = test_caompare_tol,
                test_diopi_func_name = test_forward_func_call_prefix + test_diopi_func_name,
                test_function_forward_ref_compare = test_function_forward_ref_compare
            ))

            requires_grad = False
            test_import_diopi_bp_func = ''
            for tensor in cv['tensor_para']['args']:
                if True in tensor['requires_grad']:
                    requires_grad = True
                    break
            if 'is_inplace' in cv.keys() and cv['is_inplace'] == True:
                requires_grad = False

            backward = ''
            if requires_grad:
                test_import_diopi_bp_func = f'from conformance.diopi_functions import {self._func_name}_backward'
                bp_output_data_path = output_data_path.split('.pth')[0] + '_backward.pth'
                test_diopi_bp_func_name = self._func_name + '_backward'
                backward = CaseTemplate.test_function_body_backward.substitute(env=dict(
                    test_module_name = self._module,
                    bp_output_data_path = bp_output_data_path,
                    test_diopi_bp_func_name = test_diopi_bp_func_name
                ))

            # test function
            test_function_templ = CaseTemplate.test_function_templ.substitute(env=dict(
                func_case_name = func_case_name,
                forward = forward,
                backward = backward
            ))

            test_case_items.append(test_function_templ)

            nodeid = gen_pytest_case_nodeid(self._fm.output_dir,
                                             f'test_{self._module}_{self._suite_name}_{self._func_name}.py',
                                             f'TestM{self._module}S{self._suite_name}F{self._func_name}',
                                             f'test_{func_case_name}')
            item = {'case_name': ck, 'model_name': self._module, 'pytest_nodeid': nodeid,
                    'func_name': self._func_name, 'case_config': cv, 'result': 'skipped'}
            all_case_items.append(item)

        test_diopi_head_import = CaseTemplate.test_diopi_head_import.substitute(env=dict(
            test_diopi_function_module = test_diopi_function_module,
            test_diopi_func_import = test_diopi_func_import,
            test_diopi_func_name = test_diopi_func_name,
            test_import_diopi_bp_func = test_import_diopi_bp_func
        ))

        test_class_name = f'TestM{self._module}S{self._suite_name}F{self._func_name}'
        test_class_templ = CaseTemplate.test_class_templ.substitute(env=dict(
            test_diopi_head_import = test_diopi_head_import,
            test_class_name = test_class_name,
            test_case_items = test_case_items
        ))

        file_name = f'test_{self._module}_{self._suite_name}_{self._func_name}.py'
        self._fm.write(file_name, test_class_templ)

        db_conn.will_insert_device_case(all_case_items)

if __name__ == '__main__':
    gctc = GenConfigTestCase()
    gctc.gen_test_cases()
