import numpy as np

from conformance.diopi_runtime import Tensor
from conformance.exception import InputChangedException, OutputCheckFailedException
# from utils import logger
from conformance.global_settings import glob_vars


class CheckResult(object):
    @staticmethod
    def compare_input(input1: dict, input2: dict, ignore_paras_for_input_check: list):
        input1 = {key: value for key, value in input1.items() if key not in ignore_paras_for_input_check}
        input2 = {key: value for key, value in input2.items() if key not in ignore_paras_for_input_check}
        if input1.keys() != input2.keys():
            raise InputChangedException(f"input1's keys {list(input1.keys())} is not equal to input2's keys {list(input2.keys())}.")

        passed = True
        for name, value in input1.items():
            matched = np.isclose(value, input2[name])
            mismatched_num = matched.size - np.sum(matched)
            # passed = mismatched_num <= default_cfg_dict['default_option']['mismatch_ratio_threshold'] * matched.size
            passed = mismatched_num == 0
            glob_vars.func_status[glob_vars.cur_test_func] = 'passed'
            if not passed:
                glob_vars.func_status[glob_vars.cur_test_func] = 'failed'
                debug_level = glob_vars.debug_level
                error_info = f'Run {glob_vars.cur_test_func} failed, because of inputs: {name} changed'
                if debug_level > 1:
                    error_info += (f"\n\texpect input: \n\t\t{value}\n\tbut got\n\t\t{input2[name]}")
                raise InputChangedException(error_info)

    @staticmethod
    def compare_output(output, output_reference, **kwargs):
        if isinstance(output, Tensor):
            compare_fn = CheckResult.compare_tensor
        elif isinstance(output, (list, tuple)):
            compare_fn = CheckResult.compare_list
        elif isinstance(output, dict):
            compare_fn = CheckResult.compare_dict
        elif isinstance(output, (int, float)):
            compare_fn = CheckResult.compare_num
        else:
            raise TypeError(f'Not support output type {type(output)}')
        compare_fn(output, output_reference, **kwargs)

    @staticmethod
    def compare_tensor(output, output_reference, **kwargs):
        CheckResult.allclose(output.numpy(), output_reference, **kwargs)

    @staticmethod
    def compare_list(output, output_reference, **kwargs):
        assert isinstance(output_reference, (list, tuple))
        assert len(output) == len(output_reference)
        for i in range(len(output)):
            if isinstance(output[i], Tensor):
                kwargs['name'] = "out" + str(i)
                CheckResult.allclose(output[i].numpy(), output_reference[i], **kwargs)

    @staticmethod
    def compare_dict(output, output_reference, **kwargs):
        assert isinstance(output_reference, dict)
        assert len(output) == len(output_reference)
        for k, v in output.items():
            if isinstance(v, Tensor):
                kwargs['name'] = k
                CheckResult.allclose(v.numpy(), output_reference[k], **kwargs)

    @staticmethod
    def compare_num(output, output_reference, **kwargs):
        assert isinstance(output_reference, np.ndarray), "output_reference should be type numpy.array"
        output = np.array(output)
        assert output.shape == output_reference.shape, "output and output_reference should be same shape"
        kwargs['name'] = "scalar"
        CheckResult.allclose(output, output_reference, **kwargs)

    @staticmethod
    def allclose(tensor1: np.ndarray, tensor2: np.ndarray, **kwargs) -> bool:
        var_name = kwargs.get('name', 'out')
        sum_to_compare = kwargs.get('sum_to_compare', False)
        rtol = kwargs.get('rtol', 1e-5)
        atol = kwargs.get('atol', 1e-8)
        tensor1 = np.sum(tensor1) if sum_to_compare else tensor1
        tensor2 = np.sum(tensor2) if sum_to_compare else tensor2
        matched = np.isclose(tensor1, tensor2, rtol, atol, True)
        mismatched_num = matched.size - np.sum(matched)
        # passed = mismatched_num <= default_cfg_dict['default_option']['mismatch_ratio_threshold'] * matched.size
        passed = mismatched_num == 0
        glob_vars.func_status[glob_vars.cur_test_func] = 'passed'
        if not passed:
            glob_vars.func_status[glob_vars.cur_test_func] = 'failed'
            sum1 = tensor1.sum()
            sum2 = tensor2.sum()
            mask = np.isclose(tensor1, tensor2, rtol, atol, True)
            count = np.count_nonzero(np.equal(mask, False))
            debug_level = glob_vars.debug_level
            if tensor1.dtype == np.bool_:
                max_diff = 1
                error_info = f"The count of elements that do not meet the accuracy requirement is {count}.\n" + \
                    f"Max of diff is {max_diff}.\n"
            else:
                assert tensor1.size == tensor2.size, "tensor1 element num does not equal tensor2's."
                diff = np.abs(tensor1 - tensor2)
                max_diff = np.abs(tensor1 - tensor2).max()
                max_diff_index = np.unravel_index(np.argmax(diff), diff.shape)
                max_diff_elem = tensor1[max_diff_index]
                max_diff_elem_ref = tensor2[max_diff_index]
                error_info = f"The count of elements that do not meet the accuracy requirement is {count}.\n" + \
                    f"The dtype of {var_name} is {tensor1.dtype}.\n" + \
                    f"The shape of {var_name} is {tensor1.shape}.\n" + \
                    f"The stride of {var_name} is {np.divide(tensor1.strides, tensor1.itemsize).astype(np.int32)}.\n" + \
                    f"The max of diff is {max_diff}. Specifically, the actual val is {max_diff_elem} and the expected is {max_diff_elem_ref}.\n"
            if debug_level > 0:
                error_info += f"Sum of {var_name} is {sum1}, Sum of {var_name}_ref is {sum2}, Max of diff is {max_diff}.\n"
                if debug_level > 1:
                    error_info += f"{var_name} is {tensor1},\n{var_name}_ref is {tensor2},\nMask is {mask}\n"
            raise OutputCheckFailedException(error_info)
