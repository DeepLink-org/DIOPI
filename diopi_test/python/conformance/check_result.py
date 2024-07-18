import numpy as np

from conformance.diopi_runtime import Tensor, is_dtype, to_numpy_dtype
from conformance.exception import InputChangedException, OutputCheckFailedException
from conformance.global_settings import glob_vars
from copy import deepcopy

class CheckResult(object):

    @staticmethod
    def to_numpy(data):
        r"""
        Recursively convert the tensor and scalar to numpy.
        E.g: {"input": Tensor, "other": Tensor, "input_size": 5, "dtype": Dtype.float16} -> {"input": np.ndarray, "other": np.ndarray, "input_size": np.array(5), "dtype": np.float16}
        """
        if isinstance(data, Tensor):
            data_np = data.numpy()
        elif is_dtype(data):
            data_np = to_numpy_dtype(data)
        elif isinstance(data, (int, float)):
            data_np = np.array(data)
        elif isinstance(data, list):
            data_np = [CheckResult.to_numpy(i) for i in data]
        elif isinstance(data, tuple):
            data_np = tuple(CheckResult.to_numpy(i) for i in data)
        elif isinstance(data, dict):
            data_np = {k: CheckResult.to_numpy(v) for k, v in data.items()}
        elif isinstance(data, np.ndarray):
            raise TypeError("Data type should not be ndarray, please check!")
        else:
            data_np = deepcopy(data)
        return data_np


class CheckInput(CheckResult):
    @staticmethod
    def deep_compare(input, input_reference, ignore_paras_for_input_check={}, **kwargs):
        if isinstance(input, np.ndarray):
            CheckInput.compare_tensor(input, input_reference, **kwargs)
        elif isinstance(input, (list, tuple)):
            CheckInput.compare_list(input, input_reference, **kwargs)
        elif isinstance(input, dict):
            CheckInput.compare_dict(input, input_reference, ignore_paras_for_input_check, **kwargs)
        else:
            CheckInput.compare_others(input, input_reference, **kwargs)

    @staticmethod
    def compare_tensor(input, input_reference, **kwargs):
        CheckInput.allclose(input, input_reference, **kwargs)

    @staticmethod
    def compare_list(input, input_reference, **kwargs):
        if not isinstance(input_reference, (list, tuple)):
            raise InputChangedException(f"There are elements in input that are {type(input)}, but in input_reference, they correspond to {type(input_reference)}")
        if len(input) != len(input_reference):
            raise InputChangedException(f"The length of input is {len(input)}, but which in input_reference is {len(input_reference)}")
        for i in range(len(input)):
            CheckInput.deep_compare(input[i], input_reference[i], **kwargs)

    @staticmethod
    def compare_dict(input, input_reference, ignore_paras_for_input_check, **kwargs):
        if not isinstance(input_reference, dict):
            raise InputChangedException(f"There are elements in input that are dicts, but in input_reference, they correspond to {type(input_reference)}")
        if input.keys() != input_reference.keys():
            raise InputChangedException("Input changed! The keys of input is not equal to input_reference")
        for k, v in input.items():
            if k not in ignore_paras_for_input_check:
                CheckInput.deep_compare(v, input_reference[k], **kwargs)

    @staticmethod
    def compare_others(input, input_reference, **kwargs):
        if type(input) is not type(input_reference):
            raise InputChangedException(f"There are elements in that are {type(input)}. But those in input_reference are {type(input_reference)}")
        if input != input_reference:
            raise InputChangedException(f"Input changed! Expected value should be {input_reference} but get {input}")

    @staticmethod
    def allclose(input, input_reference, **kwargs):
        passed = True
        mismatch_ratio_threshold = kwargs.get('mismatch_ratio_threshold', 1e-3)
        matched = np.isclose(input_reference, input, equal_nan=True)
        mismatched_num = matched.size - np.sum(matched)
        passed = mismatched_num <= mismatch_ratio_threshold * matched.size
        if not passed:
            glob_vars.func_status[glob_vars.cur_test_func] = 'failed'
            debug_level = glob_vars.debug_level
            error_info = f'Run {glob_vars.cur_test_func} failed, because of input changed'
            if debug_level > 1:
                error_info += (f"\n\texpect input: \n\t\t{input_reference}\n\tbut got input reference \n\t\t{input}")
            raise InputChangedException(error_info)


class CheckOutput(CheckResult):

    @staticmethod
    def deep_compare(output, output_reference, **kwargs):
        if isinstance(output, np.ndarray):
            compare_fn = CheckOutput.compare_tensor
        elif isinstance(output, (list, tuple)):
            compare_fn = CheckOutput.compare_list
        elif isinstance(output, dict):
            compare_fn = CheckOutput.compare_dict
        else:
            compare_fn = CheckOutput.compare_others
        compare_fn(output, output_reference, **kwargs)

    @staticmethod
    def compare_tensor(output, output_reference, **kwargs):
        CheckOutput.allclose(output, output_reference, **kwargs)

    @staticmethod
    def compare_list(output, output_reference, **kwargs):
        if not isinstance(output_reference, (list, tuple)):
            raise OutputCheckFailedException(f"Exist element in ouput is {type(output)}, while output_reference is {type(output_reference)}")
        if len(output) != len(output_reference):
            raise OutputCheckFailedException("Length of output is not equal to output_reference")
        for i in range(len(output)):
            kwargs['name'] = "out" + str(i)
            CheckOutput.deep_compare(output[i], output_reference[i], **kwargs)

    @staticmethod
    def compare_dict(output, output_reference, **kwargs):
        if not isinstance(output_reference, dict):
            raise OutputCheckFailedException(f"Exist element in ouput is dict, while output_reference is {type(output_reference)}")
        if output.keys() != output_reference.keys():
            raise OutputCheckFailedException("Key of output is not equal to output_reference")
        for k, v in output.items():
            kwargs['name'] = k
            CheckOutput.deep_compare(v, output_reference[k], **kwargs)

    @staticmethod
    def compare_others(output, output_reference, **kwargs):
        if type(output) is not type(output_reference):
            raise OutputCheckFailedException(f"There are elements in that are {type(output)}. But those in output_reference are {type(output_reference)}")
        if output != output_reference:
            raise OutputCheckFailedException(f"output changed! Expected value should be {output_reference} but get {output}")

    @staticmethod
    def allclose(tensor_dev: np.ndarray, tensor_ref: np.ndarray, **kwargs) -> bool:
        var_name = kwargs.get('name', 'out')
        sum_to_compare = kwargs.get('sum_to_compare', False)
        # sum_to_compare = False
        rtol = kwargs.get('rtol', 1e-5)
        atol = kwargs.get('atol', 1e-8)
        mismatch_ratio_threshold = kwargs.get('mismatch_ratio_threshold', 1e-3)
        tensor_dev = np.sum(tensor_dev) if sum_to_compare else tensor_dev
        tensor_ref = np.sum(tensor_ref) if sum_to_compare else tensor_ref
        matched = np.isclose(tensor_dev, tensor_ref, rtol, atol, equal_nan=True)
        mismatched_num = matched.size - np.sum(matched)
        passed = mismatched_num <= mismatch_ratio_threshold * matched.size
        glob_vars.func_status[glob_vars.cur_test_func] = 'passed'
        if not passed:
            glob_vars.func_status[glob_vars.cur_test_func] = 'failed'
            print(f"tensor_dev {tensor_dev}")
            print(f"tensor_ref {tensor_ref}")
            sum1 = tensor_dev.sum()
            sum2 = tensor_ref.sum()
            mask = np.isclose(tensor_dev, tensor_ref, rtol, atol, equal_nan=True)
            count = np.count_nonzero(np.equal(mask, False))
            debug_level = glob_vars.debug_level
            if debug_level < 1:
                print(f'debug_level {debug_level}')
                debug_level = 100
            if tensor_dev.dtype == np.bool_:
                max_diff = 1
                error_info = f"The count of elements that do not meet the accuracy requirement is {count}.\n" + \
                    f"\n"
            elif tensor_dev.ndim == 0 and tensor_ref.ndim == 0:
                # result is scalar array
                error_info = f"The actual val is {tensor_dev} and the expected is {tensor_ref}.\n"
            else:
                assert tensor_dev.size == tensor_ref.size, "tensor_dev element num does not equal tensor_ref's."
                error_info = f"The count of elements that do not meet the accuracy requirement is {count}.\n" + \
                    f"The dtype of {var_name} is {tensor_dev.dtype}.\n" + \
                    f"The shape of {var_name} is {tensor_dev.shape}.\n" + \
                    f"The stride of {var_name} is {np.divide(tensor_dev.strides, tensor_dev.itemsize).astype(np.int32)}.\n"
                nan_index = np.isnan(tensor_dev) | np.isnan(tensor_ref)
                nan_index[matched] = False          # mismatched nan number index
                if (len(np.argwhere(nan_index)) > 0):
                    # nan number exists
                    error_info += f"Exist mismatched nan number. E.g., the actual val is {tensor_dev[nan_index].ravel()[0]} and the expected is {tensor_ref[nan_index].ravel()[0]}.\n"
                if (not np.array_equal(nan_index, ~matched)):
                    # not all mimatched numbers are nan,exists different numbers
                    diff = np.abs(tensor_dev - tensor_ref)
                    diff[matched] = 0
                    diff[nan_index] = 0
                    max_diff = diff.max()
                    max_diff_index = np.unravel_index(np.argmax(diff), diff.shape)
                    max_diff_elem = tensor_dev[max_diff_index]
                    max_diff_elem_ref = tensor_ref[max_diff_index]
                    error_info += f"The max of diff is {max_diff}. Specifically, the actual val is {max_diff_elem} and the expected is {max_diff_elem_ref}.\n"
            if debug_level > 0:
                if np.isnan(sum1) or np.isnan(sum2):
                    error_info += f"Exists nan, {var_name} is {sum1} and {var_name}_ref is {sum2}.\n"
                else:
                    error_info += f"Sum of {var_name} is {sum1}, Sum of {var_name}_ref is {sum2}.\n"
                if debug_level > 1:
                    error_info += f"{var_name} is {tensor_dev},\n{var_name}_ref is {tensor_ref},\nMask is {mask}\n"
            raise OutputCheckFailedException(error_info)
