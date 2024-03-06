import numpy as np

from conformance.diopi_runtime import Tensor
from conformance.exception import InputChangedException, OutputCheckFailedException
from conformance.global_settings import glob_vars, default_cfg_dict


class CheckResult(object):

    @staticmethod
    def compare_input_list(input: list, input_reference: list, **kwargs):
        if len(input) != len(input_reference):
            raise InputChangedException(f"expect input list len {len(input_reference)} but {input}")

        passed = True
        mismatch_ratio_threshold = kwargs.get('mismatch_ratio_threshold', 1e-3)
        for i in range(len(input)):
            matched = np.isclose(input_reference[i], input[i], equal_nan=True)
            mismatched_num = matched.size - np.sum(matched)
            passed = mismatched_num <= mismatch_ratio_threshold * matched.size
            if not passed:
                glob_vars.func_status[glob_vars.cur_test_func] = 'failed'
                debug_level = glob_vars.debug_level
                error_info = f'Run {glob_vars.cur_test_func} failed, because of input of backward changed'
                if debug_level > 1:
                    error_info += (f"\n\texpect input: \n\t\t{input_reference[i]}\n\tbut got input reference \n\t\t{input[i]}")
                raise InputChangedException(error_info)

    @staticmethod
    def compare_input_dict(input: dict, input_reference: dict, ignore_paras_for_input_check: list, **kwargs):
        input = {key: value for key, value in input.items() if key not in ignore_paras_for_input_check and isinstance(value, np.ndarray)}
        input_reference = {key: value for key, value in input_reference.items() if key not in ignore_paras_for_input_check and isinstance(value, np.ndarray)}
        if input.keys() != input_reference.keys():
            raise InputChangedException(f"input's keys {list(input.keys())} is not equal to input_reference's keys {list(input_reference.keys())}.")

        mismatch_ratio_threshold = kwargs.get('mismatch_ratio_threshold', 1e-3)
        passed = True
        for name, value in input.items():
            matched = np.isclose(input_reference[name], value, equal_nan=True)
            mismatched_num = matched.size - np.sum(matched)
            passed = mismatched_num <= mismatch_ratio_threshold * matched.size
            if not passed:
                glob_vars.func_status[glob_vars.cur_test_func] = 'failed'
                debug_level = glob_vars.debug_level
                error_info = f'Run {glob_vars.cur_test_func} failed, because of input\'s args: {name} changed'
                if debug_level > 1:
                    error_info += (f"\n\texpect input: \n\t\t{input_reference[name]}\n\tbut got input reference \n\t\t{value}")
                raise InputChangedException(error_info)

    @staticmethod
    def compare_output(output, output_reference, **kwargs):
        if isinstance(output, np.ndarray):
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
        CheckResult.allclose(output, output_reference, **kwargs)

    @staticmethod
    def compare_list(output, output_reference, **kwargs):
        assert isinstance(output_reference, (list, tuple))
        assert len(output) == len(output_reference)
        for i in range(len(output)):
            if isinstance(output[i], Tensor) or isinstance(output[i], np.ndarray):
                kwargs['name'] = "out" + str(i)
                CheckResult.allclose(output[i], output_reference[i], **kwargs)

    @staticmethod
    def compare_dict(output, output_reference, **kwargs):
        assert isinstance(output_reference, dict)
        assert len(output) == len(output_reference)
        for k, v in output.items():
            if isinstance(v, Tensor) or isinstance(v, np.ndarray):
                kwargs['name'] = k
                CheckResult.allclose(v, output_reference[k], **kwargs)

    @staticmethod
    def compare_num(output, output_reference, **kwargs):
        assert isinstance(output_reference, np.ndarray), "output_reference should be type numpy.array"
        assert output.shape == output_reference.shape, "output and output_reference should be same shape"
        kwargs['name'] = "scalar"
        CheckResult.allclose(output, output_reference, **kwargs)

    @staticmethod
    def to_numpy(input):
        r"""Switch to numpy. When the input is an iterable object, only tensors within it will be transformed.
        E.g: {"input": Tensor, "other": Tensor, "input_size": 5} -> {"input": np.ndarray, "other": np.ndarray, "input_size": 5}
        """
        if isinstance(input, Tensor):
            input_np = input.numpy()
        elif isinstance(input, list):
            input_np = [i.numpy() if isinstance(i, Tensor) else i for i in input]
        elif isinstance(input, tuple):
            input_np = tuple(i.numpy() if isinstance(i, Tensor) else i for i in input)
        elif isinstance(input, dict):
            input_np = {k: v.numpy() if isinstance(v, Tensor) else v for k, v in input.items()}
        elif isinstance(input, (int, float)):
            input_np = np.array(input)
        else:
            raise TypeError(f'Not support output type {type(input)}')
        return input_np

    @staticmethod
    def allclose(tensor_dev: np.ndarray, tensor_ref: np.ndarray, **kwargs) -> bool:
        var_name = kwargs.get('name', 'out')
        sum_to_compare = kwargs.get('sum_to_compare', False)
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
            sum1 = tensor_dev.sum()
            sum2 = tensor_ref.sum()
            mask = np.isclose(tensor_dev, tensor_ref, rtol, atol, equal_nan=True)
            count = np.count_nonzero(np.equal(mask, False))
            debug_level = glob_vars.debug_level
            if tensor_dev.dtype == np.bool_:
                max_diff = 1
                error_info = f"The count of elements that do not meet the accuracy requirement is {count}.\n" + \
                    f"Max of diff is {max_diff}.\n"
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
                    error_info += f"Sum of {var_name} is {sum1}, Sum of {var_name}_ref is {sum2}, Max of diff is {max_diff}.\n"
                if debug_level > 1:
                    error_info += f"{var_name} is {tensor_dev},\n{var_name}_ref is {tensor_ref},\nMask is {mask}\n"
            raise OutputCheckFailedException(error_info)
