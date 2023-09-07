import pytest
import logging
import numpy as np
import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conformance'))
from conformance.gen_input import GenInputData
from conformance.gen_output import GenOutputData


case_cfg_map = {
    'test forward': {'pointwise_binary::sub_0.pth': {'name': 'sub', 'interface': ['torch'], 'is_inplace': True, 'tensor_para': {'args': [{'ins': 'input', 'shape': (), 'requires_grad': [False], 'gen_num_range': [], 'dtype': np.float64, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}, {'ins': 'other', 'shape': (), 'requires_grad': [False], 'gen_num_range': [], 'dtype': np.float64, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}], 'seq_name': ''}, 'atol': 1e-05, 'rtol': 1e-05, 'atol_half': 0.01, 'rtol_half': 0.05, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'para': {}, 'tag': [], 'saved_args': {}},
                     'pointwise_binary::sub_1.pth': {'name': 'sub', 'interface': ['torch'], 'is_inplace': True, 'tensor_para': {'args': [{'ins': 'input', 'shape': (1024,), 'requires_grad': [False], 'gen_num_range': [], 'dtype': np.int64, 'gen_fn': 'Genfunc.randint', 'gen_policy': 'default'}, {'ins': 'other', 'shape': (1024,), 'requires_grad': [False], 'gen_num_range': [], 'dtype': np.float64, 'gen_fn': dict(fn='Genfunc.uniform', high=10), 'gen_policy': 'default'}], 'seq_name': ''}, 'atol': 1e-05, 'rtol': 1e-05, 'atol_half': 0.01, 'rtol_half': 0.05, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'para': {}, 'tag': [], 'saved_args': {}},
                     'pointwise_binary::sub_2.pth': {'name': 'sub', 'interface': ['torch'], 'tag': ['scalar'], 'is_inplace': True, 'para': {'other': 0}, 'tensor_para': {'args': [{'ins': 'input', 'shape': (), 'requires_grad': [False], 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': dict(fn='Genfunc.randint', low=-10), 'gen_policy': 'default'}], 'seq_name': ''}, 'atol': 1e-05, 'rtol': 1e-05, 'atol_half': 0.01, 'rtol_half': 0.05, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'saved_args': {}}},
    'test backward': {'batch_norm::batch_norm_0.pth': {'name': 'batch_norm', 'atol': 0.001, 'rtol': 0.0001, 'atol_half': 0.1, 'rtol_half': 0.01, 'para': {'training': False, 'momentum': 0.1, 'eps': 1e-05}, 'tensor_para': {'args': [{'ins': 'input', 'shape': (2, 8, 32, 56, 56), 'requires_grad': [True], 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}, {'ins': 'running_mean', 'shape': (8,), 'requires_grad': [False], 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}, {'ins': 'running_var', 'shape': (8,), 'requires_grad': [False], 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}, {'ins': 'weight', 'requires_grad': [True], 'shape': (8,), 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}, {'ins': 'bias', 'requires_grad': [True], 'shape': (8,), 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}], 'seq_name': ''}, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'tag': [], 'saved_args': {}, 'interface': []}},
    'test no_output_ref': {'dropout::dropout_0.pth': {'name': 'dropout', 'no_output_ref': True, 'is_inplace': True, 'para': {'p': 0.5, 'training': True}, 'tensor_para': {'args': [{'ins': 'input', 'shape': (), 'dtype': np.float16, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default', 'requires_grad': [False], 'gen_num_range': []}], 'seq_name': ''}, 'atol': 1e-05, 'rtol': 1e-05, 'atol_half': 0.01, 'rtol_half': 0.05, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'tag': [], 'saved_args': {}, 'interface': []}},
    'test torch.Tensor': {'fill::fill__0.pth': {'name': 'fill_', 'interface': ['torch.Tensor'], 'para': {'value': float('-inf')}, 'tensor_para': {'args': [{'ins': 'input', 'shape': (4, 49), 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default', 'requires_grad': [False], 'gen_num_range': []}], 'seq_name': ''}, 'atol': 1e-05, 'rtol': 1e-05, 'atol_half': 0.01, 'rtol_half': 0.05, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'tag': [], 'saved_args': {}}},
    'test CustomizedTest': {'clip_grad_norm::clip_grad_norm__0.pth': {'name': 'clip_grad_norm_', 'interface': ['CustomizedTest'], 'para': {'max_norm': 5, 'norm_type': 3.0, 'error_if_nonfinite': False}, 'tensor_para': {'args': [{'ins': 'tensors', 'shape': (10, 2), 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'gen_tensor_list', 'gen_num_range': [1, 5], 'requires_grad': [False], 'tensors_num': 3}], 'seq_name': 'tensors'}, 'atol': 1e-05, 'rtol': 1e-05, 'atol_half': 0.01, 'rtol_half': 0.05, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'tag': [], 'saved_args': {}}},
    'test dtype in kwargs': {'prod::prod_0.pth': {'name': 'prod', 'interface': ['torch'], 'atol_half': 0.0001, 'rtol_half': 0.001, 'para': {'dim': 0, 'dtype': np.bool_}, 'tensor_para': {'args': [{'ins': 'input', 'shape': (2, 80, 128, 128, 1), 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default', 'requires_grad': [False], 'gen_num_range': []}], 'seq_name': ''}, 'atol': 1e-05, 'rtol': 1e-05, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'tag': [], 'saved_args': {}},
                             'prod::prod_1.pth': {'name': 'prod', 'interface': ['torch'], 'atol_half': 0.0001, 'rtol_half': 0.001, 'para': {'dim': 0, 'dtype': np.int32}, 'tensor_para': {'args': [{'ins': 'input', 'shape': (2, 80, 128, 128, 1), 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default', 'requires_grad': [False], 'gen_num_range': []}], 'seq_name': ''}, 'atol': 1e-05, 'rtol': 1e-05, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'tag': [], 'saved_args': {}}},
    # 'xfail:test gen output failed': {'conv_2d::conv2d_0.pth': {'name': 'conv2d', 'atol': 0.001, 'rtol': 0.001, 'para': {'stride': 2, 'padding': 0, 'dilation': 1, 'groups': 1}, 'tensor_para': {'args': [{'ins': 'input', 'requires_grad': [True], 'shape': (2, 256, 200, 304), 'gen_num_range': [], 'dtype': np.float64, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}, {'ins': 'weight', 'requires_grad': [True], 'shape': (12, 256, 1, 1), 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}, {'ins': 'bias', 'requires_grad': [True], 'shape': (12,), 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}], 'seq_name': ''}, 'atol_half': 0.01, 'rtol_half': 0.05, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'tag': [], 'saved_args': {}, 'interface': []},
    #                                  'conv_2d::conv2d_1.pth': {'name': 'conv2d', 'atol': 0.001, 'rtol': 0.001, 'para': {'stride': 2, 'padding': 0, 'dilation': 1, 'groups': 1}, 'tensor_para': {'args': [{'ins': 'input', 'requires_grad': [True], 'shape': (2, 256, 200, 304), 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}, {'ins': 'weight', 'requires_grad': [True], 'shape': (12, 256, 1, 1), 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}, {'ins': 'bias', 'requires_grad': [True], 'shape': (12,), 'gen_num_range': [], 'dtype': np.float32, 'gen_fn': 'Genfunc.randn', 'gen_policy': 'default'}], 'seq_name': ''}, 'atol_half': 0.01, 'rtol_half': 0.05, 'mismatch_ratio_threshold': 0.001, 'memory_format': 'NCHW', 'fp16_exact_match': False, 'train': True, 'requires_backward': [], 'tag': [], 'saved_args': {}, 'interface': []}}
    }
cache_path = os.path.join(os.path.dirname(__file__), 'cache')
inputs_path = os.path.join(cache_path, 'data/inputs')
outputs_path = os.path.join(cache_path, 'data/outputs')


class TestGenOutputData(object):
    @pytest.fixture(params=case_cfg_map.values(), ids=case_cfg_map.keys())
    def gen_and_clear_case(self, request):

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        case_cfg = request.param
        cfg_file_path = os.path.join(cache_path, f'case_items_{request.param_index}.cfg')

        with open(cfg_file_path, "wb") as f:
            pickle.dump(case_cfg, f)

        GenInputData.run(cfg_file_path, inputs_path)

        yield case_cfg, cfg_file_path

        for case_name in case_cfg:
            os.remove(os.path.join(inputs_path, case_name))
            if not case_cfg[case_name].get('no_output_ref', False):
                os.remove(os.path.join(outputs_path, case_name))
            if_backward = len([i['requires_grad'] for i in case_cfg[case_name]['tensor_para']['args'] if i['requires_grad'] == [True]]) != 0
            if if_backward:
                os.remove(os.path.join(outputs_path, case_name.split(".pth")[0] + "_backward.pth"))
        os.remove(cfg_file_path)

    def test_gen_output_cfg(self, gen_and_clear_case):
        case_cfg, cfg_file_path = gen_and_clear_case
        GenOutputData.run(cfg_file_path, inputs_path, outputs_path)
        for case_name in case_cfg:
            case_path = os.path.join(outputs_path, case_name)
            if case_cfg[case_name].get('no_output_ref', False):
                assert not os.path.exists(case_path), f'Expect file {case_path} not found'
                continue

            assert os.path.exists(case_path), f'File {case_path} not found'
            with open(case_path, 'rb') as f:
                forward_result = pickle.load(f)

            saved_backward_pth = os.path.join(outputs_path, case_name.split(".pth")[0] + "_backward.pth")
            if_backward = len([i['requires_grad'] for i in case_cfg[case_name]['tensor_para']['args'] if i['requires_grad'] == [True]]) != 0

            backward_result = None
            if if_backward:
                assert os.path.exists(saved_backward_pth), f'File {saved_backward_pth} not found'
                with open(saved_backward_pth, 'rb') as f:
                    backward_result = pickle.load(f)

            self.check_output(case_cfg[case_name], forward_result, backward_result)

    def check_output(self, cfg, forward_result, backward_result):
        # logging.info(f'case_cfg: {cfg}\nforward_result: {forward_result}\nbackward_result:{backward_result}')
        if not isinstance(forward_result, (list, tuple)):
            forward_result = [forward_result]
        assert len(forward_result) != 0, f'generate output for {cfg["name"]} failed'
        if backward_result is not None:
            grad_key = [i['ins'] for i in cfg['tensor_para']['args'] if i['requires_grad'] == [True]]
            assert len(grad_key) == len(backward_result), f'expect grad output\'s length is {len(grad_key)}, but got {len(backward_result)}'
            for key in grad_key:
                assert key in backward_result, f'generate {cfg["name"]} grad for {key} failed'
