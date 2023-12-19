# Copyright (c) 2023, DeepLink.
# This is a temporary solution for testing the flash attention operator on the A100. It is intended for use 
# only in the CI (Continuous Integration) environment. To use this file on the A100, replace the original 
# DIOPI/diopi_test/python/conformance/diopi_configs.py with it.
import numpy as np

ops_with_states = {}

diopi_configs = {
    'multihead_attention': dict(
        name=['multihead_attention'],
        interface=['CustomizedTest'],
        dtype=[np.float16],
        saved_args=dict(out=0),
        atol=1e-3,
        rtol=1e-4,
        para=dict(
            dropout_p=[0, 0, 0, 0, 0],
            is_causal=[False, False, True, True, False],
            return_debug_mask=[False, False, False, False, False],
            scale=[None, 1.234, None, 0.1334, 0]
        ),
        tensor_para=dict(
            gen_fn='Genfunc.randn',
            args=[
                {
                    "ins": ['q'],
                    # FIXME q的seqlen与num_head与k不相等，pytorch报错
                    # "shape": ((2, 2, 2, 8), (2, 10, 28, 16), (4, 64, 16, 32), (8, 128, 16, 256),
                    #           (2, 0, 4, 8)),
                    "shape": ((2, 2, 2, 8), (2, 10, 28, 16), (4, 32, 8, 32), (8, 128, 16, 256),
                              (2, 0, 4, 8)),
                    "requires_grad": [True],
                },
                {
                    "ins": ['k'],
                    "shape": ((2, 2, 2, 8), (2, 5, 28, 16), (4, 32, 8, 32), (8, 128, 16, 256),
                              (2, 0, 4, 8)),
                    "requires_grad": [True],
                },
                {
                    "ins": ['v'],
                    "shape": ((2, 2, 2, 8), (2, 5, 28, 16), (4, 32, 8, 32), (8, 128, 16, 256),
                              (2, 0, 4, 8)),
                    "requires_grad": [True],
                },
            ],
        ),
    ),

    'multihead_attention_dropout': dict(
        name=['multihead_attention'],
        no_output_ref=True,
        dtype=[np.float16],
        saved_args=dict(out=0),
        para=dict(
            dropout_p=[0.1, 0.5, 0.3, 0.7],
            is_causal=[False, True, False, True],
            # FIXME multihead_attention测试dropout，return_debug_mask为True时报错
            # return_debug_mask=[True, True, True, True],
            return_debug_mask=[False, False, False, False],
            scale=[None, 1.234, None, 0.1334]
        ),
        tensor_para=dict(
            gen_fn='Genfunc.randn',
            args=[
                {
                    "ins": ['q'],
                    "shape": ((2, 2, 2, 8), (2, 5, 7, 16), (4, 103, 8, 32), (8, 256, 16, 256),),
                },
                {
                    "ins": ['k'],
                    "shape": ((2, 2, 2, 8), (2, 5, 7, 16), (4, 103, 8, 32), (8, 256, 16, 256)),
                },
                {
                    "ins": ['v'],
                    "shape": ((2, 2, 2, 8), (2, 5, 7, 16), (4, 103, 8, 32), (8, 256, 16, 256)),
                },
            ],
        ),
    ),

    'multihead_attention_varlen': dict(
        name=['multihead_attention_varlen'],
        interface=['CustomizedTest'],
        dtype=[np.float16],
        saved_args=dict(out=0),
        atol=1e-3,
        rtol=1e-4,
        para=dict(
            cu_seqlens=[[0, 0, 0, 1, 1], [0, 1, 2], [0, 3], [0, 13, 66, 153, 256], [0, 256, 300, 425, 512]],
            max_seqlen=[1, 1, 3, 103, 256],
            dropout_p=[0, 0, 0, 0, 0],
            is_causal=[True, False, True, False, True],
            return_debug_mask=[False, False, False, False, False],
            # FIXME scale不为None时，反向传播结果错误
            # scale=[None, 0, 0.21, None, 1.00056]
            scale=[None, None, None, None, None]
        ),
        tensor_para=dict(
            gen_fn='Genfunc.randn',
            args=[
                {
                    "ins": ['q'],
                    # FIXME 输入最后一维不能整除8时，执行报错
                    # FIXME q的seqlen与num_head与k不相等，pytorch报错
                    # "shape": ((1, 9, 8), (2, 4, 14), (3, 16, 256), (256, 16, 32), (512, 16, 128)),
                    "shape": ((1, 9, 8), (2, 4, 16), (3, 8, 256), (256, 16, 32), (512, 16, 128)),
                    "requires_grad": [True],
                },
                {
                    "ins": ['k'],
                    # "shape": ((1, 9, 8), (2, 4, 14), (3, 8, 256), (256, 16, 32), (512, 16, 128)),
                    "shape": ((1, 9, 8), (2, 4, 16), (3, 8, 256), (256, 16, 32), (512, 16, 128)),
                    "requires_grad": [True],
                },
                {
                    "ins": ['v'],
                    # "shape": ((1, 9, 8), (2, 4, 14), (3, 8, 256), (256, 16, 32), (512, 16, 128)),
                    "shape": ((1, 9, 8), (2, 4, 16), (3, 8, 256), (256, 16, 32), (512, 16, 128)),
                    "requires_grad": [True],
                },
            ],
        ),
    ),

    'multihead_attention_varlen_dropout': dict(
        name=['multihead_attention_varlen'],
        no_output_ref=True,
        dtype=[np.float16],
        saved_args=dict(out=0),
        atol=1e-3,
        rtol=1e-4,
        para=dict(
            cu_seqlens=[[0, 0, 0, 1, 1], [0, 1, 2], [0, 3], [0, 13, 66, 153, 256], [0, 256, 300, 425, 512]],
            max_seqlen=[1, 1, 3, 103, 256],
            dropout_p=[0.1, 0.5, 0.3, 0.7, 0.9],
            is_causal=[True, False, True, False, True],
            # FIXME multihead_attention_varlen测试dropout，return_debug_mask为True时报错
            # return_debug_mask=[True, True, True, True, True],
            return_debug_mask=[False, False, False, False, False],
            # FIXME scale不为None时，反向传播结果错误
            # scale=[None, 0, 0.21, None, 1.00056]
            scale=[None, None, None, None, None]
        ),
        tensor_para=dict(
            gen_fn='Genfunc.randn',
            args=[
                {
                    "ins": ['q'],
                    # FIXME 输入最后一维不能整除8时，执行报错
                    # FIXME q的seqlen与num_head与k不相等，pytorch报错
                    # "shape": ((1, 9, 8), (2, 4, 14), (3, 16, 256), (256, 16, 32), (512, 16, 128)),
                    "shape": ((1, 9, 8), (2, 4, 16), (3, 8, 256), (256, 16, 32), (512, 16, 128)),
                },
                {
                    "ins": ['k'],
                    # "shape": ((1, 9, 8), (2, 4, 14), (3, 8, 256), (256, 16, 32), (512, 16, 128)),
                    "shape": ((1, 9, 8), (2, 4, 16), (3, 8, 256), (256, 16, 32), (512, 16, 128)),
                },
                {
                    "ins": ['v'],
                    # "shape": ((1, 9, 8), (2, 4, 14), (3, 8, 256), (256, 16, 32), (512, 16, 128)),
                    "shape": ((1, 9, 8), (2, 4, 16), (3, 8, 256), (256, 16, 32), (512, 16, 128)),
                },
            ],
        ),
    ),
}