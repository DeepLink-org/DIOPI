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
        atol=1e-3,
        rtol=1e-4,
        para=dict(
            dropout_p=[0, 0, 0, 0],
            is_causal=[False, False, True, False],
            return_debug_mask=[False, False, False, False],
            scale=[None, None, None, 0.1334]
        ),
        tensor_para=dict(
            gen_fn='Genfunc.randn',
            args=[
                {
                    "ins": ['q'],
                    "requires_grad": [True],
                    "shape": ((2, 2, 2, 8), (2, 5, 7, 8), (4, 103, 8, 32), (8, 256, 16, 256)),
                    "dtype": [np.float16],
                },
                {
                    "ins": ['k'],
                    "requires_grad": [True],
                    "shape": ((2, 2, 2, 8), (2, 5, 7, 8), (4, 103, 8, 32), (8, 256, 16, 256)),
                    "dtype": [np.float16],
                },
                {
                    "ins": ['v'],
                    "requires_grad": [True],
                    "shape": ((2, 2, 2, 8), (2, 5, 7, 8), (4, 103, 8, 32), (8, 256, 16, 256)),
                    "dtype": [np.float16],
                },
            ],
        ),
        saved_args=dict(out=0, softmax_lse=1),
        requires_backward=[0],
    ),
}