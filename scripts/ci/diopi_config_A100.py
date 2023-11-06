# Copyright (c) 2023, DeepLink.
# This is a temporary solution for testing the flash attention operator on the A100. It is intended for use 
# only in the CI (Continuous Integration) environment. To use this file on the A100, replace the original 
# DIOPI/diopi_test/python/conformance/diopi_configs.py with it.
import numpy as np

ops_with_states = {}

diopi_configs = {
    'multihead_attention_forward': dict(
        name=['multihead_attention_forward'],
        interface=['CustomizedTest'],
        dtype=[np.float16],
        atol=1e-3,
        rtol=1e-4,
        para=dict(
            dropout_p=[0, 0],
            is_causal=[False, False],
            return_debug_mask=[False, False],
            scale=[None, None]
        ),
        tensor_para=dict(
            gen_fn='Genfunc.randn',
            args=[
                {
                    "ins": ['q'],
                    "shape": ((2, 2, 2, 8), (2, 5, 7, 8)),
                    "dtype": [np.float16],
                },
                {
                    "ins": ['k'],
                    "shape": ((2, 2, 2, 8), (2, 5, 7, 8)),
                    "dtype": [np.float16],
                },
                {
                    "ins": ['v'],
                    "shape": ((2, 2, 2, 8), (2, 5, 7, 8)),
                    "dtype": [np.float16],
                },
            ],
        ),
    ),
}