# Copyright (c) 2024, DeepLink.
import numpy as np
from skip import Skip

device_configs = {
    "log_integer_input": dict(
        name=["log1p"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [
                        Skip(np.int16),
                        Skip(np.int32),
                        Skip(np.int64),
                        Skip(np.int8),
                        Skip(np.uint8),
                        Skip(np.float16),
                    ],
                },
            ]
        ),
    ),
    "pointwise_op_abs_input": dict(
        name=["log1p"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [
                        Skip(np.int16),
                        Skip(np.int32),
                        Skip(np.int64),
                        Skip(np.int8),
                        Skip(np.uint8),
                        Skip(np.float16),
                    ],
                },
            ]
        ),
    ),
    "log_zero_input": dict(
        name=["log1p"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [
                        Skip(np.int16),
                        Skip(np.int32),
                        Skip(np.int64),
                        Skip(np.int8),
                        Skip(np.uint8),
                        Skip(np.float16),
                    ],
                },
            ]
        ),
    ),
    "log_neg_input": dict(
        name=["log1p"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [
                        Skip(np.int16),
                        Skip(np.int32),
                        Skip(np.int64),
                        Skip(np.int8),
                        Skip(np.uint8),
                        Skip(np.float16),
                    ],
                },
            ]
        ),
    ),
}
