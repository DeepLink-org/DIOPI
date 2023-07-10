# Copyright (c) 2023, DeepLink.

from .device_config_helper import Skip
from .diopi_runtime import Dtype

device_configs = {
    'avg_pool2d': dict(
        name=["avg_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64)]
                },
            ]
        ),
    ),
}
