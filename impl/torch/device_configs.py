# Copyright (c) 2023, DeepLink.
import numpy as np
from skip import Skip

device_configs = {
    'multihead_attention': dict(
        name=['multihead_attention'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['q'],
                    "shape": [Skip((2, 2, 2, 8)),Skip((2, 5, 7, 8)),Skip((8, 256, 16, 256)),],
                },
            ]
        ),
    ),
   
}