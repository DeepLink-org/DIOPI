from skip import Skip
import numpy as np

device_configs = {
    'rms_norm': dict(
        name=['rms_norm'],
        atol=1e-3,
        rtol=1e-3,
        atol_half=1e-2,
        rtol_half=1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['bias'],
                    # Skip becasuse bias not supported.
                    "dtype": [Skip((5, )), Skip((32, )), Skip((64, )), Skip((8, )), Skip((128,)), Skip((64,)), Skip((32,)),
                              Skip((3, 5)), Skip((2, 16, 128))],
                },
            ]
        ),
    ),
}