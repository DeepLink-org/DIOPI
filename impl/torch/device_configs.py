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
                    "ins": ['input'],
                    "dtype": [Skip(np.float16)],
                },
            ]
        ),
    ),
}