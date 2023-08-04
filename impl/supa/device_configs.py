# Copyright (c) 2023, DeepLink.

from .device_config_helper import Skip
from .diopi_runtime import Dtype

device_configs = {
    'cumsum': dict(
        name=["cumsum"],
        atol=1e-3,
        rtol=1e-4,
    ),

    'silu': dict(
        name=["silu"],
        atol=1e-3,
        rtol=1e-4,
    ),

    'sum': dict(
        name=["sum"],
        atol=1e-4,
        rtol=1e-4,
    ),
}
