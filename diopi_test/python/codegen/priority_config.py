# Copyright (c) 2023, DeepLink.
import numpy as np


def priority_generator(priority):
    class Priority(object):
        def __init__(self, value) -> None:
            self._priority = priority
            self._value = value

        def __str__(self) -> str:
            return f'{self._priority}: {str(self._value)}'

        def value(self):
            return self._value

        def priority(self):
            return self._priority
    return Priority


P0 = priority_generator('P0')
P1 = priority_generator('P1')
P2 = priority_generator('P2')


priority_config = {
    # bool value
    'threshold': dict(
        name=["threshold"],
        para=dict(
            threshold=[P2(False)],
            value=[P2(True)],
        )
    ),

    # broadcast
    'pointwise_binary': dict(
        name=['add', 'sub', 'mul', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (P1((128, 64, 3, 3)), P1((2, 64, 16, 128)),
                              P1((2, 32, 130, 130))),
                }
            ],
        ),
    ),
}
