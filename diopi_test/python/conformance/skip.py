# Copyright (c) 2023, DeepLink.

class Skip(object):

    def __init__(self, value) -> None:
        self._value = value

    def __str__(self) -> str:
        return str(self._value)
    
    def value(self):
        return self._value
