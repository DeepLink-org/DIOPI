# Copyright (c) 2023, DeepLink.
# -*- coding: UTF-8 -*-


class InputChangedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class OutputCheckFailedException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
