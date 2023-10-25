import pytest
import logging
import numpy as np
import sys
import os
import pickle


from conformance.check_result import CheckResult, glob_vars, Tensor
from conformance.exception import InputChangedException, OutputCheckFailedException


glob_vars.debug_level = 0
glob_vars.cur_test_func = "diopiTestOp"

case_input = {
    "test check input": [
        {"input": np.array([1, 2, 3])},
        {"input": np.array([1, 2, 3])},
        [],
    ],
    "test check input multi args": [
        {"input": np.array([1, 2, 3]), "other": np.array([[1, 2, 3], [4, 5, 6]])},
        {"input": np.array([1, 2, 3]), "other": np.array([[1, 2, 3], [4, 5, 6]])},
        [],
    ],
    "test ignore_paras_for_input_check": [
        {"input": np.array([1, 2, 3]), "other": np.array([[1, 2, 3], [4, 5, 6]])},
        {
            "input": np.array([4, 5, 6]),
            "other": np.array([[1, 2, 3 + 1e-5], [4, 5, 6]]),
        },
        ["input"],
    ],
}

case_input_xfail = {
    "test input key": [
        {"input": np.array([1, 2, 3])},
        {"other": np.array([1, 2, 3])},
        [],
    ],
    "test input value": [
        {"input": np.array([1, 2, 3])},
        {"input": np.array([4, 5, 6])},
        [],
    ],
    "test input shape": [
        {"input": np.array([1, 2, 3])},
        {"input": np.array([[4, 5, 6]])},
        [],
    ],
}

case_output = {
    "test check output Tensor": [
        Tensor.from_numpy(np.array([1, 2, 3])),
        np.array([1, 2, 3]),
        {},
    ],
    "test check output Tensor bool": [
        Tensor.from_numpy(np.array([True])),
        np.array([True]),
        {},
    ],
    "test check output Tensor atol rtol": [
        Tensor.from_numpy(np.array([[1, 2.002, 3]])),
        np.array([[1, 2 + 1e-3 + 5e-4 * 2, 3]]),
        {"atol": 1e-3, "rtol": 5e-4},
    ],
    "test check output Tensor sum_to_compare": [
        Tensor.from_numpy(np.array([1, 2, 3])),
        np.array([0, 0, 6]),
        {"sum_to_compare": True},
    ],
    "test check output list": [
        [
            Tensor.from_numpy(np.array([1, 2, 3])),
            Tensor.from_numpy(np.array([4, 5, 6])),
        ],
        [np.array([1, 2, 3]), np.array([4, 5, 6])],
        {},
    ],
    "test check output list bool": [
        [Tensor.from_numpy(np.array([False])), Tensor.from_numpy(np.array([True]))],
        (np.array([False]), np.array([True])),
        {},
    ],
    "test check output list atol rtol": [
        [
            Tensor.from_numpy(np.array([1, 2.002, 3])),
            Tensor.from_numpy(np.array([4, 5, 6])),
        ],
        [np.array([1, 2 + 1e-3 + 5e-4 * 2, 3]), np.array([4, 5, 6])],
        {},
    ],
    "test check output list sum_to_compare": [
        (
            Tensor.from_numpy(np.array([1, 2, 3])),
            Tensor.from_numpy(np.array([4, 5, 6])),
        ),
        (np.array([0, 0, 6]), np.array([5, 5, 5])),
        {"sum_to_compare": True},
    ],
    "test check output dict": [
        {
            "out1": Tensor.from_numpy(np.array([1, 2, 3])),
            "out2": Tensor.from_numpy(np.array([4, 5, 6])),
        },
        {"out1": np.array([1, 2, 3]), "out2": np.array([4, 5, 6])},
        {},
    ],
    "test check output dict bool": [
        {
            "out1": Tensor.from_numpy(np.array([False])),
            "out2": Tensor.from_numpy(np.array([True])),
        },
        {"out1": np.array([False]), "out2": np.array([True])},
        {},
    ],
    "test check output dict atol rtol": [
        {
            "out1": Tensor.from_numpy(np.array([1, 2.002, 3])),
            "out2": Tensor.from_numpy(np.array([4, 5, 6])),
        },
        {"out1": np.array([1, 2 + 1e-3 + 5e-4 * 2, 3]), "out2": np.array([4, 5, 6])},
        {},
    ],
    "test check output dict sum_to_compare": [
        {
            "out1": Tensor.from_numpy(np.array([1, 2, 3])),
            "out2": Tensor.from_numpy(np.array([4, 5, 6])),
        },
        {"out1": np.array([0, 0, 6]), "out2": np.array([5, 5, 5])},
        {"sum_to_compare": True},
    ],
    "test check output num": [2, np.array(2), {}],
    "test check output num bool": [True, np.array(True), {}],
    "test check output num atol rtol": [
        2.002,
        np.array(2 + 1e-3 + 5e-4 * 2),
        {"atol": 1e-3, "rtol": 5e-4},
    ],
    "test check output num sum_to_compare": [2, np.array(2), {"sum_to_compare": True}],
}

case_output_xfail = {
    "test check output Tensor": [
        Tensor.from_numpy(np.array([4, 5, 6])),
        np.array([1, 2, 3]),
        {},
    ],
    "test check output Tensor bool": [
        Tensor.from_numpy(np.array([True])),
        np.array([False]),
        {},
    ],
    "test check output Tensor atol rtol": [
        Tensor.from_numpy(np.array([[1, 2.005, 3]])),
        np.array([[1, 2 + 1e-3 + 5e-4 * 2, 3]]),
        {"atol": 1e-3, "rtol": 5e-4},
    ],
    "test check output Tensor sum_to_compare": [
        Tensor.from_numpy(np.array([1, 2, 3])),
        np.array([0, 0, 7]),
        {"sum_to_compare": True},
    ],
    "test check output list": [
        [
            Tensor.from_numpy(np.array([1, 2, 3])),
            Tensor.from_numpy(np.array([4, 5, 7])),
        ],
        [np.array([1, 2, 3]), np.array([4, 5, 6])],
        {},
    ],
    "test check output list bool": [
        [Tensor.from_numpy(np.array([True])), Tensor.from_numpy(np.array([True]))],
        (np.array([False]), np.array([True])),
        {},
    ],
    "test check output list atol rtol": [
        [
            Tensor.from_numpy(np.array([1, 2.005, 3])),
            Tensor.from_numpy(np.array([4, 5, 6])),
        ],
        [np.array([1, 2 + 1e-3 + 5e-4 * 2, 3]), np.array([4, 5, 6])],
        {},
    ],
    "test check output list sum_to_compare": [
        (
            Tensor.from_numpy(np.array([1, 2, 3])),
            Tensor.from_numpy(np.array([4, 5, 6])),
        ),
        (np.array([0, 0, 6]), np.array([5, 5, 6])),
        {"sum_to_compare": True},
    ],
    "test check output dict": [
        {
            "out1": Tensor.from_numpy(np.array([1, 2, 3])),
            "out2": Tensor.from_numpy(np.array([4, 5, 7])),
        },
        {"out1": np.array([1, 2, 3]), "out2": np.array([4, 5, 6])},
        {},
    ],
    "test check output dict bool": [
        {
            "out1": Tensor.from_numpy(np.array([True])),
            "out2": Tensor.from_numpy(np.array([True])),
        },
        {"out1": np.array([False]), "out2": np.array([True])},
        {},
    ],
    "test check output dict atol rtol": [
        {
            "out1": Tensor.from_numpy(np.array([1, 2.005, 3])),
            "out2": Tensor.from_numpy(np.array([4, 5, 6])),
        },
        {"out1": np.array([1, 2 + 1e-3 + 5e-4 * 2, 3]), "out2": np.array([4, 5, 6])},
        {},
    ],
    "test check output dict sum_to_compare": [
        {
            "out1": Tensor.from_numpy(np.array([1, 2, 3])),
            "out2": Tensor.from_numpy(np.array([4, 5, 6])),
        },
        {"out1": np.array([0, 0, 6]), "out2": np.array([5, 5, 6])},
        {"sum_to_compare": True},
    ],
    "test check output num": [2, np.array(3), {}],
    "test check output num bool": [True, np.array(False), {}],
    "test check output num atol rtol": [
        2.005,
        np.array(2 + 1e-3 + 5e-4 * 2),
        {"atol": 1e-3, "rtol": 5e-4},
    ],
    "test check output num sum_to_compare": [2, np.array(5), {"sum_to_compare": True}],
    "test check output invalid output type": ["2", np.array(2), {}],
}


class TestCompareInput(object):
    @pytest.mark.parametrize(
        "input1,input2,ignore_paras_for_input_check",
        case_input.values(),
        ids=case_input.keys(),
    )
    def test_compare_input(self, input1, input2, ignore_paras_for_input_check):
        CheckResult.compare_input(input1, input2, ignore_paras_for_input_check)

    @pytest.mark.parametrize(
        "input1,input2,ignore_paras_for_input_check",
        case_input_xfail.values(),
        ids=case_input_xfail.keys(),
    )
    @pytest.mark.xfail(raises=InputChangedException, strict=True)
    def test_compare_input_xfail(self, input1, input2, ignore_paras_for_input_check):
        CheckResult.compare_input(input1, input2, ignore_paras_for_input_check)


class TestCompareOutput(object):
    @pytest.mark.parametrize(
        "out1,out2,kwargs", case_output.values(), ids=case_output.keys()
    )
    def test_compare_output(self, out1, out2, kwargs):
        CheckResult.compare_output(out1, out2, **kwargs)

    @pytest.mark.parametrize(
        "out1,out2,kwargs", case_output_xfail.values(), ids=case_output_xfail.keys()
    )
    @pytest.mark.xfail(raises=(OutputCheckFailedException, TypeError), strict=True)
    def test_compare_output_xfail(self, out1, out2, kwargs):
        CheckResult.compare_output(out1, out2, **kwargs)
