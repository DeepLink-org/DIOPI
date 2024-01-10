import pytest
import numpy as np

from codegen.case_priority import CasePriority, P0, P1, P2


case_cfg_map = {
    "test priority cfg has priority": [
        "threshold",
        dict(
            threshold=dict(
                name=["threshold"],
                para=dict(
                    threshold=[P2(False)],
                    value=[P1(True)],
                ),
                tensor_para=dict(
                    args=[
                        {
                            "ins": ['input'],
                            "shape": [P0((64,))],
                            "dtype": [P1(np.float16)],
                        },
                    ]
                ),
            ),
        ),
        dict(
            name="threshold",
            priority='P3',
            is_inplace=True,
            para=dict(
                threshold=False,
                value=0,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (64, ),
                        "dtype": np.float16,
                    },
                ]
            ),
        ),
        'P3'
    ],
    "test priority shape": [
        "threshold",
        dict(
            threshold=dict(
                name=["threshold"],
                para=dict(
                    threshold=[P2(False)],
                    value=[P1(True)],
                ),
                tensor_para=dict(
                    args=[
                        {
                            "ins": ['input'],
                            "shape": [P0((64,))],
                            "dtype": [P1(np.float16)],
                        },
                    ]
                ),
            ),
        ),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=False,
                value=0,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (64, ),
                        "dtype": np.float64,
                    },
                ]
            ),
        ),
        'P0'
    ],
    "test priority dtype first": [
        "threshold",
        dict(
            threshold=dict(
                name=["threshold"],
                para=dict(
                    threshold=[P2(False)],
                    value=[P1(True)],
                ),
                tensor_para=dict(
                    args=[
                        {
                            "ins": ['input'],
                            "shape": [P0((64,))],
                            "dtype": [P1(np.float16)],
                        },
                        {
                            "ins": ['weight'],
                            "shape": [P0((64,))],
                            "dtype": [P1(np.float16)],
                        }
                    ]
                ),
            ),
        ),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=False,
                value=0,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (32, ),
                        "dtype": np.float64,
                    },
                    {
                        "ins": 'weight',
                        "requires_grad": True,
                        "shape": (64, ),
                        "dtype": np.float16,
                    },
                ]
            ),
        ),
        'P1'
    ],
    "test priority para1": [
        "threshold",
        dict(
            threshold=dict(
                name=["threshold"],
                para=dict(
                    threshold=[P2(False)],
                    value=[P1(True)],
                ),
                tensor_para=dict(
                    args=[
                        {
                            "ins": ['input'],
                            "shape": [P0((64,))],
                            "dtype": [P1(np.float16)],
                        },
                    ]
                ),
            ),
        ),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=False,
                value=True,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (32, ),
                        "dtype": np.float32,
                    },
                ]
            ),
        ),
        'P2'
    ],
    "test priority para2": [
        "threshold",
        dict(
            threshold=dict(
                name=["threshold"],
                para=dict(
                    threshold=[P2(False)],
                    value=[P1(True)],
                ),
                tensor_para=dict(
                    args=[
                        {
                            "ins": ['input'],
                            "shape": [P0((64,))],
                            "dtype": [P1(np.float16)],
                        },
                    ]
                ),
            ),
        ),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=1,
                value=True,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (32, ),
                        "dtype": np.float32,
                    },
                ]
            ),
        ),
        'P1'
    ],
    "test priority nothing match": [
        "threshold",
        dict(
            threshold=dict(
                name=["threshold"],
                para=dict(
                    threshold=[P2(False)],
                    value=[P1(True)],
                ),
                tensor_para=dict(
                    args=[
                        {
                            "ins": ['input'],
                            "shape": [P0((64,))],
                            "dtype": [P1(np.float16)],
                        },
                    ]
                ),
            ),
        ),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=1,
                value=0,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (32, ),
                        "dtype": np.float32,
                    },
                ]
            ),
        ),
        'P0'
    ],
    "test priority no para": [
        "threshold",
        dict(
            threshold=dict(
                name=["threshold"],
                tensor_para=dict(
                    args=[
                        {
                            "ins": ['input'],
                            "shape": [P0((64,))],
                            "dtype": [P1(np.float16)],
                        },
                    ]
                ),
            ),
        ),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=1,
                value=0,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (32, ),
                        "dtype": np.float32,
                    },
                ]
            ),
        ),
        'P0'
    ],
    "test priority no tensor_para": [
        "threshold",
        dict(
            threshold=dict(
                name=["threshold"],
                para=dict(
                    threshold=[P2(False)],
                    value=[P1(True)],
                ),
            ),
        ),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=1,
                value=0,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (32, ),
                        "dtype": np.float32,
                    },
                ]
            ),
        ),
        'P0'
    ],
    "test priority cfg no para": [
        "threshold",
        dict(
            threshold=dict(
                name=["threshold"],
                para=dict(
                    threshold=[P2(False)],
                    value=[P1(True)],
                ),
                tensor_para=dict(
                    args=[
                        {
                            "ins": ['input'],
                            "shape": [P0((64,))],
                            "dtype": [P1(np.float16)],
                        },
                    ]
                ),
            ),
        ),
        dict(
            name="threshold",
            is_inplace=True,
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (32, ),
                        "dtype": np.float32,
                    },
                ]
            ),
        ),
        'P0'
    ],
    "test priority cfg no tensor para": [
        "threshold",
        dict(
            threshold=dict(
                name=["threshold"],
                para=dict(
                    threshold=[P2(False)],
                    value=[P1(True)],
                ),
                tensor_para=dict(
                    args=[
                        {
                            "ins": ['input'],
                            "shape": [P0((64,))],
                            "dtype": [P1(np.float16)],
                        },
                    ]
                ),
            ),
        ),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=1,
                value=0,
            ),
        ),
        'P0'
    ],
    "test priority cfg zero size": [
        "threshold",
        dict(),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=1,
                value=0,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (0, 2),
                        "no_contiguous": True,
                        "dtype": np.float32,
                    },
                ]
            ),
        ),
        'P2'
    ],
    "test priority cfg no contiguous": [
        "threshold",
        dict(),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=1,
                value=0,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (64, ),
                        "no_contiguous": True,
                        "dtype": np.float32,
                    },
                ]
            ),
        ),
        'P1'
    ],
    "test priority cfg stride": [
        "threshold",
        dict(),
        dict(
            name="threshold",
            is_inplace=True,
            para=dict(
                threshold=1,
                value=0,
            ),
            tensor_para=dict(
                genfunc='Genfunc.randn',
                args=[
                    {
                        "ins": 'input',
                        "requires_grad": True,
                        "shape": (64, ),
                        "stride": (1,),
                        "dtype": np.float32,
                    },
                ]
            ),
        ),
        'P1'
    ]
}


class TestPriority(object):
    @pytest.mark.parametrize('case_name,priority_cfg,case_cfg,expect_priority', case_cfg_map.values(), ids=case_cfg_map.keys())
    def test_priority_cfg(self, case_name, priority_cfg, case_cfg, expect_priority):
        actual_priority = CasePriority(
            case_name, case_cfg, priority_cfg).get_case_priority()
        assert expect_priority == actual_priority
