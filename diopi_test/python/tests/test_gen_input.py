import pytest
import logging
import numpy as np
import sys
import os
import pickle

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "conformance"))
from conformance.gen_input import GenInputData, GenPolicy


case_cfg_map = {
    "test gen_policy default": {
        "pointwise_binary::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.randn",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.randn",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "pointwise_binary::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (1024,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int64,
                        "gen_fn": "Genfunc.randint",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (1024,),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": dict(fn="Genfunc.uniform", high=10),
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "pointwise_binary::add_2.pth": {
            "name": "add",
            "interface": ["torch"],
            "tag": ["scalar"],
            "is_inplace": True,
            "para": {"other": 0},
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.float32,
                        "gen_fn": dict(fn="Genfunc.randint", low=-10),
                        "gen_policy": "default",
                    }
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "saved_args": {},
        },
    },
    "test gen_policy gen_tensor_by_value": {
        "nms::nms_0.pth": {
            "name": "nms",
            "interface": ["torchvision.ops"],
            "para": {"iou_threshold": 0.3},
            "tensor_para": {
                "args": [
                    {
                        "ins": "boxes",
                        "value": [
                            [2.4112, 0.7486, 2.4551, 2.7486],
                            [0.7486, 1.3544, 1.1294, 2.3544],
                            [1.4551, 0.1294, 1.6724, 1.3294],
                            [1.4959, 0.1086, 2.778335, 3.22],
                            [0.107706, 2.948, 2.1256, 4.525],
                            [2.7735, 2.12506, 7.0556, 8.995],
                        ],
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float32,
                        "gen_fn": "Genfunc.randn",
                        "gen_policy": "gen_tensor_by_value",
                    },
                    {
                        "ins": "scores",
                        "shape": (6,),
                        "gen_fn": "Genfunc.randn",
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.float32,
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "tag": [],
            "saved_args": {},
        },
        "nms::nms_1.pth": {
            "name": "nms",
            "interface": ["torchvision.ops"],
            "para": {"iou_threshold": 0.3},
            "tensor_para": {
                "args": [
                    {
                        "ins": "boxes",
                        "value": [
                            [2.4112, 0.7486, 2.4551, 2.7486],
                            [0.7486, 1.3544, 1.1294, 2.3544],
                            [1.4551, 0.1294, 1.6724, 1.3294],
                            [1.4959, 0.1086, 2.778335, 3.22],
                            [0.107706, 2.948, 2.1256, 4.525],
                            [2.7735, 2.12506, 7.0556, 8.995],
                        ],
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.float32,
                        "gen_fn": "Genfunc.randn",
                        "gen_policy": "gen_tensor_by_value",
                    },
                    {
                        "ins": "scores",
                        "shape": (6,),
                        "gen_fn": "Genfunc.randn",
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.float32,
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "tag": [],
            "saved_args": {},
        },
    },
    "test gen_policy gen_tensor_list": {
        "clip_grad_norm::clip_grad_norm__0.pth": {
            "name": "clip_grad_norm_",
            "interface": ["CustomizedTest"],
            "para": {"max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": True},
            "tensor_para": {
                "args": [
                    {
                        "ins": "grads",
                        "shape": (10, 3),
                        "gen_fn": "Genfunc.randn",
                        "dtype": np.float32,
                        "gen_num_range": [1, 5],
                        "requires_grad": [True],
                        "gen_policy": "gen_tensor_list",
                    }
                ],
                "seq_name": "tensors",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "tag": [],
            "saved_args": {},
        },
        "clip_grad_norm::clip_grad_norm__1.pth": {
            "name": "clip_grad_norm_",
            "interface": ["CustomizedTest"],
            "para": {"max_norm": 1.0, "norm_type": 2.0, "error_if_nonfinite": True},
            "tensor_para": {
                "args": [
                    {
                        "ins": "grads",
                        "shape": (10, 3),
                        "gen_fn": "Genfunc.randn",
                        "dtype": np.float32,
                        "gen_num_range": [1, 5],
                        "requires_grad": [True],
                        "gen_policy": "gen_tensor_list",
                    },
                    {
                        "ins": "grads_test",
                        "shape": (4,),
                        "gen_fn": dict(fn="Genfunc.uniform", low=-10, high=10),
                        "dtype": np.int64,
                        "gen_num_range": [1, 5],
                        "requires_grad": [False],
                        "gen_policy": "gen_tensor_list",
                    },
                ],
                "seq_name": "tensors",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "tag": [],
            "saved_args": {},
        },
    },
    "test gen_policy gen_tensor_list_diff_shape": {
        "cat_diff_size::cat_0.pth": {
            "name": "cat",
            "interface": ["torch"],
            "atol": 0.0001,
            "rtol": 1e-05,
            "para": {"dim": -1},
            "tensor_para": {
                "args": [
                    {
                        "ins": "tensors",
                        "requires_grad": [False],
                        "shape": ((8,), (16,)),
                        "dtype": np.int8,
                        "gen_fn": "Genfunc.randint",
                        "gen_num_range": [],
                        "gen_policy": "gen_tensor_list_diff_shape",
                    }
                ],
                "seq_name": "tensors",
            },
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "tag": [],
            "saved_args": {},
        },
        "cat_diff_size::cat_1.pth": {
            "name": "cat",
            "interface": ["torch"],
            "atol": 0.0001,
            "rtol": 1e-05,
            "para": {"dim": 0},
            "tensor_para": {
                "args": [
                    {
                        "ins": "tensors",
                        "requires_grad": [True],
                        "shape": ((2, 8), (16, 8), (3, 8), (4, 8), (1, 8)),
                        "dtype": np.float32,
                        "gen_fn": "Genfunc.randn",
                        "gen_num_range": [],
                        "gen_policy": "gen_tensor_list_diff_shape",
                    },
                    {
                        "ins": "tensors_test",
                        "requires_grad": [True],
                        "shape": ((2, 8), (16, 8), (3, 8), (4, 8), (1, 8)),
                        "dtype": np.float32,
                        "gen_fn": "Genfunc.randn",
                        "gen_num_range": [],
                        "gen_policy": "gen_tensor_list_diff_shape",
                    },
                ],
                "seq_name": "tensors",
            },
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "tag": [],
            "saved_args": {},
        },
    },
    "test gen_fn": {
        "test_rand::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.rand",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": "Genfunc.rand",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_rand::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": "Genfunc.rand",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": "Genfunc.rand",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_randn::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.randn",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": "Genfunc.randn",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_randn::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": "Genfunc.randn",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": "Genfunc.randn",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_uniform::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.uniform",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": dict(fn="Genfunc.uniform", low=-10, high=10),
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_uniform::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": dict(fn="Genfunc.uniform", low=-10),
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": dict(fn="Genfunc.uniform", high=100),
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_empty::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.empty",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": "Genfunc.empty",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_empty::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": "Genfunc.empty",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": "Genfunc.empty",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_ones::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.ones",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": "Genfunc.ones",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_ones::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": "Genfunc.ones",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": "Genfunc.ones",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_zeros::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.zeros",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": "Genfunc.zeros",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_zeros::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": "Genfunc.zeros",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": "Genfunc.zeros",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_mask::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.mask",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": "Genfunc.mask",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_mask::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": "Genfunc.mask",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": "Genfunc.mask",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_randint::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.randint",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": dict(fn="Genfunc.randint", low=-10, high=10),
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_randint::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": dict(fn="Genfunc.randint", low=-10),
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": dict(fn="Genfunc.randint", high=100),
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_positive::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.positive",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": "Genfunc.positive",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_positive::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": "Genfunc.positive",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": "Genfunc.positive",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_sym_mat::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (3, 4),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.sym_mat",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (2, 3, 4),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.float32,
                        "gen_fn": "Genfunc.sym_mat",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_sym_mat::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (2, 3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.float32,
                        "gen_fn": "Genfunc.sym_mat",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.float16,
                        "gen_fn": "Genfunc.sym_mat",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_randn_int::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.randn_int",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": dict(fn="Genfunc.randn_int", low=-10, high=10),
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_randn_int::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": dict(fn="Genfunc.randn_int", low=-10),
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": dict(fn="Genfunc.randn_int", high=100),
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_randn_complx::add_0.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (),
                        "requires_grad": [True],
                        "gen_num_range": [],
                        "dtype": np.float64,
                        "gen_fn": "Genfunc.randn_complx",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.int32,
                        "gen_fn": "Genfunc.randn_complx",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_randn_complx::add_1.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (0,),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.bool_,
                        "gen_fn": "Genfunc.randn_complx",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (3, 0),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.uint8,
                        "gen_fn": "Genfunc.randn_complx",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
        "test_randn_complx::add_2.pth": {
            "name": "add",
            "interface": ["torch"],
            "is_inplace": True,
            "tensor_para": {
                "args": [
                    {
                        "ins": "input",
                        "shape": (5, 6),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.complex64,
                        "gen_fn": "Genfunc.randn_complx",
                        "gen_policy": "default",
                    },
                    {
                        "ins": "other",
                        "shape": (5, 6),
                        "requires_grad": [False],
                        "gen_num_range": [],
                        "dtype": np.complex128,
                        "gen_fn": "Genfunc.randn_complx",
                        "gen_policy": "default",
                    },
                ],
                "seq_name": "",
            },
            "atol": 1e-05,
            "rtol": 1e-05,
            "atol_half": 0.01,
            "rtol_half": 0.05,
            "mismatch_ratio_threshold": 0.001,
            "memory_format": "NCHW",
            "fp16_exact_match": False,
            "train": True,
            "requires_backward": [],
            "para": {},
            "tag": [],
            "saved_args": {},
        },
    },
}
cache_path = os.path.join(os.path.dirname(__file__), "cache")
inputs_path = os.path.join(cache_path, "data/inputs")


class TestGenInputData(object):
    @pytest.fixture(params=case_cfg_map.values(), ids=case_cfg_map.keys())
    def gen_and_clear_case(self, request):
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        case_cfg = request.param
        cfg_file_path = os.path.join(
            cache_path, f"case_items_{request.param_index}.cfg"
        )

        with open(cfg_file_path, "wb") as f:
            pickle.dump(case_cfg, f)

        yield case_cfg, cfg_file_path

        for case_name in case_cfg:
            os.remove(os.path.join(inputs_path, case_name))
        os.remove(cfg_file_path)

    def test_gen_input_cfg(self, gen_and_clear_case):
        case_cfg, cfg_file_path = gen_and_clear_case
        GenInputData.run(cfg_file_path, inputs_path)
        for case_name in case_cfg:
            case_path = os.path.join(inputs_path, case_name)
            assert os.path.exists(case_path), f"File {case_path} not found"

            with open(case_path, "rb") as f:
                case_dict = pickle.load(f)

            self.check_cfg(case_cfg[case_name], case_dict)

    def check_cfg(self, case_cfg, case_dict):
        logging.debug(f"case_cfg: {case_cfg}\ncase_dict: {case_dict}")
        # check kwargs num
        expect_args_num = len(case_cfg["para"]) + (
            len(case_cfg["tensor_para"]["args"]) if case_cfg.get("tensor_para") else 0
        )
        actual_args_num = len(case_dict["function_paras"]["kwargs"])
        assert (
            expect_args_num == actual_args_num
        ), f"expect args num is {expect_args_num}, but got {actual_args_num}"

        # check para
        for each_para in case_cfg["para"]:
            assert (
                each_para in case_dict["function_paras"]["kwargs"]
            ), f"{each_para} disapper"
            assert (
                case_cfg["para"][each_para] == case_dict["function_paras"]["kwargs"][each_para]
            ), f"{each_para} value is wrong"

        # check tensor value
        if case_cfg.get("tensor_para"):
            for index, tensor_cfg in enumerate(case_cfg["tensor_para"]["args"]):
                name = tensor_cfg["ins"]
                gen_policy = tensor_cfg["gen_policy"]

                tensor = case_dict["function_paras"]["kwargs"][name]

                if gen_policy == GenPolicy.default:
                    expect_shape = tensor_cfg["shape"]
                    if tensor_cfg["gen_fn"] == "Genfunc.sym_mat":
                        axis = [i for i in range(len(expect_shape) - 2)] + [-1, -2]
                        mat = np.random.randn(*expect_shape)
                        expect_shape = (mat @ mat.transpose(axis)).shape
                    assert (
                        expect_shape == tensor.shape
                    ), f"expect {name}.shape is {expect_shape}, but got {tensor.shape}"
                    assert (
                        tensor_cfg["dtype"] == tensor.dtype
                    ), f"expect {name}.dtype is {tensor_cfg['dtype']}, but got {tensor.dtype}"

                    if tensor_cfg["requires_grad"] == [True]:
                        assert case_dict["function_paras"]["requires_grad"].get(
                            name, False
                        ), f"expect arg {name}.requires_grad is True, but got False"
                    else:
                        assert (
                            name not in case_dict["function_paras"]["requires_grad"]
                        ), f"expect arg {name}.requires_grad is False, but got True"
                elif gen_policy == GenPolicy.default:
                    expect_shape = np.array(tensor_cfg["value"]).shape
                    assert (
                        expect_shape == tensor.shape
                    ), f"expect {name}.shape is {expect_shape}, but got {tensor.shape}"
                    assert (
                        tensor_cfg["dtype"] == tensor.dtype
                    ), f"expect {name}.dtype is {tensor_cfg['dtype']}, but got {tensor.dtype}"

                    if tensor_cfg["requires_grad"] == [True]:
                        assert case_dict["function_paras"]["requires_grad"].get(
                            name, False
                        ), f"expect arg {name}.requires_grad is True, but got False"
                    else:
                        assert (
                            name not in case_dict["function_paras"]["requires_grad"]
                        ), f"expect arg {name}.requires_grad is False, but got True"
                elif gen_policy == GenPolicy.gen_tensor_list:
                    assert isinstance(
                        tensor, list
                    ), f"expect {name} is list, but got {type(tensor)}"
                    tensors_num = case_dict["cfg"]["tensor_para"]["args"][index][
                        "tensors_num"
                    ]
                    assert tensors_num == len(
                        tensor
                    ), f"expect {name}'s length is {tensors_num}, but got {len(tensor)}"
                    for each_tensor in tensor:
                        assert (
                            tensor_cfg["shape"] == each_tensor.shape
                        ), f"expect {name}.shape is {tensor_cfg['shape']}, but got {each_tensor.shape}"
                        assert (
                            tensor_cfg["dtype"] == each_tensor.dtype
                        ), f"expect {name}.dtype is {tensor_cfg['dtype']}, but got {each_tensor.dtype}"
                    if tensor_cfg["requires_grad"] == [True]:
                        assert case_dict["function_paras"]["requires_grad"].get(
                            name, False
                        ), f"expect arg {name}.requires_grad is True, but got False"
                    else:
                        assert (
                            name not in case_dict["function_paras"]["requires_grad"]
                        ), f"expect arg {name}.requires_grad is False, but got True"
                elif gen_policy == GenPolicy.gen_tensor_list_diff_shape:
                    assert isinstance(
                        tensor_cfg["shape"][0], (tuple, list)
                    ), f"expect {name}'s shape is nested_list"
                    assert isinstance(
                        tensor, list
                    ), f"expect {name} is list, but got {type(tensor)}"
                    assert len(tensor_cfg["shape"]) == len(
                        tensor
                    ), f"expect {name}'s length is {len(tensor_cfg['shape'])}, but got {len(tensor)}"
                    for i, each_tensor in enumerate(tensor):
                        assert (
                            tensor_cfg["shape"][i] == each_tensor.shape
                        ), f"expect {name}.shape is {tensor_cfg['shape'][i]}, but got {each_tensor.shape}"
                        assert (
                            tensor_cfg["dtype"] == each_tensor.dtype
                        ), f"expect {name}.dtype is {tensor_cfg['dtype']}, but got {each_tensor.dtype}"
                    if tensor_cfg["requires_grad"] == [True]:
                        assert case_dict["function_paras"]["requires_grad"].get(
                            name, False
                        ), f"expect arg {name}.requires_grad is True, but got False"
                    else:
                        assert (
                            name not in case_dict["function_paras"]["requires_grad"]
                        ), f"expect arg {name}.requires_grad is False, but got True"
