# Copyright (c) 2023, DeepLink.

from .device_config_helper import Skip
from .diopi_runtime import Dtype

device_configs = {
    'batch_norm': dict(
        name=["batch_norm"],
        atol=1e-2,
        rtol=1e-3,
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float16)]
                },
            ]
        ),
    ),

    'baddbmm_without_inplace': dict(
        name=["baddbmm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((2, )),),
                },
                {
                    "ins": ["batch1"],
                    "shape": (Skip((2, 0, 4)),),
                },
                {
                    "ins": ["batch2"],
                    "shape": (Skip((2, 4, 2)),),
                },
            ]
        ),
    ),
    
    'hardtanh': dict(
        name=["hardtanh"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((0, 8)), Skip((16, 0, 8))),
                },
            ],
        ),
    ),
    
    'hardtanh_int': dict(
        name=["hardtanh"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((0, 8)), Skip((16, 0, 8))),
                },
            ],
        ),
    ),
    
    'hardtanh_uint': dict(
        name=["hardtanh"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((0, 8)), Skip((16, 0, 8))),
                },
            ],
        ),
    ),

    'threshold_uint': dict(
        name=["threshold"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.uint8)],
                },
            ],
        ),
    ),

    'batch_norm_no_contiguous': dict(
        name=["batch_norm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16), Skip(Dtype.float64)]
                },
            ]
        ),
    ),

    'pow_tensor_only_0_1': dict(
        name=["pow"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["exponent"],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64),
                              Skip(Dtype.int8), Skip(Dtype.uint8)]
                },
            ]
        ),
    ),

    'pow_input_scalar': dict(
        name=["pow"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["exponent"],
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64),
                              Skip(Dtype.int8), Skip(Dtype.uint8)]
                },
            ]
        ),
    ),

    'pow_input_scalar_bool': dict(
        name=["pow"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["exponent"],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64),
                              Skip(Dtype.int8)]
                },
            ]
        ),
    ),

    'pow_tensor': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64),
                              Skip(Dtype.int8), Skip(Dtype.uint8)]
                },
            ],
        ),
    ),

    'pow_diff_dtype_cast': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64),
                              Skip(Dtype.bool)]
                },
            ],
        ),
    ),

    'pow_diff_dtype': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64)]
                },
            ],
        ),
    ),

    'mse_loss': dict(
        name=["mse_loss"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((16, 0)), Skip((4, 0, 9))),
                },
            ],
        ),
    ),

    'nll_loss': dict(
        name=["nll_loss"],
        atol=1e-3,
        rtol=1e-3,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((5, 16, 0)), Skip((0, 16,)), Skip((4, 82, 0, 3))),
                },
            ],
        ),
    ),

    'conv_2d': dict(
        name=["conv2d"],
        atol_half=1e-1,
        rtol_half=1e-1,
        tensor_para=dict(
            args=[
                {
                    "ins": ["weight"],
                    "shape": [Skip((18, 8, 12, 2)), Skip((6, 9, 3, 5)), Skip((2048, 1, 3, 3)), Skip((2, 6, 2, 3))],
                },
            ]
        ),
    ),

    'hardswish': dict(
        name=["hardswish"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'hardswish_domain': dict(
        name=["hardswish"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'conv_2d_no_contiguous': dict(
        name=["conv2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16), Skip(Dtype.float64)],
                },
            ]
        ),
    ),

    'relu_no_contiguous': dict(
        name=["relu"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float64)],
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            dilation=[Skip((4, 3)), Skip((2, 3)), Skip(2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'max_pool2d_return_indices': dict(
        name=["max_pool2d"],
        para=dict(
            dilation=[Skip((4, 3)), Skip((2, 3))],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'adaptive_max_pool2d_return_indices': dict(
        name=["adaptive_max_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'adaptive_max_pool2d': dict(
        name=["adaptive_max_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'binary_cross_entropy': dict(
        name=["binary_cross_entropy"],
        atol=1e-2,
        rtol=1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((4, 0)), Skip((9, 0, 16))),
                    "dtype": [Skip(Dtype.float16)],
                },
                {
                    "ins": ['weight'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64),
                              Skip(Dtype.uint8), Skip(Dtype.int8)],
                },
            ]
        ),
    ),

    'binary_cross_entropy_with_logits': dict(
        name=["binary_cross_entropy_with_logits"],
        atol=1e-3,
        rtol=1e-4,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": (Skip((0,)), Skip((4, 0)), Skip((9, 0, 16))),
                },
                {
                    "ins": ['weight'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64),
                              Skip(Dtype.uint8), Skip(Dtype.int8)],
                },
                {
                    "ins": ['pos_weight'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64),
                              Skip(Dtype.uint8), Skip(Dtype.int8)],
                },
            ],
        ),
    ),

    'pointwise_op': dict(
        name=['floor', 'asin', 'ceil', 'atan', 'erfinv'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'pointwise_op_int_without_inplace': dict(
        name=['atan'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64),
                              Skip(Dtype.int8)],
                },
            ],
        ),
    ),

    'pointwise_op_uint8': dict(
        name=['atan'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.uint8)],
                },
            ],
        ),
    ),

    'pointwise_binary': dict(
        name=['mul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # when dtype of input is uint8, output might overflow.
                    "dtype": [Skip(Dtype.uint8)],
                },

            ],
        ),
    ),

    'pointwise_binary_broadcast': dict(
        name=['mul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((0,)), Skip((8, 16, 1)), Skip((32, 0, 16))],
                    },
            ]
        )
    ),

    'pointwise_binary_broadcast_inplace': dict(
        name=['mul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((16, 0,)), Skip((8, 16, 0)), Skip((32, 0, 16))],
                },
            ],
        ),
    ),

    'pointwise_binary_diff_dtype': dict(
        name=['mul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.uint8)],
                },
                {
                    "ins": ['other'],
                    "dtype": [Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'pointwise_binary_diff_dtype_without_bool': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.uint8)],
                },
                {
                    "ins": ['other'],
                    "dtype": [Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'silu': dict(
        name=["silu"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((0,)), Skip((0, 16)), Skip((8, 0, 17))),
                }
            ]
        ),
    ),

    'div_rounding_mode': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'bmm': dict(
        name=['bmm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0, 12, 16)), Skip((4, 0, 6)), Skip((4, 9, 0)), Skip((5, 8, 13))),
                }
            ],
        ),
    ),

    'addmm': dict(
        name=["addmm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip(()),),
                }
            ],
        ),
    ),
    
    'addcmul': dict(
        name=["addcmul"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64),
                              Skip(Dtype.uint8), Skip(Dtype.int8)],
                },
            ],
        ),
    ),

    'addcdiv': dict(
        name=["addcdiv"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'avg_pool2d': dict(
        name=["avg_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((5, 2, 16, 7)), Skip((3, 4, 16, 7))),
                },
            ]
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((3, 16, 8)), Skip((4, 16, 12))),
                },
            ]
        ),
    ),

    'leaky_relu': dict(
        name=["leaky_relu"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((0, 8)), Skip((16, 0, 8))),
                },
            ],
        ),
    ),

    'matmul': dict(
        name=['matmul'],
        atol=1e-3,
    ),

    'clamp_tensor': dict(
        name=['clamp'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.float64), Skip(Dtype.float32)],
                },

            ],
        ),
    ),

    'reduce_op': dict(
        name=['mean', 'sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((16, 0, 9)),),
                    "dtype": [Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'reduce_partial_op': dict(
        name=['mean', 'sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((16, 0, 9)),),
                    "dtype": [Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'reduce_partial_op_1': dict(
        name=['std'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((12, 0)), Skip((9, 0, 7))),
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            # label_smoothing is not supported by camb kernel
            label_smoothing=[Skip(True), Skip(1), Skip(0.3), Skip(-1.3), Skip(0.4)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0, 16)), Skip((0, 5, 6)), Skip((4, 6, 0, 3))),
                },
            ],
        ),
    ),

    'cross_entropy_empty_tensor': dict(
        name=["cross_entropy"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((5, 0)),),
                },
            ],
        ),
    ),

    'cross_entropy_prob_target': dict(
        name=["cross_entropy"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'select': dict(
        name=["select"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64)],
                },
            ]
        ),
    ),

    'masked_scatter': dict(
        name=["masked_scatter"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'embedding': dict(
        name=["embedding"],
        para=dict(
            # The diopiEmbeddingRenorm_ function is temporarily unavailable due to the unsupported Cambrian operator.
            # Thus, to pass the test case, skip all non-None types of the max_norm parameter in the configuration file.
            max_norm=[Skip(1.0), Skip(-2), Skip(2), Skip(9), Skip(-0.5)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # when padding_idx set to 0, this case failed
                    "shape": [Skip((2, 3, 4))],
                },
            ],
        ),
    ),

    'clip_grad_norm': dict(
        name=["clip_grad_norm_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["grads"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'transpose': dict(
        name=['transpose'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64)],
                    "shape": [Skip(())],
                },
            ],
        ),
    ),

    'sigmoid': dict(
        name=["sigmoid"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((256, 0)), Skip((8, 0, 128))),
                },
            ],
        ),
    ),

    'sigmoid_focal_loss': dict(
        name=["sigmoid_focal_loss"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['inputs'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'conv_transpose2d': dict(
        name=["conv_transpose2d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((0, 16, 20, 8)),),
                }
            ]
        ),
    ),

    'nms': dict(
        name=["nms"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['scores'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'roi_align': dict(
        name=["roi_align"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'index': dict(
        name=["index"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'index_empty_tensor': dict(
        name=["index"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'index_int': dict(
        name=["index"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'sgd': dict(
        name=["sgd"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'sgd_without_buf': dict(
        name=["sgd"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'reciprocal': dict(
        name=["reciprocal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64)],
                },
            ],
        ),
    ),

    'adam': dict(
        name=['adam', 'adamw'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'cdist': dict(
        name=['cdist'],
        para=dict(
            # Currently, p must be equal 1.0 due to the limitation of Cambrian operator.
            p=[Skip(2), Skip(0), Skip(0.5), Skip(float("inf"))],
        ),
    ),

    'cdist_compute_mode': dict(
        name=['cdist'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['x1'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float64)],
                },
            ],
        ),
    ),

    'argmax': dict(
        name=['argmax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip(()),),
                },
            ],
        ),
    ),

    'argmax_same_value': dict(
        name=['argmax'],
    ),

    'adadelta': dict(
        name=["adadelta"],
        atol_half=1e-3,
        rtol_half=1e-3,
        tensor_para=dict(
            args=[
                {
                    # can't get correct result
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'rmsprop': dict(
        name=["rmsprop"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'conv3d': dict(
        name=['conv3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'max_pool3d': dict(
        name=['max_pool3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16), Skip(Dtype.float64)],
                },
            ]
        ),
    ),

    'max_pool3d_return_indices': dict(
        name=['max_pool3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16), Skip(Dtype.float64)],
                },
            ]
        ),
    ),

    'adaptive_avg_pool3d': dict(
        name=["adaptive_avg_pool3d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'adaptive_max_pool3d': dict(
        name=["adaptive_max_pool3d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'adaptive_max_pool3d_return_indices': dict(
        name=["adaptive_max_pool3d"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    # 'mm': dict(
    #     name=['mm'],
    #     atol=1e-1,
    #     rtol=1e-1
    # ),

    'index_fill': dict(
        name=['index_fill'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'index_fill_scalar': dict(
        name=['index_fill'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'norm': dict(
        name=['norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'group_norm': dict(
        name=['group_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'prod': dict(
        name=['prod'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'sort': dict(
        name=["sort"],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip(()),),
                },
            ],
        ),
    ),

    'ctc_loss': dict(
        name=["ctc_loss"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['log_probs'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
    ),

    'remainder_self_scalar': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64), Skip(Dtype.int8), Skip(Dtype.uint8)],
                },
            ],
        ),
    ),

    'remainder_self_bool': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64), Skip(Dtype.int8), Skip(Dtype.uint8)],
                },
            ],
        ),
    ),

    'remainder_tensor': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int8)],
                },
            ],
        ),
    ),

    'remainder_tensor_zero': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int8)],
                },
            ],
        ),
    ),

    'remainder_other_scalar': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64), Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'remainder_other_scalar_bool': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((4, 1))],
                },
            ],
        ),
    ),

    'gelu': dict(
        name=['gelu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((0, 8)), Skip((16, 0, 7))),
                },
            ],
        ),
    ),

    'linear': dict(
        name=["linear"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['weight'],
                    "requires_grad": [True],
                    "shape": (Skip((0, 8)),),
                },
            ]
        ),
    ),

    'one_hot': dict(
        name=["one_hot"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip(()),),
                },
            ],
        ),
    ),

    # When not performing a reduce operation, the accuracy comparison of scatter needs to be performed on the CPU
    'scatter': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float64), Skip(Dtype.float16), Skip(Dtype.int16),
                              Skip(Dtype.int32), Skip(Dtype.int64), Skip(Dtype.uint8), Skip(Dtype.int8), Skip(Dtype.bool)],
                },
            ]
        ),
    ),

    'scatter_specific': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32)],
                },
            ]
        ),
    ),

    'scatter_reduce': dict(
        name=['scatter'],
        para=dict(
            # The reduction operation of multiply is not supported by cnnl
            reduce=[Skip('multiply')],
        ),
    ),

    # When not performing a reduce operation, the accuracy comparison of scatter needs to be performed on the CPU
    'scatter_scalar': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float64)],
                },
            ]
        ),
    ),

    'scatter_reduce_scalar': dict(
        name=['scatter'],
        para=dict(
            # The reduction operation of multiply is not supported by cnnl
            reduce=[Skip('multiply')],
        ),
    ),

    'index_put': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.bool)],    # not supported by camb kernel when accumulate is true
                },
            ]
        ),
    ),

    'index_put_acc_bool_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.bool)],  # not supported by camb kernel when accumulate is true
                },
            ]
        ),
    ),

    'reduce_partial_op_2': dict(
        name=['min', 'max'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((12, 0)), Skip((2, 0, 12))),
                },
            ],
        ),
    ),

    'reduce_partial_op_3': dict(
        name=['any', 'all'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": (Skip((0,)), Skip((12, 0)), Skip((2, 0, 12))),
                },
            ],
        ),
    ),

    'random': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((0,)), Skip((4, 0)), Skip((3, 0, 9))],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float16), Skip(Dtype.int64), Skip(Dtype.int32),
                              Skip(Dtype.int16), Skip(Dtype.int8)],
                },
            ],
        ),
    ),

    'random_bool_and_uint8': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),
    
    'randperm': dict(
        name=['randperm'],
        para=dict(
            n=[Skip(1)],
        ),
    ),

    'bernoulli': dict(
        name=['bernoulli'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        atol=1e-4,
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((32,)), Skip((2, 16, 128))),
                },
            ]
        )
    ),

    'layer_norm_empty_tensor': dict(
        name=["layer_norm"],
        atol=1e-4,
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": (Skip((0,)), Skip((0, 12))),
                },
            ]
        )
    ),

    'copy': dict(
        name=["copy_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ]
        )
    ),

    'copy_different_dtype': dict(
        name=["copy_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
        )
    ),

    'interpolate': dict(
        name=["interpolate"],
        atol=1e-4,
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    # camb not supports 5d upsample nearest
                    # when shape is (2, 16, 23), can't get correct result
                    "shape": [Skip((2, 16, 23)), Skip((1, 3, 32, 224, 224))],
                },
            ]
        )
    ),

    'cholesky': dict(
        name=['cholesky_ex'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
        requires_backward=[0],
        saved_args=dict(output=0),
    ),

    'triangular_solve': dict(
        name=['triangular_solve'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
        ),
        saved_args=dict(output=0),
    ),

    'normal_': dict(
        name=["normal_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.float32), Skip(Dtype.float64)],
                },
            ]
        ),
    ),

    'normal_std_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['std'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'normal_mean_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'normal_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ]
        ),
    ),

    'lerp': dict(
        name=['lerp'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
                {
                    "ins": ['end'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'lerp_tensor': dict(
        name=['lerp'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
                {
                    "ins": ['end'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
                {
                    "ins": ['weight'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'triu': dict(
        name=['triu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'sgn': dict(
        name=['sgn'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.complex64), Skip(Dtype.complex128)],
                },
            ],
        ),
    ),

    'sgn_zero': dict(
        name=['sgn'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.complex64), Skip(Dtype.complex128)],
                },
            ],
        ),
    ),

    'amax': dict(
        name=['amax'],
        interface=['torch'],
        dtype=[Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
               Skip(Dtype.int64), Skip(Dtype.int32), Skip(Dtype.int16),
               Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
    ),

    'linalgqr': dict(
        name=['linalgqr'],
        atol=1e-4,
    ),
}
