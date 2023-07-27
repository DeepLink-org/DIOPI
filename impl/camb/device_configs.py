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

    'pow_tensor_skip_camb': dict(
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

    'pow_diff_dtype': dict(
        name=["pow"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "dtype": [Skip(Dtype.uint8)]
                },
                {
                    "ins": ["exponent"],
                    "dtype": [Skip(Dtype.float16)]
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
                    "dtype": [Skip(Dtype.float16)]
                },
            ]
        ),
    ),

    'nll_loss': dict(
        name=["nll_loss"],
        atol=1e-3,
        rtol=1e-3,
    ),

    'conv_2d': dict(
        name=["conv2d"],
        atol_half=1e-1,
        rtol_half=1e-1,
        tensor_para=dict(
            args=[
                {
                    "ins": ["weight"],
                    "shape": [Skip((2048, 1, 3, 3))],
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

    'pow_float_tensor': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 512, 38, 38))],
                    "dtype": [Skip(Dtype.float64)],
                },
                {
                    "ins": ['exponent'],
                    "shape": [Skip((2, 512, 38, 38))],
                    "dtype": [Skip(Dtype.float64)],
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
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16)],
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
    ),

    'pointwise_op': dict(
        name=['floor', 'asin', 'ceil', 'atan'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'pointwise_op_zero': dict(
        name=['ceil'],
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
        name=['asin', 'atan'],
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

    'pointwise_op_bool': dict(
        name=['asin'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.bool)],
                },
            ],
        ),
    ),

    'erfinv': dict(
        name=["erfinv"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ]
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
        atol=1e-1,
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
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },

            ],
        ),
    ),

    'reduce_partial_op': dict(
        name=['sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'reduce_partial_op_1': dict(
        name=['std'],
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            # label_smoothing is not supported by camb kernel
            label_smoothing=[Skip(0.5)],
        ),
    ),

    'cross_entropy_prob_target': dict(
        name=["cross_entropy"],
        para=dict(
            # label_smoothing is not supported by camb kernel
            label_smoothing=[Skip(0.1), Skip(0.3), Skip(0.5)],
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
            max_norm=[Skip(1.0)],
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
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
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
    ),

    'argmax_same_value': dict(
        name=['argmax'],
    ),

    'adadelta': dict(
        name=["adadelta"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
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

    'smooth_l1_loss': dict(
        name=["smooth_l1_loss"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
            ],
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
                    "dtype": [Skip(Dtype.float32), Skip(Dtype.float16)],
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

    'remainder_self_scalar_float': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "shape:" [Skip(1, 28, 28)],
                    "dtype": [Skip(Dtype.float16)],
                },
            ],
        ),
    ),

    'remainder_bool': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "dtype": [Skip(Dtype.int16), Skip(Dtype.int32), Skip(Dtype.int64), Skip(Dtype.int8)],
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
                    "dtype": [Skip(Dtype.int16)],
                },
            ],
        ),
    ),

    'remainder_other_scalar': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
    ),

    'remainder_self_scalar_int': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "shape:" [Skip(1, 28, 28)],
                },
            ],
        ),
    ),

    'remainder_scalar_bool': dict(
        name = ['remainder'],
        atol = 1e-1,
        rtol = 1e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "shape:" [Skip(4, 1)],
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
                    "dtype": [Skip(Dtype.uint8),    # overflow issue
                              Skip(Dtype.bool)],    # not supported by camb kernel when accumulate is true
                },
            ]
        ),
    ),

    # when accumulate is True and dtype of indices is bool, can't get the correct result
    'index_put_acc_bool_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16),
                              Skip(Dtype.int64), Skip(Dtype.int32),
                              Skip(Dtype.int8), Skip(Dtype.uint8), Skip(Dtype.bool)],
                },
            ]
        ),
    ),

    'random': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.int64), Skip(Dtype.int32),
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
                    "shape": [Skip((1, 3, 32, 224, 224))],
                },
            ]
        )
    ),

    'col2im': dict(
        name=["col2im"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 576, 46464))],
                },
            ]
        ),
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

    'polar': dict(
        name=["polar"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['abs'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
                },
                {
                    "ins": ['angle'],
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32)],
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
                    "dtype": [Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.float16)],
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
                    "dtype": [Skip(Dtype.complex64), Skip(Dtype.complex128), Skip(Dtype.float64), Skip(Dtype.float32), Skip(Dtype.int16)],
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
