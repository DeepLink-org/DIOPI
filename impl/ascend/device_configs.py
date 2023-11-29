# Copyright (c) 2023, DeepLink.
import numpy as np
from skip import Skip

device_configs = {
    # temp for 910B
    'nonzero': dict(
        name=["nonzero"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8),],
                },
            ],
        ),
    ),

    # temp for 910B
    'nonzero_uint': dict(
        name=["nonzero"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8),],
                },
            ],
        ),
    ),

    # temp for 910B
    'join': dict(
        name=['stack'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ],
        ),
    ),

    # temp for 910B
    'join_int': dict(
        name=['stack'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    "dtype": [Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),Skip(np.int32)],
                },
            ],
        ),
    ),

    # temp for 910B
    'topk_nonzero': dict(
        name=['topk'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": (Skip(np.float16),Skip(np.float32),Skip(np.float64),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),),
                },
            ],
        ),
    ),

    # temp for 910B
    'topk_zero': dict(
        name=['topk'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ],
        ),
    ),

    # temp for 910B
    'uniform': dict(
        name=['uniform'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ],
        ),
    ),

    # temp for 910B
    'normal_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'batch_norm': dict(
        name=['batch_norm'],
        atol=1e-2,
        rtol=1e-3,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "dtype": [Skip(np.float16),],
                    # temp for 910B
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    # temp for 910B
    'batch_norm_nan': dict(
        name=['batch_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'batch_norm_no_contiguous': dict(
        name=['batch_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "dtype": [Skip(np.float16),],
                    # temp for 910B
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'batch_norm_stats': dict(
        name=['batch_norm_stats'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 8, 32, 56, 56)),Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((2, 16)),],
                },
            ]
        ),
    ),

    'batch_norm_gather_stats_with_counts': dict(
        name=['batch_norm_gather_stats_with_counts'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 8, 32, 56, 56)),Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((2, 16)),],
                },
            ]
        ),
    ),

    'batch_norm_backward_reduce': dict(
        name=['batch_norm_backward_reduce'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['grad_output'],
                    "shape": [Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((2, 16)),],
                },
            ]
        ),
    ),

    'batch_norm_backward_elemt': dict(
        name=['batch_norm_backward_elemt'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['grad_out'],
                    "shape": [Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((2, 16)),],
                },
            ]
        ),
    ),

    'batch_norm_elemt': dict(
        name=['batch_norm_elemt'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((2, 16)),],
                },
            ]
        ),
    ),

    'baddbmm': dict(
        name=['baddbmm'],
        atol=1e-2,
        rtol=1e-2,
        # temp for 910B
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'baddbmm_without_inplace': dict(
        name=['baddbmm'],
        atol=1e-2,
        rtol=1e-2,
        # temp for 910B
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=['conv2d'],
        atol=1e-1,
        rtol=1e-2,
    ),

    'conv_2d_no_contiguous': dict(
        name=['conv2d'],
        atol=1e-1,
        rtol=1e-2,
    ),

    'hardswish': dict(
        name=['hardswish'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'hardswish_domain': dict(
        name=['hardswish'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'gelu': dict(
        name=['gelu'],
        atol=1e-3,
        rtol=1e-3,
    ),

    'gelu_specific': dict(
        name=['gelu'],
        atol=1e-3,
        rtol=1e-3,
    ),

    'avg_pool2d': dict(
        name=['avg_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),],
                },
            ]
        ),
    ),

    'avg_pool2d_float64': dict(
        name=['avg_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64),],
                },
            ]
        ),
    ),

    'max_pool2d': dict(
        name=['max_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'max_pool2d_return_indices': dict(
        name=['max_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=['adaptive_avg_pool2d'],
        atol=2e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((3,16,8)), Skip((4,16,12)), Skip((2,144,65,65))],
                },
            ]
        ),
    ),

    'adaptive_max_pool2d': dict(
        name=['adaptive_max_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float16),Skip(np.float64),],
                },
            ]
        ),
    ),

    'adaptive_max_pool2d_return_indices': dict(
        name=['adaptive_max_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float16),Skip(np.float64),],
                },
            ]
        ),
    ),

    'binary_cross_entropy': dict(
        name=['binary_cross_entropy'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((16,)),Skip((72,)),Skip((2, 11856)),Skip((2, 741, 80)),Skip((4, 4, 16, 20)),Skip((0,)),Skip((4, 0)),Skip((9, 0, 16)),],
                },
            ]
        ),
    ),

    'binary_cross_entropy_with_logits': dict(
        name=['binary_cross_entropy_with_logits'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((16,)),Skip((72,)),Skip((2, 11856)),Skip((2, 741, 80)),Skip((4, 4, 16, 20)),Skip((0,)),Skip((4, 0)),Skip((9, 0, 16)),],
                },
            ]
        ),
    ),

    'pointwise_op': dict(
        name=['erf', 'erfinv', 'exp', 'asin', 'rsqrt', 'ceil', 'atan'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),Skip((0,)),Skip((16, 0)),],
                },
            ]
        ),
    ),

    'pointwise_op_int_without_inplace': dict(
        name=['erf', 'exp', 'asin', 'rsqrt', 'atan'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
                },
            ]
        ),
    ),

    'pointwise_op_uint8': dict(
        name=['erf', 'exp', 'asin', 'rsqrt', 'atan'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),],
                },
            ]
        ),
    ),

    'pointwise_op_mask': dict(
        name=['bitwise_not'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
                },
            ]
        ),
    ),

    'pointwise_op_bool': dict(
        name=['erf', 'exp', 'asin', 'rsqrt', 'atan'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
                },
            ]
        ),
    ),

    'pointwise_op_abs_input': dict(
        name=['rsqrt'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),Skip((0,)),Skip((0, 16)),Skip((8, 0, 4)),],
                },
            ]
        ),
    ),

    'log_integer_input': dict(
        name=['log2', 'log10'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
                },
            ]
        ),
    ),

    'log_zero_input': dict(
        name=['log2', 'log10'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
                },
            ]
        ),
    ),

    'log_neg_input': dict(
        name=['log2', 'log10'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16), Skip(np.int32), Skip(np.int64), Skip(np.uint8), Skip(np.int8),],
                },
            ]
        ),
    ),

    'tanh': dict(
        name=['tanh'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()), Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),Skip((0,)),Skip((16, 0)),Skip((1, 0, 6))],
                },
            ]
        ),
    ),

    'tanh_not_float': dict(
        name=['tanh'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()), Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),Skip((0,)),Skip((16, 0)),Skip((1, 0, 6)),],
                },
            ]
        ),
    ),

    'sign': dict(
        name=['sign'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),Skip((0,)),Skip((16, 0)),],
                },
            ]
        ),
    ),

    'pointwise_op_zero': dict(
        name=['exp', 'rsqrt', 'ceil'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((16,)),Skip((8, 64)),],
                },
            ]
        ),
    ),

    'pointwise_op_without_inplace_zero': dict(
        name=['sign', 'exp', 'rsqrt'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((16,)),Skip((8, 64)),],
                },
            ]
        ),
    ),

    'silu': dict(
        name=['silu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'pow': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'pow_int': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),],
                },
            ]
        ),
    ),

    'pow_bool': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.bool_),],
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
                    "shape": [Skip(()),Skip((1,)),Skip((20267, 80)),Skip((2, 128, 3072)),Skip((2, 512, 38, 38)),Skip((0,)),Skip((0, 4)),Skip((9, 0, 3)),],
                },
            ]
        ),
    ),

    'pow_tensor_only_0_1': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1,)),Skip((20267, 80)),Skip((2, 128, 3072)),Skip((2, 512, 38, 38)),Skip((0,)),Skip((0, 4)),Skip((9, 0, 3)),],
                },
            ]
        ),
    ),

    'pow_broadcast': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((2, 1, 128)),Skip((2, 64, 1, 128)),Skip((2, 32, 130, 130)),Skip((0,)),Skip((8, 16, 1)),],
                },
            ]
        ),
    ),

    'pow_broadcast_inplace': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((64,)),Skip((2, 1024)),Skip((2, 384, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((5, 2, 32, 130, 130)),Skip((16, 0)),Skip((8, 16, 0)),Skip((32, 0, 16)),],
                },
            ]
        ),
    ),

    'pow_diff_dtype_cast': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int64),Skip(np.int32),Skip(np.int16),Skip(np.bool_),Skip(np.bool_),Skip(np.bool_),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'pow_diff_dtype': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64),Skip(np.float32),Skip(np.float16),Skip(np.int32),Skip(np.float64),Skip(np.float32),Skip(np.float32),Skip(np.int16),Skip(np.int64),],
                },
            ]
        ),
    ),

    'pow_input_scalar': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['exponent'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'pow_input_scalar_bool': dict(
        name=['pow'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['exponent'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),],
                },
            ]
        ),
    ),

   'pointwise_binary_diff_dtype_without_bool': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16)],
                },
                {
                    "ins": ['other'],
                    "dtype": [Skip(np.float16)],
                },
            ]
        ),
    ),

    'bitwise_op': dict(
        name=['bitwise_and', 'bitwise_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input', 'other'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'bitwise_op_diff_dtype': dict(
        name=['bitwise_and', 'bitwise_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'bitwise_op_broadcast': dict(
        name=['bitwise_and', 'bitwise_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'bitwise_op_scalar': dict(
        name=['bitwise_and', 'bitwise_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((128, 64, 3, 3)),Skip((2, 32, 130, 130)),Skip((0,)),Skip((0, 4)),Skip((4, 0, 5)),],
                },
            ]
        ),
    ),

    'bitwise_op_scalar_bool': dict(
        name=['bitwise_and', 'bitwise_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((128, 64, 3, 3)),Skip((2, 32, 130, 130)),Skip((0,)),],
                },
            ]
        ),
    ),

    'bmm': dict(
        name=['bmm'],
        atol=3e-2,
        rtol=3e-2,  
    ),

    'addmm': dict(
        name=['addmm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((2, 10)),Skip((768,)),Skip((1, 400)),Skip((4, 1)),Skip((1, 5)),Skip(()),Skip((0,)),],
                },
            ]
        ),
    ),

    'addcdiv_specific': dict(
        name=['addcdiv'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((128,)),Skip((576, 192)),Skip((64, 3, 3, 3)),Skip((10, 3, 5)),Skip((0,)),Skip((0, 5)),Skip((2, 0, 9)),],
                },
            ]
        ),
    ),

    'matmul': dict(
        name=['matmul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((128, 49, 128)),Skip((5,)),Skip((128, 4, 49, 32)),Skip((2, 1, 3136, 3136)),Skip((2, 784, 64)),Skip((2, 16, 8, 64)),Skip((2, 31, 6, 40, 512)),],
                },
            ]
        ),
    ),

    'fill': dict(
        name=['fill_'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16)],
                },
            ]
        ),
    ),

    'reduce_op': dict(
        name=['mean', 'sum'],
        atol=1e-3,
        rtol=1e-3,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16)],
                },
            ],
        ),
    ),

    'reduce_partial_op': dict(
        atol=1e-3,
        rtol=1e-3,
        name=['mean', 'sum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16)],
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
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'reduce_partial_op_2': dict(
        name=['min', 'max'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'max_min_equal_input': dict(
        name=['min', 'max'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),],
                },
            ]
        ),
    ),

    'max_min_all': dict(
        name=['min', 'max'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'mse_loss': dict(
        name=['mse_loss'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((64,)),Skip((2, 11856, 2)),Skip((16, 2, 2964, 2)),Skip((2964, 32)),Skip((0,)),Skip((16, 0)),Skip((4, 0, 9)),],
                },
            ]
        ),
    ),

    'nll_loss': dict(
        name=['nll_loss'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # TODO(someone): dtype float16 is not supported
                    "dtype": [Skip(np.float16)],
                },
            ]
        ),
    ),

    'nll_loss_empty_tensor': dict(
        name=['nll_loss'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # TODO(someone): dtype float16 is not supported
                    "dtype": [Skip(np.float16)],
                },
            ]
        ),
    ),

    'cross_entropy': dict(
        name=['cross_entropy'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1024, 81)),Skip((3, 9)),Skip((64, 9)),Skip((5, 9, 12, 4)),Skip((0, 16)),Skip((0, 5, 6)),Skip((4, 6, 0, 3)),],
                },
            ]
        ),
    ),

    'cross_entropy_empty_tensor': dict(
        name=['cross_entropy'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((5, 0)),],
                },
            ]
        ),
    ),

    'cross_entropy_prob_target': dict(
        name=['cross_entropy'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((3, 5, 6, 6)),Skip((1024, 81)),Skip((64, 8, 8)),Skip((3, 5, 6, 6)),Skip((1024, 81)),Skip((64, 8)),Skip((12, 0)),Skip((9, 0, 8)),],
                },
            ]
        ),
    ),

    'select': dict(
        name=['select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'select_not_float': dict(
        name=['select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'index_select': dict(
        name=['index_select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'index_select_not_float': dict(
        name=['index_select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int32),Skip(np.int16),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'masked_scatter': dict(
        name=['masked_scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'linear': dict(
        name=['linear'],
        atol = 1e-1,
        rtol = 1e-1,
    ),

    'embedding': dict(
        name=['embedding'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int64),Skip(np.int64),Skip(np.int32),],
                },
            ]
        ),
    ),

    'embedding_forward': dict(
        name=['embedding'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int64),Skip(np.int64),Skip(np.int32),],
                },
            ]
        ),
    ),

    'clip_grad_norm': dict(
        name=['clip_grad_norm_'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['grads'],
                    "dtype": [Skip(np.float32),Skip(np.float16),Skip(np.float64),],
                },
            ]
        ),
    ),

    'clip_grad_norm_diff_shape': dict(
        name=['clip_grad_norm_'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    "dtype": [Skip(np.float32),Skip(np.float16),Skip(np.float64),],
                },
            ]
        ),
    ),

    'tril': dict(
        name=['tril'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'one_hot': dict(
        name=['one_hot'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int64),],
                },
            ]
        ),
    ),

    'split': dict(
        name=['split'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),],
                },
            ]
        ),
    ),

    'split_bool': dict(
        name=['split'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "dtype": [Skip(np.bool_),],
                },
            ]
        ),
    ),

    'sort': dict(
        name=['sort'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((11400,)),Skip((12, 8)),Skip((8, 12, 9)),Skip((4, 4, 16, 20)),Skip((4, 4, 16, 2, 20)),Skip((24180,)),Skip((0,)),Skip((12, 0)),Skip((4, 0, 5)),],
                },
            ]
        ),
    ),

    'sort_same_value': dict(
        name=['sort'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((11400,)),Skip((4, 4, 16, 20)),Skip((4, 4, 16, 2, 20)),],
                },
            ]
        ),
    ),

    'transpose': dict(
        name=['transpose'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((32,)),Skip((2, 1536, 950)),Skip((16, 8)),Skip((660, 6, 49, 32)),Skip((0,)),Skip((0, 8)),Skip((16, 0, 8)),],
                },
            ]
        ),
    ),

    'where': dict(
        name=['where'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "dtype": [Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'where_broadcast': dict(
        name=['where'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "dtype": [Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'where_same_value': dict(
        name=['where'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "dtype": [Skip(np.bool_),],
                },
            ]
        ),
    ),

    'dropout': dict(
        name=['dropout'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'dropout_training': dict(
        name=['dropout'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'dropout2d': dict(
        name=['dropout2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'leaky_relu': dict(
        name=['leaky_relu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'sigmoid_focal_loss': dict(
        name=['sigmoid_focal_loss'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['inputs'],
                    "shape": [Skip(()),Skip((64,)),Skip((16, 7)),Skip((2, 11856, 2)),Skip((16, 2, 2964, 2)),Skip((0,)),Skip((6, 0)),Skip((12, 0, 4)),],
                },
            ]
        ),
    ),

    'nms': dict(
        name=['nms'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['boxes'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'roi_align': dict(
        name=['roi_align'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((6, 3, 32, 32)),Skip((2, 3, 16, 16)),],
                },
            ]
        ),
    ),

    'index': dict(
        name=['index'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'index_empty_tensor': dict(
        name=['index'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'index_int': dict(
        name=['index'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'sgd': dict(
        name=['sgd'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(np.float16), Skip(np.float32), Skip(np.float64),],
                },
            ]
        ),
    ),

    'sgd_without_buf': dict(
        name=['sgd'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "dtype": [Skip(np.float16), Skip(np.float32),],
                },
            ]
        ),
    ),

    'masked_fill_scalar_int': dict(
        name=['masked_fill'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8)],
                },
            ]
        ),
    ),

    'reciprocal': dict(
        name=['reciprocal'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16)],
                },
            ]
        ),
    ),
    
    'reciprocal_zero': dict(
        name=['reciprocal'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16)],
                },
            ]
        ),
    ),

    'conv_transpose2d': dict(
        name=['conv_transpose2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'unfold': dict(
        name=['unfold'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'unfold_int': dict(
        name=['unfold'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.bool_),Skip(np.int64),Skip(np.int32),Skip(np.int16),Skip(np.int8),Skip(np.uint8),],
                },
            ]
        ),
    ),

    'cdist': dict(
        name=['cdist'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['x1'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'cdist_compute_mode': dict(
        name=['cdist'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['x1'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'bitwise_not_uint8': dict(
        name=['bitwise_not'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8),],
                },
            ]
        ),
    ),

    'bitwise_not_int': dict(
        name=['bitwise_not'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.bool_),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),],
                },
            ]
        ),
    ),

    'adadelta': dict(
        name=['adadelta'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [Skip(()), Skip((16,)), Skip((16, 8)), Skip((2, 3, 16)), Skip((4, 32, 7, 7)), Skip((0,)), Skip((4, 0)), Skip((12, 0, 9)),],
                },
            ]
        ),
    ),

    'rmsprop': dict(
        name=['rmsprop'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [Skip(()), Skip((16,)), Skip((16, 8)), Skip((2, 3, 16)), Skip((4, 32, 7, 7)), Skip((0,)), Skip((4, 0)), Skip((12, 0, 9)),],
                },
            ]
        ),
    ),

    'smooth_l1_loss': dict(
        name=['smooth_l1_loss'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.float32),Skip(np.float64),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'conv3d': dict(
        name=['conv3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
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
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
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
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'adaptive_avg_pool3d': dict(
        name=['adaptive_avg_pool3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float16),Skip(np.float64),],
                },
            ]
        ),
    ),

    'adaptive_max_pool3d': dict(
        name=['adaptive_max_pool3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float16),Skip(np.float64),],
                },
            ]
        ),
    ),

    'adaptive_max_pool3d_return_indices': dict(
        name=['adaptive_max_pool3d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float16),Skip(np.float64),],
                },
            ]
        ),
    ),

    'masked_select': dict(
        name=['masked_select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'masked_select_not_float': dict(
        name=['masked_select'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'imum_input_nan': dict(
        name=['minimum', 'maximum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'imum_other_nan': dict(
        name=['minimum', 'maximum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'imum_broadcast': dict(
        name=['minimum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'mm': dict(
        name=['mm'],
        atol=2e-2,
        rtol=2e-2, 
    ),

    'mm_diff_dtype': dict(
        name=['mm'],
        atol=2e-2,
        rtol=2e-2, 
    ),

    'index_fill_scalar': dict(
        name=['index_fill'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'index_fill_scalar_specific': dict(
        name=['index_fill'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'index_fill_tensor': dict(
        name=['index_fill'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'index_fill_tensor_specific': dict(
        name=['index_fill'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'expand': dict(
        name=['expand'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.bool_),Skip(np.float16),Skip(np.float64),Skip(np.int64),Skip(np.int32),Skip(np.int16),Skip(np.int8),Skip(np.uint8),],
                },
            ]
        ),
    ),

    'linspace': dict(
        name=['linspace'],
        para=dict(
            start=[Skip(0),Skip(1.4),Skip(-10),Skip(False),Skip(True),Skip(2),Skip(-2.5),Skip(-10),Skip(0.0001),Skip(0),Skip(5),],
            end=[Skip(0.5),Skip(-2.4),Skip(True),Skip(100.232),Skip(False),Skip(2),Skip(-2.5),Skip(10),Skip(0.001),Skip(10),Skip(-2),],
            steps=[Skip(24),Skip(23),Skip(152),Skip(100),Skip(76),Skip(50),Skip(38),Skip(25),Skip(5),Skip(0),Skip(1),],
        ),
    ),

    'permute': dict(
        name=['permute'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'pad': dict(
        name=['pad'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'pad_not_float': dict(
        name=['pad'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'constant_pad': dict(
        name=['pad'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'constant_pad_positive': dict(
        name=['pad'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.uint8),],
                },
            ]
        ),
    ),

    'roll': dict(
        name=['roll'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.bool_),Skip(np.int64),Skip(np.int32),Skip(np.int16),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'norm': dict(
        name=['norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'group_norm': dict(
        name=['group_norm'],
        atol=5e-2,
        rtol=5e-2,
        atol_half=5e-2,
        rtol_half=5e-2,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),],
                },
            ]
        ),
    ),

    'unique': dict(
        name=['unique'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int64),Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'unique_same_value': dict(
        name=['unique'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int64),Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
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
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'ctc_loss': dict(
        name=['ctc_loss'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['log_probs'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'ctc_loss_un_padded': dict(
        name=['ctc_loss'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['log_probs'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'remainder_self_scalar': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'remainder_self_bool': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['other'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
                },
            ]
        ),
    ),

    'remainder_tensor': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'remainder_tensor_zero': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'remainder_other_scalar': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'remainder_other_scalar_bool': dict(
        name=['remainder'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
                },
            ]
        ),
    ),

    'gather': dict(
        name=['gather'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'gather_0dim': dict(
        name=['gather'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'gather_not_float': dict(
        name=['gather'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'scatter': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
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
                    "dtype": [Skip(np.float32),],
                },
            ]
        ),
    ),

    'scatter_reduce': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'scatter_scalar': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'scatter_reduce_scalar': dict(
        name=['scatter'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'index_put_acc_three_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'index_put_acc_two_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'index_put_acc_one_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'index_put_acc_bool_indices_zeros': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.int64),],
                },
            ]
        ),
    ),

    'index_put_one_indices': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'index_put_bool_indices_value': dict(
        name=['index_put'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'arange': dict(
        name=['arange'],
        para=dict(
            start=[Skip(0),Skip(0),Skip(-4),Skip(0.1),Skip(10),Skip(2.3),Skip(True),Skip(-20),Skip(90),Skip(0.001),],
            end=[Skip(91),Skip(128),Skip(5),Skip(0.5),Skip(10),Skip(2.3),Skip(100),Skip(False),Skip(-90),Skip(0.0001),],
            step=[Skip(13),Skip(1),Skip(1),Skip(0.1),Skip(True),Skip(0.5),Skip(2.1),Skip(0.5),Skip(-5.6),Skip(-1e-05),],
        ),
    ),

    'arange_default': dict(
        name=['arange'],
        para=dict(
            end=[Skip(5),Skip(10),Skip(4.0),Skip(9.0),],
        ),
    ),

    'randperm': dict(
        name=['randperm'],
        para=dict(
            n=[Skip(2),Skip(1999),Skip(640000),Skip(0),Skip(1),],
        ),
    ),

    'random': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int64),Skip(np.int32),Skip(np.int16),Skip(np.int8),],
                },
            ]
        ),
    ),

    'random_bool_and_uint8': dict(
        name=['random'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.bool_),Skip(np.uint8),],
                },
            ]
        ),
    ),

    'bernoulli': dict(
        name=['bernoulli'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'bernoulli_int': dict(
        name=['bernoulli'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int64),Skip(np.int32),Skip(np.int16),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'layer_norm': dict(
        name=['layer_norm'],
        atol=1e-2,
        rtol=1e-3,
        para=dict(
            eps=[Skip(2),],
        ),
        # temp for 910B
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                },
            ]
        ),
    ),

    'copy': dict(
        name=['copy_'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    # "shape": [Skip((2, 1, 38, 45)),Skip()],
                    # temp for 910B
                    "dtype": [Skip(np.float32), Skip(np.float64), Skip(np.float16), Skip(np.bool_), Skip(np.int64), Skip(np.int32), Skip(np.int16), Skip(np.int8), Skip(np.uint8)]
                },
            ]
        ),
    ),

    'fill_not_float': dict(
        name=["fill_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.bool_), Skip(np.int64), Skip(np.int32), Skip(np.int16), Skip(np.int8), Skip(np.uint8)]
                },
            ]
        ),
    ),

    'copy_different_dtype': dict(
        name=['copy_'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 1, 38, 45))],
                },
            ]
        ),
    ),

    'copy_broadcast': dict(
        name=['copy_'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64)],
                },
            ]
        ),
    ),

    'interpolate': dict(
        name=['interpolate'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 16, 23)),Skip((2, 256, 25, 38)),Skip((1, 3, 32, 224, 224)),Skip((2, 2, 16, 16)),Skip((2, 2, 16, 16)),Skip((2, 256, 13, 19)),Skip((3, 12, 14, 19)),Skip((2, 16, 1, 1)),Skip((2, 16, 15, 32)),Skip((1, 3, 32, 112, 112)),Skip((1, 3, 32, 112, 112)),Skip((2, 32, 32)),Skip((2, 32, 32)),Skip((2, 32, 32)),],
                },
            ]
        ),
    ),

    'col2im': dict(
        name=['col2im'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'im2col': dict(
        name=['im2col'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'flip': dict(
        name=['flip'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
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
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'triangular_solve': dict(
        name=['triangular_solve'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),],
                },
            ]
        ),
    ),

    'repeat': dict(
        name=['repeat'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'meshgrid': dict(
        name=['meshgrid'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'polar': dict(
        name=['polar'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['abs'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((64, 1, 128)),Skip((128, 64, 3, 3)),Skip((2, 32, 130, 130)),Skip((0,)),Skip((0, 3)),Skip((18, 0, 9)),],
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
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((2, 1, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((2, 32, 130, 130)),Skip((0,)),Skip((0, 3)),Skip((18, 0, 9)),],
                },
            ]
        ),
    ),

    'lerp_tensor': dict(
        name=['lerp'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((2, 1, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((2, 32, 130, 130)),Skip((0,)),Skip((0, 3)),Skip((18, 0, 9)),],
                },
            ]
        ),
    ),

    'triu': dict(
        name=['triu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                },
            ]
        ),
    ),

    'amax': dict(
        name=['amax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip(()),Skip((18,)),Skip((1024, 64)),Skip((384, 128)),Skip((64, 1, 128)),Skip((128, 64, 3, 3)),Skip((2, 32, 130, 130)),Skip((128, 64, 32, 3)),Skip((384, 128)),Skip((3, 0)),Skip((4, 0, 5)),],
                },
            ]
        ),
    ),

    'linalgqr': dict(
        name=['linalgqr'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1024, 384)),Skip((384, 1024)),Skip((64, 1, 128)),Skip((128, 64, 32, 3)),Skip((2, 32, 130, 100)),Skip((2, 32, 100, 150)),Skip((4, 2, 1024, 1024)),Skip((4, 284, 284)),Skip((64, 64)),Skip((4, 0)),Skip((0, 16)),Skip((6, 0, 0)),],
                },
            ]
        ),
    ),

    'reduce_partial_op_4': dict(
        name=['sum'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-4,
        # temp for 910B
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                },
            ],
        ),
    ),

    'rotary_emb': dict(
        name=["rotary_emb"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ],
        ),
    ),

    # temp for 910B
    'normal_': dict(
        name=["normal_"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    # temp for 910B
    'normal_std_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['std'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    # temp for 910B
    'normal_mean_tensor': dict(
        name=["normal"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(np.float64), Skip(np.float32), Skip(np.float16)],
                },
            ]
        ),
    ),

    'rms_norm': dict(
        name=["rms_norm"],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(np.float32)],
                },
            ],
        ),
    )
}
