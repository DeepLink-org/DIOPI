from .device_config_helper import Skip
from .diopi_runtime import Dtype

device_configs = {
    'batch_norm': dict(
        name=['batch_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 8, 32, 56, 56)),Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((2, 16)),],
                },
            ]
        ),
    ),

    'batch_norm_nan': dict(
        name=['batch_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 8, 32, 56, 56)),Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((16, 2)),],
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
                    "shape": [Skip((2, 8, 32, 56, 56)),Skip((2, 64, 32, 32)),Skip((2, 96, 28)),Skip((32, 16)),],
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
        tensor_para=dict(
            atol=4e-2,
            rtol=4e-2,
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((32, 64, 16)),Skip((32, 64, 32)),Skip((168, 52, 64)),Skip((2, 0, 2)),],
                },
            ]
        ),
    ),

    'baddbmm_without_inplace': dict(
        name=['baddbmm'],
        tensor_para=dict(
            atol=4e-2,
            rtol=4e-2,
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((32, 64, 16)),Skip((32, 64, 32)),Skip((168, 52, 64)),Skip((16,)),Skip((64, 32)),Skip((1, 52, 64)),Skip((32, 1, 16)),Skip((32, 64, 1)),Skip((64,)),Skip((2,)),Skip((0, 2)),],
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=['conv2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((4, 7, 12, 13)),Skip((6, 16, 19, 8)),Skip((6, 27, 12, 8)),Skip((2, 256, 200, 304)),Skip((2, 2048, 64, 64)),Skip((2, 2048, 1, 1)),Skip((2, 256, 200, 304)),Skip((0, 6, 5, 10)),],
                },
            ]
        ),
    ),

    'conv_2d_no_contiguous': dict(
        name=['conv2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 256, 200, 304)),Skip((2, 2048, 64, 64)),Skip((2, 2048, 1, 1)),Skip((2, 256, 200, 304)),],
                },
            ]
        ),
    ),

    'hardtanh': dict(
        name=['hardtanh'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'hardswish': dict(
        name=['hardswish'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'threshold': dict(
        name=['threshold'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'threshold_int': dict(
        name=['threshold'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'threshold_uint': dict(
        name=['threshold'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.uint8),],
                },
            ]
        ),
    ),

    'gelu': dict(
        name=['gelu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'gelu_specific': dict(
        name=['gelu'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'avg_pool2d': dict(
        name=['avg_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),],
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
                    "dtype": [Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=['adaptive_avg_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'adaptive_avg_pool2d_zero_size': dict(
        name=['adaptive_avg_pool2d'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.float64),],
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
        name=['erf', 'erfinv', 'exp', 'neg', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'ceil', 'atan'],
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
        name=['erf', 'exp', 'neg', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
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
        name=['erf', 'exp', 'neg', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
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
        name=['logical_not', 'bitwise_not'],
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
        name=['erf', 'exp', 'asin', 'sqrt', 'rsqrt', 'atan', 'logical_not'],
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
        name=['log', 'log2', 'log10', 'sqrt', 'rsqrt'],
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
        name=['log', 'log2', 'log10'],
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
        name=['log', 'log2', 'log10'],
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
        name=['log', 'log2', 'log10'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1,)),Skip((1024,)),Skip((364800, 4)),Skip((2, 128, 3072)),Skip((256, 128, 3, 3)),Skip((2, 31, 512, 6, 40)),],
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
        name=['exp', 'neg', 'sqrt', 'logical_not', 'rsqrt', 'ceil'],
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
        name=['sign', 'exp', 'sqrt', 'logical_not', 'rsqrt'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((16,)),Skip((8, 64)),],
                },
            ]
        ),
    ),

    'neg_without_inplace_zero': dict(
        name=['neg'],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),],
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
                    "dtype": [Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.int64),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.bool),Skip(Dtype.bool),Skip(Dtype.bool),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float64),Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int32),Skip(Dtype.float64),Skip(Dtype.float32),Skip(Dtype.float32),Skip(Dtype.int16),Skip(Dtype.int64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),],
                },
            ]
        ),
    ),

    'pointwise_binary': dict(
        name=['add', 'ne', 'le', 'lt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((2, 32, 130, 130)),Skip((0,)),],
                },
            ]
        ),
    ),

    'pointwise_binary_broadcast': dict(
        name=['add', 'div', 'ne', 'le', 'lt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((64,)),Skip((2, 1024)),Skip((2, 1, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 1, 128)),Skip((2, 32, 130, 130)),Skip((0,)),Skip((8, 16, 1)),Skip((32, 0, 16)),],
                },
            ]
        ),
    ),

    'pointwise_binary_broadcast_inplace': dict(
        name=['add', 'div', 'ne', 'le', 'lt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((64,)),Skip((2, 1024)),Skip((2, 384, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((5, 2, 32, 130, 130)),Skip((16, 0)),Skip((8, 16, 0)),Skip((32, 0, 16)),],
                },
            ]
        ),
    ),

    'pointwise_binary_diff_dtype': dict(
        name=['ne', 'le', 'lt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64),Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int64),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int8),Skip(Dtype.uint8),Skip(Dtype.bool),],
                },
            ]
        ),
    ),

    'pointwise_binary_diff_dtype_inplace': dict(
        name=['ne', 'le', 'lt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64),Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int32),Skip(Dtype.float64),Skip(Dtype.float64),Skip(Dtype.int8),Skip(Dtype.float32),Skip(Dtype.int8),],
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
                    "dtype": [Skip(Dtype.float64),Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.int32),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int8),Skip(Dtype.uint8),Skip(Dtype.float32),],
                },
            ]
        ),
    ),

    'pointwise_binary_dtype_bool': dict(
        name=['add', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1024,)),Skip((384, 128)),Skip((128, 64, 3, 3)),Skip((2, 32, 130, 130)),],
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
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.int8),Skip(Dtype.bool),],
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

    'div': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((2, 32, 130, 130)),Skip((0,)),],
                },
            ]
        ),
    ),

    'div_broadcast': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((64,)),Skip((2, 1024)),Skip((2, 1, 128)),Skip((128, 64, 3, 3)),Skip((2, 64, 1, 128)),Skip((2, 32, 130, 130)),Skip((0,)),Skip((8, 16, 1)),Skip((32, 0, 16)),],
                },
            ]
        ),
    ),

    'div_diff_dtype_inplace': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64),Skip(Dtype.float32),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'div_rounding_mode': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((128, 64, 3, 3)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'div_dtype_int_and_bool': dict(
        name=['div'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1024,)),Skip((384, 128)),Skip((128, 64, 3, 3)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'pointwise_binary_scalar': dict(
        name=['ne', 'le', 'lt'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((2, 64, 128)),Skip((128, 64, 3, 3)),Skip((128, 32, 2, 2)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'pointwise_binary_test_equal_and_logic_specific': dict(
        name=['ne', 'le', 'lt', 'ge', 'logical_and', 'logical_or'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1024,)),Skip((384, 128)),],
                },
            ]
        ),
    ),

    'sub_constant_with_alpha_and_no_contiguous': dict(
        name=['sub'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((2, 64, 128)),Skip((128, 64, 3, 3)),Skip((128, 32, 2, 2)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'pointwise_binary_constant_with_alpha_and_no_contiguous': dict(
        name=['add'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((1024,)),Skip((384, 128)),Skip((2, 64, 128)),Skip((128, 64, 3, 3)),Skip((128, 32, 2, 2)),Skip((2, 32, 130, 130)),],
                },
            ]
        ),
    ),

    'pointwise_binary_with_alpha': dict(
        name=['add'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 3)),Skip((2, 2, 4, 3)),],
                },
            ]
        ),
    ),

    'pointwise_binary_with_alpha_bool': dict(
        name=['add'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 3)),Skip((2, 2, 4, 3)),],
                },
            ]
        ),
    ),

    'bmm': dict(
        name=['bmm'],
        tensor_para=dict(
            atol=1e-3,
            rtol=1e-3,
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((16, 726, 32)),Skip((16, 100, 100)),Skip((9, 5, 5)),Skip((0, 12, 16)),Skip((4, 0, 6)),Skip((4, 9, 0)),Skip((5, 8, 13)),],
                },
            ]
        ),
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

    'addcmul': dict(
        name=['addcmul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((128,)),Skip((576, 192)),Skip((64, 3, 3, 3)),Skip((10, 3, 5)),Skip((0,)),Skip((0, 5)),Skip((2, 0, 9)),],
                },
            ]
        ),
    ),

    'addcdiv': dict(
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

    'addcdiv_addcmul_broadcast_inplace': dict(
        name=['addcdiv', 'addcmul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((3, 4)),Skip((4, 5, 5)),Skip((2, 3, 4, 5)),],
                },
            ]
        ),
    ),

    'addcdiv_addcmul_without_inplace': dict(
        name=['addcdiv', 'addcmul'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((128,)),Skip((576, 192)),Skip((64, 3, 1, 3)),Skip((10, 3, 5)),Skip((0,)),Skip((0, 5)),Skip((2, 0, 9)),],
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

    'clamp_scalar': dict(
        name=['clamp'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),],
                },
            ]
        ),
    ),

    'clamp_max_scalar': dict(
        name=['clamp_max'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'clamp_min_scalar': dict(
        name=['clamp_min'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'clamp_tensor': dict(
        name=['clamp'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.float32), Skip(Dtype.float64),],
                },
            ],
        ),
    ),

    'clamp_max_tensor': dict(
        name=['clamp_max'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.float32), Skip(Dtype.float64),],
                },
            ],
        ),
    ),

    'clamp_min_tensor': dict(
        name=['clamp_min'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.float32), Skip(Dtype.float64),],
                },
            ],
        ),
    ),

    'fill': dict(
        name=['fill_'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16)],
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
                    "dtype": [Skip(Dtype.float16)],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
                },
            ]
        ),
    ),

    'reduce_partial_op_3': dict(
        name=['any', 'all'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.bool),Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'reduce_partial_op_zeros_input': dict(
        name=['any', 'all'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.bool),Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'reduce_partial_op_ones_input': dict(
        name=['any', 'all'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.bool),Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),],
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
                    "shape": [Skip((200, 79)),Skip((200, 80)),Skip((5, 16, 0)),Skip((0, 16)),Skip((4, 82, 0, 3)),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
                },
            ]
        ),
    ),

    'linear': dict(
        name=['linear'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'log_softmax': dict(
        name=['log_softmax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'log_softmax_specific': dict(
        name=['log_softmax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'soft_max': dict(
        name=['softmax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'embedding': dict(
        name=['embedding'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int64),Skip(Dtype.int64),Skip(Dtype.int32),],
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
                    "dtype": [Skip(Dtype.int64),Skip(Dtype.int64),Skip(Dtype.int32),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.int64),],
                },
            ]
        ),
    ),

    'join': dict(
        name=['cat', 'stack'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.float64),Skip(Dtype.int16),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),Skip(Dtype.int32),],
                },
            ]
        ),
    ),

    'cat_diff_size': dict(
        name=['cat'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensors'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.float64),Skip(Dtype.int16),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),Skip(Dtype.int32),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),],
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
                    "dtype": [Skip(Dtype.bool),],
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

    'topk_nonzero': dict(
        name=['topk'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip(()),Skip((8723,)),Skip((1024, 81)),Skip((5, 4, 6)),Skip((2, 2, 64, 64)),Skip((0,)),Skip((12, 0)),Skip((4, 0, 7)),],
                },
            ]
        ),
    ),

    'topk_zero': dict(
        name=['topk'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((1,)),Skip((16,)),Skip((50, 25, 10)),],
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
                    "dtype": [Skip(Dtype.uint8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.uint8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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

    'slice': dict(
        name=['slice_op'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((7,)),Skip((128, 3, 3)),Skip((2, 3, 224, 224)),Skip((3, 2, 6, 197, 64)),],
                },
            ]
        ),
    ),

    'slice_int': dict(
        name=['slice_op'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((7,)),Skip((128, 3, 3)),Skip((2, 3, 224, 224)),Skip((3, 2, 6, 197, 64)),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.float32), Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16), Skip(Dtype.float32),],
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
                    "dtype": [Skip(Dtype.uint8)],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'reciprocal_int': dict(
        name=['reciprocal'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool)],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.bool),Skip(Dtype.int64),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int8),Skip(Dtype.uint8),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.uint8),],
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
                    "dtype": [Skip(Dtype.bool),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'argmax': dict(
        name=['argmax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),],
                },
            ]
        ),
    ),

    'argmax_same_value': dict(
        name=['argmax'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float16),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
                },
            ]
        ),
    ),

    'imum': dict(
        name=['minimum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input', 'other'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.bool),Skip(Dtype.int64),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int8),Skip(Dtype.uint8),],
                },
            ]
        ),
    ),

    'imum_input_nan': dict(
        name=['minimum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'imum_other_nan': dict(
        name=['minimum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((128, 128)),Skip((256, 8, 8)),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'imum_ones': dict(
        name=['minimum'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'mm': dict(
        name=['mm'],
        tensor_para=dict(
            atol=1e-3,
            rtol=1e-3,
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'mm_diff_dtype': dict(
        name=['mm'],
        tensor_para=dict(
            atol=1e-3,
            rtol=1e-3,
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.bool),Skip(Dtype.float16),Skip(Dtype.float64),Skip(Dtype.int64),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int8),Skip(Dtype.uint8),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.uint8),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.bool),Skip(Dtype.int64),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int8),Skip(Dtype.uint8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
                },
            ]
        ),
    ),

    'group_norm': dict(
        name=['group_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.int64),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.int64),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.int64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
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

    'uniform': dict(
        name=['uniform'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int64),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int8),],
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
                    "dtype": [Skip(Dtype.bool),Skip(Dtype.uint8),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
                    "dtype": [Skip(Dtype.int64),Skip(Dtype.int32),Skip(Dtype.int16),Skip(Dtype.int8),Skip(Dtype.uint8),Skip(Dtype.bool),],
                },
            ]
        ),
    ),

    'layer_norm': dict(
        name=['layer_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((2, 5, 3, 5)),Skip((2, 3136, 128)),Skip((2, 64)),Skip((32,)),Skip((2, 5, 3, 5)),Skip((2, 16, 128)),],
                },
            ]
        ),
    ),

    'layer_norm_empty_tensor': dict(
        name=['layer_norm'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip((0,)),Skip((0, 12)),Skip((6, 0, 9)),],
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
                    "shape": [Skip(()),Skip((8,)),Skip((12,)),Skip((192, 147)),Skip((1, 1, 384)),Skip((2, 1, 38, 45)),
                              Skip((0,12)),Skip((12,0,9))],
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
                    "shape": [Skip((192, 147)),Skip((1, 1, 384)),Skip((2, 1, 38, 45)),Skip((100,100))],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),Skip(Dtype.bool),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),Skip(Dtype.bool),],
                },
            ]
        ),
    ),

    'normal': dict(
        name=['normal'],
        para=dict(
            mean=[Skip(-1),Skip(-0.5),Skip(0),Skip(0.1),Skip(2),Skip(True),Skip(False),Skip(0.2),Skip(-2),Skip(0),],
            std=[Skip(0),Skip(0.5),Skip(1),Skip(2.3),Skip(3),Skip(True),Skip(True),Skip(0.5),Skip(0),Skip(3),],
            size=[Skip(()),Skip((1280,)),Skip((32, 160)),Skip((320, 8)),Skip((32, 80)),Skip((2, 2, 20, 16)),Skip((320, 2, 3, 3)),Skip((0,)),Skip((4, 0)),Skip((2, 0, 9)),],
        ),
    ),

    'normal_': dict(
        name=['normal_'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'normal_std_tensor': dict(
        name=['normal'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['std'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'normal_mean_tensor': dict(
        name=['normal'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'normal_tensor': dict(
        name=['normal'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['mean'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
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
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),Skip(Dtype.bool),],
                },
            ]
        ),
    ),

    'multinomial': dict(
        name=['multinomial'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float16),Skip(Dtype.float32),Skip(Dtype.float64),],
                },
            ]
        ),
    ),

    'cast_dtype': dict(
        name=['cast_dtype'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.int64),Skip(Dtype.int8),Skip(Dtype.uint8),],
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
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),Skip(Dtype.int16),Skip(Dtype.int32),Skip(Dtype.int64),Skip(Dtype.uint8),Skip(Dtype.int8),Skip(Dtype.bool),],
                },
            ]
        ),
    ),

    'isnan': dict(
        name=['isnan'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [Skip(()),Skip((128,)),Skip((1024, 64)),Skip((384, 128)),Skip((64, 1, 128)),Skip((128, 64, 3, 3)),Skip((2, 32, 130, 130)),Skip((0,)),Skip((4, 0)),Skip((12, 0, 9)),],
                },
            ]
        ),
    ),

    'isnan_input_nan': dict(
        name=['isnan'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "dtype": [Skip(Dtype.float32),Skip(Dtype.float64),Skip(Dtype.float16),],
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
    )
}
