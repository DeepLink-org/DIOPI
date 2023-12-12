import numpy as np
from skip import Skip

device_configs = {
        'batch_norm': dict(
            name=["batch_norm"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'batch_norm_nan': dict(
            name=["batch_norm"],
        ),

        'batch_norm_no_contiguous': dict(
            name=["batch_norm"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'batch_norm_stats': dict(
            name=["batch_norm_stats"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'batch_norm_gather_stats_with_counts': dict(
            name=["batch_norm_gather_stats_with_counts"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'batch_norm_backward_reduce': dict(
            name=["batch_norm_backward_reduce"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'batch_norm_backward_elemt': dict(
            name=["batch_norm_backward_elemt"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'batch_norm_elemt': dict(
            name=["batch_norm_elemt"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'baddbmm': dict(
            name=["baddbmm"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'baddbmm_without_inplace': dict(
            name=["batch_norm_stats"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'conv_2d': dict(
            name=["conv_2d"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'conv_2d_no_contiguous': dict(
            name=["conv_2d"],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'relu': dict(
            name=['relu'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
                    },
                ]
            ),
        ),

        'relu_no_contiguous': dict(
            name=['relu'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'hardtanh_int': dict(
            name=['hardtanh'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.int8),],
                    },
                ]
            ),
        ),

        'hardtanh_uint': dict(
            name=['hardtanh'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.uint8),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.int8),],
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
                        "dtype": [Skip(np.uint8),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.float64),Skip(np.float64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),Skip(np.float64),Skip(np.float64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
                    },
                    {
                        "ins": ['pos_weight'],
                        "dtype": [Skip(np.int32),Skip(np.float64),Skip(np.int64),
                                  Skip(np.int16),Skip(np.int8),Skip(np.uint8),],
                    },
                ]
            ),
        ),

        'pointwise_op': dict(
            name=['abs', 'cos', 'erf', 'erfinv', 'exp', 'floor',
                  'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'ceil', 'atan'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float16),Skip(np.float64),],
                        "shape": [Skip((0,)),Skip((16, 0)),],
                    },
                ]
            ),
        ),

        'pointwise_op_int_without_inplace': dict(
            name=['abs', 'cos', 'erf', 'exp',
                  'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.int8),],
                    },
                ]
            ),
        ),

        'pointwise_op_uint8': dict(
            name=['abs', 'cos', 'erf', 'exp',
                  'neg', 'sin', 'asin', 'sqrt', 'logical_not', 'rsqrt', 'atan'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.uint8),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'pointwise_op_bool': dict(
            name=['abs', 'cos', 'erf', 'exp',
                  'sin', 'asin', 'sqrt', 'rsqrt', 'atan', 'logical_not'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.bool_),],
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
                        "dtype": [Skip(np.float16),Skip(np.float64),],
                        "shape": [Skip((0,)),Skip((0, 16)),Skip((8, 0, 4)),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16), Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'pointwise_op_zero': dict(
            name=['abs', 'exp', 'floor', 'neg', 'sqrt', 'logical_not', 'rsqrt', 'ceil'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float16),Skip(np.float64),],
                    },
                ]
            ),
        ),

        'pointwise_op_without_inplace_zero': dict(
            name=['abs', 'sign', 'exp', 'sqrt', 'logical_not', 'rsqrt'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float16),Skip(np.float64),Skip(np.int16),
                                  Skip(np.int32),Skip(np.int64),Skip(np.int8),
                                  Skip(np.uint8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),Skip(np.int8),Skip(np.int16),
                                  Skip(np.int32),Skip(np.int64),Skip(np.uint8),],
                    },
                ]
            ),
        ),

        'sigmoid': dict(
            name=['sigmoid'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float16),Skip(np.float64),],
                        "shape": [Skip((0,)),Skip((0, 16)),Skip((8, 0, 17)),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.uint8),],
                    },
                    {
                        "ins": ['exponent'],
                        "dtype": [Skip(np.float64),Skip(np.uint8),],
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
                        "dtype": [Skip(np.uint8),],
                    },
                    {
                        "ins": ['exponent'],
                        "dtype": [Skip(np.uint8),],
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
                        "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                        "shape": [Skip((0,)),Skip((8, 16, 1)),],
                    },
                    {
                        "ins": ['exponent'],
                        "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                        "shape": [Skip((16, 0)),Skip((16, 0)),],
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
                        "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                        "shape": [Skip((0,)),Skip((8, 16, 1)),],
                    },
                    {
                        "ins": ['exponent'],
                        "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                        "shape": [Skip((16, 0)),Skip((16, 0)),],
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
                        "dtype": [Skip(np.int64),Skip(np.int32),Skip(np.int16),
                                  Skip(np.bool_),Skip(np.bool_),Skip(np.bool_),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['exponent'],
                        "dtype": [Skip(np.float64),Skip(np.int32),Skip(np.int8),
                                  Skip(np.uint8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int32),Skip(np.float64),
                                  Skip(np.int16),Skip(np.int64),],
                    },
                    {
                        "ins": ['exponent'],
                        "dtype": [Skip(np.int32),Skip(np.uint8),Skip(np.bool_),
                                  Skip(np.int64),Skip(np.float64),Skip(np.bool_),
                                  Skip(np.uint8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),Skip(np.uint8),],
                        "shape": [Skip((8,)),Skip((70, 1, 2)),Skip((0, 4)),
                                  Skip((9, 0, 6)),],
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
                        "dtype": [Skip(np.float64),Skip(np.uint8),],
                    },
                ]
            ),
        ),

        'pointwise_binary': dict(
            name=['add', 'sub', 'mul', 'eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "shape": [Skip((128, 64, 3, 3)),Skip((2, 64, 16, 128)),Skip((2, 32, 130, 130)),
                                  Skip((0,)),],
                        "dtype": [Skip(np.float64),
                                  ],
                    },
                    {
                        "ins": ['other'],
                        "shape": [Skip((1, )),Skip((64, 1, 128)),Skip((2, 32, 1, 1)),
                                  Skip((0,)),],
                        "dtype": [Skip(np.float64),Skip(np.int64),Skip(np.int32),
                                  Skip(np.int16),Skip(np.int8),Skip(np.uint8),],
                    },
                ]
            ),
        ),

        'pointwise_binary_broadcast': dict(
            name=['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float32),],
                        "shape": [Skip((0,)),Skip((8, 16, 1)),Skip((32, 0, 16)),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float32),],
                        "shape": [Skip((16, 0)),Skip((16, 0,)),Skip((0, 16)),],
                    },
                ]
            ),
        ),

        'pointwise_binary_broadcast_inplace': dict(
            name=['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float32),],
                        "shape": [Skip((16, 0,)),Skip((8, 16, 0)),Skip((32, 0, 16)),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float32),],
                        "shape": [Skip((0,)),Skip((16, 1,)),Skip((0, 16)),],
                    },
                ]
            ),
        ),

        'pointwise_binary_diff_dtype': dict(
            name=['mul', 'eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),Skip(np.int64),Skip(np.int32),
                                  Skip(np.int16),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.int32),Skip(np.uint8),Skip(np.bool_),
                                  Skip(np.int64),Skip(np.float64),Skip(np.int16),
                                  Skip(np.int8),],
                    },
                ]
            ),
        ),

        'pointwise_binary_diff_dtype_inplace': dict(
            name=['eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),Skip(np.int32),Skip(np.float64),
                                  Skip(np.float64),Skip(np.int8),Skip(np.int8),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.int32),Skip(np.uint8),Skip(np.bool_),
                                  Skip(np.int64),Skip(np.int16),Skip(np.bool_),
                                  Skip(np.uint8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int32),Skip(np.int32),
                                  Skip(np.int16),Skip(np.int8),Skip(np.uint8),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.int32),Skip(np.uint8),Skip(np.int32),
                                  Skip(np.int64),Skip(np.float64),Skip(np.uint8),
                                  Skip(np.int8),],
                    },
                ]
            ),
        ),

        'pointwise_binary_dtype_bool': dict(
            name=['add', 'mul', 'eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.bool_),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.bool_),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.int8),Skip(np.int8),Skip(np.bool_),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.uint8),Skip(np.bool_),Skip(np.int16),
                                  Skip(np.int64),Skip(np.int8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.int8),Skip(np.int8),Skip(np.bool_),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.int8),Skip(np.bool_),Skip(np.int16),
                                  Skip(np.int64),Skip(np.int8),Skip(np.int32),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.int8),],
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
                        "dtype": [Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
                        "shape": [Skip((0,)),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),],
                        "shape": [Skip((0,)),],
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
                        "shape": [Skip((0,)),Skip((8, 16, 1)),Skip((32, 0, 16)),],
                    },
                    {
                        "ins": ['other'],
                        "shape": [Skip((16, 0)),Skip((16, 0)),Skip((0, 16)),],
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
                        "dtype": [Skip(np.float64),Skip(np.float32),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),Skip(np.float32),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.int8),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'sub_scalar': dict(
            name=['sub'],
        ),

        'pointwise_binary_scalar': dict(
            name=['add', 'mul', 'div', 'eq', 'ne', 'le', 'lt', 'gt', 'ge'],
        ),

        'div_zero': dict(
            name=['div'],
        ),

        'pointwise_binary_scalar_div_zero': dict(
            name=['div'],
        ),

        'pointwise_binary_test_equal_and_logic_specific': dict(
            name=['eq', 'ne', 'le', 'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        ),

        'sub_constant_with_alpha_and_no_contiguous': dict(
            name=['sub'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "shape": [Skip((384, 128)),Skip((2, 64, 128)),Skip((128, 64, 3, 3)),
                                  Skip((128, 32, 2, 2)),Skip((2, 32, 130, 130))],
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
                        "shape": [Skip((384, 128)),Skip((2, 64, 128)),Skip((128, 64, 3, 3)),
                                  Skip((128, 32, 2, 2)),Skip((2, 32, 130, 130))],
                    },
                ]
            ),
        ),

        'pointwise_binary_with_alpha': dict(
            name=['add', 'sub'],
        ),

        'pointwise_binary_with_alpha_bool': dict(
            name=['add'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                    },
                ]
            ),
        ),

        'bmm': dict(
            name=['bmm'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['mat2'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['mat1'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['mat2'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),],
                    },
                    {
                        "ins": ['tensor1'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),],
                    },
                    {
                        "ins": ['tensor2'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['tensor1'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['tensor2'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['tensor1'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['tensor2'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['tensor1'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['tensor2'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['tensor1'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['tensor2'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float16),Skip(np.float64),],
                        "shape": [Skip((128, 49, 128)),Skip((5,)),Skip((2, 1, 3136, 3136)),
                                  Skip((2, 16, 8, 64)),Skip((2, 31, 6, 40, 512)),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float16),Skip(np.float64),],
                        "shape": [Skip((128, 384)),Skip((5,)),Skip((2, 3, 3136, 64)),
                                  Skip((2, 1, 64, 8)),Skip((512, 1)),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'clamp_max_tensor': dict(
            name=['clamp_max'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'clamp_min_tensor': dict(
            name=['clamp_min'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'fill_not_float': dict(
            name=['fill_'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'reduce_op': dict(
            name=['mean', 'sum'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),],
                    },
                ]
            ),
        ),

        'reduce_partial_op': dict(
            name=['mean', 'sum'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                        "shape": [Skip(()),Skip((2, 64, 3, 3, 3)),Skip((4, 133, 128, 128)),
                                  Skip((0,)),Skip((0, 2)),Skip((16, 0, 9)),],
                    },
                ]
            ),
        ),

        'reduce_partial_op_1': dict(
            name=['std'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'max_min_equal_input': dict(
            name=['min', 'max'],
        ),

        'max_min_all': dict(
            name=['min', 'max'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
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
                        "dtype": [Skip(np.bool_),Skip(np.float64),Skip(np.int16),
                                  Skip(np.int32),Skip(np.int64),Skip(np.uint8),
                                  Skip(np.int8),],
                    },
                ]
            ),
        ),

        'reduce_partial_op_4': dict(
            name=['sum'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.bool_),Skip(np.float64),Skip(np.int16),
                                  Skip(np.int32),Skip(np.int64),Skip(np.uint8),
                                  Skip(np.int8),],
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
                        "dtype": [Skip(np.bool_),Skip(np.float64),Skip(np.int16),
                                  Skip(np.int32),Skip(np.int64),Skip(np.uint8),
                                  Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['target'],
                        "dtype": [Skip(np.int64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['target'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['target'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['target'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),Skip(np.int32),Skip(np.int64),],
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
                        "dtype": [Skip(np.int32),Skip(np.int16),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int32),Skip(np.int64),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int32),Skip(np.int64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['mask'],
                        "dtype": [Skip(np.bool_),],
                    },
                    {
                        "ins": ['source'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'nonzero': dict(
            name=['nonzero'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'nonzero_float': dict(
            name=['nonzero'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'nonzero_int': dict(
            name=['nonzero'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.int8),],
                    },
                ]
            ),
        ),

        'nonzero_uint': dict(
            name=['nonzero'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.uint8),],
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
                        "dtype": [Skip(np.float16),Skip(np.float64),],
                        "shape": [Skip((16, 8)),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float16),Skip(np.float64),],
                        "shape": [Skip((0, 8)),],
                    },
                    {
                        "ins": ['bias'],
                        "dtype": [Skip(np.float16),Skip(np.float64),],
                        "shape": [Skip((16, 0)),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float16),Skip(np.float64),],
                        "shape": [Skip(()),Skip((0,)),Skip((0, 12)),
                                  Skip((16, 0, 7)),],
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
                        "dtype": [Skip(np.int64),Skip(np.int64),Skip(np.int32),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
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
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'clip_grad_norm': dict(
            name=['clip_grad_norm_'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['tensors'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
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

        'join': dict(
            name=['cat', 'stack'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['tensors'],
                        "dtype": [Skip(np.float64),],
                        "shape": [Skip((0, 50, 76)),Skip((0,)),Skip((16, 0)),]
                    },
                ]
            ),
        ),

        'join_int': dict(
            name=['cat', 'stack'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['tensors'],
                        "dtype": [Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),Skip(np.int32),],
                        "shape": [Skip((0, 50, 76)),Skip((0,)),Skip((16, 0)),]
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
                        "shape": [Skip(()),Skip((11400, )),Skip((12, 8)),
                                  Skip((8, 12, 9)),Skip((4, 4, 16, 20)),Skip((4, 4, 16, 2, 20)),
                                  Skip((24180,)),Skip((0,)),Skip((12, 0)),Skip((4, 0, 5)),],
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
                        "shape": [Skip((11400, )),Skip((4, 4, 16, 20)),Skip((4, 4, 16, 2, 20)),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),
                                  Skip(np.int32),],
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
                    {
                        "ins": ['input', 'other'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
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
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),],
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
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['targets'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['scores'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['boxes'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['idx1'],
                        "dtype": [Skip(np.int64),Skip(np.int32),Skip(np.int64),],
                    },
                    {
                        "ins": ['idx2'],
                        "dtype": [Skip(np.int32),],
                    },
                    {
                        "ins": ['idx3'],
                        "dtype": [Skip(np.bool_),Skip(np.uint8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['idx1'],
                        "dtype": [Skip(np.int64),],
                    },
                    {
                        "ins": ['idx2'],
                        "dtype": [Skip(np.int64),],
                    },
                    {
                        "ins": ['idx3'],
                        "dtype": [Skip(np.bool_),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                    },
                    {
                        "ins": ['idx1'],
                        "dtype": [Skip(np.int64),],
                    },
                    {
                        "ins": ['idx2'],
                        "dtype": [Skip(np.bool_),],
                    },
                    {
                        "ins": ['idx3'],
                        "dtype": [Skip(np.int32),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['buf'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'sgd_without_buf': dict(
            name=['sgd'],
        ),

        'masked_fill_scalar': dict(
            name=['masked_fill'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['mask'],
                        "dtype": [Skip(np.bool_),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                    },
                    {
                        "ins": ['mask'],
                        "dtype": [Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'masked_fill_scalar_without_inplace': dict(
            name=['masked_fill'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['mask'],
                        "dtype": [Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'masked_fill_tensor': dict(
            name=['masked_fill'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['mask'],
                        "dtype": [Skip(np.bool_),],
                    },
                    {
                        "ins": ['value'],
                        "dtype": [Skip(np.int32),Skip(np.bool_),Skip(np.uint8),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.float64),],
                    },
                ]
            ),
        ),

        'masked_fill_tensor_without_inplace': dict(
            name=['masked_fill'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                    },
                    {
                        "ins": ['mask'],
                        "dtype": [Skip(np.bool_),],
                    },
                    {
                        "ins": ['value'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'adam': dict(
            name=['adam', 'adamw'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['param', 'param_grad'],
                        "shape": [Skip(np.float64),],
                    },
                    {
                        "ins": ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'],
                        "shape": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['bias'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.bool_),Skip(np.int64),Skip(np.int32),
                                  Skip(np.int16),Skip(np.int8),Skip(np.uint8),],
                    },
                ]
            ),
        ),

        'cumsum': dict(
            name=['cumsum'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float16),Skip(np.float64),Skip(np.int16),
                                  Skip(np.int32),Skip(np.int64),Skip(np.uint8),
                                  Skip(np.int8),Skip(np.bool_),],
                        "shape": [Skip(()),Skip((0,)),Skip((5, 0)),
                                  Skip((4, 0, 12)),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['x2'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['x2'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.bool_),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int32),Skip(np.int16),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
                        "shape": [Skip(()),Skip((1024, 80)),Skip((2, 256, 256)),
                                  Skip((12, 0)),Skip((2, 0, 9)),Skip((0, 9, 8, 7)),],
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
                        "shape": [Skip((1024, 80)),Skip((2, 256, 256)),Skip((2, 1, 64, 64)),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['square_avg', 'acc_delta'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['square_avg', 'grad_avg', 'momentum_buffer'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.float64),Skip(np.float64),],
                    },
                    {
                        "ins": ['target'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['bias'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['mask'],
                        "dtype": [Skip(np.bool_),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                    },
                    {
                        "ins": ['mask'],
                        "dtype": [Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'imum': dict(
            name=['maximum', 'minimum'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input', 'other'],
                        "dtype": [Skip(np.float64),Skip(np.bool_),Skip(np.int64),
                                  Skip(np.int32),Skip(np.int16),Skip(np.int8),
                                  Skip(np.uint8),],
                    },
                ]
            ),
        ),

        'imum_input_nan': dict(
            name=['maximum', 'minimum'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'imum_other_nan': dict(
            name=['maximum', 'minimum'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'imum_broadcast': dict(
            name=['maximum', 'minimum'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'imum_ones': dict(
            name=['maximum', 'minimum'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.int32),],
                    },
                ]
            ),
        ),

        'mm': dict(
            name=['mm'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['mat2'],
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'mm_diff_dtype': dict(
            name=['mm'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['mat2'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['value'],
                        "dtype": [Skip(np.uint8),Skip(np.int16),Skip(np.float64),
                                  Skip(np.bool_),Skip(np.int64),Skip(np.int32),
                                  Skip(np.int8),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['value'],
                        "dtype": [Skip(np.int32),Skip(np.bool_),Skip(np.float64),
                                  Skip(np.int16),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.int64),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.bool_),Skip(np.float64),Skip(np.int64),
                                  Skip(np.int32),Skip(np.int16),Skip(np.int8),
                                  Skip(np.uint8),],
                    },
                ]
            ),
        ),

        'linspace': dict(
            name=['linspace'],
        ),

        'permute': dict(
            name=['permute'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),
                                  Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),Skip(np.bool_),Skip(np.int64),
                                  Skip(np.int32),Skip(np.int16),Skip(np.int8),
                                  Skip(np.uint8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['weight', 'bias'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.int64),Skip(np.float64),Skip(np.int16),
                                  Skip(np.int32),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
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
                        "dtype": [Skip(np.int64),Skip(np.float64),Skip(np.int16),
                                  Skip(np.int32),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['targets'],
                        "dtype": [Skip(np.int64),Skip(np.int64),],
                    },
                    {
                        "ins": ['input_lengths'],
                        "dtype": [Skip(np.int64),Skip(np.int64),],
                    },
                    {
                        "ins": ['target_lengths'],
                        "dtype": [Skip(np.int64),Skip(np.int64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['targets'],
                        "dtype": [Skip(np.int64),Skip(np.int64),],
                    },
                    {
                        "ins": ['input_lengths'],
                        "dtype": [Skip(np.int64),Skip(np.int64),],
                    },
                    {
                        "ins": ['target_lengths'],
                        "dtype": [Skip(np.int64),Skip(np.int64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.int32),Skip(np.bool_),Skip(np.uint8),
                                  Skip(np.int32),Skip(np.float64),Skip(np.int8),
                                  Skip(np.uint8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.int32),Skip(np.bool_),Skip(np.uint8),
                                  Skip(np.int32),Skip(np.float64),Skip(np.int8),
                                  Skip(np.uint8),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),],
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
                    {
                        "ins": ['index'],
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
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
                    },
                    {
                        "ins": ['src'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
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
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
                    },
                    {
                        "ins": ['src'],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
                    },
                    {
                        "ins": ['src'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.bool_),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.float32),],
                    },
                    {
                        "ins": ['index'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),
                                  Skip(np.int32),Skip(np.int64),Skip(np.uint8),
                                  Skip(np.int8),Skip(np.bool_),],
                        "shape": [Skip((4, 5, 0)),],
                    },
                    {
                        "ins": ['indices1', 'indices2', 'indices3'],
                        "shape": [Skip((4, 5, 0)),],
                    },
                    {
                        "ins": ['values'],
                        "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),
                                  Skip(np.int32),Skip(np.int64),Skip(np.uint8),
                                  Skip(np.int8),Skip(np.bool_),],
                        "shape": [Skip((4, 5, 0)),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['indices1', 'indices2'],
                        "shape": [Skip((2, 6, 10)),],
                    },
                    {
                        "ins": ['values'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['indices1'],
                        "shape": [Skip((2, 10)), Skip((2, 10)),],
                    },
                    {
                        "ins": ['values'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.int64),],
                    },
                    {
                        "ins": ['indices1'],
                    },
                    {
                        "ins": ['values'],
                        "dtype": [Skip(np.int64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['indices1'],
                    },
                    {
                        "ins": ['value'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),
                                  Skip(np.int32),Skip(np.int64),Skip(np.uint8),
                                  Skip(np.int8),Skip(np.bool_),],
                    },
                    {
                        "ins": ['indices1'],
                        "dtype": [Skip(np.bool_),],
                    },
                    {
                        "ins": ['indices2'],
                        "dtype": [Skip(np.bool_),],
                    },
                    {
                        "ins": ['indices3'],
                        "dtype": [Skip(np.bool_),],
                    },
                    {
                        "ins": ['value'],
                        "dtype": [Skip(np.float32),Skip(np.float64),Skip(np.float16),
                                  Skip(np.int32),Skip(np.int64),Skip(np.uint8),
                                  Skip(np.int8),Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'arange': dict(
            name=['arange'],
        ),

        'arange_default': dict(
            name=['arange'],
        ),

        'randperm': dict(
            name=['randperm'],
        ),

        'uniform': dict(
            name=['uniform'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int64),Skip(np.int32),
                                 Skip(np.int16),Skip(np.int8),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.int64),Skip(np.int32),Skip(np.int16),
                                  Skip(np.int8),Skip(np.uint8),Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['bias'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['bias'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.bool_),Skip(np.int64),
                                  Skip(np.int32),Skip(np.int16),Skip(np.int8),
                                  Skip(np.uint8),],
                        "shape": [Skip((8,)),Skip((192, 147)),Skip((2, 1, 38, 45)),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),Skip(np.bool_),Skip(np.int64),
                                  Skip(np.int32),Skip(np.int16),Skip(np.int8),
                                  Skip(np.uint8),],
                        "shape": [Skip(()),Skip((146, 1)),Skip((45, 38, 1, 2)),],
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
                        "dtype": [Skip(np.float64),Skip(np.bool_),Skip(np.int64),
                                  Skip(np.int32),Skip(np.int16),Skip(np.int8),
                                  Skip(np.uint8),],
                        "shape": [Skip((192, 147)),Skip((2, 1, 38, 45)),Skip((100, 100)),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.uint8),Skip(np.uint8),Skip(np.uint8),],
                        "shape": [Skip((146, 1)),Skip((45, 38, 1, 2)),Skip((1, 100)),],
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
                        "dtype": [Skip(np.float64),],
                        "shape": [Skip((8,)),Skip((12, 2)), Skip((192, 147, 2)),
                                  Skip((6, 5, 384)),Skip((2, 12, 38, 45, 3)),],
                    },
                    {
                        "ins": ['other'],
                        "dtype": [Skip(np.float64),],
                        "shape": [Skip(()),Skip((12, 0)), Skip((1, 147)),
                                  Skip((6, 1, 384)),Skip((2, 1, 38, 45)),]
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['A'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'normal': dict(
            name=['normal'],
        ),

        'normal_': dict(
            name=['normal_'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.float64),],
                    },
                    {
                        "ins": ['std'],
                        "dtype": [Skip(np.float64),Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.bool_),],
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
                        "dtype": [Skip(np.float16),Skip(np.float32),Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int64),Skip(np.int32),
                                  Skip(np.int16),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.bool_),Skip(np.uint8),Skip(np.int8),
                                  Skip(np.int8),],
                    },
                    {
                         "ins": ['out'],
                         "dtype": [Skip(np.int32),Skip(np.uint8), Skip(np.bool_),
                                   Skip(np.float64), Skip(np.int16),Skip(np.int8),
                                   Skip(np.int8), Skip(np.uint8), Skip(np.bool_)],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['angle'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['end'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['end'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['weight'],
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),],
                        "shape": [Skip((1024, 64)),Skip((384, 128)),Skip((64, 1, 128)),
                                  Skip((128, 64, 3, 3)),Skip((2, 32, 130, 130)),Skip((8, 9)),
                                  Skip((6, 7)),Skip((6, 6)),Skip((9, 9)),
                                  Skip((6, 8, 8)),Skip((64, 7, 28, 28)),Skip((2, 0)),
                                  Skip((12, 0)),Skip((2, 0, 9)),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
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
                        "dtype": [Skip(np.float64),Skip(np.int16),Skip(np.int32),
                                  Skip(np.int64),Skip(np.int8),Skip(np.uint8),
                                  Skip(np.bool_),],
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
                        "dtype": [Skip(np.float64),],
                    },
                ]
            ),
        ),

        'sgn': dict(
            name=['sgn'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.complex64),Skip(np.complex128),Skip(np.float64),
                                  Skip(np.int16),Skip(np.int32),Skip(np.int64),
                                  Skip(np.uint8),Skip(np.int8),Skip(np.bool_),],
                    },
                ]
            ),
        ),

        'sgn_zero': dict(
            name=['sgn'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.complex64),Skip(np.complex128),]
                    },
                ]
            ),
        ),

        'rotary_emb': dict(
            name=['rotary_emb'],
            tensor_para=dict(
                args=[
                    {
                        "ins": ['input'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['cos'],
                        "dtype": [Skip(np.float64),],
                    },
                    {
                        "ins": ['sin'],
                        "dtype": [Skip(np.float64),],
                    },
                ],
            ),
        ),

        'rms_norm': dict(
            name=['rms_norm'],
        ),

        'apply_penalty': dict(
            name=['apply_penalty'],
        ),

        'destindex_copy_kv': dict(
            name=['destindex_copy_kv'],
        ),

        'context_attention': dict(
            name=['context_attention'],
        ),

        'token_attention': dict(
            name=['token_attention'],
        ),

        'token_softmax_reducev': dict(
            name=['token_softmax_reducev'],
        ),
}
