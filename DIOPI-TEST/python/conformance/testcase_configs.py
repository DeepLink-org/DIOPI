# from functools import partial
from .testcase_parse import Genfunc
from .dtype import Dtype

configs = {
    'batch_norm': dict(
        name=["batch_norm"],
        dtype=[Dtype.float32],
        atol=1e-5,
        call_para=dict(
            args=[
                {
                    "shape": ((2, 5, 3, 5), (3, 4, 3), (2, 3)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": ((5, ), (4, ), (3, )),
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ["running_var"],
                    "value": [[1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0]],
                },
                {
                    "ins": ["weight", "bias"],
                    "shape": ((5, ), (4, ), (3, )),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=["conv2d"],
        atol=1e-5,
        rtol=1e-4,
        dtype=[Dtype.float32, Dtype.float16],
        para=dict(
            stride=[2, ],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 3, 16, 16),),
                },
                {
                    "ins": ["weight"],
                    "shape": ((6, 3, 2, 2),),
                },
                {
                    "ins": ["bias"],
                    "shape": ((6, ), None, (6,)),
                },
            ]
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        dtype=[Dtype.float32],
        atol=1e-5,
        para=dict(
            eps=[1e-6, 2e-6],
        ),
        related_para=dict(
            normalized_shape=[(5, 3, 5), (4, 3), (3,), (3,),
                              [5, 3, 5], [4, 3], [3], [3]],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 5, 3, 5), (3, 4, 3), (2, 3), (3, 3),
                              (2, 5, 3, 5), (3, 4, 3), (2, 3), (3, 3)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "shape": (None, None, (3, ), (3,),
                              None, None, (3, ), (3,)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "shape": (None, (4, 3), None, (3,),
                              None, (4, 3), None, (3,)),
                    "gen_fn": Genfunc.randn,
                },
            ]
        )
    ),

    'relu': dict(
        name=["relu"],
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 3, 14, 14), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu_': dict(
        name=["relu_"],
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 3, 14, 14), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'avg_pool2d': dict(
        name=["avg_pool2d"],
        para=dict(
            kernel_size=[(2, 2)],
            stride=[1, (1, 2)],
            padding=[(1, 1)],
            ceil_mode=[True, False],
            count_include_pad=[True, False],
            divisor_override=[None, 1, 2]
        ),
        call_para=dict(
            args=[
                {
                    "shape": ((3, 16, 22, 15), (2, 16, 10, 9), (3, 4, 10)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        related_para=dict(
            kernel_size=[(2, 2), (1, 2), 1, 3],
            stride=[(2, 1), 2, (3, 4), 2],
            dilation=[3, (2, 1), (2, 4), 2],
            ceil_mode=[False, True, True, True],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 8, 6, 9),
                              (4, 4, 8, 9),
                              (3, 4, 5, 7),
                              (3, 4, 10)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
            ]
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        atol=1e-5,
        rtol=1e-4,
        para=dict(
            output_size=[2, (2, 3), (None, 3), (3, 4)],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 3, 10, 10), (4, 8, 10, 15)),
                    "dtype": [Dtype.float32],
                },
            ]
        ),
    ),

    'adaptive_max_pool2d': dict(
        name=["adaptive_max_pool2d"],
        atol=1e-5,
        rtol=1e-4,
        para=dict(
            output_size=[2, (2, 3), (1, 3), (3, 4)],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 3, 10, 10), (4, 8, 10, 15)),
                    "dtype": [Dtype.float32],
                },
            ]
        ),
    ),

    'binary_cross_entropy': dict(
        name=["binary_cross_entropy_with_logits"],
        atol=1e-3,
        rtol=1e-4,
        dtype=[Dtype.float32],
        para=dict(
            reduction=['mean'],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((4, 4),
                              (5, 8, 20),
                              (4, 4, 16, 20)),
                },
                {
                    "ins": ['target'],
                    "requires_grad": [False],
                    "shape": ((4, 4),
                              (5, 8, 20),
                              (4, 4, 16, 20)),
                },
                {
                    "ins": ['weight'],
                    "requires_grad": [False],
                    "shape": ((4, 4),
                              (5, 8, 20),
                              (4, 4, 16, 20)),
                },
            ],
        ),
    ),

    'pointwise_op': dict(
        name=['abs', 'acos', 'asin', 'atan', 'ceil', 'cos',
              'cosh', 'erf', 'erfc', 'exp', 'expm1', 'floor',
              'log', 'log2', 'log10','neg', 'round', 'sign',
              'sin', 'sinh', 'sqrt', 'tan', 'tanh'],
        interface=['torch'],
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    'pointwise_op_inp': dict(
        name=['abs_', 'acos_', 'asin_', 'atan_', 'ceil_', 'cos_',
              'cosh_', 'exp_', 'erf_', 'erfc_',  'sign_', 'expm1_',
              'floor_', 'log_', 'log2_', 'log10_', 'neg_', 'round_',
              'sin_', 'sinh_', 'sqrt_', 'tan_', 'tanh_'],
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        interface=['tensor'],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "shape": ((5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    'sigmoid': dict(
        name=["sigmoid"],
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'pow_float_number': dict(
        name=['pow'],
        interface=['torch'],
        para=dict(
           exponent=[0.1, 2.0],
        ),
        dtype=[Dtype.float32],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    'pow_float_tensor': dict(
        name=['pow'],
        interface=['torch'],
        dtype=[Dtype.float32, Dtype.float64],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 4, 1, 6), (2, 4, 1, 6),
                              (5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
                {
                    "ins": ['exponent'],
                    "requires_grad": [True],
                    "shape": ((4, 5, 6), (1, 5, 6),
                              (5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    'pow_int_number': dict(
        name=['pow'],
        interface=['torch'],
        para=dict(
           exponent=[2, 3],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [False],
                    "shape": ((3, 3),
                              (4, 4),
                              (5, 5)),
                    "dtype": [Dtype.int32],
                    "gen_fn": dict(fn=Genfunc.randint, high=4),
                }
            ],
        ),
    ),

    'pow_int_tensor': dict(
        name=['pow'],
        interface=['torch'],
        call_para=dict(
            args=[
                {
                    "ins": ['input', 'exponent'],
                    "requires_grad": [False, False],
                    "shape": ((3, 3),
                              (4, 4),
                              (5, 5)),
                    "dtype": [Dtype.int32],
                    "gen_fn": dict(fn=Genfunc.randint, high=4),
                }
            ],
        ),
    ),

    'pointwise_binary': dict(
        name=['add', 'rsub', 'mul', 'div', 'atan2',
              'eq', 'ne', 'le', 'lt', 'gt', 'ge',
              'logical_and', 'logical_or'],
        interface=['torch'],
        dtype=[Dtype.float32],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
                {
                    "ins": ['other'],
                    "shape": ((5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    'pointwise_binary_with_alpha': dict(
        name=['add', 'rsub'],
        para=dict(
            alpha=[-2, 2.0, 4],
        ),
        interface=['torch'],
        dtype=[Dtype.float32],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
                {
                    "ins": ['other'],
                    "shape": ((5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    'pointwise_binary_constant_with_alpha_and_no_contiguous': dict(
        name=['add', 'rsub'],
        para=dict(
            alpha=[-2, 2.0, 4],
            other=[-2, 2.0, 4],
        ),
        no_contiguous=[True],
        interface=['torch'],
        dtype=[Dtype.float32],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    'bmm': dict(
        name=['bmm'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((10, 3, 4), (3, 2, 4), (9, 5, 5)),
                },
                {
                    "ins": ['mat2'],
                    "requires_grad": [True],
                    "shape": ((10, 4, 5), (3, 4, 2), (9, 5, 10)),
                },
            ],
        ),
        outs=['out'],
    ),

    'addcmul': dict(
        name=["addcmul"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            value=[1, 2, 5, 15, 0.5, 3.5],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((1, 3), (10, 3, 5), (4, 1, 1)),
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [True],
                    "shape": ((3, 1), (10, 3, 1), (1, 5)),
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [True],
                    "shape": ((1, 3), (10, 1, 5), (4, 5, 1)),
                },
            ],
        ),
    ),

    'matmul': dict(
        name=['matmul'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2,), (6, 3), (5,), (3, 4),
                              (3, 3, 5), (3, 3, 4, 4), (5, 5, 6, 7),
                              (3, 3, 5), (1, 3, 5)),
                },
                {
                    "ins": ['other'],
                    "requires_grad": [True],
                    "shape": ((2, 2), (3,), (5,), (4, 3),
                              (5, 3), (3, 3, 4, 3), (5, 5, 7, 6),
                              (1, 5, 3), (3, 5, 3)),
                },
            ],
        ),
    ),

    'clamp': dict(
        name=['clamp'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            min=[-2.0, -1.2, 0, 2],
            max=[2.3, 3.2, 5.4, 6],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5),
                              (6, 7, 8),
                              (4, 6, 10, 9)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_max': dict(
        name=["clamp_max"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            max=[2.3, 3.2, 5.4, 6],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5),
                              (6, 7, 8),
                              (4, 6, 10, 9)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_min': dict(
        name=["clamp_min"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            min=[-2.0, -1.2, 0, 2],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5),
                              (6, 7, 8),
                              (4, 6, 10, 9)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_op': dict(
        name=['mean', 'std', 'sum', 'var', 'min'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5, 6), (2, 4, 9), (5, 8, 7),
                              (6, 7, 8, 9), (3, 5, 7, 9), (1, 3, 6, 8),
                              (4, 6, 10, 9, 8), (2, 3, 5, 8, 9)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op': dict(
        name=['mean', 'sum'],
        interface=['torch'],
        para=dict(
            dim=[0, 1, 2, [0, 1], [-1, 0]],
            keepdim=[True, False]
        ),
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5, 6), (2, 4, 9), (5, 8, 7),
                              (6, 7, 8, 9), (3, 5, 7, 9), (1, 3, 6, 8),
                              (4, 6, 10, 9, 8), (2, 3, 5, 8, 9)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op_1': dict(
        name=['std', 'var'],
        interface=['torch'],
        para=dict(
            dim=[0, 1, 2, [0, 1], [-1, 0]],
            keepdim=[True, False],
            unbiased=[True, False],
        ),
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5, 6), (2, 4, 9), (5, 8, 7),
                              (6, 7, 8, 9), (3, 5, 7, 9), (1, 3, 6, 8),
                              (4, 6, 10, 9, 8), (2, 3, 5, 8, 9)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'soft_margin_loss': dict(
        name=["soft_margin_loss"],
        para=dict(
            reduction=['mean'],
        ),
        dtype=[Dtype.float32],
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "value": [[1, 1.5, 2, 2.5, 3]]
                },
                {
                    "ins": ['target'],
                    "value": [[1.0, 1.0, -1.0, -1.0, 1.0]],
                },
            ],
        ),
    ),

    'mse_loss': dict(
        name=["mse_loss"],
        para=dict(
            reduction=['mean'],
        ),
        dtype=[Dtype.float32],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5, 6), (3, 5, 6, 6)),
                },
                {
                    "ins": ['target'],
                    "shape": ((3, 5, 6), (3, 5, 6, 6)),
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            reduction=['mean'],
            #label_smoothing=[0.0, 0.5],
        ),
        dtype=[Dtype.float32],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5, 6), (3, 5, 6, 6)),
                },
                {
                    "ins": ['weight'],
                    "requires_grad": [True],
                    "shape": (None, (5,)),
                },
                {
                    "ins": ['target'],
                    "shape": ((3, 6), (3, 6, 6)),
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=5),
                    "dtype": [Dtype.int64],

                },
            ],
        ),
    ), 

    'select': dict(
        name=["select"],
        interface=['torch'],
        para=dict(
            dim=[-2, -1, 0, 1],
            index=[0, 1, 2],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "requires_grad": [True],
                    "shape": ((16, 4, 4), (64, 4, 8, 8)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'index_select': dict(
        name=["index_select"],
        interface=['torch'],
        related_para=dict(
            dim=[0, 1, 2, 3]
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((5, 3), (16, 8), (16, 4, 4), (4, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['index'],
                    "requires_grad": [False],
                    "shape": ((3,), (5,), (2,), (10,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=3)
                },
            ]
        ),
    ),

    'masked_scatter': dict(
        name=["masked_scatter"],
        interface=['torch'],
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((16,), (12, 13), (12, 13, 14), (12, 13, 14, 16)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mask'],
                    "requires_grad": [False],
                    "shape": ((16,), (12, 13), (12, 13, 14), (12, 13, 14, 1)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['source'],
                    "requires_grad": [True],
                    "shape": ((16,), (12, 13), (12, 13, 14), (12, 13, 14, 16)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn
                },
            ],
            input='input',
        ),
    ),

    'nonzero': dict(
        name=["nonzero"],
        interface=['torch'],
        dtype=[Dtype.float32],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    'linear': dict(
        name=["linear"],
        atol=1e-4,
        saved_args=['weight', 'bias'],
        para=dict(
            bias=[None]
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((97, 20), (103, 217, 20), (1, 20)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
                {
                    "ins": ['weight'],
                    "requires_grad": [True],
                    "shape": ((15, 20), (15, 20), (15, 20)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
            ]
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'soft_max': dict(
        name=["softmax"],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            dim=[1, 0],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 1025), (1025, 1025)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'soft_min': dict(
        name=["softmin"],
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),


    'sigmoid': dict(
        name=["sigmoid"],
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'softsign': dict(
        name=["softsign"],
        atol=1e-4,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'embedding': dict(
        name=["embedding"],
        para=dict(
            padding_idx=[None, 0],
            max_norm=[None, 1.0],
            norm_type=[2.0, 3.0],
            scale_grad_by_freq=[False, True],
            sparse=[False],
        ),
        call_para=dict(
            args=[
                {
                    "requires_grad": [False],
                    "shape": ((), (2, ), (2, 3), (2, 3, 4)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=10),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": ((10, 3), (10, 2), (20, 3), (20, 2)),
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32],
                },
            ],
        ),
    ),

    'clip_grad_norm': dict(
        name=["clip_grad_norm_"],
        interface=["torch.nn.utils"],
        para=dict(
            max_norm=[1.0, 5],
            norm_type=[2.0, 3.0],
            #error_if_nonfinite=[True, False], #1.7 not support
        ),
        call_para=dict(
            args=[
                {
                    "ins": ["parameters"],
                    "shape": ((10, 3), (10, 2), (20, 3), (20, 2)),
                    "gen_fn": Genfunc.rand,
                    "dtype": [Dtype.float32],
                },
            ],
        ),
    ),

   'tril': dict(
        name=["tril"],
        interface=["torch"],
        para=dict(
            diagonal=[0, -1, 1],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((6, 7), (6, 8, 8),
                             (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

   'one_hot': dict(
        name=["one_hot"],
        para=dict(
            num_classes=[-1, 6],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((6, ), (6, 8, ), (64, 7, 28,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=6),
                },
            ],
        ),
    ),

    # 'join': dict(
    #     name=['cat', 'stack'],
    #     interface=['torch'],
    #     atol=1e-4,
    #     rtol=1e-5,
    #     para=dict(
    #         dim=[-1, 0, 1],
    #     ),
    #     call_para=dict(
    #         args=[
    #             {
    #                 "ins": ['tensor'],
    #                 "requires_grad": [True],
    #                 "shape": ((2, 3),
    #                           (3, 4, 5),
    #                           (4, 5, 6, 7)),
    #                 "dtype": [Dtype.float32],
    #                 "gen_fn": Genfunc.randn,
    #                 "genrange": [2, 5]
    #             },
    #         ],
    #         seq_name='tensors',
    #     ),
    # ),

    'split': dict(
        name=["split"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            split_size_or_sections=[1, 2, 3],
            dim=[0, 1, 2]
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "requires_grad": [True],
                    "shape": ((3, 5, 6),
                              (6, 7, 8, 9),
                              (4, 6, 10, 9, 8)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
        requires_backward=[0],
    ),

    'split_with_sizes': dict(
        name=["split_with_sizes"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        related_para=dict(
            split_sizes=[[1, 2], [2, 2, 3], [5, 5]],
            dim=[0, 1, 2]
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5, 6),
                              (6, 7, 8, 9),
                              (4, 6, 10, 9, 8)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
        requires_backward=[0],
    ),

    'sort': dict(
        name=["sort"],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1],
            descending=[True, False],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
        requires_backward=[0],
    ),

    'topk_nonzero': dict(
        name=['topk'],
        interface=['torch'],
        para=dict(
            k=[1, 2, 3],
            dim=[-1, 0, 1, 2],
            largest=[True, False],
            sorted=[True, False],
        ),
        call_para=dict(
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 4, 4),
                              (5, 4, 6)),
                },
            ],
        ),
        requires_backward=[0],
    ),

   'topk_zero': dict(
        name=['topk'],
        interface=['torch'],
        para=dict(
            k=[1],
            dim=[-1, 0],
            largest=[True, False],
            sorted=[True, False],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": [(1,)],
                },
            ],
        ),
        requires_backward=[0],
    ),

    'transpose': dict(
        name=['transpose'],
        interface=['torch'],
        para=dict(
            dim0=[0, 1],
            dim1=[2, 3],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 4, 4, 3),
                              (5, 4, 6, 3, 6)),
                },
            ],
        ),
    ),

    'where': dict(
        name=['where'],
        interface=['torch'],
        call_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "requires_grad": [False],
                    "shape": [(3, ), (3, 5), (4, 5, 6)],
                    "dtype": [Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['input', 'other'],
                    "requires_grad": [True, True],
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "shape": [(3, ), (3, 5), (4, 5, 6)],
                    "gen_fn": Genfunc.randn
                },
            ],
        ),
    ),

    'where_1': dict(
        name=['where'],
        interface=['torch'],
        call_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "requires_grad": [False],
                    "shape": [(3, ), (3, 5), (4, 5, 6)],
                    "dtype": [Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
            ],
        ),
    ),

    'where_2': dict(
        name=['where'],
        interface=['torch'],
        call_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "requires_grad": [False],
                    "shape": [(1, ), (3, ), (3, ), (1, 445), (3, 5), (4, ),
                              (3, 4, 5), (3, )],
                    "dtype": [Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "dtype": [Dtype.float32, Dtype.float64],
                    "shape": [(1, ), (1, ), (3, ), (1, 445), (3, 5), (1, ), (4, 5),
                              (5, 4, 3)],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['other'],
                    "requires_grad": [True],
                    "dtype": [Dtype.float32, Dtype.float64],
                    "shape": [(1, ), (1, ), (1, ), (1, ), (1, ), (4, ), (5, ), (4, 3)],
                    "gen_fn": Genfunc.randn
                },
            ],
        ),
    ),

    'tanh': dict(
        name=["tanh"],
        atol=1e-4,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28),
                              (4, 1, 16, 16, 16)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'dropout': dict(
        name=["dropout"],
        atol=1e-4,
        para=dict(
            p=[0.1, 0.5, 1.0],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (16, 14, 14), (64, 7, 28, 28),
                              (4, 1, 16, 16, 16)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu6': dict(
        name=["relu6"],
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 3, 14, 14), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'leaky_relu': dict(
        name=["leaky_relu"],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            negative_slope=[0.01, 0.1, 1, 10]
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 3, 14, 14), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'leaky_relu_': dict(
        name=["leaky_relu_"],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            negative_slope=[0.01, 0.1, 1, 10]
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 3, 14, 14), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),
}
