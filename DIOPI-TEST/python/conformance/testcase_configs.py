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
                              [1.0, 1.0, 1.0]]
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
        test_tracking=False,
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
