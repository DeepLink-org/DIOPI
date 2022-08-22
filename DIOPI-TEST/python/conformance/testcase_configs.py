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
                    "shape": ((2, 8, 32, 56, 56), (2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": ((8, ), (64, ), (96, ), (16, )),
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ["running_var"],
                    "shape": ((8, ), (64, ), (96, ), (16, )),
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ["weight", "bias"],
                    "shape": ((8, ), (64, ), (96, ), (16, )),
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
        related_para=dict(
            stride=[2, 1, 1],
            padding=[0, 12, 0],
            dilation=[1, 12, 1],
            groups=[1, 2048, 1],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 256, 200, 304), (2, 2048, 64, 64), (2, 2048, 1, 1)),
                },
                {
                    "ins": ["weight"],
                    "shape": ((12, 256, 1, 1), (2048, 1, 3, 3), (512, 2048, 1, 1)),
                },
                {
                    "ins": ["bias"],
                    "shape": ((12, ), None, None),
                },
            ]
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        dtype=[Dtype.float32],
        atol=1e-5,
        para=dict(
            eps=[1e-5, 1e-12],
        ),
        related_para=dict(
            normalized_shape=[(5, 3, 5), (128, ), (64, )],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 5, 3, 5), (2, 3136, 128), (2, 64)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "shape": (None, (128, ), (64, )),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "shape": (None, (128, ), (64, )),
                    "gen_fn": Genfunc.randn,
                },
            ]
        )
    ),

    'relu': dict(
        name=["relu"],
        is_inplace=True,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 4096), (64, 28, 28),
                              (32, 64, 112, 112), (64, 3, 7, 28, 28)),
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
            padding=[(1, 1), 0],
            ceil_mode=[True, False],
            count_include_pad=[True, False],
            divisor_override=[None, 1, 2],
        ),
        call_para=dict(
            args=[
                {
                    "shape": ((2, 1024, 14, 14), (256, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        related_para=dict(
            kernel_size=[3, (2, 1), (2, 2), 3],
            stride=[2, (2, 1), (2, 1), 2],
            padding=[1, 0, (0, 1), 0],
            dilation=[1, 1, 1, 2],
            ceil_mode=[False, True, False, True],
            return_indices=[False, False, False, True],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 64, 352, 528),
                              (2, 256, 12, 40),
                              (2, 512, 4, 26),
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
            output_size=[(1, 1), 2, (None, 3), (3, 4)],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 2048, 8, 6), (2, 144, 65, 65)),
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
                    "shape": ((2, 144, 33, 33), (2, 16, 130, 130)),
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
            reduction=['mean', 'none'],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((72,), (2, 11856),
                              (2, 741, 80),
                              (4, 4, 16, 20)),
                },
                {
                    "ins": ['target'],
                    "requires_grad": [False],
                    "shape": ((72,), (2, 11856),
                              (2, 741, 80),
                              (4, 4, 16, 20)),
                },
                {
                    "ins": ['weight'],
                    "requires_grad": [False],
                    "shape": ((72,), (2, 11856),
                              (2, 741, 80),
                              None),
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
        is_inplace=True,
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40)),
                },
            ],
        ),
    ),

    'sigmoid': dict(
        name=["sigmoid"],
        interface=['torch'],
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((182400,), (20267, 80), (8, 200, 304),
                             (32, 16, 1, 1), (16, 32, 130, 130)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
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
                    "shape": ((1, ), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38)),
                },
                {
                    "ins": ['exponent'],
                    "requires_grad": [True],
                    "shape": ((1, ), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38)),
                },
            ],
        ),
    ),

    'pow_float_number': dict(
        name=['pow'],
        interface=['torch'],
        para=dict(
           exponent=[2, 3, 4, 0.2],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [False],
                    "shape": ((1, ), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": ((125, 1), (70, 1, 2),
                              (4, 256, 16, 16)),
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
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                },
                {
                    "ins": ['other'],
                    "shape": ((1024, ), (384, 128),
                              (1, ), (2, 32, 1, 1)),
                },
            ],
        ),
    ),

    'pointwise_binary_scalar': dict(
        name=['add', 'rsub', 'mul', 'div', 'eq', 
              'ne', 'le',  'lt', 'gt', 'ge'],
        interface=['torch'],
        dtype=[Dtype.float32],
        para=dict(
            other=[-1, 0.028, 2],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
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
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
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
                    "shape": ((16, 726, 32), (16, 100, 100), (9, 5, 5)),
                },
                {
                    "ins": ['mat2'],
                    "requires_grad": [True],
                    "shape": ((16, 32, 726), (16, 100, 32), (9, 5, 10)),
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
            value=[0.001, -0.01, 2],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((128, ), (576, 192), (64, 3, 3, 3), (10, 3, 5), (4, 1, 1)),
                },
                {
                    "ins": ["tensor1"],
                    "requires_grad": [True],
                    "shape": ((128, ), (576, 192), (64, 3, 3, 3), (10, 3, 1), (1, 5)),
                },
                {
                    "ins": ["tensor2"],
                    "requires_grad": [True],
                    "shape": ((128, ), (576, 192), (64, 3, 3, 3), (10, 1, 5), (4, 5, 1)),
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
                    "shape": ((128, 49, 128), (5,), (128, 4, 49, 32),
                              (2, 1, 3136, 3136), (2, 784, 64), (2, 31, 6, 40, 512)),
                },
                {
                    "ins": ['other'],
                    "requires_grad": [True],
                    "shape": ((128, 384), (5,), (128, 4, 32, 49),
                              (2, 1, 3136, 64), (2, 64, 784), (512, 1)),
                },
            ],
        ),
    ),

    'clamp': dict(
        name=['clamp'],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        related_para=dict(
            min=[None, -4.13, 1, 1e-12],
            max=[4.13, 26, None, 1199],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((182, ), (384, 128),
                              (1, 242991, 2),
                              (2, 4, 100, 152)),
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
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op': dict(
        name=['mean', 'sum'],
        interface=['torch'],
        related_para=dict(
            dim=[0, 1, [0, 1], 2, [-1, 0], 3],
            keepdim=[True, False, True, False, True, False],
        ),
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op_1': dict(
        name=['std', 'var'],
        interface=['torch'],
        related_para=dict(
            dim=[0, 1, [0, 1], 2, [-1, 0], 3],
            keepdim=[True, False, True, False, True, False],
            unbiased=[True, False, True, False, True, False],
        ),
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
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
            reduction=['mean', 'none'],
        ),
        dtype=[Dtype.float32],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 11856, 2), (16, 2, 2964, 2)),
                },
                {
                    "ins": ['target'],
                    "shape": ((2, 11856, 2), (16, 2, 2964, 2)),
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            reduction=['mean', 'none'],
            #label_smoothing=[0.0, 0.5],
        ),
        dtype=[Dtype.float32],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((1024, 81), (3, 5, 6, 6)),
                },
                {
                    "ins": ['weight'],
                    "requires_grad": [True],
                    "shape": (None, (5,)),
                },
                {
                    "ins": ['target'],
                    "shape": ((1024, ), (3, 6, 6)),
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
                    "shape": ((5, 3), (16, 8), (1, 800, 1216), (4, 4, 14, 14)),
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
                    "shape": ((1482, ), (5, 8, 20),
                              (4, 4, 16, 20),
                              (4, 4, 16, 2, 20)),
                },
            ],
        ),
    ),

    'nonzero': dict(
        name=["nonzero"],
        interface=['torch'],
        dtype=[Dtype.bool],
        call_para=dict(
            gen_fn=Genfunc.mask,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1482, ), (5, 8, 20),
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
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 512), (128, 49, 128), (6, 2, 100, 256),
                              (2, 31, 6, 40, 512)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
                {
                    "ins": ['weight'],
                    "requires_grad": [True],
                    "shape": ((10, 512), (384, 128), (81, 256), (1, 512)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
                {
                    "ins": ['bias'],
                    "shape": ((10, ), None, (81, ), (1,)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
            ]
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        related_para=dict(
            dim=[-1, 1, 0],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((78, 24), (2, 92, 29), (2, 150, 512, 512)),
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
            dim=[-1, 1, 0],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 128, 24), (8, 16, 49, 49)),
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
                    "shape": ((741, ), (16, 7),
                              (8, 200, 304), (2, 9, 80, 80)),
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
                    "shape": ((), (2, ), (2, 30), (2, 3, 4)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=10),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": ((10, 3), (10, 2), (93, 512), (20, 2)),
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
            num_classes=[-1, 80],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, ), (6, 8, ), (64, 7, 28,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=6),
                },
            ],
        ),
    ),

    'join': dict(
        name=['cat', 'stack'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        related_para=dict(
            dim=[-1, 1, 0, 2, 1, 1],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "requires_grad": [True],
                    "shape": ((3, ), (512, 4),
                              (0, 50, 76), (2, 31, 512),
                              (2, 512, 8, 8), (1, 64, 4, 56, 56)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                    "gen_num_range": [2, 5]
                },
            ],
            seq_name='tensors',
        ),
    ),

    'split': dict(
        name=["split"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        related_para=dict(
            split_size_or_sections=[[1, 1, 1, 1], [15200, 3800, 950, 247, 70], 3],
            dim=[-1, 0, 1]
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "requires_grad": [True],
                    "shape": ((1, 4),
                              (20267, ),
                              (4, 6, 10, 9, 8)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
        requires_backward=[0],
    ),

    'split_bool': dict(
        name=["split"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        related_para=dict(
            split_size_or_sections=[[1, 1, 1, 1], [15200, 3800, 950, 247, 70], 3],
            dim=[-1, 0, 1]
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "requires_grad": [True],
                    "shape": ((1, 4),
                              (20267, ),
                              (4, 6, 10, 9, 8)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
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
        related_para=dict(
            dim=[-1, 0, 1],
            descending=[True, False, False],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        call_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((11400, ),
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
        related_para=dict(
            k=[9, 12, 1, 3],
            dim=[-1, 0, 1, 2],
            largest=[True, False, False, False],
            sorted=[True, False, False, False],
        ),
        call_para=dict(
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((8723, ), (1024, 81),
                              (5, 4, 6), (2, 2, 64, 64)),
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
                    "shape": ((1, ), ),
                },
            ],
        ),
        requires_backward=[0],
    ),

    'transpose': dict(
        name=['transpose'],
        interface=['torch'],
        related_para=dict(
            dim0=[1, -2],
            dim1=[2, -1],
        ),
        call_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 1536, 950),
                              (660, 6, 49, 32)),
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
                    "shape": [(1024, ), (1482, 4), (4, 5, 6)],
                    "dtype": [Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['input', 'other'],
                    "requires_grad": [True, True],
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "shape": [(1024, ), (1482, 4), (4, 5, 6)],
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
                    "shape": [(1024, ), (1482, 4), (4, 5, 6)],
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

    'dropout': dict(
        name=["dropout"],
        atol=1e-4,
        related_para=dict(
            p=[0.5, 0, 0.1, 0.4],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 4096), (32, 49, 256), (2, 16, 64, 64),
                              (1, 2304, 1, 1, 1)),
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
                              (2, 32, 112, 112), (64, 3, 7, 28, 28)),
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
        is_inplace=True,
        para=dict(
            negative_slope=[0.01, 0.1, 10]
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (2, 32, 208, 304), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
