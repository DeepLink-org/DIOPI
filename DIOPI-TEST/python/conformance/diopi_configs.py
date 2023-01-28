from .config import Genfunc
from .dtype import Dtype

diopi_configs = {
    'batch_norm': dict(
        name=["batch_norm"],
        dtype=[Dtype.float32],
        atol=1e-5,
        para=dict(
            training=[False, False, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 8, 32, 56, 56), (2, 64, 32, 32), (2, 96, 28), (2, 16)),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": ((8, ), (64, ), None, (16, )),
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ["running_var"],
                    "shape": ((8, ), (64, ), None, (16, )),
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    "shape": ((8, ), (64, ), (96, ), (16, )),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'conv_2d': dict(
        name=["conv2d"],
        atol=1e-3,
        rtol=1e-3,
        dtype=[Dtype.float32, Dtype.float16],
        para=dict(
            stride=[2, 1, 1],
            padding=[0, 12, 0],
            dilation=[1, 12, 1],
            groups=[1, 2048, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((2, 256, 200, 304), (2, 2048, 64, 64), (2, 2048, 1, 1)),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": ((12, 256, 1, 1), (2048, 1, 3, 3), (512, 2048, 1, 1)),
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": ((12, ), None, None),
                },
            ]
        ),
    ),

    'relu': dict(
        name=["relu"],
        is_inplace=True,
        tensor_para=dict(
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

    'hardtanh': dict(
        name=["hardtanh"],
        is_inplace=True,
        para=dict(
            min_val=[0.0, 0.0, 0.1, 1.0],
            max_val=[6.0, 0.5, 0.2, 1.2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 4096), (64, 28, 28),
                              (2, 96, 56, 56), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'threshold': dict(
        name=["threshold"],
        is_inplace=True,
        para=dict(
            threshold=[0, 2.0, 3.1, 4.7],
            value=[-5.34, 0.0, 33, 12.4],
        ),
        tensor_para=dict(
            genfunc=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((64, ),
                              (64, 28, 28),
                              (2, 144, 28, 28),
                              (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'gelu': dict(
        name=['gelu'],
        atol=1e-4,
        rtol=1e-5,
        # approximate=['none', 'tanh'], # todo: pytroch 1.12
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
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
            kernel_size=[(2, 2), 3],
            stride=[1, (1, 2)],
            padding=[(1, 1), 0],
            ceil_mode=[True, False],
            count_include_pad=[True, False],
            divisor_override=[None, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 1024, 14, 14), (256, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[3, (2, 1), (2, 2), 3],
            stride=[2, (2, 1), (2, 1), 2],
            padding=[1, 0, (0, 1), 0],
            dilation=[1, 1, 1, 2],
            ceil_mode=[False, True, False, True],
            return_indices=[False, False, False, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
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
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 2048, 8, 6), (2, 288, 33, 33),
                              (2, 144, 65, 65), (2, 1280, 7, 7)),
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
            output_size=[2, (1, 3), (3, 4)],
            return_indices=[False, False, False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((288, 33, 33), (2, 144, 33, 33), (2, 16, 130, 130)),
                    "dtype": [Dtype.float32],
                },
            ]
        ),
    ),

    'binary_cross_entropy': dict(
        name=["binary_cross_entropy"],
        atol=1e-3,
        rtol=1e-4,
        dtype=[Dtype.float32],
        para=dict(
            reduction=['mean', 'none', 'sum', 'mean'],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.rand,
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
                    "shape": ((72,), (2, 11856),
                              (2, 741, 80),
                              (4, 4, 16, 20)),
                },
                {
                    "ins": ['weight'],
                    "shape": ((72,), (2, 11856),
                              (2, 741, 80),
                              None),
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'binary_cross_entropy_with_logits': dict(
        name=["binary_cross_entropy_with_logits"],
        atol=1e-3,
        rtol=1e-4,
        dtype=[Dtype.float32],
        para=dict(
            reduction=['mean', 'none', 'sum', 'mean'],
        ),
        tensor_para=dict(
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
                    "shape": ((72,), (2, 11856),
                              (2, 741, 80),
                              (4, 4, 16, 20)),
                },
                {
                    "ins": ['weight'],
                    "shape": ((72,), (2, 11856),
                              (2, 741, 80),
                              None),
                },
                {
                    "ins": ['pos_weight'],
                    "shape": ((72,), None, (80, ), None),
                    "gen_fn": dict(fn=Genfunc.randint, high=4),
                    "dtype": [Dtype.int64],
                },
            ],
        ),
    ),

    'pointwise_op': dict(
        name=['abs', 'cos', 'erf', 'exp', 'floor',
              'neg', 'sin', 'sqrt'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
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

    'erfinv': dict(
        name=["erfinv"],
        interface=['torch'],
        is_inplace=True,
        atol=1e-5,
        rtol=1e-4,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "requires_grad": [False],
                    "shape": ((10, ), (16, 8), (16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'pointwise_op_abs_input': dict(
        name=['log', 'log2', 'log10', 'sqrt'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.positive,
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

    'tanh': dict(
        name=['tanh'],
        interface=['torch'],
        is_inplace=True,
        saved_args=dict(output=0),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((1, ), (1024,), (364800, 4), (2, 128, 3072),
                              (256, 128, 3, 3),
                              (2, 31, 512, 6, 40)),
                },
            ],
        ),
    ),

    'sign': dict(
        name=['sign'],
        interface=['torch'],
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
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
        saved_args=dict(output=0),
        tensor_para=dict(
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
        is_inplace=True,
        dtype=[Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (20267, 80),
                              (2, 128, 3072),
                              (2, 512, 38, 38)),
                },
                {
                    "ins": ['exponent'],
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
        is_inplace=True,
        para=dict(
            exponent=[2, 3, 4, 0.2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
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
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input', 'exponent'],
                    "shape": ((125, 1), (70, 1, 2),
                              (4, 256, 16, 16)),
                    "dtype": [Dtype.int32],
                    "gen_fn": dict(fn=Genfunc.randint, high=4),
                }
            ],
        ),
    ),

    'pointwise_binary': dict(
        name=['add', 'sub', 'mul', 'div', 'eq', 'ne', 'le',
              'lt', 'gt', 'ge', 'logical_and', 'logical_or'],
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32],
        tensor_para=dict(
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
        name=['add', 'sub', 'mul', 'div', 'eq',
              'ne', 'le', 'lt', 'gt', 'ge'],
        interface=['torch'],
        tag=['scalar'],
        is_inplace=True,
        dtype=[Dtype.float32],
        para=dict(
            other=[-1, 0.028, 2, 1.0],
        ),
        tensor_para=dict(
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
        name=['add'],
        para=dict(
            alpha=[-2, 2.0, 4, 1],
            other=[-2, 2.0, 4, 1],
        ),
        interface=['torch'],
        is_inplace=True,
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1024, ), (384, 128),
                              (128, 64, 3, 3),
                              (2, 32, 130, 130)),
                    'no_contiguous': [True],
                },
            ],
        ),
    ),

    'bmm': dict(
        name=['bmm'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 726, 32), (16, 100, 100), (9, 5, 5)),
                },
                {
                    "ins": ['mat2'],
                    "shape": ((16, 32, 726), (16, 100, 32), (9, 5, 10)),
                },
            ],
        ),
    ),

    'addmm': dict(
        name=["addmm"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            alpha=[0.001, -0.01, 1],
            beta=[0.001, -0.01, 1],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((10, ), (768, ), (400,)),
                },
                {
                    "ins": ["mat1"],
                    "shape": ((2, 2048), (2, 768), (1, 2304)),
                },
                {
                    "ins": ["mat2"],
                    "shape": ((2048, 10), (768, 768), (2304, 400)),
                },
            ],
        ),
    ),

    'addcmul': dict(
        name=["addcmul"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            value=[0.001, -0.01, 2, 1, 1],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((128, ), (576, 192), (64, 3, 3, 3), (10, 3, 5), (4, 1, 1)),
                },
                {
                    "ins": ["tensor1"],
                    "shape": ((128, ), (576, 192), (64, 3, 3, 3), (10, 3, 1), (1, 5)),
                },
                {
                    "ins": ["tensor2"],
                    "shape": ((128, ), (576, 192), (64, 3, 3, 3), (10, 1, 5), (4, 5, 1)),
                },
            ],
        ),
    ),

    'addcdiv': dict(
        name=["addcdiv"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            value=[-0.001, -0.01, 2, -1e-5, 1],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, ), (93, 512), (256, 256, 2, 2), (10, 3, 5), (4, 1, 1)),
                },
                {
                    "ins": ["tensor1"],
                    "shape": ((64, ), (93, 512), (256, 256, 2, 2), (10, 3, 1), (1, 5)),
                },
                {
                    "ins": ["tensor2"],
                    "shape": ((64, ), (93, 512), (256, 256, 2, 2), (10, 1, 5), (4, 5, 1)),
                },
            ],
        ),
    ),

    'matmul': dict(
        name=['matmul'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((128, 49, 128), (5,), (128, 4, 49, 32),
                              (2, 1, 3136, 3136), (2, 784, 64), (2, 31, 6, 40, 512)),
                },
                {
                    "ins": ['other'],
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
        para=dict(
            min=[None, -4.13, 1, 1e-12],
            max=[4.13, 26, None, 1199],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((182, ), (384, 128),
                              (1, 242991, 2),
                              (2, 4, 100, 152)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_tensor': dict(
        name=['clamp'],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float64],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((182, ), (384, 128),
                              (1, 242991, 2),
                              (2, 4, 100, 152)),
                },
                {
                    "ins": ['min'],
                    "shape": ((182, ), (384, 1),
                              None, (2, 4, 100, 152)),
                },
                {
                    "ins": ['max'],
                    "shape": (None, (384, 128), (1, 1, 2), None),
                },
            ],
        ),
    ),

    'fill': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[-100, 0.0, float("-inf"), -100, 0.0, float("-inf")],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726), (2, 31, 6, 40, 1)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_op': dict(
        name=['mean', 'sum'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
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
        para=dict(
            dim=[0, 1, [0, 1], 2, [-1, 0], 3],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op_1': dict(
        name=['std'],
        interface=['torch'],
        para=dict(
            dim=[0, 1, [0, 1], 2, [-1, 0], 3],
            unbiased=[True, False, True, False, True, False],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op_2': dict(
        name=['min', 'max'],
        interface=['torch'],
        para=dict(
            dim=[0, 1, 1, 2, -1, 3],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reduce_partial_op_3': dict(
        name=['any', 'all'],
        interface=['torch'],
        para=dict(
            dim=[0, 1, 0, 2, -1, 3],
        ),
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'reduce_partial_op_4': dict(
        name=['min', 'max', 'any', 'all'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((64, ), (169, 4), (17100, 2), (1, 1, 384),
                              (4, 133, 128, 128), (2, 64, 3, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.bool],
                    "gen_fn": Genfunc.randn,
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
        tensor_para=dict(
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

    'nll_loss': dict(
        name=["nll_loss"],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            reduction=['none', 'mean', 'sum'],
            ignore_index=[-100, 92, 255],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((200, 81), (2, 92, 29), (2, 150, 512, 512)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['target'],
                    "shape": ((200, ), (2, 29), (2, 512, 512)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=80),
                },
                {
                    "ins": ['weight'],
                    "shape": ((81, ), (92, ), None),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.ones,
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        atol=1e-1,
        rtol=1e-2,
        para=dict(
            reduction=['mean', 'none', 'sum', 'none'],
            ignore_index=[0, -100, 0, -100],
            label_smoothing=[0.0, 0.0, 0.5, 0.5],
        ),
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((1024, 81), (3, 8, 64, 64), (64, 32), (3, 5, 6, 6)),
                },
                {
                    "ins": ['weight'],
                    "shape": (None, (8,), None, (5,)),
                },
                {
                    "ins": ['target'],
                    "shape": ((1024, ), (3, 64, 64), (64, ), (3, 6, 6)),
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=5),
                    "dtype": [Dtype.int64],

                },
            ],
        ),
    ),

    'cross_entropy_prob_target': dict(
        name=["cross_entropy"],
        atol=1e-3,
        rtol=1e-4,
        para=dict(
            reduction=['sum', 'mean', 'none'],
            label_smoothing=[0.1, 0.3, 0.5],
        ),
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5, 6, 6), (1024, 81), (64, 8, 8)),
                },
                {
                    "ins": ['weight'],
                    "shape": ((5,), None, (8,)),
                },
                {
                    "ins": ['target'],
                    "shape": ((3, 5, 6, 6), (1024, 81), (64, 8, 8)),
                    "gen_fn": Genfunc.rand,
                },
            ],
        ),
    ),

    'select': dict(
        name=["select"],
        interface=['torch'],
        para=dict(
            dim=[-2, 1],
            index=[0, 2],
        ),
        tensor_para=dict(
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
        para=dict(
            dim=[0, 1, 2, 3]
        ),
        tensor_para=dict(
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
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16,), (12, 13), (12, 13, 14), (12, 13, 14, 16)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mask'],
                    "shape": ((16,), (12, 13), (12, 13, 14), (12, 13, 14, 1)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['source'],
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
        tensor_para=dict(
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

    'nonzero_bool': dict(
        name=["nonzero"],
        interface=['torch'],
        dtype=[Dtype.bool],
        tensor_para=dict(
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
        rtol=1e-5,
        atol_half=1e-1,
        rtol_half=1e-2,
        tensor_para=dict(
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
                    "requires_grad": [True],
                    "shape": ((10, ), None, (81, ), (1,)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
            ]
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1, 1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
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
        saved_args=dict(output=0),
        para=dict(
            dim=[-1, 1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 24), (2, 128, 24), (8, 16, 49, 49)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'embedding': dict(
        name=["embedding"],
        para=dict(
            padding_idx=[None, 0, None, 0],
            max_norm=[None, 1.0, None, 1.0],
            norm_type=[2.0, 1.0, 2.0, 3.0],
            scale_grad_by_freq=[False, True, False, True],
            sparse=[False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
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
            max_norm=[1.0, 5, 2.0, 10],
            norm_type=[2.0, 3.0, 2.0, 2.0],
            # error_if_nonfinite=[True, False], # 1.7 not support
        ),
        tensor_para=dict(
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
        tensor_para=dict(
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
            num_classes=[-1, -1, 80],
        ),
        tensor_para=dict(
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
        para=dict(
            dim=[-1, 1, 0, 2, 1, 1],
        ),
        tensor_para=dict(
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
        para=dict(
            split_size_or_sections=[[1, 1, 1, 1], [15200, 3800, 950, 247, 70], 3],
            dim=[-1, 0, 1]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "shape": ((1, 4),
                              (20267, ),
                              (4, 6, 10, 9, 8)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'split_bool': dict(
        name=["split"],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            split_size_or_sections=[[1, 1, 1, 1], [15200, 3800, 950, 247, 70], 3],
            dim=[-1, 0, 1]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['tensor'],
                    "shape": ((1, 4),
                              (20267, ),
                              (4, 6, 10, 9, 8)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'sort': dict(
        name=["sort"],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, 1],
            descending=[True, False, False],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
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
    ),

    'topk_nonzero': dict(
        name=['topk'],
        interface=['torch'],
        para=dict(
            k=[9, 12, 1, 3],
            dim=[-1, 0, 1, 2],
            largest=[True, False, False, False],
            sorted=[True, False, False, False],
        ),
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((8723, ), (1024, 81),
                              (5, 4, 6), (2, 2, 64, 64)),
                },
            ],
        ),
    ),

    'topk_zero': dict(
        name=['topk'],
        interface=['torch'],
        para=dict(
            k=[1, 1],
            dim=[-1, 0],
            largest=[True, False],
            sorted=[True, False],
        ),
        dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (1, )),
                },
            ],
        ),
    ),

    'transpose': dict(
        name=['transpose'],
        interface=['torch'],
        para=dict(
            dim0=[1, -2],
            dim1=[2, -1],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 1536, 950),
                              (660, 6, 49, 32)),
                },
            ],
        ),
    ),

    'where': dict(
        name=['where'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "shape": [(1024, ), (1482, 4), (4, 5, 6)],
                    "dtype": [Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['input', 'other'],
                    "dtype": [Dtype.float16, Dtype.float32, Dtype.float64],
                    "shape": [(1024, ), (1482, 4), (4, 5, 6)],
                    "gen_fn": Genfunc.randn
                },
            ],
        ),
    ),

    'where_2': dict(
        name=['where'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['condition'],
                    "shape": [(1, ), (3, ), (3, ), (1, 445), (3, 5), (4, ),
                              (3, 4, 5), (3, )],
                    "dtype": [Dtype.uint8, Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['input'],
                    "dtype": [Dtype.float32, Dtype.float64],
                    "shape": [(1, ), (1, ), (3, ), (1, 445), (3, 5), (1, ), (4, 5),
                              (5, 4, 3)],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['other'],
                    "dtype": [Dtype.float32, Dtype.float64],
                    "shape": [(1, ), (1, ), (1, ), (1, ), (1, ), (4, ), (5, ), (4, 3)],
                    "gen_fn": Genfunc.randn
                },
            ],
        ),
    ),

    'dropout': dict(
        name=["dropout"],
        no_output_ref=True,
        is_inplace=True,
        para=dict(
            p=[0.5, 0, 0.1, 0.4],
        ),
        tensor_para=dict(
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

    'dropout2d': dict(
        name=["dropout2d"],
        no_output_ref=True,
        is_inplace=True,
        para=dict(
            p=[0.5, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((32, 49, 256), (32, 16, 64, 64)),
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
            negative_slope=[0.01, 0.1, 10, 1]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((16, 7), (64, 28, 28),
                              (2, 32, 208, 304), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sigmoid_focal_loss': dict(
        name=["sigmoid_focal_loss"],
        interface=["torchvision.ops"],
        dtype=[Dtype.float32, Dtype.float64],
        para=dict(
            alpha=[0.25, 0.1, 0.9],
            gamma=[2, 0.1, 10],
            reduction=["mean", "sum", "none"],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['inputs'],
                    "requires_grad": [True],
                    "shape": ((16, 7), (2, 11856, 2), (16, 2, 2964, 2)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['targets'],
                    "shape": ((16, 7), (2, 11856, 2), (16, 2, 2964, 2)),
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'nms': dict(
        name=["nms"],
        interface=["torchvision.ops"],
        dtype=[Dtype.float32, Dtype.float64],
        para=dict(
            iou_threshold=[0.3, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['boxes'],
                    "value": ([[2.4112, 0.7486, 2.4551, 2.7486],
                              [0.7486, 1.3544, 1.1294, 2.3544],
                              [1.4551, 0.1294, 1.6724, 1.3294],
                              [1.4959, 0.1086, 2.778335, 3.22],
                              [0.107706, 2.948, 2.1256, 4.525],
                              [2.7735, 2.12506, 7.0556, 8.995]],

                              [[1.5, 2.2, 2.77, 3.2],
                              [2.5, 5.9, 10.6, 14.55]],),
                },
                {
                    "ins": ['scores'],
                    "shape": ((6, ), (2, )),
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'roi_align': dict(
        name=["roi_align"],
        interface=["torchvision.ops"],
        dtype=[Dtype.float32, Dtype.float64],
        para=dict(
            output_size=[(5, 6), 3],
            spatial_scale=[0.8, 1.0],
            sampling_ratio=[1, -1],
            aligned=[True, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((6, 3, 32, 32), (2, 3, 16, 16)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['boxes'],
                    "value": ([[1, 2.4112, 0.7486, 2.4551, 2.7486],
                              [0, 0.7486, 1.3544, 1.1294, 2.3544],
                              [2, 1.4551, 0.1294, 1.6724, 1.3294],
                              [5, 1.4959, 0.1086, 2.778335, 3.22],
                              [2, 0.107706, 2.948, 2.1256, 4.525],
                              [4, 2.7735, 2.12506, 7.0556, 8.995]],

                              [[0, 1.5, 2.2, 2.77, 3.2],
                              [1, 2.5, 5.9, 10.6, 14.55]],),
                },
            ],
        ),
    ),

    'slice': dict(
        name=["slice_op"],
        interface=["CustomizedTest"],
        dtype=[Dtype.float32, Dtype.float64],
        para=dict(
            index=(slice(0, 3, 1), slice(0, 3, 1), slice(0, 4, 2), slice(-3, -2, 1)),
            dim=[0, 1, 2, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((7, ), (128, 3, 3), (2, 3, 224, 224), (3, 2, 6, 197, 64)),
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'index': dict(
        name=["index"],
        interface=["CustomizedTest"],
        # input[idx1,idx2,idx3] input[...,idx3] input[idx,...,idx3]
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((128, 2, 2), (2, 3, 224, 224), (3, 2, 6, 197, 64)),
                    "requires_grad": [True],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['idx1'],
                    "shape": ((1, ), None, (1, )),
                    "gen_fn": dict(fn=Genfunc.randint, high=3),
                    "dtype": [Dtype.int64],
                },
                {
                    "ins": ['idx2'],
                    "shape": ((1, ), None, None),
                    "gen_fn": dict(fn=Genfunc.randint, high=2),
                    "dtype": [Dtype.int64],
                },
                {
                    "ins": ['idx3'],
                    "shape": ((2, ), (224, 224), (64, )),
                    "gen_fn": Genfunc.mask,
                    "dtype": [Dtype.bool],
                },
            ],
        ),
    ),

    'sgd': dict(
        name=["sgd"],
        interface=["CustomizedTest"],
        atol_half=1e-4,
        rtol_half=1e-3,
        para=dict(
            nesterov=[False, True],
            lr=[0.1, 0.1],
            momentum=[0.01, 0.01],
            weight_decay=[0, 0.1],
            dampening=[0.1, 0],
        ),
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float16],
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [(2, 3, 16), (4, 32, 7, 7)],
                    "gen_fn": Genfunc.rand,
                },
                {
                    "ins": ['buf'],
                    "shape": [(2, 3, 16), (4, 32, 7, 7)],
                    "gen_fn": Genfunc.zeros,
                },
            ]
        ),
    ),

    'masked_fill': dict(
        name=["masked_fill"],
        interface=["torch"],
        is_inplace=True,
        para=dict(
            value=[-100, 0.0, float("-inf"), -100, 0.0, float("-inf")],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726), (2, 31, 6, 40, 1)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mask'],
                    "shape": ((1,), (64,), (4, 49), (1276, 49, 49), (2, 1, 1, 726), (2, 31, 6, 40, 1)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
            ],
        ),
    ),

    'masked_fill_scalar': dict(
        name=["masked_fill"],
        interface=["torch"],
        is_inplace=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (64,), (4, 49), (1276, 49, 49), (2, 8, 726, 726), (2, 31, 6, 40, 1)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mask'],
                    "shape": ((1,), (64,), (4, 49), (1276, 49, 49), (2, 1, 1, 726), (2, 31, 6, 40, 1)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
                {
                    "ins": ['value'],
                    # masked_fill_ only supports a 0-dimensional value tensor
                    "shape": ((), (), (), (), (), ()),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.ones
                },
            ],
        ),
    ),

    'reciprocal': dict(
        name=["reciprocal"],
        interface=['torch'],
        is_inplace=True,
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (182,), (64, 128), (2, 1, 640, 640)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'adam': dict(
        name=['adam', 'adamw'],
        interface=["CustomizedTest"],
        atol=1e-4,
        rtol=1e-3,
        atol_half=1e-4,
        rtol_half=1e-3,

        para=dict(
            lr=[0.1, 0.1],
            beta1=[0.9, 0.8],
            beta2=[0.99, 0.88],
            eps=[1e-08, 1e-09],
            step=[1, 4],
            weight_decay=[0, 0.1],
            amsgrad=[False, True],
        ),
        tensor_para=dict(
            dtype=[Dtype.float16, Dtype.float32, Dtype.float64],
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [(2, 3, 16), (4, 32, 7, 7)],
                    "gen_fn": Genfunc.rand,
                },
                {
                    "ins": ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'],
                    "shape": [(2, 3, 16), (4, 32, 7, 7)],
                    "gen_fn": Genfunc.zeros,
                },
            ]
        ),
    ),

    'conv_transpose2d': dict(
        name=["conv_transpose2d"],
        atol=1e-5,
        para=dict(
            stride=[1, 2, 1, 2],
            padding=[0, 1, 0, 1],
            output_padding=[0, 1, 0, 1],
            groups=[1, 8, 1, 1],
            dilation=[1, 2, 1, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": ((2, 256, 14, 14), (2, 128, 32, 32),
                              (2, 64, 160, 160), (2, 64, 320, 320)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ["weight"],
                    "shape": ((256, 256, 2, 2), (128, 128, 4, 4),
                              (64, 64, 2, 2), (64, 1, 2, 2)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ["bias"],
                    "shape": ((256,), None, (64,), (1,)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'unfold': dict(
        name=["unfold"],
        interface=['torch.Tensor'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            dimension=[2, 1, 2],
            size=[2, 2, 2],
            step=[1, 1, 1],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 128, 56, 56), (2, 512, 14, 14), (2, 96, 200, 304)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'cumsum': dict(
        name=["cumsum"],
        interface=['torch'],
        atol=1e-6,
        rtol=1e-5,
        dtype=[Dtype.float32],
        para=dict(
            dim=[1, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 22, 33), (2, 2, 10, 16)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cdist': dict(
        name=['cdist'],
        interface=['torch'],
        saved_args=dict(output=0),
        para=dict(
            p=[1, 2],
            compute_mode=['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['x1'],
                    "requires_grad": [True],
                    "shape": ((100, 4), (2, 256, 256)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['x2'],
                    "shape": ((100, 4), (2, 256, 256)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'bitwise_not': dict(
        name=['bitwise_not'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (100, 4), (2, 256, 256)),
                    "dtype": [Dtype.bool, Dtype.int32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'argmax': dict(
        name=['argmax'],
        interface=["torch"],
        para=dict(
            dim=[-1, 0, 1, None],
            keepdim=[True, False, True, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1,), (1024, 80), (2, 256, 256), (2, 1, 64, 64)),
                    "dtype": [Dtype.float32, Dtype.int32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'adadelta': dict(
        name=["adadelta"],
        interface=["CustomizedTest"],
        atol_half=1e-4,
        rtol_half=1e-3,
        para=dict(
            lr=[0.1, 0.1],
            rho=[0.9, 0.88],
            eps=[1e-6, 1e-6],
            weight_decay=[0, 0.1],
        ),
        tensor_para=dict(
            dtype=[Dtype.float32, Dtype.float16],
            args=[
                {
                    "ins": ['param', 'param_grad'],
                    "shape": [(2, 3, 16), (4, 32, 7, 7)],
                    "gen_fn": Genfunc.rand,
                },
                {
                    "ins": ['square_avg', 'acc_delta'],
                    "shape": [(2, 3, 16), (4, 32, 7, 7)],
                    "gen_fn": Genfunc.zeros,
                },
            ]
        ),
    ),

    'smooth_l1_loss': dict(
        name=["smooth_l1_loss"],
        para=dict(
            reduction=['mean', 'none', 'sum'],
            beta=[1.0, 0.5, 0.1]
        ),
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 11856, 2), (16, 2, 2964, 2), (2, 16, 128, 128)),
                },
                {
                    "ins": ['target'],
                    "shape": ((2, 11856, 2), (16, 2, 2964, 2), (2, 16, 128, 128)),
                },
            ],
        ),
    ),

    'conv3d': dict(
        name=['conv3d'],
        atol=1e-2,
        interface=['torch'],
        para=dict(
            stride=[1, (2, 1, 1), 3, 1],
            padding=[0, (1, 0, 1), 0, (1, 0, 1)],
            dilation=[1, (2, 1, 1), 1, (2, 1, 1)],
            groups=[1, 2, 2, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((1, 3, 4, 224, 224), (1, 16, 32, 56, 56),
                              (1, 128, 4, 56, 56), (1, 256, 4, 56, 56)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": ((64, 3, 1, 7, 7), (16, 8, 5, 1, 1),
                              (64, 64, 1, 3, 3), (64, 256, 1, 1, 1)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": (None, (16,), (64,), (64,)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'max_pool3d': dict(
        name=['max_pool3d'],
        para=dict(
            kernel_size=[(3, 2, 2), (1, 2, 3), 1],
            stride=[(2, 1, 2), 2, (2, 3, 4)],
            dilation=[2, (2, 1, 3), (2, 2, 2)],
            ceil_mode=[False, False, False],
            return_indices=[False, False, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((9, 6, 6, 8, 6),
                              (4, 6, 8, 9, 12),
                              (6, 9, 8, 10, 7)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
            ]
        ),
    ),

    'adaptive_avg_pool3d': dict(
        name=["adaptive_avg_pool3d"],
        atol=1e-5,
        rtol=1e-4,
        para=dict(
            output_size=[(1, 1, 1), 2, (None, 3, 3), (3, 4, 4)],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((1, 2048, 4, 7, 7), (2, 512, 4, 4),
                              (2, 1024, 14, 14), (2, 720, 17, 17)),
                    "dtype": [Dtype.float32],
                },
            ]
        ),
    ),

    'adaptive_max_pool3d': dict(
        name=["adaptive_max_pool3d"],
        atol=1e-5,
        rtol=1e-4,
        para=dict(
            output_size=[2, (1, 3, 2), (3, 4, 4)],
            return_indices=[False, False, False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((1, 2048, 4, 7, 7), (2, 512, 4, 4), (2, 1024, 14, 14)),
                    "dtype": [Dtype.float32],
                },
            ]
        ),
    ),

    'masked_select': dict(
        name=['masked_select'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((1, ), (16,), (8, 48), (4, 128, 128), (256, 8, 8)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.ones,
                },
                {
                    "ins": ['mask'],
                    "shape": ((1, ), (16,), (8, 48), (4, 128, 128), (256, 8, 8)),
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask
                },
            ],
        ),
    ),

    'imum': dict(
        name=['maximum', 'minimum'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input', 'other'],
                    "shape": ((1, ), (16,), (8, 48), (4, 128, 128), (256, 8, 8)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mm': dict(
        name=['mm'],
        interface=['torch'],
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((8, 48), (4, 128), (256, 8)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['mat2'],
                    "shape": ((48, 128), (128, 128), (8, 1)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'index_fill': dict(
        name=['index_fill'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[0, 1, 2, 3],
            value=[1, -1, 2.0, 5]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 3), (16, 8), (16, 4, 4), (4, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['index'],
                    "shape": ((3,), (5,), (2,), (10,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=3)
                },
            ]
        ),
    ),

    'index_fill_scalar': dict(
        name=['index_fill'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[0, 1, 2, 3],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((5, 3), (16, 8), (16, 4, 4), (4, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.randn
                },
                {
                    "ins": ['value'],
                    # index_fill_ only supports a 0-dimensional value tensor
                    "shape": ((), (), (), ()),
                    "dtype": [Dtype.float32, Dtype.float64, Dtype.float16],
                    "gen_fn": Genfunc.ones
                },
                {
                    "ins": ['index'],
                    "shape": ((3,), (5,), (2,), (10,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=3)
                },
            ]
        ),
    ),

    'expand': dict(
        name=['expand'],
        interface=['torch.Tensor'],
        para=dict(
            size=[(60800, 3), (-1, 4), (-1, 8, -1), (7, 3, -1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(60800, 1), (100, 1), (70, 1, 2), (3, 1)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.bool],
                },
            ],
        ),
    ),

    'linspace': dict(
        name=['linspace'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-3,
        rtol_half=1e-3,
        atol_half=1e-4,
        para=dict(
            start=[0, 0, -1, -1, -1, -1, -1, -1],
            end=[0.5, 0.0, 1, 1, 1, 1, 1, 1],
            steps=[24, 23, 152, 100, 76, 50, 38, 25],
        ),
    ),

    'permute': dict(
        name=['permute'],
        interface=['torch'],
        para=dict(
            dims=[(0, 1, 3, 2, 4, 5), (2, 0, 1), (0, 2, 3, 1), (1, 0)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(2, 8, 7, 8, 7, 128), (49, 49, 4), (2, 3, 200, 304), (20267, 1)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'pad': dict(
        name=['pad'],
        para=dict(
            pad=[(0, 3), (0, 1, 0, 1), (1, 1, 1, 1), (0, 193)],
            mode=['circular', 'replicate', 'reflect', 'replicate'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(2, 56, 56), (2, 3, 260, 260), (2, 144, 65, 65), (3, 576, 862)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'constant_pad': dict(
        name=['pad'],
        para=dict(
            pad=[(0, 3), (0, 1, 0, 1), (1, 1, 1, 1), (0, 193)],
            mode=['constant', 'constant', 'constant', 'constant'],
            value=[100, 0, -1, 1]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": [(2, 56, 56), (2, 3, 260, 260), (2, 144, 65, 65), (3, 576, 862)],
                    "gen_fn": Genfunc.randn,
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'roll': dict(
        name=['roll'],
        interface=['torch'],
        para=dict(
            shifts=[1, (0, 1), (3, 3), (-3, -3)],
            dims=[None, (0, 1), (1, 2), (1, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "shape": ((4,), (8, 32), (2, 14, 14, 512), (2, 56, 56, 128)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'norm': dict(
        name=['norm'],
        interface=['torch'],
        para=dict(
            p=[2.5, float('inf'), -float('inf')],
            dim=[None, (0, 1), (1, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "shape": ((128, ), (384, 128), (256, 512, 1, 1)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'group_norm': dict(
        name=['group_norm'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            num_groups=[32, 32, 32, 32],
            eps=[1e-05, 1e-05, 1e-05, 1e-05]
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((2, 256, 100, 152), (2, 256, 7, 10),
                              (2, 256, 24, 24), (2, 256, 12, 12)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ["weight", "bias"],
                    "requires_grad": [True],
                    "shape": ((256,), (256,),
                              (256,), (256,)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'unique': dict(
        name=['unique'],
        interface=['torch'],
        para=dict(
            sorted=[True, False, True, False],
            return_inverse=[True, False, True, False],
            return_counts=[True, False, True, False],
            dim=[-1, 0, -1, 0],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((0,), (252,), (2, 256), (4, 64, 128)),
                    "dtype": [Dtype.int64, Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'prod': dict(
        name=['prod'],
        interface=['torch'],
        para=dict(
            dim=[-1, 0, -1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((56, 1), (70, 1, 2), (2, 512, 38, 38), (2, 80, 128, 128, 1)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'ctc_loss': dict(
        name=["ctc_loss"],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            reduction=['none', 'mean', 'sum'],
            blank=[0, 0, 0],
            zero_infinity=[True, False, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['log_probs'],
                    "requires_grad": [True],
                    "shape": ((26, 2, 38), (26, 2, 38), (26, 2, 38)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['targets'],
                    "shape": ((2, 10), (2, 14), (2, 11)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=80),
                },
                {
                    "ins": ['input_lengths'],
                    "shape": ((2, ), (2, ), (2, )),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=1, high=26),
                },
                {
                    "ins": ['target_lengths'],
                    "shape": ((2, ), (2, ), (2, )),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=1, high=10),
                },
            ],
        ),
    ),

    'remainder': dict(
        name=['remainder'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            self=[4.3, 10.1, 5.2, 100.],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['other'],
                    "shape": ((6, ), (4, 1), (1, 28, 28),
                              (16, 3, 7, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'remainder_tensor': dict(
        name=['remainder'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((6, ), (4, 5), (14, 28, 28),
                              (16, 1, 7, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ['other'],
                    "shape": ((6, ), (4, 1), (1, 28, 28),
                              (16, 3, 7, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'remainder_scalar': dict(
        name=['remainder'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            other=[4.3, 10.1, 5.2, 100.],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((6, ), (4, 5), (14, 28, 28),
                              (16, 1, 7, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'gather': dict(
        name=['gather'],
        interface=['torch'],
        para=dict(
            dim=[0, 1, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((16, 4, 4), (64, 4, 14, 14), (64, 4, 16, 16)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },

                {
                    "ins": ['index'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14), (64, 4, 16, 16)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },

            ],
        ),
    ),

    'scatter': dict(
        name=['scatter'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[2, 1],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 4, 4), (2, 8, 64, 64)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['index'],
                    "shape": ((16, 4, 4), (2, 8, 1, 1)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['src'],
                    "shape": ((16, 4, 4), (2, 8, 4, 4)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'scatter_reduce': dict(
        name=['scatter'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[2, 1],
            reduce=['add', 'multiply']
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 4, 4), (2, 8, 64, 64)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['index'],
                    "shape": ((16, 4, 4), (2, 8, 1, 1)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['src'],
                    "shape": ((16, 4, 4), (2, 8, 4, 4)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'scatter_scalar': dict(
        name=['scatter'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[2, 1],
            value=[-100, float("-inf")],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['index'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
            ]
        ),
    ),

    'scatter_reduce_scalar': dict(
        name=['scatter'],
        interface=['torch'],
        is_inplace=True,
        para=dict(
            dim=[2, 1],
            value=[-100, float("-inf")],
            reduce=['add', 'multiply']
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['index'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
            ]
        ),
    ),

    'index_put_acc': dict(
        name=['index_put'],
        interface=['CustomizedTest'],
        is_inplace=True,
        para=dict(
            accumulate=[True, True]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['indices1', 'indices2'],
                    "shape": ((16, 4), (64, 4)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['values'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),

    'index_put_acc_one_indices': dict(
        name=['index_put'],
        interface=['CustomizedTest'],
        is_inplace=True,
        para=dict(
            accumulate=[True, True]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['indices1'],
                    "shape": ((16,), (64,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['values'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ]
        ),
    ),


    'index_put': dict(
        name=['index_put'],
        interface=['CustomizedTest'],
        is_inplace=True,
        para=dict(
            accumulate=[True, False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['indices1', 'indices2'],
                    "shape": ((16, 4), (64, 4)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['values'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    # sample test for index_put when acc is False
                    "gen_fn": Genfunc.ones,
                },
            ]
        ),
    ),

    'index_put_one_indices': dict(
        name=['index_put'],
        interface=['CustomizedTest'],
        is_inplace=True,
        para=dict(
            accumulate=[True, False]
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
                {
                    "ins": ['indices1'],
                    "shape": ((16,), (64,)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=4),
                },
                {
                    "ins": ['values'],
                    "shape": ((16, 4, 4), (64, 4, 14, 14)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    # sample test for index_put when acc is False
                    "gen_fn": Genfunc.ones,
                },
            ]
        ),
    ),

    'arange': dict(
        name=['arange'],
        interface=['torch'],
        para=dict(
            start=[0, 0, -4, 0.1],
            end=[91, 128, 5, 0.5],
            step=[13, 1, 1, 0.1],
        ),
    ),

    'randperm': dict(
        name=['randperm'],
        no_output_ref=True,
        para=dict(
            n=[2, 1999, 640000],
        ),
    ),

    'uniform': dict(
        name=['uniform'],
        no_output_ref=True,
        para={
            'start': [0.5, -0.12499999999999999, -0.25, -0.04811252243246881],
            'end': [1.5, 0.12499999999999999, 0.25, 0.04811252243246881],
        },
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (64, 64), (16, 1, 3, 3), (96, 48, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'random': dict(
        name=['random'],
        no_output_ref=True,
        para={
            'start': [0, 3, -1, 0],
            'end': [2, None, 1, None],
        },
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (64, 64), (16, 1, 3, 3), (96, 48, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.int64],
                },
            ],
        ),
    ),

    'bernoulli': dict(
        name=['bernoulli'],
        no_output_ref=True,
        is_inplace=True,
        para=dict(
            p=[None, 0.5, None, None],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.rand,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((1, ), (64, 64), (16, 1, 3, 3), (96, 48, 3, 3)),
                    "dtype": [Dtype.float32, Dtype.float64],
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        dtype=[Dtype.float32],
        atol=1e-5,
        para=dict(
            eps=[1e-5, 1e-5, 1e-12],
            normalized_shape=[(5, 3, 5), (128, ), (64, )],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((2, 5, 3, 5), (2, 3136, 128), (2, 64)),
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": (None, (128, ), (64, )),
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": (None, (128, ), (64, )),
                },
            ]
        )
    ),

    'copy': dict(
        name=["copy_"],
        interface=['torch.Tensor'],
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ["input"],
                    "shape": ((192, 147), (1, 1, 384), (2, 1, 38, 45)),
                    "no_contiguous": [True],
                },
                {
                    "ins": ["other"],
                    "shape": ((147, 1), (384, 1, 1), (45, 38, 1, 2)),
                },
            ]
        )
    ),

    'interpolate': dict(
        name=["interpolate"],
        dtype=[Dtype.float32],
        para=dict(
            mode=['nearest', 'bilinear', 'nearest', 'bicubic', 'trilinear', 'linear'],
            size=[(50, 76), (25, 38), (4, 224, 224), (64, 64), (4, 224, 112), (64, )],
            align_corners=[None, False, None, True, True, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": ((2, 256, 25, 38), (2, 256, 13, 19), (1, 3, 32, 224, 224), (2, 16, 1, 1),
                              (1, 3, 32, 224, 224), (2, 32, 32)),
                },
            ]
        )
    ),

    'col2im': dict(
        name=["col2im"],
        interface=['CustomizedTest'],
        para=dict(
            output_size=[(352, 528), (12, 40), (4, 26), 10],
            kernel_size=[3, (2, 1), (2, 2), 3],
            stride=[2, (2, 1), (2, 1), 2],
            padding=[1, 0, (0, 1), 0],
            dilation=[1, 1, 1, 2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 576, 46464),
                              (2, 512, 240),
                              (2, 2048, 54),
                              (3, 36, 9)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
            ]
        ),
    ),

    'im2col': dict(
        name=["im2col"],
        interface=['CustomizedTest'],
        para=dict(
            kernel_size=[3, (2, 1), (2, 2), 3],
            stride=[2, (2, 1), (2, 1), 2],
            padding=[1, 0, (0, 1), 0],
            dilation=[1, 1, 1, 2],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 64, 352, 528),
                              (2, 256, 12, 40),
                              (2, 512, 4, 26),
                              (3, 4, 10, 10)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
            ]
        ),
    ),

    'flip': dict(
        name=['flip'],
        interface=['torch'],
        para=dict(
            dims=[(1,), (-2, -1), (0, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "shape": ((49, 49), (12, 13, 14), (12, 13, 14, 16)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cholesky': dict(
        name=['cholesky_ex'],
        interface=['torch.linalg'],
        para=dict(
            upper=[True, False],
            check_errors=[True, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 3, 3), (2, 3, 3)),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.sym_mat,
                },
            ],
        ),
        requires_backward=[0],
        saved_args=dict(output=0),
    ),

    'triangular_solve': dict(
        name=['triangular_solve'],
        interface=['torch'],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            upper=[True, False, True, False],
            transpose=[True, False, True, False],
            unitriangular=[True, False, True, False],
        ),
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((2, 2, 2), (3, 3), (7, 6, 5), (7, 2, 1)),
                    "dtype": [Dtype.float32],
                },
                {
                    "ins": ['A'],
                    "requires_grad": [True],
                    "shape": ((2, 2, 2), (5, 3, 3), (7, 6, 6), (2, 2)),
                    "dtype": [Dtype.float32],
                },
            ],
        ),
        saved_args=dict(output=0),
    ),

}
