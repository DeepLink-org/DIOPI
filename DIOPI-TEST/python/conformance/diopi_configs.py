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
                    "gen_fn": Genfunc.rand,
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
        atol=1e-4,
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

    'max_pool2d_with_indices': dict(
        name=["max_pool2d"],
        requires_backward=[0],
        saved_args=dict(indices=1),
        para=dict(
            kernel_size=[3, (2, 1), (2, 2), 3],
            stride=[2, (2, 1), (2, 1), 2],
            padding=[1, 0, (0, 1), 0],
            dilation=[1, 1, 1, 2],
            ceil_mode=[False, True, False, True],
            return_indices=[True, True, True, True],
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
        requires_backward=[0],
        saved_args=dict(indices=1),
        para=dict(
            output_size=[2, (1, 3), (3, 4)],
            return_indices=[True, True, True]
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
        name=['abs', 'cos', 'erf', 'exp', 'floor',
              'log', 'log2', 'log10', 'neg', 'sin',
              'sqrt'],
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
        name=['add', 'mul', 'div', 'eq', 'ne', 'le', 'lt',
              'gt', 'ge', 'logical_and', 'logical_or'],
        interface=['torch'],
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
        name=['add', 'mul', 'div', 'eq',
              'ne', 'le',  'lt', 'gt', 'ge'],
        interface=['torch'],
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
        no_contiguous=[True],
        interface=['torch'],
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
            keepdim=[True, False, True, False, True, False],
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
            keepdim=[True, False, True, False, True, False],
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
            keepdim=[True, False, True, False, True, False],
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
            keepdim=[True, False, True, False, True, False],
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
                    "requires_grad": [False],
                    "shape": ((200, ), (2, 29), (2, 512, 512)),
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=0, high=80),
                },
                {
                    "ins": ['weight'],
                    "requires_grad": [False],
                    "shape": ((81, ), (92, ), None),
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.ones,
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            reduction=['mean', 'none'],
            ignore_index=[0, -100],
            label_smoothing=[0.0, 0.5],
        ),
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((1024, 81), (3, 5, 6, 6)),
                },
                {
                    "ins": ['weight'],
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

    'cross_entropy_prob_target': dict(
        name=["cross_entropy"],
        para=dict(
            reduction=['sum'],
            label_smoothing=[0.1],
        ),
        dtype=[Dtype.float32],
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "requires_grad": [True],
                    "shape": ((3, 5, 6, 6), ),
                },
                {
                    "ins": ['weight'],
                    "shape": ((5,), ),
                },
                {
                    "ins": ['target'],
                    "shape": ((3, 5, 6, 6), ),
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
        tensor_para=dict(
            gen_fn=Genfunc.randn,
            args=[
                {
                    "ins": ['input'],
                    "shape": ((2, 512), (128, 49, 128), (6, 2, 100, 256),
                              (2, 31, 6, 40, 512)),
                    "dtype": [Dtype.float16, Dtype.float32],
                },
                {
                    "ins": ['weight'],
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
        requires_backward=[0],
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
        requires_backward=[0],
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
        requires_backward=[0],
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
        requires_backward=[0],
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
        requires_backward=[0],
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
        name=["test_dropout"],
        interface=["CustomizedTest"],
        atol=2e-2,
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
}
