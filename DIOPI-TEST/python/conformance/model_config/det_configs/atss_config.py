from ...config import Genfunc
from ...dtype import Dtype

atss_config = {
    'conv2d': dict(
        name=["conv2d"],
        para=dict(
            bias=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            stride=[(2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(3, 3), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (3, 3), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 3, 800, 1344), (2, 64, 200, 336), (2, 64, 200, 336), (2, 64, 200, 336), (2, 256, 200, 336), (2, 256, 200, 336), (2, 128, 200, 336), (2, 128, 100, 168), (2, 256, 200, 336), (2, 512, 100, 168), (2, 128, 100, 168), (2, 512, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 512, 100, 168), (2, 1024, 50, 84), (2, 256, 50, 84), (2, 1024, 50, 84), (2, 512, 50, 84), (2, 512, 25, 42), (2, 1024, 50, 84), (2, 2048, 25, 42), (2, 512, 25, 42), (2, 256, 100, 168), (2, 256, 25, 42), (2, 256, 13, 21), (2, 256, 7, 11), (1, 3, 800, 1344), (1, 64, 200, 336), (1, 64, 200, 336), (1, 64, 200, 336), (1, 256, 200, 336), (1, 256, 200, 336), (1, 128, 200, 336), (1, 128, 100, 168), (1, 256, 200, 336), (1, 512, 100, 168), (1, 128, 100, 168), (1, 512, 100, 168), (1, 256, 100, 168), (1, 256, 50, 84), (1, 512, 100, 168), (1, 1024, 50, 84), (1, 256, 50, 84), (1, 1024, 50, 84), (1, 512, 50, 84), (1, 512, 25, 42), (1, 1024, 50, 84), (1, 2048, 25, 42), (1, 512, 25, 42), (1, 256, 100, 168), (1, 256, 25, 42), (1, 256, 13, 21), (1, 256, 7, 11)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (512, 512, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (512, 512, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'batch_norm': dict(
        name=["batch_norm"],
        para=dict(
            training=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 64, 400, 672), (2, 64, 200, 336), (2, 256, 200, 336), (2, 128, 200, 336), (2, 128, 100, 168), (2, 512, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 1024, 50, 84), (2, 512, 50, 84), (2, 512, 25, 42), (2, 2048, 25, 42), (1, 64, 400, 672), (1, 64, 200, 336), (1, 256, 200, 336), (1, 128, 200, 336), (1, 128, 100, 168), (1, 512, 100, 168), (1, 256, 100, 168), (1, 256, 50, 84), (1, 1024, 50, 84), (1, 512, 50, 84), (1, 512, 25, 42), (1, 2048, 25, 42)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,), (64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,), (64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,), (64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,), (64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        para=dict(
            inplace=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 64, 400, 672), (2, 64, 200, 336), (2, 256, 200, 336), (2, 128, 200, 336), (2, 128, 100, 168), (2, 512, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 1024, 50, 84), (2, 512, 50, 84), (2, 512, 25, 42), (2, 2048, 25, 42), (2, 256, 25, 42), (2, 256, 13, 21), (2, 256, 7, 11), (1, 64, 400, 672), (1, 64, 200, 336), (1, 256, 200, 336), (1, 128, 200, 336), (1, 128, 100, 168), (1, 512, 100, 168), (1, 256, 100, 168), (1, 256, 50, 84), (1, 1024, 50, 84), (1, 512, 50, 84), (1, 512, 25, 42), (1, 2048, 25, 42), (1, 256, 25, 42), (1, 256, 13, 21), (1, 256, 7, 11)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            ceil_mode=[False, False],
            return_indices=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 64, 400, 672), (1, 64, 400, 672)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add': dict(
        name=["add"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 200, 336), (2, 512, 100, 168), (2, 1024, 50, 84), (2, 2048, 25, 42), (1, 256, 200, 336), (1, 512, 100, 168), (1, 1024, 50, 84), (1, 2048, 25, 42)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 256, 200, 336), (2, 512, 100, 168), (2, 1024, 50, 84), (2, 2048, 25, 42), (1, 256, 200, 336), (1, 512, 100, 168), (1, 1024, 50, 84), (1, 2048, 25, 42)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'conv2d_1': dict(
        name=["conv2d"],
        para=dict(
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 512, 100, 168), (2, 1024, 50, 84), (2, 2048, 25, 42), (2, 256, 100, 168), (2, 256, 50, 84), (2, 256, 25, 42), (2, 256, 25, 42), (2, 256, 13, 21), (2, 256, 100, 168), (2, 256, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 256, 50, 84), (2, 256, 50, 84), (2, 256, 25, 42), (2, 256, 25, 42), (2, 256, 25, 42), (2, 256, 13, 21), (2, 256, 13, 21), (2, 256, 13, 21), (2, 256, 7, 11), (2, 256, 7, 11), (2, 256, 7, 11), (1, 512, 100, 168), (1, 1024, 50, 84), (1, 2048, 25, 42), (1, 256, 100, 168), (1, 256, 50, 84), (1, 256, 25, 42), (1, 256, 25, 42), (1, 256, 13, 21), (1, 256, 100, 168), (1, 256, 100, 168), (1, 256, 100, 168), (1, 256, 50, 84), (1, 256, 50, 84), (1, 256, 50, 84), (1, 256, 25, 42), (1, 256, 25, 42), (1, 256, 25, 42), (1, 256, 13, 21), (1, 256, 13, 21), (1, 256, 13, 21), (1, 256, 7, 11), (1, 256, 7, 11), (1, 256, 7, 11)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 512, 1, 1), (256, 1024, 1, 1), (256, 2048, 1, 1), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (80, 256, 3, 3), (4, 256, 3, 3), (1, 256, 3, 3), (80, 256, 3, 3), (4, 256, 3, 3), (1, 256, 3, 3), (80, 256, 3, 3), (4, 256, 3, 3), (1, 256, 3, 3), (80, 256, 3, 3), (4, 256, 3, 3), (1, 256, 3, 3), (80, 256, 3, 3), (4, 256, 3, 3), (1, 256, 3, 3), (256, 512, 1, 1), (256, 1024, 1, 1), (256, 2048, 1, 1), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (80, 256, 3, 3), (4, 256, 3, 3), (1, 256, 3, 3), (80, 256, 3, 3), (4, 256, 3, 3), (1, 256, 3, 3), (80, 256, 3, 3), (4, 256, 3, 3), (1, 256, 3, 3), (80, 256, 3, 3), (4, 256, 3, 3), (1, 256, 3, 3), (80, 256, 3, 3), (4, 256, 3, 3), (1, 256, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (80,), (4,), (1,), (80,), (4,), (1,), (80,), (4,), (1,), (80,), (4,), (1,), (80,), (4,), (1,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (80,), (4,), (1,), (80,), (4,), (1,), (80,), (4,), (1,), (80,), (4,), (1,), (80,), (4,), (1,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        para=dict(
            size=[(50, 84), (100, 168), (50, 84), (100, 168)],
            mode=['nearest', 'nearest', 'nearest', 'nearest'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 25, 42), (2, 256, 50, 84), (1, 256, 25, 42), (1, 256, 50, 84)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_1': dict(
        name=["add"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 50, 84), (2, 256, 100, 168), (1, 1, 4), (1, 1, 4), (1, 1, 4), (1, 1, 4), (1, 1, 4), (22400, 1), (17,), (22400,), (22400, 1), (26,), (176,), (176, 4), (176, 2), (218,), (218, 4), (218, 2), (17, 4), (17, 2), (9,), (9, 4), (9, 2), (), (1, 256, 50, 84), (1, 256, 100, 168), (177, 4), (177, 2), (25, 4), (25, 2), (1, 4), (1, 2), (203, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 256, 50, 84), (2, 256, 100, 168), (16800, 1, 4), (4200, 1, 4), (1050, 1, 4), (273, 1, 4), (77, 1, 4), (1, 17), (17,), (22400,), (1, 26), (26,), (176,), (1, 4), (176, 2), (218,), (1, 4), (218, 2), (1, 4), (17, 2), (9,), (1, 4), (9, 2), (), (1, 256, 50, 84), (1, 256, 100, 168), (1, 4), (177, 2), (1, 4), (25, 2), (1, 4), (1, 2), (203, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'group_norm': dict(
        name=["group_norm"],
        para=dict(
            num_groups=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 256, 100, 168), (2, 256, 50, 84), (2, 256, 25, 42), (2, 256, 13, 21), (2, 256, 7, 11), (1, 256, 100, 168), (1, 256, 50, 84), (1, 256, 25, 42), (1, 256, 13, 21), (1, 256, 7, 11)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 4, 100, 168), (2, 4, 50, 84), (2, 4, 25, 42), (2, 4, 13, 21), (2, 4, 7, 11), (22400,), (17,), (22400, 17), (26,), (22400, 26), (33600, 80), (176,), (176, 4), (176, 2), (8400, 80), (218,), (218, 4), (218, 2), (2100, 80), (17, 4), (17, 2), (546, 80), (154, 80), (9,), (9, 4), (9, 2), (1, 4, 100, 168), (1, 4, 50, 84), (1, 4, 25, 42), (1, 4, 13, 21), (1, 4, 7, 11), (177, 4), (177, 2), (25, 4), (25, 2), (1, 4), (1, 2), (203,), (203,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (), (), (), (22400,), (17,), (22400, 17), (26,), (22400, 26), (33600, 1), (176,), (1, 4), (176, 2), (8400, 1), (218,), (1, 4), (218, 2), (2100, 1), (1, 4), (17, 2), (546, 1), (154, 1), (9,), (1, 4), (9, 2), (), (), (), (), (), (1, 4), (177, 2), (1, 4), (25, 2), (1, 4), (1, 2), (203,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'arange': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            start=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            end=[168, 100, 84, 50, 42, 25, 21, 13, 11, 7],
        ),
    ),

    'mul_1': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(168,), (100,), (84,), (50,), (42,), (25,), (21,), (13,), (11,), (7,), (176, 2), (218, 2), (17, 2), (), (9, 2), (177, 2), (25, 2), (1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'stack': dict(
        name=["stack"],
        interface=["torch"],
        para=dict(
            dim=[-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((16800,), (16800,), (16800,), (16800,)), ((4200,), (4200,), (4200,), (4200,)), ((1050,), (1050,), (1050,), (1050,)), ((273,), (273,), (273,), (273,)), ((77,), (77,), (77,), (77,)), ((17,), (17,)), ((22400,), (22400,)), ((45, 17), (45, 17), (45, 17), (45, 17)), ((26,), (26,)), ((45, 26), (45, 26), (45, 26), (45, 26)), ((176,), (176,)), ((218,), (218,)), ((9,), (9,))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'logical_and': dict(
        name=["logical_and"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16800,), (4200,), (1050,), (273,), (77,), (45, 17), (45, 26), (33600,), (176,), (8400,), (218,), (2100,), (17,), (546,), (154,), (9,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["other"],
                    "shape": [(16800,), (4200,), (1050,), (273,), (77,), (45, 17), (45, 26), (33600,), (176,), (8400,), (218,), (2100,), (17,), (546,), (154,), (9,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'expand': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(16800, 1), (4200, 1), (1050, 1), (273, 1), (77, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16800, 1), (4200, 1), (1050, 1), (273, 1), (77, 1)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((16800, 4), (4200, 4), (1050, 4), (273, 4), (77, 4)), ((177, 4), (25, 4), (0, 4), (1, 4), (0, 4)), ((177,), (25,), (0,), (1,), (0,))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cat_1': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((16800,), (4200,), (1050,), (273,), (77,))],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'any': dict(
        name=["any"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400,), (176,), (218,), (17,), (9,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'split': dict(
        name=["split"],
        interface=["torch"],
        para=dict(
            split_size_or_sections=[[16800, 4200, 1050, 273, 77]],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensor"],
                    "shape": [(22400,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16800,), (4200,), (1050,), (273,), (77,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'sub': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400,), (17,), (22400, 17, 2), (22400, 17), (22400, 1, 2), (45, 17), (17,), (26,), (22400, 26, 2), (22400, 26), (22400, 1, 2), (45, 26), (26,), (176,), (176, 2), (218,), (218, 2), (17, 2), (9,), (9, 2), (177, 2), (25, 2), (1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(22400,), (17,), (22400, 17, 2), (22400, 17), (1, 17, 2), (17,), (45, 17), (26,), (22400, 26, 2), (22400, 26), (1, 26, 2), (26,), (45, 26), (176,), (176, 2), (218,), (218, 2), (17, 2), (9,), (9, 2), (177, 2), (25, 2), (1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'maximum': dict(
        name=["maximum"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400, 1, 2), (22400, 17), (22400, 1, 2), (22400, 26), (176, 2), (176,), (218, 2), (218,), (17, 2), (17,), (9, 2), (9,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 17, 2), (1,), (1, 26, 2), (1,), (176, 2), (1,), (218, 2), (1,), (17, 2), (1,), (9, 2), (1,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'minimum': dict(
        name=["minimum"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400, 1, 2), (22400, 1, 2), (176, 2), (218, 2), (17, 2), (9, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 17, 2), (1, 26, 2), (176, 2), (218, 2), (17, 2), (9, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp': dict(
        name=["clamp"],
        interface=["torch.Tensor"],
        para=dict(
            min=[0, 0, 0, 0, 0, 0],
            max=[None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400, 17, 2), (22400, 26, 2), (176, 2), (218, 2), (17, 2), (9, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div': dict(
        name=["div"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400, 17), (22400, 26), (176,), (218,), (17,), (9,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(22400, 17), (22400, 26), (176,), (218,), (17,), (9,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_1': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[2.0, 2.0, 2.0, 420.0000001192093, 2, 1.0000001192092896, 2, 2, 275.7549743652344],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(17,), (22400,), (26,), (), (176,), (), (218,), (9,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'pow': dict(
        name=["pow"],
        interface=["torch.Tensor"],
        para=dict(
            exponent=[2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400, 17, 2), (22400, 26, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sum_1': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[-1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400, 17, 2), (22400, 26, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sqrt': dict(
        name=["sqrt"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400, 17), (22400, 26), (176,), (218,), (17,), (9,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'topk': dict(
        name=["topk"],
        interface=["torch.Tensor"],
        para=dict(
            k=[9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            dim=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            largest=[False, False, False, False, False, False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16800, 17), (4200, 17), (1050, 17), (273, 17), (77, 17), (16800, 26), (4200, 26), (1050, 26), (273, 26), (77, 26)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_2': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 16800, 21000, 22050, 22323, 1, 0, 16800, 21000, 22050, 22323, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(9, 17), (9, 17), (9, 17), (9, 17), (9, 17), (166,), (9, 26), (9, 26), (9, 26), (9, 26), (9, 26), (254,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'cat_2': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((9, 17), (9, 17), (9, 17), (9, 17), (9, 17)), ((9, 26), (9, 26), (9, 26), (9, 26), (9, 26))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'arange_1': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            end=[17, 26],
        ),
    ),

    'mean': dict(
        name=["mean"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(45, 17), (45, 26)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'std': dict(
        name=["std"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(45, 17), (45, 26)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'ge': dict(
        name=["ge"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(45, 17), (45, 26)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 17), (1, 26)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_3': dict(
        name=["add"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 22400, 44800, 67200, 89600, 112000, 134400, 156800, 179200, 201600, 224000, 246400, 268800, 291200, 313600, 336000, 358400, 380800, 403200, 425600, 448000, 470400, 492800, 515200, 537600, 560000],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,), (45,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'expand_1': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(17, 22400), (26, 22400)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 22400), (1, 22400)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'min': dict(
        name=["min"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[1, 1, -1, -1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(45, 4, 17), (45, 4, 26), (176, 2), (218, 2), (17, 2), (9, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'gt': dict(
        name=["gt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.01, 0.01, 0, 0, 0, 0, 0.05, 0.05, 0.05, 0.05, 0.05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(45, 17), (45, 26), (176,), (218,), (17,), (9,), (16800, 80), (4200, 80), (1050, 80), (273, 80), (77, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'transpose': dict(
        name=["transpose"],
        interface=["torch.Tensor"],
        para=dict(
            dim0=[0 for i in range(4)],
            dim1=[1 for i in range(4)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400, 17), (17, 22400), (22400, 26), (26, 22400)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max': dict(
        name=["max"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[1, 1, -1, -1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400, 17), (22400, 26), (176, 2), (218, 2), (17, 2), (9, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'ne': dict(
        name=["ne"],
        interface=["torch.Tensor"],
        para=dict(
            other=[-100000000, -100, -100, -100, -100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400,), (176,), (218,), (17,), (9,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'gt_1': dict(
        name=["gt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'nonzero': dict(
        name=["nonzero"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'sub_1': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(166,), (254,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'unique': dict(
        name=["unique"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(166,), (22234,), (254,), (22146,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'eq': dict(
        name=["eq"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(22400,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'stack_1': dict(
        name=["stack"],
        interface=["torch"],
        para=dict(
            dim=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((22400, 4), (22400, 4)), ((22400,), (22400,))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'stack_2': dict(
        name=["stack"],
        interface=["torch"],
        para=dict(
            dim=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((22400,), (22400,))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'permute': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[(0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 80, 100, 168), (2, 4, 100, 168), (2, 1, 100, 168), (2, 80, 50, 84), (2, 4, 50, 84), (2, 1, 50, 84), (2, 80, 25, 42), (2, 4, 25, 42), (2, 1, 25, 42), (2, 80, 13, 21), (2, 4, 13, 21), (2, 1, 13, 21), (2, 80, 7, 11), (2, 4, 7, 11), (2, 1, 7, 11)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sum_2': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(33600, 80), (176,), (8400, 80), (218,), (2100, 80), (17,), (546, 80), (546, 4), (546,), (), (154, 80), (9,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_2': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1.0, 2.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'ge_1': dict(
        name=["ge"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(33600,), (8400,), (2100,), (546,), (154,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'lt': dict(
        name=["lt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[80, 80, 80, 80, 80],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(33600,), (8400,), (2100,), (546,), (154,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'nonzero_1': dict(
        name=["nonzero"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(33600,), (8400,), (2100,), (546,), (154,), (16800, 80), (4200, 80), (1050, 80), (273, 80), (77, 80)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'clamp_1': dict(
        name=["clamp"],
        interface=["torch.Tensor"],
        para=dict(
            min=[-4.135166556742356, -4.135166556742356, -4.135166556742356, -4.135166556742356, -4.135166556742356, -4.135166556742356, -4.135166556742356],
            max=[4.135166556742356, 4.135166556742356, 4.135166556742356, 4.135166556742356, 4.135166556742356, 4.135166556742356, 4.135166556742356],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(176, 2), (218, 2), (17, 2), (9, 2), (177, 2), (25, 2), (1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'exp': dict(
        name=["exp"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(176, 2), (218, 2), (17, 2), (9, 2), (177, 2), (25, 2), (1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat_3': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[-1, -1, -1, -1, -1, -1, -1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((176, 2), (176, 2)), ((218, 2), (218, 2)), ((17, 2), (17, 2)), ((9, 2), (9, 2)), ((177, 2), (177, 2)), ((25, 2), (25, 2)), ((1, 2), (1, 2)), ((80, 4), (80, 1))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'sub_2': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(176,), (218,), (17,), (9,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'ge_2': dict(
        name=["ge"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(176,), (218,), (17,), (9,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'binary_cross_entropy_with_logits': dict(
        name=["binary_cross_entropy_with_logits"],
        para=dict(
            reduction=['none', 'none', 'none', 'none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(176,), (218,), (17,), (9,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(176,), (218,), (17,), (9,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_4': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_2': dict(
        name=["clamp"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        para=dict(
            min=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mean_1': dict(
        name=["mean"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_3': dict(
        name=["mul"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(154, 80), (546, 80), (2100, 80), (8400, 80), (33600, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(154, 80), (546, 80), (2100, 80), (8400, 80), (33600, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sgd': dict(
        name=["sgd"],
        interface=["CustomizedTest"],
        para=dict(
            nesterov=[False for i in range(28)],
            lr=[1.000000000000001e-05 for i in range(28)],
            momentum=[0.9 for i in range(28)],
            weight_decay=[0.0001 for i in range(28)],
            dampening=[0 for i in range(28)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256,), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (256, 2048, 1, 1), (80, 256, 3, 3), (80,), (4, 256, 3, 3), (4,), (1, 256, 3, 3), (1,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256,), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (256, 2048, 1, 1), (80, 256, 3, 3), (80,), (4, 256, 3, 3), (4,), (1, 256, 3, 3), (1,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'permute_1': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[(1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4, 100, 168), (1, 100, 168), (80, 100, 168), (4, 50, 84), (1, 50, 84), (80, 50, 84), (4, 25, 42), (1, 25, 42), (80, 25, 42), (4, 13, 21), (1, 13, 21), (80, 13, 21), (4, 7, 11), (1, 7, 11), (80, 7, 11)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sigmoid': dict(
        name=["sigmoid"],
        interface=["torch.Tensor"],
        saved_args=dict(output=0),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16800,), (16800, 80), (4200,), (4200, 80), (1050,), (1050, 80), (273,), (273, 80), (77,), (77, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sort': dict(
        name=["sort"],
        interface=["torch.Tensor"],
        para=dict(
            descending=[True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(177,), (25,), (0,), (1,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_3': dict(
        name=["clamp"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        para=dict(
            min=[0, 0, 0, 0, 0, 0],
            max=[1333, 800, 1333, 800, 1333, 800],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(177, 2), (177, 2), (25, 2), (25, 2), (1, 2), (1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_2': dict(
        name=["div"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(203, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(4,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat_4': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((177,), (25,), (0,), (1,), (0,))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'max_1': dict(
        name=["max"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(203, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat_5': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((80, 4), (80, 1))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

}
