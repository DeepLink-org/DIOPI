from ...config import Genfunc
from ...dtype import Dtype

solo_config = {
    'conv2d': dict(
        name=["conv2d"],
        para=dict(
            bias=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            stride=[(2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(3, 3), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (3, 3), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 3, 800, 1344), (2, 64, 200, 336), (2, 64, 200, 336), (2, 64, 200, 336), (2, 256, 200, 336), (2, 256, 200, 336), (2, 128, 200, 336), (2, 128, 100, 168), (2, 256, 200, 336), (2, 512, 100, 168), (2, 128, 100, 168), (2, 512, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 512, 100, 168), (2, 1024, 50, 84), (2, 256, 50, 84), (2, 1024, 50, 84), (2, 512, 50, 84), (2, 512, 25, 42), (2, 1024, 50, 84), (2, 2048, 25, 42), (2, 512, 25, 42), (2, 258, 100, 168), (2, 256, 100, 168), (2, 256, 40, 40), (2, 256, 36, 36), (2, 258, 50, 84), (2, 256, 24, 24), (2, 258, 25, 42), (2, 256, 25, 42), (2, 256, 16, 16), (2, 256, 12, 12), (1, 3, 800, 1344), (1, 64, 200, 336), (1, 64, 200, 336), (1, 64, 200, 336), (1, 256, 200, 336), (1, 256, 200, 336), (1, 128, 200, 336), (1, 128, 100, 168), (1, 256, 200, 336), (1, 512, 100, 168), (1, 128, 100, 168), (1, 512, 100, 168), (1, 256, 100, 168), (1, 256, 50, 84), (1, 512, 100, 168), (1, 1024, 50, 84), (1, 256, 50, 84), (1, 1024, 50, 84), (1, 512, 50, 84), (1, 512, 25, 42), (1, 1024, 50, 84), (1, 2048, 25, 42), (1, 512, 25, 42), (1, 258, 100, 168), (1, 256, 100, 168), (1, 256, 40, 40), (1, 256, 36, 36), (1, 258, 50, 84), (1, 256, 24, 24), (1, 258, 25, 42), (1, 256, 25, 42), (1, 256, 16, 16), (1, 256, 12, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (512, 512, 3, 3), (256, 258, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 258, 3, 3), (256, 256, 3, 3), (256, 258, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (512, 512, 3, 3), (256, 258, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 258, 3, 3), (256, 256, 3, 3), (256, 258, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3)],
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
            inplace=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 64, 400, 672), (2, 64, 200, 336), (2, 256, 200, 336), (2, 128, 200, 336), (2, 128, 100, 168), (2, 512, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 1024, 50, 84), (2, 512, 50, 84), (2, 512, 25, 42), (2, 2048, 25, 42), (2, 256, 40, 40), (2, 256, 36, 36), (2, 256, 24, 24), (2, 256, 25, 42), (2, 256, 16, 16), (2, 256, 12, 12), (1, 64, 400, 672), (1, 64, 200, 336), (1, 256, 200, 336), (1, 128, 200, 336), (1, 128, 100, 168), (1, 512, 100, 168), (1, 256, 100, 168), (1, 256, 50, 84), (1, 1024, 50, 84), (1, 512, 50, 84), (1, 512, 25, 42), (1, 2048, 25, 42), (1, 256, 40, 40), (1, 256, 36, 36), (1, 256, 24, 24), (1, 256, 25, 42), (1, 256, 16, 16), (1, 256, 12, 12)],
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
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 256, 200, 336), (2, 512, 100, 168), (2, 1024, 50, 84), (2, 2048, 25, 42), (2, 256, 200, 336), (2, 256, 100, 168), (2, 256, 50, 84), (2, 256, 25, 42), (2, 256, 200, 336), (2, 256, 40, 40), (2, 256, 200, 336), (2, 256, 36, 36), (2, 256, 100, 168), (2, 256, 24, 24), (2, 256, 50, 84), (2, 256, 16, 16), (2, 256, 50, 84), (2, 256, 12, 12), (1, 256, 200, 336), (1, 512, 100, 168), (1, 1024, 50, 84), (1, 2048, 25, 42), (1, 256, 200, 336), (1, 256, 100, 168), (1, 256, 50, 84), (1, 256, 25, 42), (1, 256, 200, 336), (1, 256, 40, 40), (1, 256, 200, 336), (1, 256, 36, 36), (1, 256, 100, 168), (1, 256, 24, 24), (1, 256, 50, 84), (1, 256, 16, 16), (1, 256, 50, 84), (1, 256, 12, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 256, 1, 1), (256, 512, 1, 1), (256, 1024, 1, 1), (256, 2048, 1, 1), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (1600, 256, 1, 1), (80, 256, 3, 3), (1296, 256, 1, 1), (80, 256, 3, 3), (576, 256, 1, 1), (80, 256, 3, 3), (256, 256, 1, 1), (80, 256, 3, 3), (144, 256, 1, 1), (80, 256, 3, 3), (256, 256, 1, 1), (256, 512, 1, 1), (256, 1024, 1, 1), (256, 2048, 1, 1), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (1600, 256, 1, 1), (80, 256, 3, 3), (1296, 256, 1, 1), (80, 256, 3, 3), (576, 256, 1, 1), (80, 256, 3, 3), (256, 256, 1, 1), (80, 256, 3, 3), (144, 256, 1, 1), (80, 256, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (1600,), (80,), (1296,), (80,), (576,), (80,), (256,), (80,), (144,), (80,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (1600,), (80,), (1296,), (80,), (576,), (80,), (256,), (80,), (144,), (80,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        para=dict(
            size=[(50, 84), (100, 168), (200, 336), 40, 36, 24, 16, 12, (50, 84), (100, 168), (200, 336), 40, (200, 336), 36, (200, 336), 24, (200, 336), 16, (200, 336), 12, (200, 336)],
            mode=['nearest', 'nearest', 'nearest', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest', 'nearest', 'nearest', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 25, 42), (2, 256, 50, 84), (2, 256, 100, 168), (2, 256, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 256, 25, 42), (2, 256, 25, 42), (1, 256, 25, 42), (1, 256, 50, 84), (1, 256, 100, 168), (1, 256, 100, 168), (1, 1600, 200, 336), (1, 256, 100, 168), (1, 1296, 200, 336), (1, 256, 50, 84), (1, 576, 100, 168), (1, 256, 25, 42), (1, 256, 50, 84), (1, 256, 25, 42), (1, 144, 50, 84)],
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
                    "shape": [(2, 256, 50, 84), (2, 256, 100, 168), (2, 256, 200, 336), (), (7,), (30,), (26,), (22,), (16,), (1, 256, 50, 84), (1, 256, 100, 168), (1, 256, 200, 336)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 256, 50, 84), (2, 256, 100, 168), (2, 256, 200, 336), (), (7,), (30,), (26,), (22,), (16,), (1, 256, 50, 84), (1, 256, 100, 168), (1, 256, 200, 336)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d_1': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[1, 1],
            stride=[2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 256, 25, 42), (1, 256, 25, 42)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'interpolate_1': dict(
        name=["interpolate"],
        para=dict(
            size=[(100, 168), (25, 42), (100, 168), (25, 42)],
            mode=['bilinear', 'bilinear', 'bilinear', 'bilinear'],
            align_corners=[False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 200, 336), (2, 256, 13, 21), (1, 256, 200, 336), (1, 256, 13, 21)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'linspace': dict(
        name=["linspace"],
        interface=["torch"],
        para=dict(
            start=[-1, -1, -1, -1, -1, -1],
            end=[1, 1, 1, 1, 1, 1],
            steps=[168, 100, 84, 50, 42, 25],
        ),
    ),

    'expand': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[[2, 1, -1, -1], [2, 1, -1, -1], [2, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1], [1, 1, -1, -1]],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100, 168), (50, 84), (25, 42), (100, 168), (50, 84), (25, 42)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((2, 1, 100, 168), (2, 1, 100, 168)), ((2, 256, 100, 168), (2, 2, 100, 168)), ((2, 1, 50, 84), (2, 1, 50, 84)), ((2, 256, 50, 84), (2, 2, 50, 84)), ((2, 1, 25, 42), (2, 1, 25, 42)), ((2, 256, 25, 42), (2, 2, 25, 42)), ((1, 1, 100, 168), (1, 1, 100, 168)), ((1, 256, 100, 168), (1, 2, 100, 168)), ((1, 1, 50, 84), (1, 1, 50, 84)), ((1, 256, 50, 84), (1, 2, 50, 84)), ((1, 1, 25, 42), (1, 1, 25, 42)), ((1, 256, 25, 42), (1, 2, 25, 42))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'group_norm': dict(
        name=["group_norm"],
        para=dict(
            num_groups=[32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 256, 100, 168), (2, 256, 40, 40), (2, 256, 36, 36), (2, 256, 50, 84), (2, 256, 24, 24), (2, 256, 25, 42), (2, 256, 16, 16), (2, 256, 12, 12), (1, 256, 100, 168), (1, 256, 40, 40), (1, 256, 36, 36), (1, 256, 50, 84), (1, 256, 24, 24), (1, 256, 25, 42), (1, 256, 16, 16), (1, 256, 12, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'interpolate_2': dict(
        name=["interpolate"],
        para=dict(
            scale_factor=[2, 2, 2, 2, 2, 2],
            mode=['bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 100, 168), (2, 256, 50, 84), (2, 256, 25, 42), (1, 256, 100, 168), (1, 256, 50, 84), (1, 256, 25, 42)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(7,), (3,), (), (4,), (6,), (2,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(7,), (3,), (), (4,), (6,), (2,)],
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
                    "shape": [(7,), (6,), (7, 67200), (30, 67200), (26, 16800), (22, 4200), (16, 4200)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(7,), (6,), (7, 67200), (30, 67200), (26, 16800), (22, 4200), (16, 4200)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sqrt': dict(
        name=["sqrt"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7,), (6,)],
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
            other=[80, 80, 80, 80, 80, 1, 1.1920928955078125e-07],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(40, 40), (36, 36), (24, 24), (16, 16), (12, 12), (), ()],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'ge': dict(
        name=["ge"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1, 48, 96, 192, 384, 1, 48, 96, 192, 384],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7,), (7,), (7,), (7,), (7,), (6,), (6,), (6,), (6,), (6,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'le': dict(
        name=["le"],
        interface=["torch.Tensor"],
        para=dict(
            other=[96, 192, 384, 768, 2048, 96, 192, 384, 768, 2048],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7,), (7,), (7,), (7,), (7,), (6,), (6,), (6,), (6,), (6,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'logical_and': dict(
        name=["logical_and"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7,), (6,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["other"],
                    "shape": [(7,), (6,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'nonzero': dict(
        name=["nonzero"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7,), (6,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'mul_1': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.5, 0.5, 0.5, 0.5, 2, 3.0, 2, 3.0, 2, 3.0, 2, 3.0, 2, 3.0, 1.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3,), (4,), (2,), (6,), (7,), (7,), (30,), (30,), (26,), (26,), (22,), (22,), (16,), (16,), ()],
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
            other=[0.2, 0.2, 0.2, 0.2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3,), (4,), (2,), (6,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[-1, -1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 800, 1344), (4, 800, 1344), (2, 800, 1344), (6, 800, 1344)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'sum_1': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[-1, -1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 800), (4, 800), (2, 800), (6, 800)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'gt': dict(
        name=["gt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3,), (4,), (2,), (6,), ()],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'arange': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            end=[800, 1344],
        ),
    ),

    'sum_2': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(800, 1344), (3200,), (2592,), (1152,), (512,), (288,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'clamp': dict(
        name=["clamp"],
        interface=["torch.Tensor"],
        para=dict(
            min=[1e-06],
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

    'mul_3': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(800, 1344), (800, 1344)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["other"],
                    "shape": [(800, 1), (1344,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'sum_3': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(800, 1344)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
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
                    "shape": [()],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
                {
                    "ins": ["other"],
                    "shape": [()],
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
            other=[1344, 800],
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

    'div_2': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.027777777777777776, 0.041666666666666664, 0.0625, 0.08333333333333333, 0.025],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_3': dict(
        name=["div"],
        interface=["torch"],
        para=dict(
            other=[0.027777777777777776, 0.041666666666666664, 0.0625, 0.08333333333333333, 0.025],
            rounding_mode=['trunc', 'trunc', 'trunc', 'trunc', 'trunc'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat_1': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((0, 200, 336), (7, 200, 336)), ((10, 200, 336), (20, 200, 336)), ((14, 100, 168), (12, 100, 168)), ((22, 50, 84), (0, 50, 84)), ((16, 50, 84), (0, 50, 84))],
                    "dtype": [Dtype.uint8],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cat_2': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((0, 200, 336), (7, 200, 336)), ((10, 200, 336), (20, 200, 336)), ((14, 100, 168), (12, 100, 168)), ((22, 50, 84), (0, 50, 84)), ((16, 50, 84), (0, 50, 84)), ((1600, 80), (1296, 80), (576, 80), (256, 80), (144, 80)), ((1600, 200, 336), (1296, 200, 336), (576, 200, 336), (256, 200, 336), (144, 200, 336))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cat_3': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((1600,), (1600,)), ((1296,), (1296,)), ((576,), (576,)), ((256,), (256,)), ((144,), (144,))],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cat_4': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((1600,), (1600,)), ((1296,), (1296,)), ((576,), (576,)), ((256,), (256,)), ((144,), (144,))],
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
            dims=[(0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 80, 40, 40), (2, 80, 36, 36), (2, 80, 24, 24), (2, 80, 16, 16), (2, 80, 12, 12), (1, 80, 40, 40), (1, 80, 36, 36), (1, 80, 24, 24), (1, 80, 16, 16), (1, 80, 12, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_3': dict(
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
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'add_4': dict(
        name=["add"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
                {
                    "ins": ["other"],
                    "shape": [()],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
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
                    "shape": [(7, 200, 336), (30, 200, 336), (26, 100, 168), (22, 50, 84), (16, 50, 84), (1, 1600, 200, 336), (1, 80, 40, 40), (1, 1296, 200, 336), (1, 80, 36, 36), (1, 576, 100, 168), (1, 80, 24, 24), (1, 256, 50, 84), (1, 80, 16, 16), (1, 144, 50, 84), (1, 80, 12, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sum_4': dict(
        name=["sum"],
        interface=["torch"],
        para=dict(
            dim=[1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7, 67200), (30, 67200), (26, 16800), (22, 4200), (16, 4200)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_5': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.001, 0.001, 0.001, 0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7,), (30,), (26,), (22,), (16,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_4': dict(
        name=["div"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7,), (30,), (26,), (22,), (16,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(7,), (30,), (26,), (22,), (16,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sub_1': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7,), (30,), (26,), (22,), (16,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat_5': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((7,), (30,), (26,), (22,), (16,)), ((3200, 80), (2592, 80), (1152, 80), (512, 80), (288, 80))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'sum_5': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(101,), (7744, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_5': dict(
        name=["div"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [()],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'cat_6': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((3200,), (2592,), (1152,), (512,), (288,))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'mean': dict(
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

    'add_6': dict(
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

    'mul_4': dict(
        name=["mul"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(7744, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(7744, 80)],
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
            nesterov=[False for i in range(33)],
            lr=[1.000000000000001e-05 for i in range(33)],
            momentum=[0.9 for i in range(33)],
            weight_decay=[0.0001 for i in range(33)],
            dampening=[0 for i in range(33)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256,), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (256, 256, 1, 1), (256, 2048, 1, 1), (256, 258, 3, 3), (1600, 256, 1, 1), (1600,), (1296, 256, 1, 1), (1296,), (576, 256, 1, 1), (576,), (144, 256, 1, 1), (144,), (80, 256, 3, 3), (80,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256,), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (256, 256, 1, 1), (256, 2048, 1, 1), (256, 258, 3, 3), (1600, 256, 1, 1), (1600,), (1296, 256, 1, 1), (1296,), (576, 256, 1, 1), (576,), (144, 256, 1, 1), (144,), (80, 256, 3, 3), (80,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d_2': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[2, 2, 2, 2, 2],
            stride=[1, 1, 1, 1, 1],
            padding=[1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 80, 40, 40), (1, 80, 36, 36), (1, 80, 24, 24), (1, 80, 16, 16), (1, 80, 12, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'eq': dict(
        name=["eq"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 80, 40, 40), (1, 80, 36, 36), (1, 80, 24, 24), (1, 80, 16, 16), (1, 80, 12, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 80, 40, 40), (1, 80, 36, 36), (1, 80, 24, 24), (1, 80, 16, 16), (1, 80, 12, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_5': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 80, 40, 40), (1, 80, 36, 36), (1, 80, 24, 24), (1, 80, 16, 16), (1, 80, 12, 12)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 80, 40, 40), (1, 80, 36, 36), (1, 80, 24, 24), (1, 80, 16, 16), (1, 80, 12, 12)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'gt_1': dict(
        name=["gt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3872, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
