from ...config import Genfunc
from ...dtype import Dtype

faster_rcnn_r50_config = {
    'conv2d': dict(
        name=["conv2d"],
        para=dict(
            bias=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            stride=[(2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
            padding=[(3, 3), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (3, 3), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 3, 800, 1344), (2, 64, 200, 336), (2, 64, 200, 336), (2, 64, 200, 336), (2, 256, 200, 336), (2, 256, 200, 336), (2, 128, 200, 336), (2, 128, 100, 168), (2, 256, 200, 336), (2, 512, 100, 168), (2, 128, 100, 168), (2, 512, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 512, 100, 168), (2, 1024, 50, 84), (2, 256, 50, 84), (2, 1024, 50, 84), (2, 512, 50, 84), (2, 512, 25, 42), (2, 1024, 50, 84), (2, 2048, 25, 42), (2, 512, 25, 42), (1, 3, 800, 1344), (1, 64, 200, 336), (1, 64, 200, 336), (1, 64, 200, 336), (1, 256, 200, 336), (1, 256, 200, 336), (1, 128, 200, 336), (1, 128, 100, 168), (1, 256, 200, 336), (1, 512, 100, 168), (1, 128, 100, 168), (1, 512, 100, 168), (1, 256, 100, 168), (1, 256, 50, 84), (1, 512, 100, 168), (1, 1024, 50, 84), (1, 256, 50, 84), (1, 1024, 50, 84), (1, 512, 50, 84), (1, 512, 25, 42), (1, 1024, 50, 84), (1, 2048, 25, 42), (1, 512, 25, 42)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (512, 512, 3, 3), (64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (512, 512, 3, 3)],
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
                    "shape": [(2, 64, 400, 672), (2, 64, 200, 336), (2, 256, 200, 336), (2, 128, 200, 336), (2, 128, 100, 168), (2, 512, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 1024, 50, 84), (2, 512, 50, 84), (2, 512, 25, 42), (2, 2048, 25, 42), (2, 256, 25, 42), (2, 256, 13, 21), (1024, 1024), (1, 64, 400, 672), (1, 64, 200, 336), (1, 256, 200, 336), (1, 128, 200, 336), (1, 128, 100, 168), (1, 512, 100, 168), (1, 256, 100, 168), (1, 256, 50, 84), (1, 1024, 50, 84), (1, 512, 50, 84), (1, 512, 25, 42), (1, 2048, 25, 42), (1, 256, 25, 42), (1, 256, 13, 21), (1000, 1024)],
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
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 256, 200, 336), (2, 512, 100, 168), (2, 1024, 50, 84), (2, 2048, 25, 42), (2, 256, 200, 336), (2, 256, 100, 168), (2, 256, 50, 84), (2, 256, 25, 42), (2, 256, 200, 336), (2, 256, 200, 336), (2, 256, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 256, 50, 84), (2, 256, 25, 42), (2, 256, 25, 42), (2, 256, 13, 21), (2, 256, 13, 21), (2, 256, 13, 21), (1, 256, 200, 336), (1, 512, 100, 168), (1, 1024, 50, 84), (1, 2048, 25, 42), (1, 256, 200, 336), (1, 256, 100, 168), (1, 256, 50, 84), (1, 256, 25, 42), (1, 256, 200, 336), (1, 256, 200, 336), (1, 256, 100, 168), (1, 256, 100, 168), (1, 256, 50, 84), (1, 256, 50, 84), (1, 256, 25, 42), (1, 256, 25, 42), (1, 256, 13, 21), (1, 256, 13, 21), (1, 256, 13, 21)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 256, 1, 1), (256, 512, 1, 1), (256, 1024, 1, 1), (256, 2048, 1, 1), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (3, 256, 1, 1), (12, 256, 1, 1), (3, 256, 1, 1), (12, 256, 1, 1), (3, 256, 1, 1), (12, 256, 1, 1), (3, 256, 1, 1), (12, 256, 1, 1), (256, 256, 3, 3), (3, 256, 1, 1), (12, 256, 1, 1), (256, 256, 1, 1), (256, 512, 1, 1), (256, 1024, 1, 1), (256, 2048, 1, 1), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (3, 256, 1, 1), (12, 256, 1, 1), (3, 256, 1, 1), (12, 256, 1, 1), (3, 256, 1, 1), (12, 256, 1, 1), (3, 256, 1, 1), (12, 256, 1, 1), (256, 256, 3, 3), (3, 256, 1, 1), (12, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (3,), (12,), (3,), (12,), (3,), (12,), (3,), (12,), (256,), (3,), (12,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (3,), (12,), (3,), (12,), (3,), (12,), (3,), (12,), (256,), (3,), (12,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        para=dict(
            size=[(50, 84), (100, 168), (200, 336), (50, 84), (100, 168), (200, 336)],
            mode=['nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 256, 25, 42), (2, 256, 50, 84), (2, 256, 100, 168), (1, 256, 25, 42), (1, 256, 50, 84), (1, 256, 100, 168)],
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
                    "shape": [(2, 256, 50, 84), (2, 256, 100, 168), (2, 256, 200, 336), (1, 3, 4), (1, 3, 4), (1, 3, 4), (1, 3, 4), (1, 3, 4), (2, 1), (21,), (10, 1), (48,), (8819, 4), (8819, 2), (), (8819, 4), (2, 1), (10, 1), (6,), (12,), (1, 256, 50, 84), (1, 256, 100, 168), (1, 256, 200, 336), (4819, 4), (4819, 2), (4819, 4), (80000, 4), (80000, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 256, 50, 84), (2, 256, 100, 168), (2, 256, 200, 336), (67200, 1, 4), (16800, 1, 4), (4200, 1, 4), (1050, 1, 4), (273, 1, 4), (1, 268569), (21,), (1, 268569), (48,), (1, 4), (8819, 2), (), (8819, 1), (1, 1000), (1, 1000), (6,), (12,), (1, 256, 50, 84), (1, 256, 100, 168), (1, 256, 200, 336), (1, 4), (4819, 2), (4819, 1), (1, 4), (80000, 2)],
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

    'arange': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            start=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            end=[336, 200, 168, 100, 84, 50, 42, 25, 21, 13],
        ),
    ),

    'mul': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[4, 4, 8, 8, 16, 16, 32, 32, 64, 64, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(336,), (200,), (168,), (100,), (84,), (50,), (42,), (25,), (21,), (13,), (21,), (48,), (8819, 2), (6,), (12,), (4819, 2), (80000, 2)],
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
            dim=[-1, -1, -1, -1, -1, -1, -1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((67200,), (67200,), (67200,), (67200,)), ((16800,), (16800,), (16800,), (16800,)), ((4200,), (4200,), (4200,), (4200,)), ((1050,), (1050,), (1050,), (1050,)), ((273,), (273,), (273,), (273,)), ((21,), (21,), (21,), (21,)), ((48,), (48,), (48,), (48,)), ((6,), (6,), (6,), (6,)), ((12,), (12,), (12,), (12,))],
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
                    "shape": [(67200,), (16800,), (4200,), (1050,), (273,), (268569,), (403200,), (100800,), (25200,), (6300,), (1638,), (8819,), (1000,), (1024,), (4819,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["other"],
                    "shape": [(67200,), (16800,), (4200,), (1050,), (273,), (268569,), (403200,), (100800,), (25200,), (6300,), (1638,), (8819,), (1000,), (1024,), (4819,)],
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
            size=[(67200, 3), (16800, 3), (4200, 3), (1050, 3), (273, 3), (403200, 1), (100800, 1), (25200, 1), (6300, 1), (1638, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(67200, 1), (16800, 1), (4200, 1), (1050, 1), (273, 1), (403200, 1), (100800, 1), (25200, 1), (6300, 1), (1638, 1)],
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
                    "shape": [((201600, 4), (50400, 4), (12600, 4), (3150, 4), (819, 4)), ((2000,), (2000,), (2000,), (2000,), (819,)), ((2000, 4), (2000, 4), (2000, 4), (2000, 4), (819, 4)), ((2,), (1000,)), ((10,), (1000,)), ((6, 4), (506, 4)), ((12, 4), (500, 4)), ((1000,), (1000,), (1000,), (1000,), (819,)), ((1000, 4), (1000, 4), (1000, 4), (1000, 4), (819, 4))],
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
                    "shape": [((201600,), (50400,), (12600,), (3150,), (819,))],
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
                    "shape": [(268569,), (1024,)],
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
                    "shape": [(2,), (268569,), (2, 268569, 2), (2, 268569), (21,), (10,), (10, 268569, 2), (10, 268569), (48,), (403200, 4), (100800, 4), (25200, 4), (6300, 4), (1638, 4), (8819, 2), (8819,), (1000,), (2, 1000, 2), (2, 1000), (10, 1000, 2), (10, 1000), (1024,), (6,), (12,), (18, 4), (4819, 2), (4819,), (80000, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2,), (268569,), (2, 268569, 2), (2, 268569), (21,), (10,), (10, 268569, 2), (10, 268569), (48,), (403200, 4), (100800, 4), (25200, 4), (6300, 4), (1638, 4), (8819, 2), (8819,), (1000,), (2, 1000, 2), (2, 1000), (10, 1000, 2), (10, 1000), (1024,), (6,), (12,), (18, 4), (4819, 2), (4819,), (80000, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_1': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2,), (268569,), (2, 268569), (10,), (10, 268569), (403200, 1), (403200, 4), (100800, 1), (100800, 4), (25200, 1), (25200, 4), (6300, 1), (6300, 4), (1638, 1), (1638, 4), (8819, 4), (8819, 2), (8819,), (1000,), (2, 1000), (10, 1000), (1024,), (18, 4), (4819, 4), (4819, 2), (4819,), (80000, 4), (80000, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2,), (268569,), (2, 268569), (10,), (10, 268569), (403200, 1), (403200, 4), (100800, 1), (100800, 4), (25200, 1), (25200, 4), (6300, 1), (6300, 4), (1638, 1), (1638, 4), (1, 4), (8819, 2), (), (1000,), (2, 1000), (10, 1000), (1024,), (18, 4), (1, 4), (4819, 2), (), (1, 4), (80000, 2)],
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
                    "shape": [(2, 1, 2), (2, 268569), (10, 1, 2), (10, 268569), (2, 1, 2), (2, 1000), (10, 1, 2), (10, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 268569, 2), (1,), (1, 268569, 2), (1,), (1, 1000, 2), (1,), (1, 1000, 2), (1,)],
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
                    "shape": [(2, 1, 2), (10, 1, 2), (2, 1, 2), (10, 1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 268569, 2), (1, 268569, 2), (1, 1000, 2), (1, 1000, 2)],
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
            min=[0, 0, 0, 0],
            max=[None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 268569, 2), (10, 268569, 2), (2, 1000, 2), (10, 1000, 2)],
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
                    "shape": [(2, 268569), (21,), (10, 268569), (48,), (2, 1000), (10, 1000), (6,), (12,), (1000, 80, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 268569), (21,), (10, 268569), (48,), (2, 1000), (10, 1000), (6,), (12,), (4,)],
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
            dim=[0, 1, 0, 1, 0, 1, 0, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 268569), (2, 268569), (10, 268569), (10, 268569), (2, 1000), (2, 1000), (10, 1000), (10, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'ge': dict(
        name=["ge"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 0.7, 0.3, 0, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(268569,), (268569,), (), (1000,), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'lt': dict(
        name=["lt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.3, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(268569,), (1000,)],
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
            other=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(0,), (15,), (4,), (2,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
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
                    "shape": [(268569,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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

    'gt': dict(
        name=["gt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(268569,), (1000,), (1002,), (1010,)],
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
                    "shape": [(268569,), (403200,), (100800,), (25200,), (6300,), (1638,), (1000,), (1002,), (1010,), (1024,), (80000,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
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
                    "shape": [(21,), (235,), (48,), (208,), (6,), (506,), (12,), (500,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'eq_1': dict(
        name=["eq"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 0, 0, 0, 1, 2, 3, 0, 1, 2, 3],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(268569,), (1002,), (1010,), (1024,), (1024,), (1024,), (1024,), (1000,), (1000,), (1000,), (1000,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'randperm': dict(
        name=["randperm"],
        no_output_ref=True,
        para=dict(
            n=[268197, 267083, 996, 998],
        ),
    ),

    'sub_1': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(21,), (48,), (4,), (6,), (2,), (12,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'log': dict(
        name=["log"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(21,), (48,), (6,), (12,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sub_2': dict(
        name=["sub"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(21, 4), (48, 4), (6, 4), (12, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 4), (1, 4), (1, 4), (1, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_1': dict(
        name=["div"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(21, 4), (48, 4), (6, 4), (12, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 4), (1, 4), (1, 4), (1, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'stack_1': dict(
        name=["stack"],
        interface=["torch"],
        para=dict(
            dim=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((268569,), (268569,))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'stack_2': dict(
        name=["stack"],
        interface=["torch"],
        para=dict(
            dim=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((268569,), (268569,)), ((268569, 4), (268569, 4))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(2, 3, 200, 336), (2, 12, 200, 336), (2, 3, 100, 168), (2, 12, 100, 168), (2, 3, 50, 84), (2, 12, 50, 84), (2, 3, 25, 42), (2, 12, 25, 42), (2, 3, 13, 21), (2, 12, 13, 21)],
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
            other=[0, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(403200,), (100800,), (25200,), (6300,), (1638,), (1024,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'ne': dict(
        name=["ne"],
        interface=["torch.Tensor"],
        para=dict(
            other=[-100, -100, -100, -100, -100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(403200,), (100800,), (25200,), (6300,), (1638,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'lt_1': dict(
        name=["lt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1, 1, 1, 1, 1, 80],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(403200,), (100800,), (25200,), (6300,), (1638,), (1024,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'mul_2': dict(
        name=["mul"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(403200, 1), (100800, 1), (25200, 1), (6300, 1), (1638, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(403200, 1), (100800, 1), (25200, 1), (6300, 1), (1638, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'binary_cross_entropy_with_logits': dict(
        name=["binary_cross_entropy_with_logits"],
        para=dict(
            reduction=['none', 'none', 'none', 'none', 'none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(403200, 1), (100800, 1), (25200, 1), (6300, 1), (1638, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(403200, 1), (100800, 1), (25200, 1), (6300, 1), (1638, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(403200, 1), (403200, 4), (100800, 1), (100800, 4), (25200, 1), (25200, 4), (6300, 1), (6300, 4), (1638, 1), (1638, 4), (1024,), (18, 4)],
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
            other=[512.0000001192093, 56, 1024.0000001192093, 56],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (1024,), (), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_3': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1.0],
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

    'abs': dict(
        name=["abs"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(403200, 4), (100800, 4), (25200, 4), (6300, 4), (1638, 4), (18, 4)],
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
            dims=[(1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 200, 336), (12, 200, 336), (3, 100, 168), (12, 100, 168), (3, 50, 84), (12, 50, 84), (3, 25, 42), (12, 25, 42), (3, 13, 21), (12, 13, 21)],
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
                    "shape": [(201600,), (50400,), (12600,), (3150,), (819,)],
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
                    "shape": [(201600,), (50400,), (12600,), (3150,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp_1': dict(
        name=["clamp"],
        interface=["torch.Tensor"],
        para=dict(
            min=[-4.135166556742356, 0, -4.135166556742356, 0, -4.135166556742356],
            max=[4.135166556742356, 3, 4.135166556742356, 3, 4.135166556742356],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8819, 2), (1024,), (4819, 2), (1000,), (80000, 2)],
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
                    "shape": [(8819, 2), (4819, 2), (80000, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat_2': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[-1, 1, 1, 0, 0, -1, -1, 1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((8819, 2), (8819, 2)), ((3746, 4), (3746, 1)), ((3784, 4), (3784, 1)), ((2, 4), (1000, 4)), ((10, 4), (1000, 4)), ((512, 1), (512, 4)), ((4819, 2), (4819, 2)), ((2128, 4), (2128, 1)), ((1000, 1), (1000, 4)), ((80000, 2), (80000, 2))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'clamp_2': dict(
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
                    "shape": [(8819, 2), (8819, 2), (4819, 2), (4819, 2), (80000, 2), (80000, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat_3': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((2000,), (2000,), (2000,), (2000,), (819,)), ((2,), (1000,)), ((10,), (1000,)), ((1000,), (1000,), (1000,), (1000,), (819,))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'gt_1': dict(
        name=["gt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 0, 0, 0.05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8819,), (1024,), (4819,), (80000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'all': dict(
        name=["all"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8819,), (4819,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'max_1': dict(
        name=["max"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8819, 4), (4819, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat_4': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[-1, -1, 0, 0, 0, -1, 0, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((3746, 4), (3746, 1)), ((3784, 4), (3784, 1)), ((512, 5), (512, 5)), ((512,), (512,)), ((512, 4), (512, 4)), ((2128, 4), (2128, 1)), ((1000, 5),), ((0, 4), (0, 1))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'arange_1': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            start=[1, 1],
            end=[3, 11],
            dtype=[Dtype.int64, Dtype.int64],
        ),
    ),

    'cat_5': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((2,), (1000,)), ((10,), (1000,))],
                    "dtype": [Dtype.uint8],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'sqrt': dict(
        name=["sqrt"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024,), (1000,)],
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
            other=[1e-06, 1e-06],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024,), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'log2': dict(
        name=["log2"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024,), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'floor': dict(
        name=["floor"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024,), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'linear': dict(
        name=["linear"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1024, 12544), (1024, 1024), (1024, 1024), (1024, 1024), (1000, 12544), (1000, 1024), (1000, 1024), (1000, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(1024, 12544), (1024, 1024), (81, 1024), (320, 1024), (1024, 12544), (1024, 1024), (81, 1024), (320, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(1024,), (1024,), (81,), (320,), (1024,), (1024,), (81,), (320,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cat_6': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((512,), (512,))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'sum_1': dict(
        name=["sum"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            ignore_index=[-100],
            reduction=['none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1024, 81)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(1024,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'topk': dict(
        name=["topk"],
        interface=["torch.Tensor"],
        para=dict(
            k=[1],
            dim=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024, 81)],
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
            dim0=[0 for i in range(1)],
            dim1=[1 for i in range(1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024, 1)],
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
            size=[(1, 1024)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 1024)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'eq_2': dict(
        name=["eq"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 1024)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 1024)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'sum_2': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024,)],
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
        para=dict(
            other=[0.09765625],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mean': dict(
        name=["mean"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (1,)],
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

    'sgd': dict(
        name=["sgd"],
        interface=["CustomizedTest"],
        para=dict(
            nesterov=[False for i in range(32)],
            lr=[2.000000000000002e-05 for i in range(32)],
            momentum=[0.9 for i in range(32)],
            weight_decay=[0.0001 for i in range(32)],
            dampening=[0 for i in range(32)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256,), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (256, 256, 1, 1), (256, 2048, 1, 1), (3, 256, 1, 1), (3,), (12, 256, 1, 1), (12,), (81, 1024), (81,), (320, 1024), (320,), (1024, 12544), (1024, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256,), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (256, 256, 1, 1), (256, 2048, 1, 1), (3, 256, 1, 1), (3,), (12, 256, 1, 1), (12,), (81, 1024), (81,), (320, 1024), (320,), (1024, 12544), (1024, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'split': dict(
        name=["split"],
        interface=["torch"],
        para=dict(
            split_size_or_sections=[(1000,), (1000,), (1000,)],
            dim=[0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensor"],
                    "shape": [(1000, 5), (1000, 81), (1000, 320)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'softmax': dict(
        name=["softmax"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1000, 81)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'arange_2': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            end=[80],
            dtype=[Dtype.int64],
        ),
    ),

    'expand_2': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(1000, 80)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 80)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

}
