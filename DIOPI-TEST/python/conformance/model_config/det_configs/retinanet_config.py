from ...config import Genfunc
from ...dtype import Dtype

retinanet_config = {
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
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(2, 512, 100, 168), (2, 1024, 50, 84), (2, 2048, 25, 42), (2, 256, 100, 168), (2, 256, 50, 84), (2, 256, 25, 42), (2, 2048, 25, 42), (2, 256, 13, 21), (2, 256, 100, 168), (2, 256, 100, 168), (2, 256, 50, 84), (2, 256, 50, 84), (2, 256, 25, 42), (2, 256, 25, 42), (2, 256, 13, 21), (2, 256, 13, 21), (2, 256, 13, 21), (2, 256, 7, 11), (2, 256, 7, 11), (2, 256, 7, 11), (1, 512, 100, 168), (1, 1024, 50, 84), (1, 2048, 25, 42), (1, 256, 100, 168), (1, 256, 50, 84), (1, 256, 25, 42), (1, 2048, 25, 42), (1, 256, 13, 21), (1, 256, 100, 168), (1, 256, 100, 168), (1, 256, 50, 84), (1, 256, 50, 84), (1, 256, 25, 42), (1, 256, 25, 42), (1, 256, 13, 21), (1, 256, 13, 21), (1, 256, 13, 21), (1, 256, 7, 11), (1, 256, 7, 11), (1, 256, 7, 11)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 512, 1, 1), (256, 1024, 1, 1), (256, 2048, 1, 1), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 2048, 3, 3), (256, 256, 3, 3), (720, 256, 3, 3), (36, 256, 3, 3), (720, 256, 3, 3), (36, 256, 3, 3), (720, 256, 3, 3), (36, 256, 3, 3), (256, 256, 3, 3), (720, 256, 3, 3), (36, 256, 3, 3), (256, 256, 3, 3), (720, 256, 3, 3), (36, 256, 3, 3), (256, 512, 1, 1), (256, 1024, 1, 1), (256, 2048, 1, 1), (256, 256, 3, 3), (256, 256, 3, 3), (256, 256, 3, 3), (256, 2048, 3, 3), (256, 256, 3, 3), (720, 256, 3, 3), (36, 256, 3, 3), (720, 256, 3, 3), (36, 256, 3, 3), (720, 256, 3, 3), (36, 256, 3, 3), (256, 256, 3, 3), (720, 256, 3, 3), (36, 256, 3, 3), (256, 256, 3, 3), (720, 256, 3, 3), (36, 256, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (720,), (36,), (720,), (36,), (720,), (36,), (256,), (720,), (36,), (256,), (720,), (36,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (256,), (720,), (36,), (720,), (36,), (720,), (36,), (256,), (720,), (36,), (256,), (720,), (36,)],
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
                    "shape": [(2, 256, 50, 84), (2, 256, 100, 168), (1, 9, 4), (1, 9, 4), (1, 9, 4), (1, 9, 4), (1, 9, 4), (2, 1), (71,), (34,), (), (1, 256, 50, 84), (1, 256, 100, 168)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 256, 50, 84), (2, 256, 100, 168), (16800, 1, 4), (4200, 1, 4), (1050, 1, 4), (273, 1, 4), (77, 1, 4), (1, 201600), (71,), (34,), (), (1, 256, 50, 84), (1, 256, 100, 168)],
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

    'mul': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 0.5, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(168,), (100,), (84,), (50,), (42,), (25,), (21,), (13,), (11,), (7,), (71,), (34,)],
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
            dim=[-1, -1, -1, -1, -1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((16800,), (16800,), (16800,), (16800,)), ((4200,), (4200,), (4200,), (4200,)), ((1050,), (1050,), (1050,), (1050,)), ((273,), (273,), (273,), (273,)), ((77,), (77,), (77,), (77,)), ((71,), (71,), (71,), (71,)), ((34,), (34,), (34,), (34,))],
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
                    "shape": [(16800,), (4200,), (1050,), (273,), (77,), (201600,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["other"],
                    "shape": [(16800,), (4200,), (1050,), (273,), (77,), (201600,)],
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
            size=[(16800, 9), (4200, 9), (1050, 9), (273, 9), (77, 9)],
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
                    "shape": [((151200, 4), (37800, 4), (9450, 4), (2457, 4), (693, 4)), ((0, 4), (0, 4), (0, 4), (0, 4), (0, 4)), ((0,), (0,), (0,), (0,), (0,))],
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
                    "shape": [((151200,), (37800,), (9450,), (2457,), (693,))],
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
                    "shape": [(201600,)],
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
                    "shape": [(2,), (201600,), (2, 201600, 2), (2, 201600), (71,), (34,), (302400, 4), (75600, 4), (18900, 4), (4914, 4), (1386, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2,), (201600,), (2, 201600, 2), (2, 201600), (71,), (34,), (302400, 4), (75600, 4), (18900, 4), (4914, 4), (1386, 4)],
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
                    "shape": [(2,), (201600,), (2, 201600), (302400, 80), (302400, 4), (75600, 80), (75600, 4), (18900, 80), (18900, 4), (4914, 80), (4914, 4), (1386, 80), (1386, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2,), (201600,), (2, 201600), (302400, 1), (302400, 4), (75600, 1), (75600, 4), (18900, 1), (18900, 4), (4914, 1), (4914, 4), (1386, 1), (1386, 4)],
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
                    "shape": [(2, 1, 2), (2, 201600)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 201600, 2), (1,)],
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
                    "shape": [(2, 1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 201600, 2)],
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
            min=[0],
            max=[None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 201600, 2)],
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
                    "shape": [(2, 201600), (71,), (34,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2, 201600), (71,), (34,)],
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
            dim=[0, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2, 201600), (2, 201600)],
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
            other=[0, 0.5, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(201600,), (201600,), ()],
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
            other=[0.4],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(201600,)],
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
            other=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(71,), (34,)],
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
                    "shape": [(201600,)],
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
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(201600,)],
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
                    "shape": [(201600,)],
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
                    "shape": [(71,), (34,)],
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
                    "shape": [(71,), (201428,), (34,), (201483,)],
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
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(201600,)],
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
                    "shape": [(71,), (34,)],
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
                    "shape": [(71, 4), (34, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 4), (1, 4)],
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
                    "shape": [(71, 4), (34, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 4), (1, 4)],
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
                    "shape": [((201600,), (201600,))],
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
                    "shape": [((201600,), (201600,)), ((201600, 4), (201600, 4))],
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
                    "shape": [(2, 720, 100, 168), (2, 36, 100, 168), (2, 720, 50, 84), (2, 36, 50, 84), (2, 720, 25, 42), (2, 36, 25, 42), (2, 720, 13, 21), (2, 36, 13, 21), (2, 720, 7, 11), (2, 36, 7, 11)],
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
                    "shape": [(302400, 80), (302400, 4), (75600, 80), (75600, 4), (18900, 80), (18900, 4), (4914, 80), (4914, 4), (1386, 80), (1386, 4)],
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
            other=[105.00000011920929],
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

    'mul_2': dict(
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
                    "shape": [(302400, 4), (75600, 4), (18900, 4), (4914, 4), (1386, 4)],
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
                    "shape": [()],
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
                    "shape": [(1386, 80), (4914, 80), (18900, 80), (75600, 80), (302400, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1386, 80), (4914, 80), (18900, 80), (75600, 80), (302400, 80)],
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
            nesterov=[False for i in range(26)],
            lr=[1.000000000000001e-05 for i in range(26)],
            momentum=[0.9 for i in range(26)],
            weight_decay=[0.0001 for i in range(26)],
            dampening=[0 for i in range(26)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256,), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (256, 2048, 1, 1), (256, 2048, 3, 3), (720, 256, 3, 3), (720,), (36, 256, 3, 3), (36,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256,), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (512, 2048, 1, 1), (256, 2048, 1, 1), (256, 2048, 3, 3), (720, 256, 3, 3), (720,), (36, 256, 3, 3), (36,)],
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
                    "shape": [(36, 100, 168), (720, 100, 168), (36, 50, 84), (720, 50, 84), (36, 25, 42), (720, 25, 42), (36, 13, 21), (720, 13, 21), (36, 7, 11), (720, 7, 11)],
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
                    "shape": [(151200, 80), (37800, 80), (9450, 80), (2457, 80), (693, 80)],
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
            other=[0.05, 0.05, 0.05, 0.05, 0.05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(151200, 80), (37800, 80), (9450, 80), (2457, 80), (693, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'nonzero_1': dict(
        name=["nonzero"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(151200, 80), (37800, 80), (9450, 80), (2457, 80), (693, 80)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'sort': dict(
        name=["sort"],
        interface=["torch.Tensor"],
        para=dict(
            descending=[True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(0,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_3': dict(
        name=["div"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(0, 4)],
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

    'cat_2': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((0,), (0,), (0,), (0,), (0,))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cat_3': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((0, 4), (0, 1))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

}
