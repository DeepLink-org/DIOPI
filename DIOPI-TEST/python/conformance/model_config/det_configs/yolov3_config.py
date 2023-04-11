from ...config import Genfunc
from ...dtype import Dtype

yolov3_config = {
    'conv2d': dict(
        name=["conv2d"],
        para=dict(
            bias=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            stride=[(1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(8, 3, 320, 320), (8, 32, 320, 320), (8, 64, 160, 160), (8, 32, 160, 160), (8, 64, 160, 160), (8, 128, 80, 80), (8, 64, 80, 80), (8, 128, 80, 80), (8, 256, 40, 40), (8, 128, 40, 40), (8, 256, 40, 40), (8, 512, 20, 20), (8, 256, 20, 20), (8, 512, 20, 20), (8, 1024, 10, 10), (8, 512, 10, 10), (8, 512, 10, 10), (8, 768, 20, 20), (8, 256, 20, 20), (8, 384, 40, 40), (1, 3, 320, 320), (1, 32, 320, 320), (1, 64, 160, 160), (1, 32, 160, 160), (1, 64, 160, 160), (1, 128, 80, 80), (1, 64, 80, 80), (1, 128, 80, 80), (1, 256, 40, 40), (1, 128, 40, 40), (1, 256, 40, 40), (1, 512, 20, 20), (1, 256, 20, 20), (1, 512, 20, 20), (1, 1024, 10, 10), (1, 512, 10, 10), (1, 512, 10, 10), (1, 768, 20, 20), (1, 256, 20, 20), (1, 384, 40, 40)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(32, 3, 3, 3), (64, 32, 3, 3), (32, 64, 1, 1), (64, 32, 3, 3), (128, 64, 3, 3), (64, 128, 1, 1), (128, 64, 3, 3), (256, 128, 3, 3), (128, 256, 1, 1), (256, 128, 3, 3), (512, 256, 3, 3), (256, 512, 1, 1), (512, 256, 3, 3), (1024, 512, 3, 3), (512, 1024, 1, 1), (1024, 512, 3, 3), (256, 512, 1, 1), (256, 768, 1, 1), (128, 256, 1, 1), (128, 384, 1, 1), (32, 3, 3, 3), (64, 32, 3, 3), (32, 64, 1, 1), (64, 32, 3, 3), (128, 64, 3, 3), (64, 128, 1, 1), (128, 64, 3, 3), (256, 128, 3, 3), (128, 256, 1, 1), (256, 128, 3, 3), (512, 256, 3, 3), (256, 512, 1, 1), (512, 256, 3, 3), (1024, 512, 3, 3), (512, 1024, 1, 1), (1024, 512, 3, 3), (256, 512, 1, 1), (256, 768, 1, 1), (128, 256, 1, 1), (128, 384, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'batch_norm': dict(
        name=["batch_norm"],
        para=dict(
            training=[False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(8, 32, 320, 320), (8, 64, 160, 160), (8, 32, 160, 160), (8, 128, 80, 80), (8, 64, 80, 80), (8, 256, 40, 40), (8, 128, 40, 40), (8, 512, 20, 20), (8, 256, 20, 20), (8, 1024, 10, 10), (8, 512, 10, 10), (8, 512, 10, 10), (8, 1024, 10, 10), (8, 256, 10, 10), (8, 256, 20, 20), (8, 512, 20, 20), (8, 128, 20, 20), (8, 128, 40, 40), (8, 256, 40, 40), (1, 32, 320, 320), (1, 64, 160, 160), (1, 32, 160, 160), (1, 128, 80, 80), (1, 64, 80, 80), (1, 256, 40, 40), (1, 128, 40, 40), (1, 512, 20, 20), (1, 256, 20, 20), (1, 1024, 10, 10), (1, 512, 10, 10), (1, 256, 10, 10), (1, 128, 20, 20)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": [(32,), (64,), (32,), (128,), (64,), (256,), (128,), (512,), (256,), (1024,), (512,), (512,), (1024,), (256,), (256,), (512,), (128,), (128,), (256,), (32,), (64,), (32,), (128,), (64,), (256,), (128,), (512,), (256,), (1024,), (512,), (256,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "shape": [(32,), (64,), (32,), (128,), (64,), (256,), (128,), (512,), (256,), (1024,), (512,), (512,), (1024,), (256,), (256,), (512,), (128,), (128,), (256,), (32,), (64,), (32,), (128,), (64,), (256,), (128,), (512,), (256,), (1024,), (512,), (256,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(32,), (64,), (32,), (128,), (64,), (256,), (128,), (512,), (256,), (1024,), (512,), (512,), (1024,), (256,), (256,), (512,), (128,), (128,), (256,), (32,), (64,), (32,), (128,), (64,), (256,), (128,), (512,), (256,), (1024,), (512,), (256,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(32,), (64,), (32,), (128,), (64,), (256,), (128,), (512,), (256,), (1024,), (512,), (512,), (1024,), (256,), (256,), (512,), (128,), (128,), (256,), (32,), (64,), (32,), (128,), (64,), (256,), (128,), (512,), (256,), (1024,), (512,), (256,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'leaky_relu': dict(
        name=["leaky_relu"],
        para=dict(
            inplace=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 32, 320, 320), (8, 64, 160, 160), (8, 32, 160, 160), (8, 128, 80, 80), (8, 64, 80, 80), (8, 256, 40, 40), (8, 128, 40, 40), (8, 512, 20, 20), (8, 256, 20, 20), (8, 1024, 10, 10), (8, 512, 10, 10), (8, 256, 10, 10), (8, 128, 20, 20), (1, 32, 320, 320), (1, 64, 160, 160), (1, 32, 160, 160), (1, 128, 80, 80), (1, 64, 80, 80), (1, 256, 40, 40), (1, 128, 40, 40), (1, 512, 20, 20), (1, 256, 20, 20), (1, 1024, 10, 10), (1, 512, 10, 10), (1, 256, 10, 10), (1, 128, 20, 20)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add': dict(
        name=["add"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 64, 160, 160), (8, 128, 80, 80), (8, 256, 40, 40), (8, 512, 20, 20), (8, 1024, 10, 10), (1, 3, 4), (1, 3, 4), (1, 3, 4), (1,), (3,), (9,), (2,), (1, 1), (3, 1), (4,), (9, 1), (10,), (2, 1), (8, 300), (8, 1200), (8, 4800), (), (1, 64, 160, 160), (1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20), (1, 1024, 10, 10), (6300, 2), (6300, 2), (1, 6300)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(8, 64, 160, 160), (8, 128, 80, 80), (8, 256, 40, 40), (8, 512, 20, 20), (8, 1024, 10, 10), (100, 1, 4), (400, 1, 4), (1600, 1, 4), (1,), (3,), (9,), (2,), (1, 6300), (1, 6300), (4,), (1, 6300), (10,), (1, 6300), (8, 300), (8, 1200), (8, 4800), (), (1, 64, 160, 160), (1, 128, 80, 80), (1, 256, 40, 40), (1, 512, 20, 20), (1, 1024, 10, 10), (6300, 2), (1, 6300, 2), (1, 6300)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_1': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (1,), (4,), (8,), (3,), (2,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        para=dict(
            scale_factor=[2, 2, 2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 256, 10, 10), (8, 128, 20, 20), (1, 256, 10, 10), (1, 128, 20, 20)],
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
            dim=[1, 1, 1, 1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((8, 256, 20, 20), (8, 512, 20, 20)), ((8, 128, 40, 40), (8, 256, 40, 40)), ((1, 256, 20, 20), (1, 512, 20, 20)), ((1, 128, 40, 40), (1, 256, 40, 40)), ((0, 4), (0, 1))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'conv2d_1': dict(
        name=["conv2d"],
        para=dict(
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(8, 1024, 10, 10), (8, 512, 20, 20), (8, 256, 40, 40), (1, 1024, 10, 10), (1, 512, 20, 20), (1, 256, 40, 40)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(255, 1024, 1, 1), (255, 512, 1, 1), (255, 256, 1, 1), (255, 1024, 1, 1), (255, 512, 1, 1), (255, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(255,), (255,), (255,), (255,), (255,), (255,)],
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
            start=[0, 0, 0],
            end=[10, 20, 40],
        ),
    ),

    'mul': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[32, 16, 8, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 35.0, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(10,), (20,), (40,), (1,), (3,), (9,), (2,), (4,), (10,), (), (6300, 2)],
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
                    "shape": [((100,), (100,), (100,), (100,)), ((400,), (400,), (400,), (400,)), ((1600,), (1600,), (1600,), (1600,)), ((1,), (1,), (1,), (1,)), ((4,), (4,), (4,), (4,)), ((10,), (10,), (10,), (10,)), ((2,), (2,), (2,), (2,)), ((3,), (3,), (3,), (3,)), ((1, 6300), (1, 6300), (1, 6300), (1, 6300))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'div': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16, 8],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (1,), (1,), (3,), (3,), (3,), (9,), (9,), (9,), (2,), (2,), (2,)],
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
                    "shape": [(1,), (3,), (9,), (2,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_1': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[10, 20, 40, 10, 20, 40, 10, 20, 40, 10, 20, 40],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (1,), (1,), (3,), (3,), (3,), (9,), (9,), (9,), (2,), (2,), (2,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'add_2': dict(
        name=["add"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (3,), (9,), (2,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
                {
                    "ins": ["other"],
                    "shape": [(1,), (3,), (9,), (2,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'expand': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(100, 3), (400, 3), (1600, 3)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(100, 1), (400, 1), (1600, 1)],
                    "dtype": [Dtype.uint8],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'cat_1': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((300, 4), (1200, 4), (4800, 4)), ((300,), (1200,), (4800,))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cat_2': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((300,), (1200,), (4800,))],
                    "dtype": [Dtype.uint8],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cat_3': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((300,), (1200,), (4800,))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'sub': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (6300,), (1, 6300, 2), (1, 6300), (3,), (3, 6300, 2), (3, 6300), (4,), (9,), (9, 6300, 2), (9, 6300), (10,), (2,), (2, 6300, 2), (2, 6300), (6300, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1,), (6300,), (1, 6300, 2), (1, 6300), (3,), (3, 6300, 2), (3, 6300), (4,), (9,), (9, 6300, 2), (9, 6300), (10,), (2,), (2, 6300, 2), (2, 6300), (6300, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_2': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (6300,), (1, 6300), (3,), (3, 6300), (9,), (9, 6300), (2,), (2, 6300), (8, 300, 1), (8, 300, 80), (8, 300), (8, 300, 1), (8, 300, 2), (8, 300, 2), (8, 1200, 1), (8, 1200, 80), (8, 1200), (8, 1200, 1), (8, 1200, 2), (8, 1200, 2), (8, 4800, 1), (8, 4800, 80), (8, 4800), (8, 4800, 1), (8, 4800, 2), (8, 4800, 2), (1, 6300, 2), (6300, 2), (119840,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1,), (6300,), (1, 6300), (3,), (3, 6300), (9,), (9, 6300), (2,), (2, 6300), (8, 300, 80), (8, 300, 80), (8, 300), (8, 300, 2), (8, 300, 2), (8, 300, 1), (8, 1200, 80), (8, 1200, 80), (8, 1200), (8, 1200, 2), (8, 1200, 2), (8, 1200, 1), (8, 4800, 80), (8, 4800, 80), (8, 4800), (8, 4800, 2), (8, 4800, 2), (8, 4800, 1), (6300, 1), (1, 6300, 2), (119840,)],
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
                    "shape": [(1, 1, 2), (1, 6300), (3, 1, 2), (3, 6300), (9, 1, 2), (9, 6300), (2, 1, 2), (2, 6300)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 6300, 2), (1,), (1, 6300, 2), (1,), (1, 6300, 2), (1,), (1, 6300, 2), (1,)],
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
                    "shape": [(1, 1, 2), (3, 1, 2), (9, 1, 2), (2, 1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 6300, 2), (1, 6300, 2), (1, 6300, 2), (1, 6300, 2)],
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
            min=[0, 1e-06, 0, 1e-06, 0, 1e-06, 0, 1e-06, 1e-06],
            max=[None, 0.999999, None, 0.999999, None, 0.999999, None, 0.999999, 0.999999],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 6300, 2), (1,), (3, 6300, 2), (4,), (9, 6300, 2), (10,), (2, 6300, 2), (2,), (3,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_1': dict(
        name=["div"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 6300), (1,), (3, 6300), (4,), (9, 6300), (10,), (2, 6300), (2,), (3,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 6300), (1,), (3, 6300), (4,), (9, 6300), (10,), (2, 6300), (2,), (3,)],
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
                    "shape": [(1, 6300), (1, 6300), (3, 6300), (3, 6300), (9, 6300), (9, 6300), (2, 6300), (2, 6300)],
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
            other=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.005],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(6300,), (8, 300, 80), (8, 300), (8, 300, 2), (8, 1200, 80), (8, 1200), (8, 1200, 2), (8, 4800, 80), (8, 4800), (8, 4800, 2), (6300,)],
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
            other=[0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(6300,)],
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
                    "shape": [(6300,), (8, 300, 80), (8, 300), (8, 300, 2), (8, 1200, 80), (8, 1200), (8, 1200, 2), (8, 4800, 80), (8, 4800), (8, 4800, 2)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["other"],
                    "shape": [(6300,), (8, 300, 80), (8, 300), (8, 300, 2), (8, 1200, 80), (8, 1200), (8, 1200, 2), (8, 4800, 80), (8, 4800), (8, 4800, 2)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'bitwise_not': dict(
        name=["bitwise_not"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(6300,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'gt': dict(
        name=["gt"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.5, 0, 1.0, 0.05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(6300,), (), (), (119840,)],
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
                    "shape": [(6300,)],
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
                    "shape": [(6300,)],
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
                    "shape": [(6300,), (119840,)],
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
                    "shape": [(1,), (6293,), (4,), (6272,), (6297,), (10,), (6277,), (2,), (6294,), (6288,), (3,), (6285,), (6289,)],
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
                    "shape": [(6300,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
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
                    "shape": [(1,), (4,), (10,), (2,), (3,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'clamp_1': dict(
        name=["clamp"],
        interface=["torch.Tensor"],
        para=dict(
            min=[1e-06, 1e-06, 1e-06, 1e-06, 1e-06],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (4,), (10,), (2,), (3,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(1,), (4,), (10,), (2,), (3,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_2': dict(
        name=["div"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (4,), (10,), (2,), (3,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1,), (4,), (10,), (2,), (3,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'add_3': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.5, 0.5, 0.5, 0.5, 0.5, 1e-06],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (4,), (10,), (2,), (3,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'one_hot': dict(
        name=["one_hot"],
        para=dict(
            num_classes=[80, 80, 80, 80],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1,), (3,), (9,), (2,)],
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
            dim=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((6300, 85), (6300, 85), (6300, 85), (6300, 85), (6300, 85), (6300, 85), (6300, 85), (6300, 85))],
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
                    "shape": [((6300,), (6300,), (6300,), (6300,), (6300,), (6300,), (6300,), (6300,))],
                    "dtype": [Dtype.uint8],
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
            dims=[(0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 255, 10, 10), (8, 255, 20, 20), (8, 255, 40, 40), (1, 255, 10, 10), (1, 255, 20, 20), (1, 255, 40, 40)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_1': dict(
        name=["max"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 300), (8, 1200), (8, 4800)],
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
            other=[-100, -100, -100, -100, -100, -100, -100, -100, -100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 300, 80), (8, 300), (8, 300, 2), (8, 1200, 80), (8, 1200), (8, 1200, 2), (8, 4800, 80), (8, 4800), (8, 4800, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'binary_cross_entropy_with_logits': dict(
        name=["binary_cross_entropy_with_logits"],
        para=dict(
            reduction=['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 300, 80), (8, 300), (8, 300, 2), (8, 1200, 80), (8, 1200), (8, 1200, 2), (8, 4800, 80), (8, 4800), (8, 4800, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(8, 300, 80), (8, 300), (8, 300, 2), (8, 1200, 80), (8, 1200), (8, 1200, 2), (8, 4800, 80), (8, 4800), (8, 4800, 2)],
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
                    "shape": [(8, 300, 80), (8, 300), (8, 300, 2), (8, 1200, 80), (8, 1200), (8, 1200, 2), (8, 4800, 80), (8, 4800), (8, 4800, 2)],
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

    'mse_loss': dict(
        name=["mse_loss"],
        para=dict(
            reduction=['none', 'none', 'none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 300, 2), (8, 1200, 2), (8, 4800, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(8, 300, 2), (8, 1200, 2), (8, 4800, 2)],
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

    'norm': dict(
        name=["norm"],
        interface=["torch"],
        para=dict(
            p=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 3, 3, 3), (32,), (64, 32, 3, 3), (64,), (32, 64, 1, 1), (128, 64, 3, 3), (128,), (64, 128, 1, 1), (256, 128, 3, 3), (256,), (128, 256, 1, 1), (512, 256, 3, 3), (512,), (256, 512, 1, 1), (1024, 512, 3, 3), (1024,), (512, 1024, 1, 1), (256, 768, 1, 1), (128, 384, 1, 1), (255, 1024, 1, 1), (255,), (255, 512, 1, 1), (255, 256, 1, 1), (222,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'stack_3': dict(
        name=["stack"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ())],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'div_3': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[35.0],
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

    'reciprocal': dict(
        name=["reciprocal"],
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

    'clamp_2': dict(
        name=["clamp"],
        interface=["torch"],
        para=dict(
            max=[1.0],
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
                    "shape": [(32, 3, 3, 3), (32,), (64, 32, 3, 3), (64,), (32, 64, 1, 1), (128, 64, 3, 3), (128,), (64, 128, 1, 1), (256, 128, 3, 3), (256,), (128, 256, 1, 1), (512, 256, 3, 3), (512,), (256, 512, 1, 1), (1024, 512, 3, 3), (1024,), (512, 1024, 1, 1), (256, 768, 1, 1), (128, 384, 1, 1), (255, 1024, 1, 1), (255,), (255, 512, 1, 1), (255, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()],
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
            nesterov=[False for i in range(23)],
            lr=[9.999999999999998e-05 for i in range(23)],
            momentum=[0.9 for i in range(23)],
            weight_decay=[0.0005 for i in range(23)],
            dampening=[0 for i in range(23)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(32, 3, 3, 3), (32,), (64, 32, 3, 3), (64,), (32, 64, 1, 1), (128, 64, 3, 3), (128,), (64, 128, 1, 1), (256, 128, 3, 3), (256,), (128, 256, 1, 1), (512, 256, 3, 3), (512,), (256, 512, 1, 1), (1024, 512, 3, 3), (1024,), (512, 1024, 1, 1), (256, 768, 1, 1), (128, 384, 1, 1), (255, 1024, 1, 1), (255,), (255, 512, 1, 1), (255, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(32, 3, 3, 3), (32,), (64, 32, 3, 3), (64,), (32, 64, 1, 1), (128, 64, 3, 3), (128,), (64, 128, 1, 1), (256, 128, 3, 3), (256,), (128, 256, 1, 1), (512, 256, 3, 3), (512,), (256, 512, 1, 1), (1024, 512, 3, 3), (1024,), (512, 1024, 1, 1), (256, 768, 1, 1), (128, 384, 1, 1), (255, 1024, 1, 1), (255,), (255, 512, 1, 1), (255, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sigmoid': dict(
        name=["sigmoid"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        saved_args=dict(output=0),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 300, 2), (1, 1200, 2), (1, 4800, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'expand_1': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(300,), (1200,), (4800,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), ()],
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
            dim=[1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((1, 300, 85), (1, 1200, 85), (1, 4800, 85)), ((1, 6300, 80), (1, 6300, 1))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'sigmoid_1': dict(
        name=["sigmoid"],
        interface=["torch.Tensor"],
        saved_args=dict(output=0),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(1, 6300), (1, 6300, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sub_2': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 6300, 2)],
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
                    "shape": [(1, 6300, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_4': dict(
        name=["div"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 6300, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 1, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'expand_2': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(1498, 80, 4)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1498, 1, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'arange_1': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            end=[80],
            dtype=[Dtype.int64],
        ),
    ),

    'expand_3': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(1498, 80)],
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

    'expand_4': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(1498, 80)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1498, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
