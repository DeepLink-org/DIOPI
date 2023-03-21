from ...config import Genfunc
from ...dtype import Dtype

ssd300_config = {
    'conv2d': dict(
        name=["conv2d"],
        para=dict(
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (6, 6), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (6, 6), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (6, 6), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (6, 6), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(8, 3, 300, 300), (8, 64, 300, 300), (8, 64, 150, 150), (8, 128, 150, 150), (8, 128, 75, 75), (8, 256, 75, 75), (8, 256, 38, 38), (8, 512, 38, 38), (8, 512, 19, 19), (8, 512, 19, 19), (8, 1024, 19, 19), (8, 1024, 19, 19), (8, 256, 19, 19), (8, 512, 10, 10), (8, 128, 10, 10), (8, 256, 5, 5), (8, 128, 5, 5), (8, 256, 3, 3), (8, 128, 3, 3), (8, 512, 38, 38), (8, 512, 38, 38), (8, 1024, 19, 19), (8, 1024, 19, 19), (8, 512, 10, 10), (8, 512, 10, 10), (8, 256, 5, 5), (8, 256, 5, 5), (8, 256, 3, 3), (8, 256, 3, 3), (8, 256, 1, 1), (8, 256, 1, 1), (1, 3, 300, 300), (1, 64, 300, 300), (1, 64, 150, 150), (1, 128, 150, 150), (1, 128, 75, 75), (1, 256, 75, 75), (1, 256, 38, 38), (1, 512, 38, 38), (1, 512, 19, 19), (1, 512, 19, 19), (1, 1024, 19, 19), (1, 1024, 19, 19), (1, 256, 19, 19), (1, 512, 10, 10), (1, 128, 10, 10), (1, 256, 5, 5), (1, 128, 5, 5), (1, 256, 3, 3), (1, 128, 3, 3), (1, 512, 38, 38), (1, 512, 38, 38), (1, 1024, 19, 19), (1, 1024, 19, 19), (1, 512, 10, 10), (1, 512, 10, 10), (1, 256, 5, 5), (1, 256, 5, 5), (1, 256, 3, 3), (1, 256, 3, 3), (1, 256, 1, 1), (1, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 3, 3), (64, 64, 3, 3), (128, 64, 3, 3), (128, 128, 3, 3), (256, 128, 3, 3), (256, 256, 3, 3), (512, 256, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3), (1024, 512, 3, 3), (1024, 1024, 1, 1), (256, 1024, 1, 1), (512, 256, 3, 3), (128, 512, 1, 1), (256, 128, 3, 3), (128, 256, 1, 1), (256, 128, 3, 3), (128, 256, 1, 1), (256, 128, 3, 3), (324, 512, 3, 3), (16, 512, 3, 3), (486, 1024, 3, 3), (24, 1024, 3, 3), (486, 512, 3, 3), (24, 512, 3, 3), (486, 256, 3, 3), (24, 256, 3, 3), (324, 256, 3, 3), (16, 256, 3, 3), (324, 256, 3, 3), (16, 256, 3, 3), (64, 3, 3, 3), (64, 64, 3, 3), (128, 64, 3, 3), (128, 128, 3, 3), (256, 128, 3, 3), (256, 256, 3, 3), (512, 256, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3), (1024, 512, 3, 3), (1024, 1024, 1, 1), (256, 1024, 1, 1), (512, 256, 3, 3), (128, 512, 1, 1), (256, 128, 3, 3), (128, 256, 1, 1), (256, 128, 3, 3), (128, 256, 1, 1), (256, 128, 3, 3), (324, 512, 3, 3), (16, 512, 3, 3), (486, 1024, 3, 3), (24, 1024, 3, 3), (486, 512, 3, 3), (24, 512, 3, 3), (486, 256, 3, 3), (24, 256, 3, 3), (324, 256, 3, 3), (16, 256, 3, 3), (324, 256, 3, 3), (16, 256, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (128,), (128,), (256,), (256,), (512,), (512,), (512,), (1024,), (1024,), (256,), (512,), (128,), (256,), (128,), (256,), (128,), (256,), (324,), (16,), (486,), (24,), (486,), (24,), (486,), (24,), (324,), (16,), (324,), (16,), (64,), (64,), (128,), (128,), (256,), (256,), (512,), (512,), (512,), (1024,), (1024,), (256,), (512,), (128,), (256,), (128,), (256,), (128,), (256,), (324,), (16,), (486,), (24,), (486,), (24,), (486,), (24,), (324,), (16,), (324,), (16,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        para=dict(
            inplace=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 64, 300, 300), (8, 128, 150, 150), (8, 256, 75, 75), (8, 512, 38, 38), (8, 512, 19, 19), (8, 1024, 19, 19), (8, 256, 19, 19), (8, 512, 10, 10), (8, 128, 10, 10), (8, 256, 5, 5), (8, 128, 5, 5), (8, 256, 3, 3), (8, 128, 3, 3), (8, 256, 1, 1), (1, 64, 300, 300), (1, 128, 150, 150), (1, 256, 75, 75), (1, 512, 38, 38), (1, 512, 19, 19), (1, 1024, 19, 19), (1, 256, 19, 19), (1, 512, 10, 10), (1, 128, 10, 10), (1, 256, 5, 5), (1, 128, 5, 5), (1, 256, 3, 3), (1, 128, 3, 3), (1, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[2, 2, 2, 2, 3, 2, 2, 2, 2, 3],
            stride=[2, 2, 2, 2, 1, 2, 2, 2, 2, 1],
            padding=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            dilation=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ceil_mode=[True, True, True, True, False, True, True, True, True, False],
            return_indices=[False, False, False, False, False, False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(8, 64, 300, 300), (8, 128, 150, 150), (8, 256, 75, 75), (8, 512, 38, 38), (8, 512, 19, 19), (1, 64, 300, 300), (1, 128, 150, 150), (1, 256, 75, 75), (1, 512, 38, 38), (1, 512, 19, 19)],
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
                    "shape": [(8, 512, 38, 38), (1, 512, 38, 38)],
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
            dim=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 512, 38, 38), (1, 512, 38, 38)],
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
                    "shape": [(8, 1, 38, 38), (1, 1, 38, 38)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add': dict(
        name=["add"],
        interface=["torch.Tensor"],
        para=dict(
            other=[1e-10, 1e-10],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 1, 38, 38), (1, 1, 38, 38)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'expand': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(8, 512, 38, 38), (1, 512, 38, 38)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 512, 1, 1), (1, 512, 1, 1)],
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
                    "shape": [(8, 512, 38, 38), (14,), (8732,), (14, 8732), (9,), (9, 8732), (6,), (6, 8732), (1,), (1, 8732), (3,), (3, 8732), (5,), (5, 8732), (8732, 4), (1, 512, 38, 38), (1000, 4), (1000, 2), (672, 4), (672, 2), (2672,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(8, 512, 38, 38), (14,), (8732,), (14, 8732), (9,), (9, 8732), (6,), (6, 8732), (1,), (1, 8732), (3,), (3, 8732), (5,), (5, 8732), (8732, 4), (1, 512, 38, 38), (1, 4), (1000, 2), (1, 4), (672, 2), ()],
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
                    "shape": [(8, 512, 38, 38), (14, 8732), (71,), (19,), (9, 8732), (13,), (6, 8732), (34,), (1, 8732), (2,), (8,), (3, 8732), (17,), (5, 8732), (30,), (1, 512, 38, 38)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(8, 1, 38, 38), (14, 8732), (71,), (19,), (9, 8732), (13,), (6, 8732), (34,), (1, 8732), (2,), (8,), (3, 8732), (17,), (5, 8732), (30,), (1, 1, 38, 38)],
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
            start=[0, 0, 0, 0, 0, 0],
            end=[38, 19, 10, 5, 3, 1],
        ),
    ),

    'mul_1': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[8, 16, 32, 64, 100, 300, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(38,), (19,), (10,), (5,), (3,), (1,), (71,), (19,), (13,), (34,), (2,), (8,), (17,), (30,), (1000, 2), (672, 2)],
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
            dim=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((1444,), (1444,), (1444,), (1444,)), ((361,), (361,), (361,), (361,)), ((100,), (100,), (100,), (100,)), ((25,), (25,), (25,), (25,)), ((9,), (9,), (9,), (9,)), ((1,), (1,), (1,), (1,)), ((71,), (71,), (71,), (71,)), ((19,), (19,), (19,), (19,)), ((13,), (13,), (13,), (13,)), ((34,), (34,), (34,), (34,)), ((2,), (2,), (2,), (2,)), ((8,), (8,), (8,), (8,)), ((17,), (17,), (17,), (17,)), ((30,), (30,), (30,), (30,))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'add_1': dict(
        name=["add"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1, 4, 4), (1, 6, 4), (1, 6, 4), (1, 6, 4), (1, 4, 4), (1, 4, 4), (14, 1), (71,), (19,), (9, 1), (13,), (6, 1), (34,), (1, 1), (2,), (8,), (3, 1), (17,), (5, 1), (30,), (), (1000, 4), (1000, 2), (672, 4), (672, 2), (2672, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1444, 1, 4), (361, 1, 4), (100, 1, 4), (25, 1, 4), (9, 1, 4), (1, 1, 4), (1, 8732), (71,), (19,), (1, 8732), (13,), (1, 8732), (34,), (1, 8732), (2,), (8,), (1, 8732), (17,), (1, 8732), (30,), (), (1, 4), (1000, 2), (1, 4), (672, 2), (2672, 1)],
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
                    "shape": [(1444,), (361,), (100,), (25,), (9,), (1,), (8732,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["other"],
                    "shape": [(1444,), (361,), (100,), (25,), (9,), (1,), (8732,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'expand_1': dict(
        name=["expand"],
        interface=["torch.Tensor"],
        para=dict(
            size=[(1444, 4), (361, 6), (100, 6), (25, 6), (9, 4), (1, 4)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1444, 1), (361, 1), (100, 1), (25, 1), (9, 1), (1, 1)],
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
                    "shape": [((5776, 4), (2166, 4), (600, 4), (150, 4), (36, 4), (4, 4)), ((1000, 4), (1000, 4), (672, 4), (0, 4), (0, 4), (0, 4)), ((1000,), (1000,), (672,), (0,), (0,), (0,))],
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
                    "shape": [((5776,), (2166,), (600,), (150,), (36,), (4,))],
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
                    "shape": [(8732,)],
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
                    "shape": [(14,), (8732,), (14, 8732, 2), (14, 8732), (71,), (19,), (9,), (9, 8732, 2), (9, 8732), (13,), (6,), (6, 8732, 2), (6, 8732), (34,), (1,), (1, 8732, 2), (1, 8732), (2,), (8,), (3,), (3, 8732, 2), (3, 8732), (17,), (5,), (5, 8732, 2), (5, 8732), (30,), (8732, 4), (1000, 2), (672, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(14,), (8732,), (14, 8732, 2), (14, 8732), (71,), (19,), (9,), (9, 8732, 2), (9, 8732), (13,), (6,), (6, 8732, 2), (6, 8732), (34,), (1,), (1, 8732, 2), (1, 8732), (2,), (8,), (3,), (3, 8732, 2), (3, 8732), (17,), (5,), (5, 8732, 2), (5, 8732), (30,), (8732, 4), (1000, 2), (672, 2)],
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
                    "shape": [(14, 1, 2), (14, 8732), (9, 1, 2), (9, 8732), (6, 1, 2), (6, 8732), (1, 1, 2), (1, 8732), (3, 1, 2), (3, 8732), (5, 1, 2), (5, 8732)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 8732, 2), (1,), (1, 8732, 2), (1,), (1, 8732, 2), (1,), (1, 8732, 2), (1,), (1, 8732, 2), (1,), (1, 8732, 2), (1,)],
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
                    "shape": [(14, 1, 2), (9, 1, 2), (6, 1, 2), (1, 1, 2), (3, 1, 2), (5, 1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 8732, 2), (1, 8732, 2), (1, 8732, 2), (1, 8732, 2), (1, 8732, 2), (1, 8732, 2)],
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
                    "shape": [(14, 8732, 2), (9, 8732, 2), (6, 8732, 2), (1, 8732, 2), (3, 8732, 2), (5, 8732, 2)],
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
            dim=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(14, 8732), (14, 8732), (9, 8732), (9, 8732), (6, 8732), (6, 8732), (1, 8732), (1, 8732), (3, 8732), (3, 8732), (5, 8732), (5, 8732)],
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
            other=[0, 0.5, 0.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8732,), (8732,), ()],
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
            other=[0.5, 1.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8732,), (8732, 4)],
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
            other=[1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(71,), (8,), (11,), (34,), (2,), (17,), (30,)],
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
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8732,)],
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
                    "shape": [(8732,)],
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
            other=[1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(71,), (19,), (13,), (34,), (2,), (8,), (17,), (30,)],
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
                    "shape": [(71,), (8661,), (19,), (8713,), (13,), (8719,), (34,), (8698,), (2,), (8730,), (8,), (8724,), (17,), (8715,), (30,), (8702,)],
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
            other=[0, 80],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8732,), (8732,)],
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
                    "shape": [(71,), (19,), (13,), (34,), (2,), (8,), (17,), (30,)],
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
                    "shape": [(71, 4), (19, 4), (13, 4), (34, 4), (2, 4), (8, 4), (17, 4), (30, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)],
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
                    "shape": [(71, 4), (19, 4), (13, 4), (34, 4), (2, 4), (8, 4), (17, 4), (30, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4)],
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
                    "shape": [((8732,), (8732,), (8732,), (8732,), (8732,), (8732,), (8732,), (8732,))],
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
                    "shape": [((8732,), (8732,), (8732,), (8732,), (8732,), (8732,), (8732,), (8732,)), ((8732, 4), (8732, 4), (8732, 4), (8732, 4), (8732, 4), (8732, 4), (8732, 4), (8732, 4))],
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
            dims=[(0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1), (0, 2, 3, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 324, 38, 38), (8, 486, 19, 19), (8, 486, 10, 10), (8, 486, 5, 5), (8, 324, 3, 3), (8, 324, 1, 1), (8, 16, 38, 38), (8, 24, 19, 19), (8, 24, 10, 10), (8, 24, 5, 5), (8, 16, 3, 3), (8, 16, 1, 1)],
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
            dim=[1, -1, -2, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((8, 5776, 81), (8, 2166, 81), (8, 600, 81), (8, 150, 81), (8, 36, 81), (8, 4, 81)), ((8, 5776), (8, 2166), (8, 600), (8, 150), (8, 36), (8, 4)), ((8, 5776, 4), (8, 2166, 4), (8, 600, 4), (8, 150, 4), (8, 36, 4), (8, 4, 4)), ((1701, 4), (1701, 1))],
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
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((8, 5776), (8, 2166), (8, 600), (8, 150), (8, 36), (8, 4))],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        para=dict(
            reduction=['none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(8732, 81)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(8732,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'ge_1': dict(
        name=["ge"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8732,)],
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
            other=[80],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8732,)],
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
            k=[213, 57, 39, 102, 6, 24, 51, 90],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8661,), (8713,), (8719,), (8698,), (8730,), (8724,), (8715,), (8702,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sum_1': dict(
        name=["sum"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(71,), (213,), (8732, 4), (19,), (57,), (13,), (39,), (34,), (102,), (2,), (6,), (8,), (24,), (17,), (51,), (30,), (90,)],
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
            other=[194, 1.0, 194.0000001192093],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (8732, 4), ()],
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
                    "shape": [(8732, 4)],
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
            other=[0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8732, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sub_3': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8732, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'where': dict(
        name=["where"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["condition"],
                    "shape": [(8732, 4)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["input"],
                    "shape": [(8732, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(8732, 4)],
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
                    "shape": [(1,), ()],
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

    'sgd': dict(
        name=["sgd"],
        interface=["CustomizedTest"],
        para=dict(
            nesterov=[False for i in range(32)],
            lr=[2.000000000000002e-06 for i in range(32)],
            momentum=[0.9 for i in range(32)],
            weight_decay=[0.0005 for i in range(32)],
            dampening=[0 for i in range(32)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(64, 3, 3, 3), (64,), (64, 64, 3, 3), (128, 64, 3, 3), (128,), (128, 128, 3, 3), (256, 128, 3, 3), (256,), (256, 256, 3, 3), (512, 256, 3, 3), (512,), (512, 512, 3, 3), (1024, 512, 3, 3), (1024,), (1024, 1024, 1, 1), (256, 1024, 1, 1), (128, 512, 1, 1), (128, 256, 1, 1), (324, 512, 3, 3), (324,), (486, 1024, 3, 3), (486,), (486, 512, 3, 3), (486, 256, 3, 3), (324, 256, 3, 3), (16, 512, 3, 3), (16,), (24, 1024, 3, 3), (24,), (24, 512, 3, 3), (24, 256, 3, 3), (16, 256, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(64, 3, 3, 3), (64,), (64, 64, 3, 3), (128, 64, 3, 3), (128,), (128, 128, 3, 3), (256, 128, 3, 3), (256,), (256, 256, 3, 3), (512, 256, 3, 3), (512,), (512, 512, 3, 3), (1024, 512, 3, 3), (1024,), (1024, 1024, 1, 1), (256, 1024, 1, 1), (128, 512, 1, 1), (128, 256, 1, 1), (324, 512, 3, 3), (324,), (486, 1024, 3, 3), (486,), (486, 512, 3, 3), (486, 256, 3, 3), (324, 256, 3, 3), (16, 512, 3, 3), (16,), (24, 1024, 3, 3), (24,), (24, 512, 3, 3), (24, 256, 3, 3), (16, 256, 3, 3)],
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
            dims=[(1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0), (1, 2, 0)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 38, 38), (324, 38, 38), (24, 19, 19), (486, 19, 19), (24, 10, 10), (486, 10, 10), (24, 5, 5), (486, 5, 5), (16, 3, 3), (324, 3, 3), (16, 1, 1), (324, 1, 1)],
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
            other=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(5776, 80), (2166, 80), (600, 80), (150, 80), (36, 80), (4, 80)],
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
                    "shape": [(5776, 80), (2166, 80), (600, 80), (150, 80), (36, 80), (4, 80)],
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
            descending=[True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(76774,), (19898,), (672,), (0,)],
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
            min=[-4.135166556742356, -4.135166556742356],
            max=[4.135166556742356, 4.135166556742356],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1000, 2), (672, 2)],
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
                    "shape": [(1000, 2), (672, 2)],
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
            dim=[-1, -1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((1000, 2), (1000, 2)), ((672, 2), (672, 2)), ((1701, 4), (1701, 1))],
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
            min=[0, 0],
            max=[300, 300],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1000, 2), (672, 2)],
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
                    "shape": [(2672, 4)],
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

    'cat_5': dict(
        name=["cat"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((1000,), (1000,), (672,), (0,), (0,), (0,))],
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
                    "shape": [(2672, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
