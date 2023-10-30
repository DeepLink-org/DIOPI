from ...config import Genfunc
from ...diopi_runtime import Dtype

efficientnet_config = {
    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        interface=["torch.nn.functional"],
        para=dict(
            size=[(1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(32, 288, 17, 17), (16, 144, 65, 65)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1, 0.0001, 0.0001, 0.0001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 120, 17, 17), (32, 48, 33, 33), (48,), (1408,), (1000, 1408)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(15, 120, 17, 17), (32, 48, 33, 33), (48,), (1408,), (1000, 1408)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_case_2': dict(
        name=["add"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            alpha=[1, -0.1, -0.1, -0.1, 1, -0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 6, 1, 1), (16, 1, 3, 3), (16,), (22,), (1000, 1408), (1000, 1408)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(32, 6, 1, 1), (16, 1, 3, 3), (16,), (22,), (1000, 1408), (1000, 1408)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_case_3': dict(
        name=["add"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[1],
            alpha=[1],
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

    'add_case_4': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            other=[0],
            alpha=[1],
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

    'argmax': dict(
        name=["argmax"],
        interface=["torch"],
        para=dict(
            dim=[1, 1],
            keepdim=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(32, 1000), (16, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'batch_norm': dict(
        name=["batch_norm"],
        atol=1e-01,
        rtol=1e-02,
        atol_half=1e-01,
        rtol_half=1e-02,
        interface=["torch.nn.functional"],
        para=dict(
            training=[True, True],
            momentum=[0.1, 0.1],
            eps=[0.001, 0.001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 96, 65, 65), (32, 32, 130, 130)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(96,), (32,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(96,), (32,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad":[False],
                    "shape": [(96,), (32,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "requires_grad":[False],
                    "shape": [(96,), (32,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.positive,
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(864,), (32,), (32,), (288,), (32,), (32,), (256,), (8,), (256,), (32,), (512,), (16,), (16,), (144,), (16,), (16,), (64,), (4,), (64,), (16,), (256,), (16,), (16,), (1536,), (96,), (96,), (864,), (96,), (96,), (384,), (4,), (384,), (96,), (2304,), (24,), (24,), (3456,), (144,), (144,), (1296,), (144,), (144,), (864,), (6,), (864,), (144,), (3456,), (24,), (24,), (3456,), (144,), (144,), (1296,), (144,), (144,), (864,), (6,), (864,), (144,), (3456,), (24,), (24,), (3456,), (144,), (144,), (3600,), (144,), (144,), (864,), (6,), (864,), (144,), (6912,), (48,), (48,), (13824,), (288,), (288,), (7200,), (288,), (288,), (3456,), (12,), (3456,), (288,), (13824,), (48,), (48,), (13824,), (288,), (288,), (7200,), (288,), (288,), (3456,), (12,), (3456,), (288,), (13824,), (48,), (48,), (13824,), (288,), (288,), (2592,), (288,), (288,), (3456,), (12,), (3456,), (288,), (25344,), (88,), (88,), (46464,), (528,), (528,), (4752,), (528,), (528,), (11616,), (22,), (11616,), (528,), (46464,), (88,), (88,), (46464,), (528,), (528,), (4752,), (528,), (528,), (11616,), (22,), (11616,), (528,), (46464,), (88,), (88,), (46464,), (528,), (528,), (4752,), (528,), (528,), (11616,), (22,), (11616,), (528,), (46464,), (88,), (88,), (46464,), (528,), (528,), (13200,), (528,), (528,), (11616,), (22,), (11616,), (528,), (63360,), (120,), (120,), (86400,), (720,), (720,), (18000,), (720,), (720,), (21600,), (30,), (21600,), (720,), (86400,), (120,), (120,), (86400,), (720,), (720,), (18000,), (720,), (720,), (21600,), (30,), (21600,), (720,), (86400,), (120,), (120,), (86400,), (720,), (720,), (18000,), (720,), (720,), (21600,), (30,), (21600,), (720,), (86400,), (120,), (120,), (86400,), (720,), (720,), (18000,), (720,), (720,), (21600,), (30,), (21600,), (720,), (149760,), (208,), (208,), (259584,), (1248,), (1248,), (31200,), (1248,), (1248,), (64896,), (52,), (64896,), (1248,), (259584,), (208,), (208,), (259584,), (1248,), (1248,), (31200,), (1248,), (1248,), (64896,), (52,), (64896,), (1248,), (259584,), (208,), (208,), (259584,), (1248,), (1248,), (31200,), (1248,), (1248,), (64896,), (52,), (64896,), (1248,), (259584,), (208,), (208,), (259584,), (1248,), (1248,), (31200,), (1248,), (1248,), (64896,), (52,), (64896,), (1248,), (259584,), (208,), (208,), (259584,), (1248,), (1248,), (11232,), (1248,), (1248,), (64896,), (52,), (64896,), (1248,), (439296,), (352,), (352,), (743424,), (2112,), (2112,), (19008,), (2112,), (2112,), (185856,), (88,), (185856,), (2112,), (743424,), (352,), (352,), (495616,), (1408,), (1408,), (1408000,), (1000,), (3,), (3,), (32,), (32,), (32,), (32,), (16,), (16,), (16,), (16,), (16,), (16,), (96,), (96,), (96,), (96,), (24,), (24,), (144,), (144,), (144,), (144,), (24,), (24,), (144,), (144,), (144,), (144,), (24,), (24,), (144,), (144,), (144,), (144,), (48,), (48,), (288,), (288,), (288,), (288,), (48,), (48,), (288,), (288,), (288,), (288,), (48,), (48,), (288,), (288,), (288,), (288,), (88,), (88,), (528,), (528,), (528,), (528,), (88,), (88,), (528,), (528,), (528,), (528,), (88,), (88,), (528,), (528,), (528,), (528,), (88,), (88,), (528,), (528,), (528,), (528,), (120,), (120,), (720,), (720,), (720,), (720,), (120,), (120,), (720,), (720,), (720,), (720,), (120,), (120,), (720,), (720,), (720,), (720,), (120,), (120,), (720,), (720,), (720,), (720,), (208,), (208,), (1248,), (1248,), (1248,), (1248,), (208,), (208,), (1248,), (1248,), (1248,), (1248,), (208,), (208,), (1248,), (1248,), (1248,), (1248,), (208,), (208,), (1248,), (1248,), (1248,), (1248,), (208,), (208,), (1248,), (1248,), (1248,), (1248,), (352,), (352,), (2112,), (2112,), (2112,), (2112,), (352,), (352,), (1408,), (1408,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'conv2d': dict(
        name=["conv2d"],
        atol=1e-03,
        rtol=1e-03,
        interface=["torch.nn.functional"],
        para=dict(
            stride=[(1, 1), (1, 1)],
            padding=[(0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 144, 1, 1), (32, 720, 9, 9)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(6, 144, 1, 1), (208, 720, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(6,), None],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div': dict(
        name=["div"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 3, 260, 260), (15, 3, 260, 260), (16, 3, 260, 260)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (3, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_case_2': dict(
        name=["div"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 32, 32, 1, 15, 15],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (), (), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'fill_': dict(
        name=["fill_"],
        interface=["torch.Tensor"],
        para=dict(
            value=[0, 0, 1, 0, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 96, 131, 131), (16, 288, 35, 35), (720,), (4,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'flip': dict(
        name=["flip"],
        interface=["torch"],
        para=dict(
            dims=[(1,), (1,), (1,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 3, 260, 260), (16, 3, 260, 260), (15, 3, 260, 260)],
                    "dtype": [Dtype.uint8],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'linear': dict(
        name=["linear"],
        atol=1e-03,
        rtol=1e-04,
        atol_half=1e-01,
        rtol_half=1e-02,
        interface=["torch.nn.functional"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 1408), (32, 1408), (15, 1408), (16, 1408)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(1000, 1408), (1000, 1408), (1000, 1408), (1000, 1408)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(1000,), (1000,), (1000,), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        interface=["torch.nn.functional"],
        saved_args=dict(output=0),
        para=dict(
            dim=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 1000), (15, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mean': dict(
        name=["mean"],
        interface=["torch"],
        para=dict(
            dim=[None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(96, 16, 1, 1), (2112, 352, 1, 1), (1000,), (96,), (1000, 1408), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul': dict(
        name=["mul"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 96, 65, 65), (32, 720, 17, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(15, 96, 1, 1), (32, 720, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_case_2': dict(
        name=["mul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.9, 0.9, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2112, 88, 1, 1), (288, 1, 5, 5), (208,), (22,), (1000, 1408)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_case_3': dict(
        name=["mul"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1408,), (352,), (), (), (96, 4, 1, 1), (2112, 88, 1, 1), (1000, 1408)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'nll_loss': dict(
        name=["nll_loss"],
        interface=["torch.nn.functional"],
        para=dict(
            weight=[None, None],
            reduction=['none', 'none'],
            ignore_index=[-100, -100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 1000), (15, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "requires_grad":[False],
                    "shape": [(32,), (15,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'normal_': dict(
        name=["normal_"],
        no_output_ref=True,
        interface=["torch.nn.functional"],
        para=dict(
            mean=[0, 0, 0],
            std=[0.353553, 0.150756, 0.01],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 16, 1, 1), (88, 2112, 1, 1), (1000, 1408)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sigmoid': dict(
        name=["sigmoid"],
        interface=["torch"],
        saved_args=dict(output=0),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 8, 1, 1), (32, 720, 9, 9)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'softmax': dict(
        name=["softmax"],
        interface=["torch.nn.functional"],
        saved_args=dict(output=0),
        para=dict(
            dim=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(32, 1000), (16, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sub': dict(
        name=["sub"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 3, 260, 260), (15, 3, 260, 260), (16, 3, 260, 260)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (3, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        interface=["torch"],
        para=dict(
            dim=[[2, 3], [2, 3], None, None],
            keepdim=[True, True, False, False],
            dtype=[None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 144, 65, 65), (15, 2112, 9, 9), (32,), (15,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
