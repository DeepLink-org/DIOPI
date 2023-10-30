from ...config import Genfunc
from ...diopi_runtime import Dtype

yolov3_config = {
    'add': dict(
        name=["add"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1, 1, 0.0005, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(17,), (19,), (8, 256, 28, 40), (64, 128, 1, 1), (52, 1), (18, 1), (1, 3, 4), (1, 3, 4), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(17,), (19,), (8, 256, 28, 40), (64, 128, 1, 1), (1, 6300), (1, 6300), (1280, 1, 4), (100, 1, 4), ()],
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
            alpha=[1, -0.000864932, -0.00014097, -0.000617309, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 256, 36, 40), (255, 256, 1, 1), (64,), (64,), (8, 840, 80), (8, 960, 2), (8, 960), (8, 3360)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(8, 256, 36, 40), (255, 256, 1, 1), (64,), (64,), (8, 840, 80), (8, 960, 2), (8, 960), (8, 3360)],
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
            other=[0.5, 0.5, 0, 1e-06],
            alpha=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(91,), (8,), (), ()],
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
            start=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            end=[32, 28, 40, 7, 9, 18, 10, 20, 14, 16, 36, 8, 6, 12, 24],
            step=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
            training=[False, False],
            momentum=[0.1, 0.1],
            eps=[1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(8, 32, 224, 320), (8, 64, 160, 160)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(32,), (64,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(32,), (64,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad":[False],
                    "shape": [(32,), (64,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "requires_grad":[False],
                    "shape": [(32,), (64,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.positive,
                },
            ],
        ),
    ),

    'bitwise_and': dict(
        name=["bitwise_and"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(8, 3840, 80), (8, 1200, 80), (3780,), (5040,), (8, 4800), (8, 3360)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["other"],
                    "requires_grad":[False],
                    "shape": [(8, 3840, 80), (8, 1200, 80), (3780,), (5040,), (8, 4800), (8, 3360)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'bitwise_not': dict(
        name=["bitwise_not"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(5040,), (6300,), (5670,), (4410,), (3780,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'cat': dict(
        name=["cat"],
        interface=["torch"],
        para=dict(
            dim=[0, 0, 1, 1, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(270,), (1080,), (4320,)], [(300,), (1200,), (4800,)], [(8, 128, 40, 28), (8, 256, 40, 28)], [(8, 256, 20, 20), (8, 512, 20, 20)], [(3,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(864,), (32,), (32,), (18432,), (64,), (64,), (2048,), (32,), (32,), (18432,), (64,), (64,), (73728,), (128,), (128,), (8192,), (64,), (64,), (73728,), (128,), (128,), (8192,), (64,), (64,), (73728,), (128,), (128,), (294912,), (256,), (256,), (32768,), (128,), (128,), (294912,), (256,), (256,), (32768,), (128,), (128,), (294912,), (256,), (256,), (32768,), (128,), (128,), (294912,), (256,), (256,), (32768,), (128,), (128,), (294912,), (256,), (256,), (32768,), (128,), (128,), (294912,), (256,), (256,), (32768,), (128,), (128,), (294912,), (256,), (256,), (32768,), (128,), (128,), (294912,), (256,), (256,), (32768,), (128,), (128,), (294912,), (256,), (256,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (1179648,), (512,), (512,), (4718592,), (1024,), (1024,), (524288,), (512,), (512,), (4718592,), (1024,), (1024,), (524288,), (512,), (512,), (4718592,), (1024,), (1024,), (524288,), (512,), (512,), (4718592,), (1024,), (1024,), (524288,), (512,), (512,), (4718592,), (1024,), (1024,), (524288,), (512,), (512,), (4718592,), (1024,), (1024,), (524288,), (512,), (512,), (4718592,), (1024,), (1024,), (524288,), (512,), (512,), (131072,), (256,), (256,), (196608,), (256,), (256,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (1179648,), (512,), (512,), (131072,), (256,), (256,), (32768,), (128,), (128,), (49152,), (128,), (128,), (294912,), (256,), (256,), (32768,), (128,), (128,), (294912,), (256,), (256,), (32768,), (128,), (128,), (4718592,), (1024,), (1024,), (1179648,), (512,), (512,), (294912,), (256,), (256,), (261120,), (255,), (130560,), (255,), (65280,), (255,), (3,), (32,), (32,), (64,), (64,), (32,), (32,), (64,), (64,), (128,), (128,), (64,), (64,), (128,), (128,), (64,), (64,), (128,), (128,), (256,), (256,), (128,), (128,), (256,), (256,), (128,), (128,), (256,), (256,), (128,), (128,), (256,), (256,), (128,), (128,), (256,), (256,), (128,), (128,), (256,), (256,), (128,), (128,), (256,), (256,), (128,), (128,), (256,), (256,), (128,), (128,), (256,), (256,), (512,), (512,), (256,), (256,), (512,), (512,), (256,), (256,), (512,), (512,), (256,), (256,), (512,), (512,), (256,), (256,), (512,), (512,), (256,), (256,), (512,), (512,), (256,), (256,), (512,), (512,), (256,), (256,), (512,), (512,), (256,), (256,), (512,), (512,), (1024,), (1024,), (512,), (512,), (1024,), (1024,), (512,), (512,), (1024,), (1024,), (512,), (512,), (1024,), (1024,), (512,), (512,), (1024,), (1024,), (512,), (512,), (1024,), (1024,), (512,), (512,), (1024,), (1024,), (512,), (512,), (256,), (256,), (256,), (256,), (512,), (512,), (256,), (256,), (512,), (512,), (256,), (256,), (128,), (128,), (128,), (128,), (256,), (256,), (128,), (128,), (256,), (256,), (128,), (128,), (1024,), (1024,), (512,), (512,), (256,), (256,)]],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'clamp_min': dict(
        name=["clamp_min"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            min=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(8, 4320, 80), (2, 2880, 80), (8, 240), (2, 180)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'clamp': dict(
        name=["clamp"],
        atol=1e-04,
        rtol=1e-05,
        interface=["torch"],
        para=dict(
            min=[0, 0, 1e-06, 1e-06, None],
            max=[None, None, None, 0.999999, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(34, 6300, 2), (20, 6300, 2), (36,), (67,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'conv2d': dict(
        name=["conv2d"],
        atol=1e-03,
        rtol=1e-03,
        interface=["torch.nn.functional"],
        para=dict(
            stride=[(1, 1), (1, 1)],
            padding=[(1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(8, 32, 160, 112), (8, 3, 320, 224)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(64, 32, 3, 3), (32, 3, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[False],
                    "shape": [None, None],
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
                    "shape": [(49, 5670), (60, 6300), (3, 320, 211), (3, 186, 320), (42,), (14,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(49, 5670), (60, 6300), (3, 1, 1), (3, 1, 1), (42,), (14,)],
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
            other=[8, 16, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15,), (45,), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'eq': dict(
        name=["eq"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4410,), (5040,), (6300,), (5670,), (3780,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'eq_case_2': dict(
        name=["eq"],
        interface=["torch"],
        para=dict(
            other=[0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(5670,), (5040,), (4410,), (6300,), (3780,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'exp': dict(
        name=["exp"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(2, 720, 80), (8, 240, 80), (2, 180), (8, 3360)],
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
            value=[-1, -1, 1, 0, 0, 0, 42, 86],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4410,), (3780,), (8, 270, 2), (3, 320, 224), (43, 80), (12, 80), (), ()],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
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
                    "requires_grad":[False],
                    "shape": [(51,), (8,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'ge': dict(
        name=["ge"],
        interface=["torch"],
        para=dict(
            other=[0, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 840), (8, 1080), (5670,), (6300,), (8, 3360, 80), (8, 240, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'gt': dict(
        name=["gt"],
        interface=["torch"],
        para=dict(
            other=[0, 1, 0, 0.5, 0, 0.5, 0.5, 0, 0, 0.5, 0, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (5040,), (5040,), (4410,), (4410,), (6300,), (6300,), (5670,), (5670,), (3780,), (3780,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'index': dict(
        name=["index"],
        interface=["CustomizedTest"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(13, 4), (34, 80), (42,), (12,), (3, 320, 305), (3, 266, 320)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["indices"],
                    "requires_grad":[False],
                    "shape": [(26,), (25,), (48,), (9,), (3,), (3,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, low=-3, high=3),
                },
            ],
        ),
    ),

    'index_put': dict(
        name=["index_put"],
        interface=["CustomizedTest"],
        para=dict(
            accumulate=[False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(400,), (1600,), (52, 6300), (5040, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["indices1"],
                    "requires_grad":[False],
                    "shape": [(31,), (33,), (6300,), (37,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
                {
                    "ins": ["values"],
                    "requires_grad":[False],
                    "shape": [(), (), (), (37, 80)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'le': dict(
        name=["le"],
        interface=["torch"],
        para=dict(
            other=[0.5, 0.5, 0.5, 0.5, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4410,), (5670,), (6300,), (5040,), (3780,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'leaky_relu': dict(
        name=["leaky_relu"],
        interface=["torch.nn.functional"],
        is_inplace=[True],
        para=dict(
            negative_slope=[0.1, 0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(8, 128, 20, 20), (8, 1024, 8, 10)],
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
                    "requires_grad":[False],
                    "shape": [(35,), (74,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.positive,
                },
            ],
        ),
    ),

    'log_case_2': dict(
        name=["log"],
        interface=["torch"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(2, 2880, 80), (8, 4320, 80), (2, 2880), (8, 960)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.positive,
                },
            ],
        ),
    ),

    'mse_loss': dict(
        name=["mse_loss"],
        interface=["torch.nn.functional"],
        para=dict(
            reduction=['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(8, 3360, 2), (8, 840, 2), (8, 210, 2), (8, 3840, 2), (8, 4800, 2), (8, 300, 2), (8, 1080, 2), (8, 270, 2), (8, 4320, 2), (8, 240, 2), (8, 1200, 2), (8, 960, 2), (2, 720, 2), (2, 2880, 2), (2, 180, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "requires_grad":[False],
                    "shape": [(8, 3360, 2), (8, 840, 2), (8, 210, 2), (8, 3840, 2), (8, 4800, 2), (8, 300, 2), (8, 1080, 2), (8, 270, 2), (8, 4320, 2), (8, 240, 2), (8, 1200, 2), (8, 960, 2), (2, 720, 2), (2, 2880, 2), (2, 180, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max': dict(
        name=["max"],
        interface=["torch"],
        para=dict(
            dim=[0, 1],
            keepdim=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(26, 5040), (1, 6300)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_case_2': dict(
        name=["max"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(8, 210), (8, 4800), (40,), (74,)],
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
                    "requires_grad":[False],
                    "shape": [(20, 5670), (3, 5670), (28, 1, 2), (11, 1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "requires_grad":[False],
                    "shape": [(1,), (1,), (1, 5040, 2), (1, 5670, 2)],
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
            dim=[None, None, None, None, None],
            keepdim=[False, False, False, False, False],
            dtype=[None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(256, 128, 3, 3), (128, 384, 1, 1), (256,), (32,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'min': dict(
        name=["min"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(66,), (29,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
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
                    "requires_grad":[False],
                    "shape": [(30, 1, 2), (27, 1, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "requires_grad":[False],
                    "shape": [(1, 5040, 2), (1, 6300, 2)],
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
                    "shape": [(49, 6300), (3, 6300), (8, 960, 80), (8, 960, 1), (53,), (10,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(49, 6300), (3, 6300), (8, 960, 80), (8, 960, 80), (53,), (10,)],
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
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(256, 512, 1, 1), (255, 512, 1, 1), (8, 4800, 2), (8, 240, 2), (8, 270), (8, 1200), (64,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (8, 4800, 2), (8, 240, 2), (8, 270), (8, 1200), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_case_3': dict(
        name=["mul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.9, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 32, 3, 3), (512, 1024, 1, 1), (255,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_case_4': dict(
        name=["mul"],
        interface=["torch"],
        para=dict(
            other=[10, 20, 1, 1, 35, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(37,), (48,), (256, 128, 3, 3), (64, 128, 1, 1), (), ()],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'ne': dict(
        name=["ne"],
        interface=["torch"],
        para=dict(
            other=[-100, -100, -100, -100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 960, 2), (8, 210, 80), (2, 2880), (8, 840)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'neg': dict(
        name=["neg"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 840, 80), (8, 240, 80), (2, 180), (2, 180)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "requires_grad":[False],
                    "shape": [(5670,), (5040,), (4410,), (6300,), (3780,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'norm': dict(
        name=["norm"],
        interface=["torch"],
        para=dict(
            p=[2, 2, 2, 2],
            dim=[(0, 1, 2, 3), (0, 1, 2, 3), (0,), (0,)],
            keepdim=[False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(256, 128, 3, 3), (255, 512, 1, 1), (1024,), (64,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'normal_': dict(
        name=["normal_"],
        no_output_ref=True,
        interface=["torch.nn.functional"],
        para=dict(
            mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            std=[0.0146583, 0.01, 0.0879497, 0.0207299, 0.01, 0.12438, 0.0879497, 0.01, 0.0621898, 0.01, 0.0293166, 0.01, 0.0829198, 0.12438, 0.01],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1024, 512, 3, 3), (1024, 512, 3, 3), (256, 512, 1, 1), (512, 256, 3, 3), (512, 256, 3, 3), (128, 256, 1, 1), (256, 768, 1, 1), (255, 1024, 1, 1), (512, 1024, 1, 1), (256, 128, 3, 3), (256, 128, 3, 3), (255, 512, 1, 1), (32, 3, 3, 3), (128, 384, 1, 1), (255, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'reciprocal': dict(
        name=["reciprocal"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'repeat': dict(
        name=["repeat"],
        interface=["torch.Tensor"],
        para=dict(
            repeats=[(1, 20), (1, 20), (1080,), (180,), (9,), (10,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(12, 1), (20, 1), (), (), (10,), (10,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'scatter': dict(
        name=["scatter"],
        interface=["torch"],
        para=dict(
            dim=[-1, -1],
            value=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(13, 80), (9, 80)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
                {
                    "ins": ["index"],
                    "requires_grad":[False],
                    "shape": [(13, 1), (9, 1)],
                    "dtype": [Dtype.int64],
                    "gen_fn": dict(fn=Genfunc.randint, high=80),
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
                    "shape": [(8, 1080, 2), (8, 300, 80), (2, 180), (8, 240)],
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
            dim=[1, 1, 0, 0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(60,), (60,), (60,), (60,)], [(2,), (2,), (2,), (2,)], [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()], [(5040,), (5040,), (5040,), (5040,), (5040,), (5040,), (5040,), (5040,)], [(5670, 85), (5670, 85), (5670, 85), (5670, 85), (5670, 85), (5670, 85), (5670, 85), (5670, 85)], [(3780, 85), (3780, 85)], [(3, 192, 320), (3, 192, 320)]],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'sub': dict(
        name=["sub"],
        interface=["torch"],
        para=dict(
            alpha=[1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 320, 134), (56, 5670, 2), (48, 6300), (36, 6300), (58,), (19,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (56, 5670, 2), (48, 6300), (36, 6300), (58,), (19,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'sub_case_2': dict(
        name=["sub"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            alpha=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 270, 80), (8, 270, 2), (2, 2880), (8, 210)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(8, 270, 80), (8, 270, 2), (2, 2880), (8, 210)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'sub_case_3': dict(
        name=["sub"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1],
            alpha=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(55,), (48,), ()],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'sum': dict(
        name=["sum"],
        interface=["torch"],
        para=dict(
            dim=[None, None, None, None],
            keepdim=[False, False, False, False],
            dtype=[None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(8, 240), (8, 300), (8, 4320, 80), (2, 720, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'unique': dict(
        name=["unique"],
        interface=["torch"],
        para=dict(
            sorted=[True, True],
            return_inverse=[False, False],
            return_counts=[False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(43,), (5640,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        interface=["torch.nn.functional"],
        para=dict(
            mode=['nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest'],
            size=[(20, 16), (20, 18), (40, 28), (20, 20), (28, 40), (40, 40), (36, 40), (18, 20), (32, 40), (20, 14), (40, 32), (14, 20), (16, 20), (40, 36), (12, 20), (24, 40)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(8, 256, 10, 8), (8, 256, 10, 9), (8, 128, 20, 14), (8, 256, 10, 10), (8, 128, 14, 20), (8, 128, 20, 20), (8, 128, 18, 20), (8, 256, 9, 10), (8, 128, 16, 20), (8, 256, 10, 7), (8, 128, 20, 16), (8, 256, 7, 10), (8, 256, 8, 10), (8, 128, 20, 18), (2, 256, 6, 10), (2, 128, 12, 20)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
