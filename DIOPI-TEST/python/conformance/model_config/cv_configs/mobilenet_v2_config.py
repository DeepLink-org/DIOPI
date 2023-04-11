from ...config import Genfunc
from ...dtype import Dtype

mobilenet_v2_config = {
    'randperm': dict(
        name=["randperm"],
        no_output_ref=True,
        para=dict(
            n=[1281167],
        ),
    ),

    'conv2d': dict(
        name=["conv2d"],
        para=dict(
            bias=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            stride=[(2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(1, 1), (1, 1), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 32, 1, 1, 96, 1, 1, 144, 1, 144, 1, 1, 192, 1, 192, 1, 1, 384, 1, 1, 1, 576, 1, 576, 1, 1, 960, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 3, 224, 224), (32, 32, 112, 112), (32, 32, 112, 112), (32, 16, 112, 112), (32, 96, 112, 112), (32, 96, 56, 56), (32, 24, 56, 56), (32, 144, 56, 56), (32, 144, 56, 56), (32, 144, 56, 56), (32, 144, 28, 28), (32, 32, 28, 28), (32, 192, 28, 28), (32, 192, 28, 28), (32, 192, 28, 28), (32, 192, 14, 14), (32, 64, 14, 14), (32, 384, 14, 14), (32, 384, 14, 14), (32, 384, 14, 14), (32, 96, 14, 14), (32, 576, 14, 14), (32, 576, 14, 14), (32, 576, 14, 14), (32, 576, 7, 7), (32, 160, 7, 7), (32, 960, 7, 7), (32, 960, 7, 7), (32, 960, 7, 7), (32, 320, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(32, 3, 3, 3), (32, 1, 3, 3), (16, 32, 1, 1), (96, 16, 1, 1), (96, 1, 3, 3), (24, 96, 1, 1), (144, 24, 1, 1), (144, 1, 3, 3), (24, 144, 1, 1), (144, 1, 3, 3), (32, 144, 1, 1), (192, 32, 1, 1), (192, 1, 3, 3), (32, 192, 1, 1), (192, 1, 3, 3), (64, 192, 1, 1), (384, 64, 1, 1), (384, 1, 3, 3), (64, 384, 1, 1), (96, 384, 1, 1), (576, 96, 1, 1), (576, 1, 3, 3), (96, 576, 1, 1), (576, 1, 3, 3), (160, 576, 1, 1), (960, 160, 1, 1), (960, 1, 3, 3), (160, 960, 1, 1), (320, 960, 1, 1), (1280, 320, 1, 1)],
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
            other=[1],
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

    'batch_norm': dict(
        name=["batch_norm"],
        para=dict(
            training=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 32, 112, 112), (32, 16, 112, 112), (32, 96, 112, 112), (32, 96, 56, 56), (32, 24, 56, 56), (32, 144, 56, 56), (32, 144, 28, 28), (32, 32, 28, 28), (32, 192, 28, 28), (32, 192, 14, 14), (32, 64, 14, 14), (32, 384, 14, 14), (32, 96, 14, 14), (32, 576, 14, 14), (32, 576, 7, 7), (32, 160, 7, 7), (32, 960, 7, 7), (32, 320, 7, 7), (32, 1280, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": [(32,), (16,), (96,), (96,), (24,), (144,), (144,), (32,), (192,), (192,), (64,), (384,), (96,), (576,), (576,), (160,), (960,), (320,), (1280,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "shape": [(32,), (16,), (96,), (96,), (24,), (144,), (144,), (32,), (192,), (192,), (64,), (384,), (96,), (576,), (576,), (160,), (960,), (320,), (1280,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(32,), (16,), (96,), (96,), (24,), (144,), (144,), (32,), (192,), (192,), (64,), (384,), (96,), (576,), (576,), (160,), (960,), (320,), (1280,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(32,), (16,), (96,), (96,), (24,), (144,), (144,), (32,), (192,), (192,), (64,), (384,), (96,), (576,), (576,), (160,), (960,), (320,), (1280,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'hardtanh': dict(
        name=["hardtanh"],
        para=dict(
            min_val=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            max_val=[6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0],
            inplace=[True, True, True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 32, 112, 112), (32, 96, 112, 112), (32, 96, 56, 56), (32, 144, 56, 56), (32, 144, 28, 28), (32, 192, 28, 28), (32, 192, 14, 14), (32, 384, 14, 14), (32, 576, 14, 14), (32, 576, 7, 7), (32, 960, 7, 7), (32, 1280, 7, 7)],
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
                    "shape": [(32, 24, 56, 56), (32, 32, 28, 28), (32, 64, 14, 14), (32, 96, 14, 14), (32, 160, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(32, 24, 56, 56), (32, 32, 28, 28), (32, 64, 14, 14), (32, 96, 14, 14), (32, 160, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        para=dict(
            output_size=[(1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 1280, 7, 7)],
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
                    "shape": [(32, 1280)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(1000, 1280)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
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
                    "shape": [(32, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(32,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
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
                    "shape": [(32,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[32],
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

    'mul': dict(
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

    'add_2': dict(
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
            nesterov=[False for i in range(42)],
            lr=[0.045 for i in range(42)],
            momentum=[0.9 for i in range(42)],
            weight_decay=[4e-05 for i in range(42)],
            dampening=[0 for i in range(42)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(32, 3, 3, 3), (32,), (32, 1, 3, 3), (16, 32, 1, 1), (16,), (96, 16, 1, 1), (96,), (96, 1, 3, 3), (24, 96, 1, 1), (24,), (144, 24, 1, 1), (144,), (144, 1, 3, 3), (24, 144, 1, 1), (32, 144, 1, 1), (192, 32, 1, 1), (192,), (192, 1, 3, 3), (32, 192, 1, 1), (64, 192, 1, 1), (64,), (384, 64, 1, 1), (384,), (384, 1, 3, 3), (64, 384, 1, 1), (96, 384, 1, 1), (576, 96, 1, 1), (576,), (576, 1, 3, 3), (96, 576, 1, 1), (160, 576, 1, 1), (160,), (960, 160, 1, 1), (960,), (960, 1, 3, 3), (160, 960, 1, 1), (320, 960, 1, 1), (320,), (1280, 320, 1, 1), (1280,), (1000, 1280), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(32, 3, 3, 3), (32,), (32, 1, 3, 3), (16, 32, 1, 1), (16,), (96, 16, 1, 1), (96,), (96, 1, 3, 3), (24, 96, 1, 1), (24,), (144, 24, 1, 1), (144,), (144, 1, 3, 3), (24, 144, 1, 1), (32, 144, 1, 1), (192, 32, 1, 1), (192,), (192, 1, 3, 3), (32, 192, 1, 1), (64, 192, 1, 1), (64,), (384, 64, 1, 1), (384,), (384, 1, 3, 3), (64, 384, 1, 1), (96, 384, 1, 1), (576, 96, 1, 1), (576,), (576, 1, 3, 3), (96, 576, 1, 1), (160, 576, 1, 1), (160,), (960, 160, 1, 1), (960,), (960, 1, 3, 3), (160, 960, 1, 1), (320, 960, 1, 1), (320,), (1280, 320, 1, 1), (1280,), (1000, 1280), (1000,)],
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
            dim=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 1000), (16, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
