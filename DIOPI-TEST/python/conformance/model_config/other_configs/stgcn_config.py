from ...config import Genfunc
from ...dtype import Dtype

stgcn_config = {
    'randperm': dict(
        name=["randperm"],
        no_output_ref=True,
        para=dict(
            n=[40090],
        ),
    ),

    'permute': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[(0, 4, 3, 1, 2), (0, 1, 3, 4, 2), (0, 4, 3, 1, 2), (0, 1, 3, 4, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 3, 300, 17, 2), (16, 2, 17, 3, 300), (10, 3, 300, 17, 2), (10, 2, 17, 3, 300)],
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
            training=[True, True, True, True, True, True, True, True, True, True, True, True],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 51, 300), (32, 64, 300, 17), (32, 128, 150, 17), (32, 128, 300, 17), (32, 256, 75, 17), (32, 256, 150, 17), (20, 51, 300), (20, 64, 300, 17), (20, 128, 150, 17), (20, 128, 300, 17), (20, 256, 75, 17), (20, 256, 150, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": [(51,), (64,), (128,), (128,), (256,), (256,), (51,), (64,), (128,), (128,), (256,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "shape": [(51,), (64,), (128,), (128,), (256,), (256,), (51,), (64,), (128,), (128,), (256,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(51,), (64,), (128,), (128,), (256,), (256,), (51,), (64,), (128,), (128,), (256,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(51,), (64,), (128,), (128,), (256,), (256,), (51,), (64,), (128,), (128,), (256,), (256,)],
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
                    "shape": [(3, 17, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 17, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'conv2d': dict(
        name=["conv2d"],
        para=dict(
            stride=[(1, 1), (1, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(0, 0), (4, 0), (0, 0), (0, 0), (0, 0), (4, 0), (0, 0), (4, 0), (0, 0), (0, 0), (4, 0), (0, 0), (4, 0), (0, 0), (0, 0), (4, 0), (0, 0), (0, 0), (0, 0), (4, 0), (0, 0), (4, 0), (0, 0), (0, 0), (4, 0), (0, 0), (4, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 3, 300, 17), (32, 64, 300, 17), (32, 64, 300, 17), (32, 64, 300, 17), (32, 64, 300, 17), (32, 128, 300, 17), (32, 128, 150, 17), (32, 128, 150, 17), (32, 128, 150, 17), (32, 128, 150, 17), (32, 256, 150, 17), (32, 256, 75, 17), (32, 256, 75, 17), (16, 256, 1, 1), (20, 3, 300, 17), (20, 64, 300, 17), (20, 64, 300, 17), (20, 64, 300, 17), (20, 64, 300, 17), (20, 128, 300, 17), (20, 128, 150, 17), (20, 128, 150, 17), (20, 128, 150, 17), (20, 128, 150, 17), (20, 256, 150, 17), (20, 256, 75, 17), (20, 256, 75, 17), (10, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(192, 3, 1, 1), (64, 64, 9, 1), (192, 64, 1, 1), (128, 64, 1, 1), (384, 64, 1, 1), (128, 128, 9, 1), (384, 128, 1, 1), (128, 128, 9, 1), (256, 128, 1, 1), (768, 128, 1, 1), (256, 256, 9, 1), (768, 256, 1, 1), (256, 256, 9, 1), (60, 256, 1, 1), (192, 3, 1, 1), (64, 64, 9, 1), (192, 64, 1, 1), (128, 64, 1, 1), (384, 64, 1, 1), (128, 128, 9, 1), (384, 128, 1, 1), (128, 128, 9, 1), (256, 128, 1, 1), (768, 128, 1, 1), (256, 256, 9, 1), (768, 256, 1, 1), (256, 256, 9, 1), (60, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(192,), (64,), (192,), (128,), (384,), (128,), (384,), (128,), (256,), (768,), (256,), (768,), (256,), (60,), (192,), (64,), (192,), (128,), (384,), (128,), (384,), (128,), (256,), (768,), (256,), (768,), (256,), (60,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        para=dict(
            inplace=[True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 64, 300, 17), (32, 128, 300, 17), (32, 128, 150, 17), (32, 256, 150, 17), (32, 256, 75, 17), (20, 64, 300, 17), (20, 128, 300, 17), (20, 128, 150, 17), (20, 256, 150, 17), (20, 256, 75, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'dropout': dict(
        name=["dropout"],
        no_output_ref=True,
        para=dict(
            p=[0, 0, 0, 0, 0, 0],
            training=[True, True, True, True, True, True],
            inplace=[True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 64, 300, 17), (32, 128, 150, 17), (32, 256, 75, 17), (20, 64, 300, 17), (20, 128, 150, 17), (20, 256, 75, 17)],
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
            other=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 64, 300, 17), (20, 64, 300, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(32, 64, 300, 17), (32, 128, 150, 17), (32, 256, 75, 17), (20, 64, 300, 17), (20, 128, 150, 17), (20, 256, 75, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(32, 64, 300, 17), (32, 128, 150, 17), (32, 256, 75, 17), (20, 64, 300, 17), (20, 128, 150, 17), (20, 256, 75, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        para=dict(
            output_size=[(1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 256, 75, 17), (20, 256, 75, 17)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mean': dict(
        name=["mean"],
        interface=["torch.Tensor"],
        para=dict(
            dim=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 2, 256, 1, 1), (10, 2, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'cross_entropy': dict(
        name=["cross_entropy"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16, 60), (10, 60)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(16,), (10,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'mul_1': dict(
        name=["mul"],
        is_inplace=[True],
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

    'mean_2': dict(
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

    'div': dict(
        name=["div"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        para=dict(
            other=[1],
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

    'div_1': dict(
        name=["div"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        para=dict(
            other=[1],
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
            nesterov=[True for i in range(21)],
            lr=[0.1 for i in range(21)],
            momentum=[0.9 for i in range(21)],
            weight_decay=[0.0001 for i in range(21)],
            dampening=[0 for i in range(21)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(51,), (192, 3, 1, 1), (192,), (64,), (64, 64, 9, 1), (192, 64, 1, 1), (384, 64, 1, 1), (384,), (128,), (128, 128, 9, 1), (128, 64, 1, 1), (384, 128, 1, 1), (768, 128, 1, 1), (768,), (256,), (256, 256, 9, 1), (256, 128, 1, 1), (768, 256, 1, 1), (3, 17, 17), (60, 256, 1, 1), (60,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(51,), (192, 3, 1, 1), (192,), (64,), (64, 64, 9, 1), (192, 64, 1, 1), (384, 64, 1, 1), (384,), (128,), (128, 128, 9, 1), (128, 64, 1, 1), (384, 128, 1, 1), (768, 128, 1, 1), (768,), (256,), (256, 256, 9, 1), (256, 128, 1, 1), (768, 256, 1, 1), (3, 17, 17), (60, 256, 1, 1), (60,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_2': dict(
        name=["mul"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(51,), (192, 3, 1, 1), (192,), (64,), (64, 64, 9, 1), (192, 64, 1, 1), (384, 64, 1, 1), (384,), (128,), (128, 128, 9, 1), (128, 64, 1, 1), (384, 128, 1, 1), (768, 128, 1, 1), (768,), (256,), (256, 256, 9, 1), (256, 128, 1, 1), (768, 256, 1, 1), (3, 17, 17), (60, 256, 1, 1), (60,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
