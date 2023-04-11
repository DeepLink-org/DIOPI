from ...config import Genfunc
from ...dtype import Dtype

seresnet50_config = {
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
            bias=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            stride=[(2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
            padding=[(3, 3), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 3, 224, 224), (32, 64, 56, 56), (32, 64, 56, 56), (32, 64, 56, 56), (32, 256, 56, 56), (32, 256, 56, 56), (32, 128, 56, 56), (32, 128, 28, 28), (32, 256, 56, 56), (32, 512, 28, 28), (32, 128, 28, 28), (32, 512, 28, 28), (32, 256, 28, 28), (32, 256, 14, 14), (32, 512, 28, 28), (32, 1024, 14, 14), (32, 256, 14, 14), (32, 1024, 14, 14), (32, 512, 14, 14), (32, 512, 7, 7), (32, 1024, 14, 14), (32, 2048, 7, 7), (32, 512, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 7, 7), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128, 128, 3, 3), (512, 128, 1, 1), (512, 256, 1, 1), (128, 512, 1, 1), (128, 128, 3, 3), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (256, 256, 3, 3), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048, 1024, 1, 1), (512, 2048, 1, 1), (512, 512, 3, 3)],
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
                    "shape": [(32, 64, 112, 112), (32, 64, 56, 56), (32, 256, 56, 56), (32, 128, 56, 56), (32, 128, 28, 28), (32, 512, 28, 28), (32, 256, 28, 28), (32, 256, 14, 14), (32, 1024, 14, 14), (32, 512, 14, 14), (32, 512, 7, 7), (32, 2048, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (256,), (128,), (128,), (512,), (256,), (256,), (1024,), (512,), (512,), (2048,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        para=dict(
            inplace=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 64, 112, 112), (32, 64, 56, 56), (32, 16, 1, 1), (32, 256, 56, 56), (32, 128, 56, 56), (32, 128, 28, 28), (32, 32, 1, 1), (32, 512, 28, 28), (32, 256, 28, 28), (32, 256, 14, 14), (32, 64, 1, 1), (32, 1024, 14, 14), (32, 512, 14, 14), (32, 512, 7, 7), (32, 128, 1, 1), (32, 2048, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[3],
            stride=[2],
            padding=[1],
            dilation=[1],
            ceil_mode=[False],
            return_indices=[False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 64, 112, 112)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        para=dict(
            output_size=[1, 1, 1, 1, (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 256, 56, 56), (32, 512, 28, 28), (32, 1024, 14, 14), (32, 2048, 7, 7), (32, 2048, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'conv2d_1': dict(
        name=["conv2d"],
        para=dict(
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 256, 1, 1), (32, 16, 1, 1), (32, 512, 1, 1), (32, 32, 1, 1), (32, 1024, 1, 1), (32, 64, 1, 1), (32, 2048, 1, 1), (32, 128, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(16, 256, 1, 1), (256, 16, 1, 1), (32, 512, 1, 1), (512, 32, 1, 1), (64, 1024, 1, 1), (1024, 64, 1, 1), (128, 2048, 1, 1), (2048, 128, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(16,), (256,), (32,), (512,), (64,), (1024,), (128,), (2048,)],
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
                    "requires_grad": [True],
                    "shape": [(32, 256, 1, 1), (32, 512, 1, 1), (32, 1024, 1, 1), (32, 2048, 1, 1)],
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
                    "shape": [(32, 256, 56, 56), (32, 512, 28, 28), (32, 1024, 14, 14), (32, 2048, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(32, 256, 1, 1), (32, 512, 1, 1), (32, 1024, 1, 1), (32, 2048, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_1': dict(
        name=["add"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 256, 56, 56), (32, 512, 28, 28), (32, 1024, 14, 14), (32, 2048, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(32, 256, 56, 56), (32, 512, 28, 28), (32, 1024, 14, 14), (32, 2048, 7, 7)],
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
                    "shape": [(32, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(1000, 2048)],
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

    'mul_1': dict(
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
            nesterov=[False for i in range(38)],
            lr=[0.1 for i in range(38)],
            momentum=[0.9 for i in range(38)],
            weight_decay=[0.0001 for i in range(38)],
            dampening=[0 for i in range(38)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(64, 3, 7, 7), (64,), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (256,), (16, 256, 1, 1), (16,), (256, 16, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (32, 512, 1, 1), (32,), (512, 32, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (64, 1024, 1, 1), (1024, 64, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (128, 2048, 1, 1), (2048, 128, 1, 1), (512, 2048, 1, 1), (1000, 2048), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(64, 3, 7, 7), (64,), (64, 64, 1, 1), (64, 64, 3, 3), (256, 64, 1, 1), (256,), (16, 256, 1, 1), (16,), (256, 16, 1, 1), (64, 256, 1, 1), (128, 256, 1, 1), (128,), (128, 128, 3, 3), (512, 128, 1, 1), (512,), (512, 256, 1, 1), (32, 512, 1, 1), (32,), (512, 32, 1, 1), (128, 512, 1, 1), (256, 512, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (1024,), (1024, 512, 1, 1), (64, 1024, 1, 1), (1024, 64, 1, 1), (256, 1024, 1, 1), (512, 1024, 1, 1), (512, 512, 3, 3), (2048, 512, 1, 1), (2048,), (2048, 1024, 1, 1), (128, 2048, 1, 1), (2048, 128, 1, 1), (512, 2048, 1, 1), (1000, 2048), (1000,)],
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
