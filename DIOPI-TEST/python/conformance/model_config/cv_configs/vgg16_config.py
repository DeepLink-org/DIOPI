from ...config import Genfunc
from ...dtype import Dtype

vgg16_config = {
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
            stride=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 3, 224, 224), (32, 64, 224, 224), (32, 64, 112, 112), (32, 128, 112, 112), (32, 128, 56, 56), (32, 256, 56, 56), (32, 256, 28, 28), (32, 512, 28, 28), (32, 512, 14, 14)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 3, 3), (64, 64, 3, 3), (128, 64, 3, 3), (128, 128, 3, 3), (256, 128, 3, 3), (256, 256, 3, 3), (512, 256, 3, 3), (512, 512, 3, 3), (512, 512, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (128,), (128,), (256,), (256,), (512,), (512,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        para=dict(
            inplace=[True, True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 64, 224, 224), (32, 128, 112, 112), (32, 256, 56, 56), (32, 512, 28, 28), (32, 512, 14, 14), (32, 4096)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        para=dict(
            kernel_size=[2, 2, 2, 2, 2],
            stride=[2, 2, 2, 2, 2],
            padding=[0, 0, 0, 0, 0],
            dilation=[1, 1, 1, 1, 1],
            ceil_mode=[False, False, False, False, False],
            return_indices=[False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(32, 64, 224, 224), (32, 128, 112, 112), (32, 256, 56, 56), (32, 512, 28, 28), (32, 512, 14, 14)],
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
                    "shape": [(32, 25088), (32, 4096), (32, 4096)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(4096, 25088), (4096, 4096), (1000, 4096)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(4096,), (4096,), (1000,)],
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
            p=[0.5],
            training=[True],
            inplace=[False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 4096)],
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

    'add': dict(
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
            nesterov=[False for i in range(17)],
            lr=[0.01 for i in range(17)],
            momentum=[0.9 for i in range(17)],
            weight_decay=[0.0001 for i in range(17)],
            dampening=[0 for i in range(17)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(64, 3, 3, 3), (64,), (64, 64, 3, 3), (128, 64, 3, 3), (128,), (128, 128, 3, 3), (256, 128, 3, 3), (256,), (256, 256, 3, 3), (512, 256, 3, 3), (512,), (512, 512, 3, 3), (4096, 25088), (4096,), (4096, 4096), (1000, 4096), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(64, 3, 3, 3), (64,), (64, 64, 3, 3), (128, 64, 3, 3), (128,), (128, 128, 3, 3), (256, 128, 3, 3), (256,), (256, 256, 3, 3), (512, 256, 3, 3), (512,), (512, 512, 3, 3), (4096, 25088), (4096,), (4096, 4096), (1000, 4096), (1000,)],
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
