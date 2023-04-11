from ...config import Genfunc
from ...dtype import Dtype

dbnet_config = {
    'mean': dict(
        name=["mean"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 3, 7, 7), (64,), (64, 64, 3, 3), (128, 64, 3, 3), (128,), (128, 128, 3, 3), (128, 64, 1, 1), (256, 128, 3, 3), (256,), (256, 256, 3, 3), (256, 128, 1, 1), (512, 256, 3, 3), (512,), (512, 512, 3, 3), (512, 256, 1, 1), (256, 64, 1, 1), (256, 256, 1, 1), (256, 512, 1, 1), (64, 256, 3, 3), (64, 64, 2, 2), (64, 1, 2, 2), (1,), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'normal_': dict(
        name=["normal_"],
        no_output_ref=True,
        para=dict(
            size=[(256, 64, 1, 1), (256, 128, 1, 1), (256, 256, 1, 1), (256, 512, 1, 1), (64, 256, 3, 3)],
            mean=[0, 0, 0, 0, 0],
            std=[0.08838834764831845, 0.08838834764831845, 0.08838834764831845, 0.08838834764831845, 0.05892556509887897],
        ),
    ),

    'randperm': dict(
        name=["randperm"],
        no_output_ref=True,
        para=dict(
            n=[1000],
        ),
    ),

    'sub': dict(
        name=["sub"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(3, 640, 640), (16, 640, 640)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (16, 640, 640)],
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
                    "shape": [(3, 640, 640), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'stack': dict(
        name=["stack"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640), (3, 640, 640))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'conv2d': dict(
        name=["conv2d"],
        para=dict(
            bias=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            stride=[(2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (2, 2), (1, 1), (2, 2), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            padding=[(3, 3), (1, 1), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16, 3, 640, 640), (16, 64, 160, 160), (16, 64, 160, 160), (16, 128, 80, 80), (16, 64, 160, 160), (16, 128, 80, 80), (16, 256, 40, 40), (16, 128, 80, 80), (16, 256, 40, 40), (16, 512, 20, 20), (16, 256, 40, 40), (16, 64, 160, 160), (16, 128, 80, 80), (16, 256, 40, 40), (16, 512, 20, 20), (16, 256, 160, 160), (16, 256, 80, 80), (16, 256, 40, 40), (16, 256, 20, 20)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 7, 7), (64, 64, 3, 3), (128, 64, 3, 3), (128, 128, 3, 3), (128, 64, 1, 1), (256, 128, 3, 3), (256, 256, 3, 3), (256, 128, 1, 1), (512, 256, 3, 3), (512, 512, 3, 3), (512, 256, 1, 1), (256, 64, 1, 1), (256, 128, 1, 1), (256, 256, 1, 1), (256, 512, 1, 1), (64, 256, 3, 3), (64, 256, 3, 3), (64, 256, 3, 3), (64, 256, 3, 3)],
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
            training=[True, True, True, True, True],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16, 64, 320, 320), (16, 64, 160, 160), (16, 128, 80, 80), (16, 256, 40, 40), (16, 512, 20, 20)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": [(64,), (64,), (128,), (256,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "shape": [(64,), (64,), (128,), (256,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (128,), (256,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(64,), (64,), (128,), (256,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        para=dict(
            inplace=[True, True, True, True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 64, 320, 320), (16, 64, 160, 160), (16, 128, 80, 80), (16, 256, 40, 40), (16, 512, 20, 20)],
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
                    "shape": [(16, 64, 320, 320)],
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
                    "shape": [(16, 64, 160, 160), (16, 128, 80, 80), (16, 256, 40, 40), (16, 512, 20, 20)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(16, 64, 160, 160), (16, 128, 80, 80), (16, 256, 40, 40), (16, 512, 20, 20)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'interpolate': dict(
        name=["interpolate"],
        para=dict(
            size=[(40, 40), (80, 80), (160, 160), (160, 160), (160, 160), (160, 160), (160, 160)],
            mode=['nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest', 'nearest'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 256, 20, 20), (16, 256, 40, 40), (16, 256, 80, 80), (16, 64, 160, 160), (16, 64, 80, 80), (16, 64, 40, 40), (16, 64, 20, 20)],
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
                    "shape": [(16, 256, 40, 40), (16, 256, 80, 80), (16, 256, 160, 160), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(16, 256, 40, 40), (16, 256, 80, 80), (16, 256, 160, 160), ()],
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
            dim=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [((16, 64, 160, 160), (16, 64, 160, 160), (16, 64, 160, 160), (16, 64, 160, 160))],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'conv_transpose2d': dict(
        name=["conv_transpose2d"],
        para=dict(
            stride=[(2, 2), (2, 2)],
            padding=[(0, 0), (0, 0)],
            output_padding=[(0, 0), (0, 0)],
            groups=[1, 1],
            dilation=[(1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 64, 160, 160), (16, 64, 320, 320)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "shape": [(64, 64, 2, 2), (64, 1, 2, 2)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "shape": [(64,), (1,)],
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
                    "shape": [(16, 640, 640), (16, 1, 640, 640)],
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
            other=[-50, 2, 5.0, 10.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 640, 640), (), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'exp': dict(
        name=["exp"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 640, 640)],
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
            other=[1.0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 640, 640), ()],
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
                    "shape": [(16, 640, 640)],
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
                    "shape": [(14,), (2,), (5,), (3,), (8,), (7,), (13,), (4,), (79,), (9,)],
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
                    "shape": [(14,), (2,), (5,), (3,), (8,), (7,), (13,), (4,), (79,), (9,)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
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
                    "shape": [(16, 640, 640), (16, 409600)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(16, 640, 640), (16, 409600)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(16, 640, 640), ()],
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
                    "shape": [(16, 640, 640), (79842,), (16, 409600)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max': dict(
        name=["max"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 640, 640)],
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

    'min': dict(
        name=["min"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 640, 640)],
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

    'binary_cross_entropy_with_logits': dict(
        name=["binary_cross_entropy_with_logits"],
        para=dict(
            weight=[None],
            reduction=['none'],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 640, 640)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(16, 640, 640)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'topk': dict(
        name=["topk"],
        interface=["torch"],
        para=dict(
            k=[79842],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(6553600,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div_1': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[106456.000001, 1],
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

    'smooth_l1_loss': dict(
        name=["smooth_l1_loss"],
        para=dict(
            reduction=['none'],
            beta=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(16, 640, 640)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "shape": [(16, 640, 640)],
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
            other=[1e-06],
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
            nesterov=[False for i in range(22)],
            lr=[0.007 for i in range(22)],
            momentum=[0.9 for i in range(22)],
            weight_decay=[0.0001 for i in range(22)],
            dampening=[0 for i in range(22)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(64, 3, 7, 7), (64,), (64, 64, 3, 3), (128, 64, 3, 3), (128,), (128, 128, 3, 3), (128, 64, 1, 1), (256, 128, 3, 3), (256,), (256, 256, 3, 3), (256, 128, 1, 1), (512, 256, 3, 3), (512,), (512, 512, 3, 3), (512, 256, 1, 1), (256, 64, 1, 1), (256, 256, 1, 1), (256, 512, 1, 1), (64, 256, 3, 3), (64, 64, 2, 2), (64, 1, 2, 2), (1,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["buf"],
                    "shape": [(64, 3, 7, 7), (64,), (64, 64, 3, 3), (128, 64, 3, 3), (128,), (128, 128, 3, 3), (128, 64, 1, 1), (256, 128, 3, 3), (256,), (256, 256, 3, 3), (256, 128, 1, 1), (512, 256, 3, 3), (512,), (512, 512, 3, 3), (512, 256, 1, 1), (256, 64, 1, 1), (256, 256, 1, 1), (256, 512, 1, 1), (64, 256, 3, 3), (64, 64, 2, 2), (64, 1, 2, 2), (1,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
