from ...config import Genfunc
from ...diopi_runtime import Dtype

repvgg_config = {
    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        interface=["torch.nn.functional"],
        para=dict(
            size=[(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(16, 1280, 7, 7), (80, 1280, 7, 7), (256, 1280, 7, 7), (15, 1280, 7, 7)],
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
            alpha=[0.0001, 1, 0.0001, 0.0001, 0.0001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(48, 48, 1, 1), (16, 192, 14, 14), (96,), (192,), (1000, 1280)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(48, 48, 1, 1), (16, 192, 14, 14), (96,), (192,), (1000, 1280)],
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
            alpha=[-0.0999899, -0.0999884, -0.0999932, -0.0999964, -0.0999875, -0.0999926],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(1000,), (192,), (192, 96, 3, 3), (48, 48, 3, 3), (1000, 1280), (1000, 1280)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(1000,), (192,), (192, 96, 3, 3), (48, 48, 3, 3), (1000, 1280), (1000, 1280)],
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
            other=[0, 0, 0],
            alpha=[1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(15, 48, 112, 112), (256, 192, 14, 14), ()],
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
                    "shape": [(256, 1000), (80, 1000)],
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
            training=[False, False],
            momentum=[0.1, 0.1],
            eps=[1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(80, 1280, 7, 7), (80, 48, 112, 112)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(1280,), (48,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(1280,), (48,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad":[False],
                    "shape": [(1280,), (48,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "requires_grad":[False],
                    "shape": [(1280,), (48,)],
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
                    "shape": [[(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(1296,), (48,), (48,), (144,), (48,), (48,), (20736,), (48,), (48,), (2304,), (48,), (48,), (48,), (48,), (20736,), (48,), (48,), (2304,), (48,), (48,), (41472,), (96,), (96,), (4608,), (96,), (96,), (96,), (96,), (82944,), (96,), (96,), (9216,), (96,), (96,), (96,), (96,), (82944,), (96,), (96,), (9216,), (96,), (96,), (96,), (96,), (82944,), (96,), (96,), (9216,), (96,), (96,), (165888,), (192,), (192,), (18432,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (192,), (192,), (331776,), (192,), (192,), (36864,), (192,), (192,), (2211840,), (1280,), (1280,), (245760,), (1280,), (1280,), (1280000,), (1000,), (3,), (3,), (48,), (48,), (48,), (48,), (48,), (48,), (48,), (48,), (48,), (48,), (48,), (48,), (48,), (48,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (96,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (192,), (1280,), (1280,), (1280,), (1280,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]],
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
            stride=[(2, 2), (2, 2)],
            padding=[(0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(16, 192, 14, 14), (256, 192, 14, 14)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(1280, 192, 1, 1), (1280, 192, 1, 1)],
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
                    "shape": [(16, 3, 224, 224), (80, 3, 224, 224), (256, 3, 224, 224), (15, 3, 224, 224)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)],
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
            other=[16, 16, 1, 1, 1, 15, 15],
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
            value=[1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(192,), (192,), (96,), (96,), (48,), (48,), (1000,), (), (1280,), (1280,)],
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
            dims=[(1,), (1,), (1,), (1,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 3, 224, 224), (256, 3, 224, 224), (15, 3, 224, 224), (80, 3, 224, 224)],
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
                    "shape": [(16, 1280), (15, 1280), (80, 1280), (256, 1280)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(1000, 1280), (1000, 1280), (1000, 1280), (1000, 1280)],
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
                    "shape": [(16, 1000), (15, 1000)],
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
                    "shape": [(96, 48, 1, 1), (96, 96, 1, 1), (96,), (192,), (), (1000, 1280)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul': dict(
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
                    "shape": [(48,), (96,), (48, 3, 1, 1), (1280, 192, 1, 1), (1000, 1280)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_case_2': dict(
        name=["mul"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(48, 3, 1, 1), (96, 48, 3, 3), (192,), (1000,), (), (), (1000, 1280)],
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
                    "shape": [(16, 1000), (15, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "requires_grad":[False],
                    "shape": [(16,), (15,)],
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
            mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            std=[0.144338, 0.204124, 0.204124, 0.0481125, 0.0340207, 0.0481125, 0.01, 0.0680414, 0.0395285, 0.0131762, 0.0340207, 0.102062, 0.102062, 0.144338, 0.0680414],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(96, 48, 1, 1), (48, 3, 1, 1), (48, 48, 1, 1), (96, 96, 3, 3), (192, 192, 3, 3), (96, 48, 3, 3), (1000, 1280), (48, 3, 3, 3), (1280, 192, 1, 1), (1280, 192, 3, 3), (192, 96, 3, 3), (192, 96, 1, 1), (192, 192, 1, 1), (96, 96, 1, 1), (48, 48, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        interface=["torch.nn.functional"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 48, 56, 56), (16, 192, 14, 14)],
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
                    "shape": [(80, 1000), (256, 1000)],
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
            alpha=[1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16, 3, 224, 224), (15, 3, 224, 224), (80, 3, 224, 224), (256, 3, 224, 224)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1), (3, 1, 1), (3, 1, 1), (3, 1, 1)],
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
            dim=[None, None],
            keepdim=[False, False],
            dtype=[None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(16,), (15,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
