from ...config import Genfunc
from ...diopi_runtime import Dtype

resnet50_config = {
    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        interface=["torch.nn.functional"],
        para=dict(
            output_size=[(1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 2048, 7, 7)],
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
            alpha=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(2048,), (2048, 512, 1, 1), (512, 128, 1, 1), (64,), (1024,), (1024, 512, 1, 1), (1024, 256, 1, 1), (512, 256, 1, 1), (1000, 2048), (512, 2048, 1, 1), (64, 3, 7, 7), (512, 512, 3, 3), (128,), (64, 64, 1, 1), (64, 64, 3, 3), (256, 512, 1, 1), (512, 1024, 1, 1), (64, 256, 1, 1), (128, 512, 1, 1), (1000,), (256,), (2048, 1024, 1, 1), (256, 256, 3, 3), (128, 256, 1, 1), (256, 64, 1, 1), (256, 1024, 1, 1), (512,), (128, 128, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(2048,), (2048, 512, 1, 1), (512, 128, 1, 1), (64,), (1024,), (1024, 512, 1, 1), (1024, 256, 1, 1), (512, 256, 1, 1), (1000, 2048), (512, 2048, 1, 1), (64, 3, 7, 7), (512, 512, 3, 3), (128,), (64, 64, 1, 1), (64, 64, 3, 3), (256, 512, 1, 1), (512, 1024, 1, 1), (64, 256, 1, 1), (128, 512, 1, 1), (1000,), (256,), (2048, 1024, 1, 1), (256, 256, 3, 3), (128, 256, 1, 1), (256, 64, 1, 1), (256, 1024, 1, 1), (512,), (128, 128, 3, 3)],
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
            alpha=[-0.1, -0.1, -0.1, -0.1, -0.1, 1, 1, 1, -0.1, 1, 1, 1, 1, -0.1, -0.1, -0.1, -0.1, -0.1, 1, -0.1, -0.1, -0.1, -0.1, 1, 1, 1, -0.1, 1, -0.1, 1, 1, -0.1, 1, -0.1, -0.1, 1, 1, 1, 1, 1, 1, -0.1, 1, -0.1, -0.1, 1, 1, 1, 1, 1, 1, 1, 1, -0.1, 1, 1, 1, 1, 1, 1, 1, -0.1, -0.1, -0.1, -0.1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 256, 1, 1), (64, 3, 7, 7), (256,), (512, 2048, 1, 1), (1024, 512, 1, 1), (64,), (32, 256, 56, 56), (1024, 256, 1, 1), (512,), (512, 1024, 1, 1), (128,), (32, 2048, 7, 7), (2048,), (128, 512, 1, 1), (512, 128, 1, 1), (128,), (128, 256, 1, 1), (1000, 2048), (128, 256, 1, 1), (64,), (512, 256, 1, 1), (512, 1024, 1, 1), (1000,), (128, 128, 3, 3), (512, 512, 3, 3), (256,), (1024, 256, 1, 1), (512, 128, 1, 1), (256, 512, 1, 1), (32, 64, 56, 56), (256, 512, 1, 1), (2048,), (256, 64, 1, 1), (2048, 512, 1, 1), (2048, 1024, 1, 1), (64, 64, 3, 3), (32, 1024, 14, 14), (1024,), (32, 2048, 7, 7), (256, 256, 3, 3), (64, 64, 1, 1), (256, 64, 1, 1), (32, 1024, 14, 14), (64, 64, 1, 1), (512, 512, 3, 3), (64, 3, 7, 7), (32, 512, 28, 28), (64, 256, 1, 1), (512,), (2048, 1024, 1, 1), (1000,), (1000, 2048), (128, 512, 1, 1), (256, 256, 3, 3), (32, 256, 56, 56), (32, 512, 28, 28), (512, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (2048, 512, 1, 1), (512, 2048, 1, 1), (256, 1024, 1, 1), (64, 64, 3, 3), (1024,), (128, 128, 3, 3)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(64, 256, 1, 1), (64, 3, 7, 7), (256,), (512, 2048, 1, 1), (1024, 512, 1, 1), (64,), (32, 256, 56, 56), (1024, 256, 1, 1), (512,), (512, 1024, 1, 1), (128,), (32, 2048, 7, 7), (2048,), (128, 512, 1, 1), (512, 128, 1, 1), (128,), (128, 256, 1, 1), (1000, 2048), (128, 256, 1, 1), (64,), (512, 256, 1, 1), (512, 1024, 1, 1), (1000,), (128, 128, 3, 3), (512, 512, 3, 3), (256,), (1024, 256, 1, 1), (512, 128, 1, 1), (256, 512, 1, 1), (32, 64, 56, 56), (256, 512, 1, 1), (2048,), (256, 64, 1, 1), (2048, 512, 1, 1), (2048, 1024, 1, 1), (64, 64, 3, 3), (32, 1024, 14, 14), (1024,), (32, 2048, 7, 7), (256, 256, 3, 3), (64, 64, 1, 1), (256, 64, 1, 1), (32, 1024, 14, 14), (64, 64, 1, 1), (512, 512, 3, 3), (64, 3, 7, 7), (32, 512, 28, 28), (64, 256, 1, 1), (512,), (2048, 1024, 1, 1), (1000,), (1000, 2048), (128, 512, 1, 1), (256, 256, 3, 3), (32, 256, 56, 56), (32, 512, 28, 28), (512, 256, 1, 1), (1024, 512, 1, 1), (256, 1024, 1, 1), (2048, 512, 1, 1), (512, 2048, 1, 1), (256, 1024, 1, 1), (64, 64, 3, 3), (1024,), (128, 128, 3, 3)],
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

    'batch_norm': dict(
        name=["batch_norm"],
        atol=1e-01,
        rtol=1e-02,
        atol_half=1e-01,
        rtol_half=1e-02,
        interface=["torch.nn.functional"],
        para=dict(
            training=[True, True, True, True, True, True, True, True, True, True, True, True],
            momentum=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 64, 56, 56), (32, 128, 28, 28), (32, 256, 56, 56), (32, 256, 28, 28), (32, 1024, 14, 14), (32, 256, 14, 14), (32, 2048, 7, 7), (32, 512, 28, 28), (32, 64, 112, 112), (32, 512, 7, 7), (32, 128, 56, 56), (32, 512, 14, 14)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(64,), (128,), (256,), (256,), (1024,), (256,), (2048,), (512,), (64,), (512,), (128,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(64,), (128,), (256,), (256,), (1024,), (256,), (2048,), (512,), (64,), (512,), (128,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad":[False],
                    "shape": [(64,), (128,), (256,), (256,), (1024,), (256,), (2048,), (512,), (64,), (512,), (128,), (512,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "requires_grad":[False],
                    "shape": [(64,), (128,), (256,), (256,), (1024,), (256,), (2048,), (512,), (64,), (512,), (128,), (512,)],
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
            dim=[0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]],
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
            stride=[(1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (2, 2), (1, 1), (1, 1)],
            padding=[(1, 1), (0, 0), (1, 1), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (0, 0), (3, 3), (0, 0), (0, 0)],
            dilation=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)],
            groups=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 64, 56, 56), (32, 256, 56, 56), (32, 128, 56, 56), (32, 128, 28, 28), (32, 512, 7, 7), (32, 512, 14, 14), (32, 512, 28, 28), (32, 2048, 7, 7), (32, 128, 28, 28), (32, 1024, 14, 14), (32, 256, 28, 28), (32, 256, 14, 14), (32, 512, 7, 7), (32, 64, 56, 56), (32, 512, 28, 28), (32, 256, 56, 56), (32, 64, 56, 56), (32, 512, 28, 28), (32, 256, 14, 14), (32, 1024, 14, 14), (32, 3, 224, 224), (32, 256, 56, 56), (32, 1024, 14, 14)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(64, 64, 3, 3), (64, 256, 1, 1), (128, 128, 3, 3), (128, 128, 3, 3), (2048, 512, 1, 1), (512, 512, 3, 3), (128, 512, 1, 1), (512, 2048, 1, 1), (512, 128, 1, 1), (512, 1024, 1, 1), (256, 256, 3, 3), (1024, 256, 1, 1), (512, 512, 3, 3), (256, 64, 1, 1), (256, 512, 1, 1), (512, 256, 1, 1), (64, 64, 1, 1), (1024, 512, 1, 1), (256, 256, 3, 3), (2048, 1024, 1, 1), (64, 3, 7, 7), (128, 256, 1, 1), (256, 1024, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[False],
                    "shape": [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
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
                    "shape": [(32, 3, 224, 224)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1)],
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
            other=[32, 32, 1, 1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (), ()],
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
            value=[0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(512,), (1024,), (64,), (128,), (512,), (), (1024,), (256,), (2048,), (1000,), (64,), (256,), (128,), (2048,)],
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
            dims=[(1,)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 3, 224, 224)],
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
                    "shape": [(32, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(1000, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(1000,)],
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
            dim=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        name=["max_pool2d"],
        interface=["torch.nn.functional"],
        requires_backward=[0],
        para=dict(
            kernel_size=[(3, 3), (3, 3)],
            stride=[(2, 2), (2, 2)],
            padding=[(1, 1), (1, 1)],
            dilation=[(1, 1), (1, 1)],
            ceil_mode=[False, False],
            return_indices=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 64, 112, 112), (12, 64, 112, 112)],
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
            dim=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
            keepdim=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (512, 1024, 1, 1), (512, 128, 1, 1), (2048, 1024, 1, 1), (256, 1024, 1, 1), (128, 128, 3, 3), (512, 2048, 1, 1), (1024, 512, 1, 1), (512, 256, 1, 1), (64, 3, 7, 7), (256,), (1024,), (128, 512, 1, 1), (64, 64, 3, 3), (256, 256, 3, 3), (128, 256, 1, 1), (1000,), (256, 512, 1, 1), (512, 512, 3, 3), (1024, 256, 1, 1), (1000, 2048), (512,), (256, 64, 1, 1), (2048, 512, 1, 1), (128,), (64, 256, 1, 1), (64, 64, 1, 1), (2048,), (64,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul': dict(
        name=["mul"],
        interface=["torch"],
    ),

    'mul_case_2': dict(
        name=["mul"],
        interface=["torch"],
        is_inplace=[True],
        para=dict(
            other=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 256, 1, 1), (256, 256, 3, 3), (512, 128, 1, 1), (1024, 512, 1, 1), (64, 3, 7, 7), (2048, 1024, 1, 1), (256, 64, 1, 1), (1024,), (2048, 512, 1, 1), (512, 1024, 1, 1), (2048,), (256, 512, 1, 1), (256, 1024, 1, 1), (512, 512, 3, 3), (64, 256, 1, 1), (512, 2048, 1, 1), (64, 64, 1, 1), (128, 512, 1, 1), (1000,), (512,), (64, 64, 3, 3), (512, 256, 1, 1), (1000, 2048), (256,), (1024, 256, 1, 1), (64,), (128,), (128, 128, 3, 3)],
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
            other=[1, 1],
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

    'nll_loss': dict(
        name=["nll_loss"],
        interface=["torch.nn.functional"],
        para=dict(
            weight=[None],
            reduction=['none'],
            ignore_index=[-100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "requires_grad":[False],
                    "shape": [(32,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'normal_': dict(
        name=["normal_"],
        no_output_ref=True,
        interface=["torch.Tensor"],
        para=dict(
            mean=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            std=[0.125, 0.0252538, 0.0883883, 0.0883883, 0.0625, 0.03125, 0.0625, 0.0416667, 0.125, 0.0208333, 0.0883883, 0.0589256, 0.01, 0.0625, 0.176777, 0.0294628, 0.0625, 0.0441942, 0.03125, 0.0441942, 0.176777],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 512, 1, 1), (64, 3, 7, 7), (256, 64, 1, 1), (256, 512, 1, 1), (512, 256, 1, 1), (2048, 1024, 1, 1), (512, 1024, 1, 1), (128, 128, 3, 3), (128, 256, 1, 1), (512, 512, 3, 3), (256, 1024, 1, 1), (64, 64, 3, 3), (1000, 2048), (512, 2048, 1, 1), (64, 64, 1, 1), (256, 256, 3, 3), (512, 128, 1, 1), (1024, 512, 1, 1), (2048, 512, 1, 1), (1024, 256, 1, 1), (64, 256, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu': dict(
        name=["relu"],
        interface=["torch.nn.functional"],
        is_inplace=[True],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 256, 28, 28), (32, 1024, 14, 14), (32, 64, 112, 112), (32, 2048, 7, 7), (32, 256, 14, 14), (32, 128, 56, 56), (32, 128, 28, 28), (32, 256, 56, 56), (32, 512, 28, 28), (32, 512, 7, 7), (32, 512, 14, 14), (32, 64, 56, 56)],
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
            alpha=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(32, 3, 224, 224)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 1, 1)],
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
            dim=[None],
            keepdim=[False],
            dtype=[None],
        ),
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

}
