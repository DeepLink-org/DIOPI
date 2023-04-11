from ...config import Genfunc
from ...dtype import Dtype

swin_transformer_config = {
    'linspace': dict(
        name=["linspace"],
        interface=["torch"],
        para=dict(
            start=[0],
            end=[0.5],
            steps=[24],
        ),
    ),

    'arange': dict(
        name=["arange"],
        interface=["torch"],
        para=dict(
            start=[0, 0],
            end=[91, 7],
            step=[13, 1],
        ),
    ),

    'randperm': dict(
        name=["randperm"],
        no_output_ref=True,
        para=dict(
            n=[1281167],
        ),
    ),

    'one_hot': dict(
        name=["one_hot"],
        para=dict(
            num_classes=[1000],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64,)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'randperm_1': dict(
        name=["randperm"],
        no_output_ref=True,
        para=dict(
            n=[64],
        ),
    ),

    'mul': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.5864556760204082, 0.41354432397959184],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 1000), (64, 1000)],
                    "dtype": [Dtype.int64],
                    "gen_fn": Genfunc.randint,
                },
            ],
        ),
    ),

    'add': dict(
        name=["add"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 1000), (4096, 4, 49, 49), (64, 3136, 128), (64, 64, 4, 49, 49), (1024, 8, 49, 49), (64, 784, 256), (64, 16, 8, 49, 49), (256, 16, 49, 49), (64, 196, 512), (64, 4, 16, 49, 49), (64, 32, 49, 49), (64, 49, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(64, 1000), (1, 4, 49, 49), (64, 3136, 128), (1, 64, 1, 49, 49), (1, 8, 49, 49), (64, 784, 256), (1, 16, 1, 49, 49), (1, 16, 49, 49), (64, 196, 512), (1, 4, 1, 49, 49), (1, 32, 49, 49), (64, 49, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'conv2d': dict(
        name=["conv2d"],
        para=dict(
            stride=[(4, 4)],
            padding=[(0, 0)],
            dilation=[(1, 1)],
            groups=[1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 3, 224, 224)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(128, 3, 4, 4)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'transpose': dict(
        name=["transpose"],
        interface=["torch.Tensor"],
        para=dict(
            dim0=[1, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 1],
            dim1=[2, -1, 2, 2, -1, 2, 2, -1, 2, 2, -1, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 128, 3136), (4096, 4, 49, 32), (4096, 4, 49, 32), (64, 512, 784), (1024, 8, 49, 32), (1024, 8, 49, 32), (64, 1024, 196), (256, 16, 49, 32), (256, 16, 49, 32), (64, 2048, 49), (64, 32, 49, 32), (64, 32, 49, 32)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'layer_norm': dict(
        name=["layer_norm"],
        para=dict(
            normalized_shape=[(128,), (512,), (256,), (1024,), (512,), (2048,), (1024,)],
            eps=[1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 3136, 128), (64, 784, 512), (64, 784, 256), (64, 196, 1024), (64, 196, 512), (64, 49, 2048), (64, 49, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(128,), (512,), (256,), (1024,), (512,), (2048,), (1024,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(128,), (512,), (256,), (1024,), (512,), (2048,), (1024,)],
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
            p=[0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            training=[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
            inplace=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 3136, 128), (4096, 4, 49, 49), (4096, 49, 128), (64, 3136, 512), (1024, 8, 49, 49), (1024, 49, 256), (64, 784, 1024), (64, 784, 256), (256, 16, 49, 49), (256, 49, 512), (64, 196, 2048), (64, 196, 512), (64, 32, 49, 49), (64, 49, 1024), (64, 49, 4096)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'pad': dict(
        name=["pad"],
        para=dict(
            pad=[(0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 56, 56, 128), (64, 28, 28, 256), (64, 14, 14, 512), (64, 7, 7, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'permute': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[(0, 1, 3, 2, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 3, 2, 4, 5), (0, 1, 3, 2, 4, 5)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 8, 7, 8, 7, 128), (64, 8, 8, 7, 7, 128), (1, 8, 7, 8, 7, 1), (64, 4, 7, 4, 7, 256), (64, 4, 4, 7, 7, 256), (1, 4, 7, 4, 7, 1), (64, 2, 7, 2, 7, 512), (64, 2, 2, 7, 7, 512), (1, 2, 7, 2, 7, 1), (64, 1, 7, 1, 7, 1024), (64, 1, 1, 7, 7, 1024)],
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
                    "shape": [(4096, 49, 128), (4096, 49, 128), (64, 3136, 128), (64, 3136, 512), (1024, 49, 256), (1024, 49, 256), (64, 784, 256), (64, 784, 1024), (256, 49, 512), (256, 49, 512), (64, 196, 512), (64, 196, 2048), (64, 49, 1024), (64, 49, 1024), (64, 49, 1024), (64, 49, 4096), (64, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(384, 128), (128, 128), (512, 128), (128, 512), (768, 256), (256, 256), (1024, 256), (256, 1024), (1536, 512), (512, 512), (2048, 512), (512, 2048), (3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096), (1000, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad": [True],
                    "shape": [(384,), (128,), (512,), (128,), (768,), (256,), (1024,), (256,), (1536,), (512,), (2048,), (512,), (3072,), (1024,), (4096,), (1024,), (1000,)],
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
            dims=[(2, 0, 3, 1, 4), (2, 0, 3, 1, 4), (2, 0, 3, 1, 4), (2, 0, 3, 1, 4)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4096, 49, 3, 4, 32), (1024, 49, 3, 8, 32), (256, 49, 3, 16, 32), (64, 49, 3, 32, 32)],
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
            other=[0.1767766952966369, 0.1767766952966369, 0.1767766952966369, 0.1767766952966369, 0.9, 5.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4096, 4, 49, 32), (1024, 8, 49, 32), (256, 16, 49, 32), (64, 32, 49, 32), (64, 1000), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'matmul': dict(
        name=["matmul"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4096, 4, 49, 32), (4096, 4, 49, 49), (1024, 8, 49, 32), (1024, 8, 49, 49), (256, 16, 49, 32), (256, 16, 49, 49), (64, 32, 49, 32), (64, 32, 49, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(4096, 4, 32, 49), (4096, 4, 49, 32), (1024, 8, 32, 49), (1024, 8, 49, 32), (256, 16, 32, 49), (256, 16, 49, 32), (64, 32, 32, 49), (64, 32, 49, 32)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'permute_2': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[(2, 0, 1), (2, 0, 1), (2, 0, 1), (2, 0, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(49, 49, 4), (49, 49, 8), (49, 49, 16), (49, 49, 32)],
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
            dim=[-1, -1, -1, -1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(4096, 4, 49, 49), (1024, 8, 49, 49), (256, 16, 49, 49), (64, 32, 49, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'gelu': dict(
        name=["gelu"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 3136, 512), (64, 784, 1024), (64, 196, 2048), (64, 49, 4096)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'roll': dict(
        name=["roll"],
        interface=["torch"],
        para=dict(
            shifts=[(-3, -3), (3, 3), (-3, -3), (3, 3), (-3, -3), (3, 3), (0, 0)],
            dims=[(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 56, 56, 128), (64, 56, 56, 128), (64, 28, 28, 256), (64, 28, 28, 256), (64, 14, 14, 512), (64, 14, 14, 512), (64, 7, 7, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(64, 1, 49), (16, 1, 49), (4, 1, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(64, 49, 1), (16, 49, 1), (4, 49, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'ne': dict(
        name=["ne"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 49, 49), (16, 49, 49), (4, 49, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'masked_fill': dict(
        name=["masked_fill"],
        interface=["torch.Tensor"],
        para=dict(
            value=[-100.0, 0.0, -100.0, 0.0, -100.0, 0.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 49, 49), (64, 49, 49), (16, 49, 49), (16, 49, 49), (4, 49, 49), (4, 49, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["mask"],
                    "shape": [(64, 49, 49), (64, 49, 49), (16, 49, 49), (16, 49, 49), (4, 49, 49), (4, 49, 49)],
                    "dtype": [Dtype.bool],
                    "gen_fn": Genfunc.mask,
                },
            ],
        ),
    ),

    'eq': dict(
        name=["eq"],
        interface=["torch.Tensor"],
        para=dict(
            other=[0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 49, 49), (16, 49, 49), (4, 49, 49)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'uniform': dict(
        name=["uniform"],
        no_output_ref=True,
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 1, 1)],
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
            other=[0.9782608691602945, 0.9565217383205891, 0.9347826093435287, 0.9130434766411781, 0.8913043439388275, 0.8695652186870575, 0.8478260785341263, 0.8260869532823563, 0.8043478280305862, 0.782608687877655, 0.760869562625885, 0.739130437374115, 0.717391312122345, 0.695652186870575, 0.6739130318164825, 0.6521739065647125, 0.6304347813129425, 0.6086956560611725, 0.5869565308094025, 0.5652174055576324, 0.54347825050354, 0.52173912525177, 0.5, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), ()],
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
            other=[0.9782608691602945, 0.9565217383205891, 0.9347826093435287, 0.9130434766411781, 0.8913043439388275, 0.8695652186870575, 0.8478260785341263, 0.8260869532823563, 0.8043478280305862, 0.782608687877655, 0.760869562625885, 0.739130437374115, 0.717391312122345, 0.695652186870575, 0.6739130318164825, 0.6521739065647125, 0.6304347813129425, 0.6086956560611725, 0.5869565308094025, 0.5652174055576324, 0.54347825050354, 0.52173912525177, 0.5],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 3136, 128), (64, 784, 256), (64, 784, 256), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 196, 512), (64, 49, 1024), (64, 49, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'floor': dict(
        name=["floor"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 1, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'mul_2': dict(
        name=["mul"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 3136, 128), (64, 784, 256), (64, 196, 512), (64, 49, 1024), (64, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'permute_3': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[[0, 3, 1, 2], [0, 3, 1, 2], [0, 3, 1, 2]],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 56, 56, 128), (64, 28, 28, 256), (64, 14, 14, 512)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'im2col': dict(
        name=["im2col"],
        interface=["CustomizedTest"],
        para=dict(
            kernel_size=[(2, 2), (2, 2), (2, 2)],
            dilation=[(1, 1), (1, 1), (1, 1)],
            padding=[(0, 0), (0, 0), (0, 0)],
            stride=[(2, 2), (2, 2), (2, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 128, 56, 56), (64, 256, 28, 28), (64, 512, 14, 14)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'linear_1': dict(
        name=["linear"],
        para=dict(
            bias=[None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad": [True],
                    "shape": [(64, 784, 512), (64, 196, 1024), (64, 49, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad": [True],
                    "shape": [(256, 512), (512, 1024), (1024, 2048)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'permute_4': dict(
        name=["permute"],
        interface=["torch.Tensor"],
        para=dict(
            dims=[(0, 3, 1, 2)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 7, 7, 1024)],
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
                    "shape": [(64, 1024, 7, 7)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'add_2': dict(
        name=["add"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        para=dict(
            other=[0.0001],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'neg': dict(
        name=["neg"],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 1000)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        saved_args=dict(output=0),
        para=dict(
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 1000)],
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
            dim=[-1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(64, 1000)],
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
                    "shape": [(64,)],
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
            other=[64],
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

    'mul_3': dict(
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

    'norm': dict(
        name=["norm"],
        interface=["torch"],
        para=dict(
            p=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 3, 4, 4), (128,), (169, 4), (384, 128), (384,), (128, 128), (512, 128), (512,), (128, 512), (256, 512), (256,), (169, 8), (768, 256), (768,), (256, 256), (1024, 256), (1024,), (256, 1024), (512, 1024), (169, 16), (1536, 512), (1536,), (512, 512), (2048, 512), (2048,), (512, 2048), (1024, 2048), (169, 32), (3072, 1024), (3072,), (1024, 1024), (4096, 1024), (4096,), (1024, 4096), (1000, 1024), (1000,), (329,)],
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
                    "shape": [((), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ())],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

    'add_3': dict(
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

    'div_2': dict(
        name=["div"],
        interface=["torch.Tensor"],
        para=dict(
            other=[5.0],
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

    'reciprocal': dict(
        name=["reciprocal"],
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

    'clamp': dict(
        name=["clamp"],
        interface=["torch"],
        para=dict(
            max=[1.0],
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

    'mul_4': dict(
        name=["mul"],
        is_inplace=[True],
        interface=["torch.Tensor"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 3, 4, 4), (128,), (169, 4), (384, 128), (384,), (128, 128), (512, 128), (512,), (128, 512), (256, 512), (256,), (169, 8), (768, 256), (768,), (256, 256), (1024, 256), (1024,), (256, 1024), (512, 1024), (169, 16), (1536, 512), (1536,), (512, 512), (2048, 512), (2048,), (512, 2048), (1024, 2048), (169, 32), (3072, 1024), (3072,), (1024, 1024), (4096, 1024), (4096,), (1024, 4096), (1000, 1024), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ()],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'adamw': dict(
        name=["adamw"],
        interface=["CustomizedTest"],
        para=dict(
            step=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            amsgrad=[False for i in range(21)],
            beta1=[0.9 for i in range(21)],
            beta2=[0.999 for i in range(21)],
            lr=[1.000000000000001e-06 for i in range(21)],
            weight_decay=[0.05 for i in range(21)],
            eps=[1e-08 for i in range(21)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(128, 3, 4, 4), (384, 128), (128, 128), (512, 128), (128, 512), (256, 512), (768, 256), (256, 256), (1024, 256), (256, 1024), (512, 1024), (1536, 512), (512, 512), (2048, 512), (512, 2048), (1024, 2048), (3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096), (1000, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"],
                    "shape": [(128, 3, 4, 4), (384, 128), (128, 128), (512, 128), (128, 512), (256, 512), (768, 256), (256, 256), (1024, 256), (256, 1024), (512, 1024), (1536, 512), (512, 512), (2048, 512), (512, 2048), (1024, 2048), (3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096), (1000, 1024)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'adamw_1': dict(
        name=["adamw"],
        interface=["CustomizedTest"],
        para=dict(
            step=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            amsgrad=[False for i in range(15)],
            beta1=[0.9 for i in range(15)],
            beta2=[0.999 for i in range(15)],
            lr=[1.000000000000001e-06 for i in range(15)],
            weight_decay=[0.0 for i in range(15)],
            eps=[1e-08 for i in range(15)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["param", "param_grad"],
                    "shape": [(128,), (169, 4), (384,), (512,), (256,), (169, 8), (768,), (1024,), (169, 16), (1536,), (2048,), (169, 32), (3072,), (4096,), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"],
                    "shape": [(128,), (169, 4), (384,), (512,), (256,), (169, 8), (768,), (1024,), (169, 16), (1536,), (2048,), (169, 32), (3072,), (4096,), (1000,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

}
