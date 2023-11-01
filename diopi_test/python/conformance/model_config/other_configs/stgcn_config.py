from ...config import Genfunc
from ...diopi_runtime import Dtype

stgcn_config = {
    'adaptive_avg_pool2d': dict(
        name=["adaptive_avg_pool2d"],
        interface=["torch.nn.functional"],
        para=dict(
            output_size=[(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(14, 256, 25, 25), (14, 256, 25, 25), (32, 256, 25, 25), (32, 256, 25, 25)],
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
            alpha=[1, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(14, 128, 50, 25), (256, 256, 9, 1), (192,), (128,), (60, 256), (3, 25, 25), (3, 25, 25)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(14, 128, 50, 25), (256, 256, 9, 1), (192,), (128,), (60, 256), (3, 25, 25), (3, 25, 25)],
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
            alpha=[-0.0998104, -0.0997045, -0.0997628, -0.0996279, -0.099284, -0.0997342, -0.0998837, -0.0996702],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128,), (256,), (64, 64, 9, 1), (256, 256, 9, 1), (60, 256), (60, 256), (3, 25, 25), (3, 25, 25)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(128,), (256,), (64, 64, 9, 1), (256, 256, 9, 1), (60, 256), (60, 256), (3, 25, 25), (3, 25, 25)],
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
                    "shape": [(32, 64, 100, 25), (32, 64, 100, 25), ()],
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
            dim=[-1, -1],
            keepdim=[True, True],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(16, 60), (7, 60)],
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
            training=[True, False, True, False],
            momentum=[0.1, 0.1, 0.1, 0.1],
            eps=[1e-05, 1e-05, 1e-05, 1e-05],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(14, 256, 50, 25), (14, 256, 25, 25), (32, 75, 100), (14, 75, 100)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(256,), (256,), (75,), (75,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(256,), (256,), (75,), (75,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "requires_grad":[False],
                    "shape": [(256,), (256,), (75,), (75,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_var"],
                    "requires_grad":[False],
                    "shape": [(256,), (256,), (75,), (75,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.positive,
                },
            ],
        ),
    ),

    'bmm': dict(
        name=["bmm"],
        interface=["torch"],
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(1, 204800, 25), (1, 89600, 75)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["mat2"],
                    "requires_grad":[True],
                    "shape": [(1, 25, 75), (1, 75, 25)],
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
            dim=[0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(75,), (75,), (1875,), (64,), (64,), (576,), (192,), (36864,), (64,), (64,), (64,), (1875,), (64,), (64,), (12288,), (192,), (36864,), (64,), (64,), (64,), (1875,), (64,), (64,), (12288,), (192,), (36864,), (64,), (64,), (64,), (1875,), (64,), (64,), (12288,), (192,), (36864,), (64,), (64,), (64,), (1875,), (128,), (128,), (24576,), (384,), (147456,), (128,), (128,), (128,), (8192,), (128,), (128,), (128,), (1875,), (128,), (128,), (49152,), (384,), (147456,), (128,), (128,), (128,), (1875,), (128,), (128,), (49152,), (384,), (147456,), (128,), (128,), (128,), (1875,), (256,), (256,), (98304,), (768,), (589824,), (256,), (256,), (256,), (32768,), (256,), (256,), (256,), (1875,), (256,), (256,), (196608,), (768,), (589824,), (256,), (256,), (256,), (1875,), (256,), (256,), (196608,), (768,), (589824,), (256,), (256,), (256,), (15360,), (60,), (75,), (75,), (1875,), (64,), (64,), (64,), (64,), (1875,), (64,), (64,), (64,), (64,), (1875,), (64,), (64,), (64,), (64,), (1875,), (64,), (64,), (64,), (64,), (1875,), (128,), (128,), (128,), (128,), (128,), (128,), (1875,), (128,), (128,), (128,), (128,), (1875,), (128,), (128,), (128,), (128,), (1875,), (256,), (256,), (256,), (256,), (256,), (256,), (1875,), (256,), (256,), (256,), (256,), (1875,), (256,), (256,), (256,), (256,)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,), (1,)]],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
            stride=[(2, 1), (2, 1)],
            padding=[(0, 0), (4, 0)],
            dilation=[(1, 1), (1, 1)],
            groups=[1, 1],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(32, 64, 100, 25), (32, 256, 50, 25)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(128, 64, 1, 1), (256, 256, 9, 1)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(128,), (256,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'div': dict(
        name=["div"],
        interface=["torch"],
        para=dict(
            other=[1, 1, 1, 2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(), (), (), (7, 2, 256), (16, 2, 256)],
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
            value=[0, 1, 1, 0, 0, 1, 1, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(256,), (256,), (64,), (64,), (60,), (), (128,), (128,)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
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
                    "shape": [(7, 256), (7, 256), (16, 256), (16, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["weight"],
                    "requires_grad":[True],
                    "shape": [(60, 256), (60, 256), (60, 256), (60, 256)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["bias"],
                    "requires_grad":[True],
                    "shape": [(60,), (60,), (60,), (60,)],
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
                    "shape": [(7, 60), (16, 60)],
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
            dim=[None, None, None, None, [1], [1], None, None, None],
            keepdim=[False, False, False, False, False, False, False, False, False],
            dtype=[None, None, None, None, None, None, None, None, None],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(128, 128, 9, 1), (64, 64, 9, 1), (64,), (128,), (7, 1, 60), (7, 2, 256), (), (), (60, 256)],
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
                    "shape": [(3, 25, 25), (3, 25, 25)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["other"],
                    "shape": [(3, 25, 25), (3, 25, 25)],
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
            other=[0.9, 0.9, 0.9, 0.9, 0.9, 1, 0.9],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(60, 256), (384,), (64,), (768, 128, 1, 1), (128, 64, 1, 1), (), (3, 25, 25)],
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
                    "shape": [(768, 256, 1, 1), (192, 3, 1, 1), (256,), (192,), (60, 256), (), (3, 25, 25)],
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
            reduction=['mean', 'mean'],
            ignore_index=[-100, -100],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[True],
                    "shape": [(16, 60), (7, 60)],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["target"],
                    "requires_grad":[False],
                    "shape": [(16,), (7,)],
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
            mean=[0, 0, 0, 0, 0, 0],
            std=[0.0294628, 0.125, 0.01, 0.0589256, 0.0883883, 0.0416667],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "shape": [(256, 256, 9, 1), (128, 64, 1, 1), (60, 256), (64, 64, 9, 1), (256, 128, 1, 1), (128, 128, 9, 1)],
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
                    "shape": [(14, 256, 50, 25), (14, 128, 100, 25)],
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
            dim=[2, 2],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["input"],
                    "requires_grad":[False],
                    "shape": [(16, 1, 60), (7, 1, 60)],
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
            dim=[0, 0, 0, 0],
        ),
        tensor_para=dict(
            args=[
                {
                    "ins": ["tensors"],
                    "shape": [[(1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3)], [(1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3)], [(1,), (1,), (1,), (1,), (1,), (1,), (1,)], [(1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3), (1, 2, 100, 25, 3)]],
                    "dtype": [Dtype.float32],
                    "gen_fn": Genfunc.randn,
                },
            ],
            seq_name='tensors',
        ),
    ),

}
