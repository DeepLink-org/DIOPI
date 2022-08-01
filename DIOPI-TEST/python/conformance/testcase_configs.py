# from functools import partial
from .testcase_parse import Genfunc
from .dtype import Dtype

configs = {
    'batch_norm': dict(
        name=["batch_norm"],
        dtype=[Dtype.float32, Dtype.float64],
        atol=1e-5,
        call_para=dict(
            args=[
                {
                    "shape": ((2, 5, 3, 5), (3, 4, 3), (2, 3)),
                    "gen_fn": Genfunc.randn,
                },
                {
                    "ins": ["running_mean"],
                    "shape": ((5, ), (4, ), (3, )),
                    "gen_fn": Genfunc.zeros,
                },
                {
                    "ins": ["running_var"],
                    "value": [[1.0, 1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0, 1.0],
                              [1.0, 1.0, 1.0]]
                },
                {
                    "ins": ["weight", "bias"],
                    "shape": ((5, ), (4, ), (3, )),
                    "gen_fn": Genfunc.randn,
                },
            ]
        ),
    ),

    'soft_margin_loss': dict(
        name=["soft_margin_loss"],
        para=dict(
            reduction=['elementwise_mean'],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "value": [[1, 1.5, 2, 2.5, 3]]
                },
                {
                    "ins": ['target'],
                    "value": [[1.0, 1.0, -1.0, -1.0, 1.0]],
                },
            ],
        ),
    ),

    'log_softmax': dict(
        name=["log_softmax"],
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'soft_max': dict(
        name=["softmax"],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            dim=[1],
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 1025), (1025, 1025)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'soft_min': dict(
        name=["softmin"],
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'tanhshrink': dict(
        ex_base='tanhshrink',
        name=["tanhshrink"],
    ),

    'sigmoid': dict(
        name=["sigmoid"],
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'softsign': dict(
        name=["softsign"],
        atol=1e-4,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'tanh': dict(
        name=["tanh"],
        atol=1e-4,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 14, 14), (64, 7, 28, 28),
                              (4, 1, 16, 16, 16)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'soft_max_2': dict(
        name=["softmax"],
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 1025), (64, 28, 28), (0, ),
                              (16, 14, 14), (64, 7, 28, 28),
                              (16, 128, 4096), (32, 128, 2048)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'relu6': dict(
        name=["relu6"],
        atol=1e-4,
        rtol=1e-5,
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 3, 14, 14), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),

    'max_pool2d': dict(
        ex_base="max_pool2d",
        name=["max_pool2d"],
    ),
    'max_pool2d_native': dict(
        ex_base="max_pool2d_native",
        name=["max_pool2d"],
    ),
    'max_pool3d': dict(
        ex_base="max_pool3d",
        name=["max_pool3d"],
    ),
    'max_pool3d_native': dict(
        ex_base="max_pool3d_native",
        name=["max_pool3d"],
    ),

    'logsigmoid': dict(
        ex_base="logsigmoid",
        name=["logsigmoid"],
    ),

    'multilabel_soft_margin_loss': dict(
        ex_base="multilabel_soft_margin_loss",
        name=["multilabel_soft_margin_loss"],
    ),

    'celu': dict(
        ex_base="celu",
        name=["celu"],
    ),

    'rrelu_1': dict(
        name=["rrelu"],
        ex_base='rrelu_inplace',
        rtol=1e-5,
    ),

    'hard': dict(
        name=["hardshrink",
              "hardtanh"],
        ex_base="hard",
        rtol=1e-5,
    ),
    'fold': dict(
        name=["fold"],
        ex_base="fold",
        rtol=1e-5,
    ),
    'unfold': dict(
        name=["unfold"],
        ex_base="unfold",
        rtol=1e-5,
    ),

    'leaky_relu_': dict(
        name=["leaky_relu_"],
        atol=1e-4,
        rtol=1e-5,
        para=dict(
            negative_slope=[0.01, 0.1, 1, 10]
        ),
        call_para=dict(
            args=[
                {
                    "ins": ['input'],
                    "shape": ((16, 7), (64, 28, 28),
                              (16, 3, 14, 14), (64, 3, 7, 28, 28)),
                    "dtype": [Dtype.float32, Dtype.float64],
                    "gen_fn": Genfunc.randn,
                },
            ],
        ),
    ),
}
