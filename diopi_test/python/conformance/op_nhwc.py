# Copyright (c) 2023, DeepLink.

# Note : 1. aten's cuda implementation doesn't support 3-dim nhwc Tensor
#        adaptive_max_pool2d（3d), max_pool3d,
#        adaptive_avg_pool3d, interpolate doesn't support nhwc memory format
#        avg_pool2d backward can't compute right along the edges
#        2. For camb test, adaptive_max_pool2d/max_pool2d need indices being int32
#        Only conv2d, bn, adaptive_avg_pool2d, adaptive_max_pool2d can be tested, because
#        the rest have't been implemented.
nhwc_op = {'conv2d': ["2d", "input", 'weight'],
           'conv3d': ["3d", "input", 'weight'],
           'batch_norm': ['input'],
           'adaptive_avg_pool2d': ["2d", 'input'],
           'adaptive_max_pool2d': ["2d", 'input'],
           'adaptive_avg_pool3d': ["3d", 'input'],
           'adaptive_max_pool3d': ["3d", 'input'],
           'avg_pool2d': ["2d", 'input'],
           'max_pool2d': ["2d", 'input'],
           # 'avg_pool3d': ["3d", 'input'], diopi doesn't hava avg_pool3d test
           'max_pool3d': ["3d", 'input'],
           # both embedding
           'interpolate': ['input'],
           'pad': ['input'],
           'roi_align': ['input']}
