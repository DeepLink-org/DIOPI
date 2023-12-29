# Copyright (c) 2023, DeepLink.

# set nhwc
def compute_nhwc_stride_2d(sizes, itemsize=1):
    dim = len(sizes)
    strides = [itemsize for i in range(dim)]
    assert dim == 3 or dim == 4, "not supported dim"
    if dim == 3:
        strides[0] = itemsize
        strides[2] = strides[0] * sizes[0]
        strides[1] = strides[2] * sizes[2]
    elif dim == 4:
        strides[1] = itemsize
        strides[3] = strides[0] * sizes[1]
        strides[2] = strides[3] * sizes[3]
        strides[0] = strides[2] * sizes[2]
    return strides


def compute_nhwc_stride_3d(sizes, itemsize=1):
    dim = len(sizes)
    strides = [itemsize for i in range(dim)]
    assert dim == 4 or dim == 5, "not supported dim"
    if dim == 4:
        strides[0] = itemsize
        strides[3] = strides[0] * sizes[0]
        strides[2] = strides[3] * sizes[3]
        strides[1] = strides[2] * sizes[2]
    elif dim == 5:
        strides[1] = itemsize
        strides[4] = strides[0] * sizes[1]
        strides[3] = strides[4] * sizes[4]
        strides[2] = strides[3] * sizes[3]
        strides[0] = strides[2] * sizes[2]
    return strides


def compute_nhwc_stride(size, itemsize=1, name=None):
    if name == '2d':
        return compute_nhwc_stride_2d(size, itemsize)
    if name == '3d':
        return compute_nhwc_stride_3d(size, itemsize)

    dim = len(size)
    if dim < 5:
        return compute_nhwc_stride_2d(size, itemsize)
    else:
        return compute_nhwc_stride_3d(size, itemsize)
