import conformance.diopi_functions as F
import re
from conformance.diopi_functions import abs, cos, erf, exp, floor, log, log2, log10, neg, nonzero,\
                                        sign, sin, sqrt, add, bmm, div, eq, fill_, ge, gt, le, lt,\
                                        logical_and, logical_or, matmul, mul, ne, pow, sub, binary_cross_entropy_with_logits,\
                                        cross_entropy, mse_loss, nll_loss, leaky_relu, relu, sigmoid, hardtanh, threshold,\
                                        gelu, tanh, softmax, log_softmax, mean, min, max, std, sum, all, any, addcdiv, addcmul,\
                                        addmm, adaptive_avg_pool2d, avg_pool2d, max_pool2d, adaptive_max_pool2d, batch_norm,\
                                        cat, clamp, clip_grad_norm_, conv2d, dropout, embedding, index_select, masked_scatter,\
                                        linear, one_hot, select, sort, split, stack, topk, transpose, tril, where

from conformance.diopi_functions import sigmoid_focal_loss, nms, slice_op, index, sgd, roi_align #TODO(ht): add slice_op and index doc
from conformance.diopi_functions import arange, randperm, uniform, random, bernoulli, masked_fill, adamw, adam, adadelta, conv_transpose2d, \
                                        cumsum, cdist, reciprocal, bitwise_not, argmax, smooth_l1_loss, maximum, minimum, mm, conv3d, \
                                        expand, unfold, masked_select, index_fill, linspace, roll, norm, group_norm, layer_norm,\
                                        adaptive_avg_pool3d, adaptive_max_pool3d, max_pool3d, permute, copy_, gather, remainder,\
                                        ctc_loss, index_put, scatter, interpolate, pad, unique, prod
from conformance.diopi_functions import erfinv, im2col, col2im, flip, cholesky_ex, triangular_solve


def parse_kwargs(desc):
    """Maps a description of args to a dictionary of {argname: description}.
    Input:
        ('    weight (Tensor): a weight tensor\n' +
         '        Some optional description')
    Output: {
        'weight': \
        'weight (Tensor): a weight tensor\n        Some optional description'
    }
    """
    # Split on exactly 4 spaces after a newline
    regx = re.compile(r"\n\s{4}(?!\s)")
    kwargs = [section.strip() for section in regx.split(desc)]
    kwargs = [section for section in kwargs if len(section) > 0]
    return {desc.split(' ')[0]: desc for desc in kwargs}


def merge_dicts(*dicts):
    return {x: d[x] for d in dicts for x in d}


common_args = parse_kwargs("""
    input (Tensor): the input tensor.
    generator (:class:`Generator`, optional): a pseudorandom number generator for sampling
    out (Tensor, optional): the output tensor.
    memory_format (:class:`memory_format`, optional): the desired memory format of
        returned tensor. Default: ``preserve_format``.
""")

reduceops_common_args = merge_dicts(common_args, parse_kwargs("""
    dtype (:class:`dtype`, optional): the desired data type of returned tensor.
        If specified, the input tensor is casted to :attr:`dtype` before the operation
        is performed. This is useful for preventing data type overflows. Default: None.
    keepdim (bool): whether the output tensor has :attr:`dim` retained or not.
"""))

multi_dim_common = merge_dicts(reduceops_common_args, parse_kwargs("""
    dim (int or tuple of ints): the dimension or dimensions to reduce.
"""), {'keepdim_details': """
If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension(s) :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`squeeze`), resulting in the
output tensor having 1 (or ``len(dim)``) fewer dimension(s).
"""})

single_dim_common = merge_dicts(reduceops_common_args, parse_kwargs("""
    dim (int): the dimension to reduce.
"""), {'keepdim_details': """If :attr:`keepdim` is ``True``, the output tensor is of the same size
as :attr:`input` except in the dimension :attr:`dim` where it is of size 1.
Otherwise, :attr:`dim` is squeezed, resulting in
the output tensor having 1 fewer dimension than :attr:`input`."""})

def add_docstr(attr, docstr):
    assert hasattr(F, attr)
    fuc = getattr(F, attr)
    fuc.__doc__ = docstr


add_docstr("abs", r"""
Computes the absolute value of each element in :attr:`input`.

.. math::
    \text{out}_{i} = |\text{input}_{i}|
""" + r"""
Args:
    {input}

Example::

    >>> abs(tensor([-1, -2, 3]))
    tensor([ 1,  2,  3])
""".format(**common_args))

add_docstr("cos",
           r"""
Returns a new tensor with the cosine  of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \cos(\text{input}_{i})
""" + r"""
Args:
    {input}

Example::

    >>> a = randn(4)
    >>> a
    tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
    >>> cos(a)
    tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
""".format(**common_args))

add_docstr("sin",
        r"""
Returns a new tensor with the sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin(\text{input}_{i})
""" + r"""
Args:
    {input}

Example::

    >>> a = randn(4)
    >>> a
    tensor([-0.5461,  0.1347, -2.7266, -0.2746])
    >>> sin(a)
    tensor([-0.5194,  0.1343, -0.4032, -0.2711])
""")



add_docstr("where",
           r"""
Return a tensor of elements selected from either :attr:`x` or :attr:`y`, depending on :attr:`condition`.

The operation is defined as:

.. math::
    \text{out}_i = \begin{cases}
        \text{x}_i & \text{if } \text{condition}_i \\
        \text{y}_i & \text{otherwise} \\
    \end{cases}

.. note::
    The tensors :attr:`condition`, :attr:`x`, :attr:`y` must be :ref:`broadcastable <broadcasting-semantics>`.

.. note::
    Currently valid scalar and tensor combination are
    1. Scalar of floating dtype and double
    2. Scalar of integral dtype and long
    3. Scalar of complex dtype and complex128

Arguments:
    condition (BoolTensor): When True (nonzero), yield x, otherwise yield y
    x (Tensor or Scalar): value (if :attr:x is a scalar) or values selected at indices
                          where :attr:`condition` is ``True``
    y (Tensor or Scalar): value (if :attr:x is a scalar) or values selected at indices
                          where :attr:`condition` is ``False``

Returns:
    Tensor: A tensor of shape equal to the broadcasted shape of :attr:`condition`, :attr:`x`, :attr:`y`

Example::

    >>> x = randn(3, 2)
    >>> y = ones(3, 2)
    >>> x
    tensor([[-0.4620,  0.3139],
            [ 0.3898, -0.7197],
            [ 0.0478, -0.1657]])
    >>> where(x > 0, x, y)
    tensor([[ 1.0000,  0.3139],
            [ 0.3898,  1.0000],
            [ 0.0478,  1.0000]])
    >>> x = randn(2, 2, dtype=double)
    >>> x
    tensor([[ 1.0779,  0.0383],
            [-0.8785, -1.1089]], dtype=float64)
    >>> where(x > 0, x, 0.)
    tensor([[1.0779, 0.0383],
            [0.0000, 0.0000]], dtype=float64)
""")

add_docstr("tril",
           r"""
Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
:attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.

The lower triangular part of the matrix is defined as the elements on and
below the diagonal.

The argument :attr:`diagonal` controls which diagonal to consider. If
:attr:`diagonal` = 0, all elements on and below the main diagonal are
retained. A positive value includes just as many diagonals above the main
diagonal, and similarly a negative value excludes just as many diagonals below
the main diagonal. The main diagonal are the set of indices
:math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` where
:math:`d_{1}, d_{2}` are the dimensions of the matrix.
""" + r"""
Args:
    {input}
    diagonal (int, optional): the diagonal to consider

Example::

    >>> a = randn(3, 3)
    >>> a
    tensor([[-1.0813, -0.8619,  0.7105],
            [ 0.0935,  0.1380,  2.2112],
            [-0.3409, -0.9828,  0.0289]])
    >>> tril(a)
    tensor([[-1.0813,  0.0000,  0.0000],
            [ 0.0935,  0.1380,  0.0000],
            [-0.3409, -0.9828,  0.0289]])

    >>> b = randn(4, 6)
    >>> b
    tensor([[ 1.2219,  0.5653, -0.2521, -0.2345,  1.2544,  0.3461],
            [ 0.4785, -0.4477,  0.6049,  0.6368,  0.8775,  0.7145],
            [ 1.1502,  3.2716, -1.1243, -0.5413,  0.3615,  0.6864],
            [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0978]])
    >>> tril(b, diagonal=1)
    tensor([[ 1.2219,  0.5653,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.4785, -0.4477,  0.6049,  0.0000,  0.0000,  0.0000],
            [ 1.1502,  3.2716, -1.1243, -0.5413,  0.0000,  0.0000],
            [-0.0614, -0.7344, -1.3164, -0.7648, -1.4024,  0.0000]])
    >>> tril(b, diagonal=-1)
    tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 0.4785,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
            [ 1.1502,  3.2716,  0.0000,  0.0000,  0.0000,  0.0000],
            [-0.0614, -0.7344, -1.3164,  0.0000,  0.0000,  0.0000]])
""".format(**common_args))


add_docstr("transpose",
           r"""
Returns a tensor that is a transposed version of :attr:`input`.
The given dimensions :attr:`dim0` and :attr:`dim1` are swapped.

The resulting :attr:`out` tensor shares it's underlying storage with the
:attr:`input` tensor, so changing the content of one would change the content
of the other.

Args:
    {input}
    dim0 (int): the first dimension to be transposed
    dim1 (int): the second dimension to be transposed

Example::

    >>> x = randn(2, 3)
    >>> x
    tensor([[ 1.0028, -0.9893,  0.5809],
            [-0.1669,  0.7299,  0.4942]])
    >>> transpose(x, 0, 1)
    tensor([[ 1.0028, -0.1669],
            [-0.9893,  0.7299],
            [ 0.5809,  0.4942]])
""".format(**common_args))


add_docstr("topk",
           r"""
Returns the :attr:`k` largest elements of the given :attr:`input` tensor along
a given dimension.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`largest` is ``False`` then the `k` smallest elements are returned.

A namedtuple of `(values, indices)` is returned, where the `indices` are the indices
of the elements in the original `input` tensor.

The boolean option :attr:`sorted` if ``True``, will make sure that the returned
`k` elements are themselves sorted

Args:
    {input}
    k (int): the k in "top-k"
    dim (int, optional): the dimension to sort along
    largest (bool, optional): controls whether to return largest or
           smallest elements
    sorted (bool, optional): controls whether to return the elements
           in sorted order

Example::

    >>> x = arange(1., 6.)
    >>> x
    tensor([ 1.,  2.,  3.,  4.,  5.])
    >>> topk(x, 3)
    return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
""".format(**common_args))


add_docstr("stack",
           r"""
Concatenates a sequence of tensors along a new dimension.

All tensors need to be of the same size.

Arguments:
    tensors (sequence of Tensors): sequence of tensors to concatenate
    dim (int): dimension to insert. Has to be between 0 and the number
        of dimensions of concatenated tensors (inclusive)
""")


add_docstr("split", r"""Splits the tensor into chunks. Each chunk is a view of the original tensor.

    If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
    be split into equally sized chunks (if possible). Last chunk will be smaller if
    the tensor size along the given dimension :attr:`dim` is not divisible by
    :attr:`split_size`.

    If :attr:`split_size_or_sections` is a list, then :attr:`tensor` will be split
    into ``len(split_size_or_sections)`` chunks with sizes in :attr:`dim` according
    to :attr:`split_size_or_sections`.

    Args:
        tensor (Tensor): tensor to split.
        split_size_or_sections (int) or (list(int)): size of a single chunk or
            list of sizes for each chunk
        dim (int): dimension along which to split the tensor.

    Example::

        >>> a = arange(10).reshape(5,2)
        >>> a
        tensor([[0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9]])
        >>> split(a, 2)
        (tensor([[0, 1],
                 [2, 3]]),
         tensor([[4, 5],
                 [6, 7]]),
         tensor([[8, 9]]))
        >>> split(a, [1,4])
        (tensor([[0, 1]]),
         tensor([[2, 3],
                 [4, 5],
                 [6, 7],
                 [8, 9]]))
    """)

add_docstr("index_select",
           r"""
Returns a new tensor which indexes the :attr:`input` tensor along dimension
:attr:`dim` using the entries in :attr:`index` which is a `LongTensor`.

The returned tensor has the same number of dimensions as the original tensor
(:attr:`input`).  The :attr:`dim`\ th dimension has the same size as the length
of :attr:`index`; other dimensions have the same size as in the original tensor.

.. note:: The returned tensor does **not** use the same storage as the original
          tensor.  If :attr:`out` has a different shape than expected, we
          silently change it to the correct shape, reallocating the underlying
          storage if necessary.

Args:
    {input}
    dim (int): the dimension in which we index
    index (LongTensor): the 1-D tensor containing the indices to index

Example::

    >>> x = randn(3, 4)
    >>> x
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-0.4664,  0.2647, -0.1228, -1.1068],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> indices = tensor([0, 2])
    >>> index_select(x, 0, indices)
    tensor([[ 0.1427,  0.0231, -0.5414, -1.0009],
            [-1.1734, -0.6571,  0.7230, -0.6004]])
    >>> index_select(x, 1, indices)
    tensor([[ 0.1427, -0.5414],
            [-0.4664, -0.1228],
            [-1.1734,  0.7230]])
""".format(**common_args))


add_docstr("sort",
           r"""
Sorts the elements of the :attr:`input` tensor along a given dimension
in ascending order by value.

If :attr:`dim` is not given, the last dimension of the `input` is chosen.

If :attr:`descending` is ``True`` then the elements are sorted in descending
order by value.

A namedtuple of (values, indices) is returned, where the `values` are the
sorted values and `indices` are the indices of the elements in the original
`input` tensor.

Args:
    {input}
    dim (int, optional): the dimension to sort along
    descending (bool, optional): controls the sorting order (ascending or descending)

Example::

    >>> x = randn(3, 4)
    >>> sorted, indices = sort(x)
    >>> sorted
    tensor([[-0.2162,  0.0608,  0.6719,  2.3332],
            [-0.5793,  0.0061,  0.6058,  0.9497],
            [-0.5071,  0.3343,  0.9553,  1.0960]])
    >>> indices
    tensor([[ 1,  0,  2,  3],
            [ 3,  1,  0,  2],
            [ 0,  3,  1,  2]])

    >>> sorted, indices = sort(x, 0)
    >>> sorted
    tensor([[-0.5071, -0.2162,  0.6719, -0.5793],
            [ 0.0608,  0.0061,  0.9497,  0.3343],
            [ 0.6058,  0.9553,  1.0960,  2.3332]])
    >>> indices
    tensor([[ 2,  0,  0,  1],
            [ 0,  1,  1,  2],
            [ 1,  2,  2,  0]])
""".format(**common_args))


add_docstr('select',
               r"""
Slices the :attr:`self` tensor along the selected dimension at the given index.
This function returns a view of the original tensor with the given dimension removed.

Args:
    dim (int): the dimension to slice
    index (int): the index to select with

.. note::

    :meth:`select` is equivalent to slicing. For example,
    ``tensor.select(0, index)`` is equivalent to ``tensor[index]`` and
    ``tensor.select(2, index)`` is equivalent to ``tensor[:,:,index]``.
""")


add_docstr("one_hot", r"""
Takes LongTensor with index values of shape ``(*)`` and returns a tensor
of shape ``(*, num_classes)`` that have zeros everywhere except where the
index of last dimension matches the corresponding value of the input tensor,
in which case it will be 1.

See also `One-hot on Wikipedia`_ .

.. _One-hot on Wikipedia:
    https://en.wikipedia.org/wiki/One-hot

Arguments:
    tensor (LongTensor): class values of any shape.
    num_classes (int):  Total number of classes. If set to -1, the number
        of classes will be inferred as one greater than the largest class
        value in the input tensor.

Returns:
    LongTensor that has one more dimension with 1 values at the
    index of last dimension indicated by the input, and 0 everywhere
    else.

Examples:
    >>> F.one_hot(arange(0, 5) % 3)
    tensor([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]])
    >>> F.one_hot(arange(0, 5) % 3, num_classes=5)
    tensor([[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]])
    >>> F.one_hot(arange(0, 6).view(3,2) % 3)
    tensor([[[1, 0, 0],
             [0, 1, 0]],
            [[0, 0, 1],
             [1, 0, 0]],
            [[0, 1, 0],
             [0, 0, 1]]])
""")

add_docstr("linear", r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:

        - Input: :math:`(N, *, in_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out_features, in_features)`
        - Bias: :math:`(out_features)`
        - Output: :math:`(N, *, out_features)`
    """)


add_docstr('masked_scatter',
               r"""
Copies elements from :attr:`source` into :attr:`self` tensor at positions where
the :attr:`mask` is True.
The shape of :attr:`mask` must be :ref:`broadcastable <broadcasting-semantics>`
with the shape of the underlying tensor. The :attr:`source` should have at least
as many elements as the number of ones in :attr:`mask`

Args:
    mask (BoolTensor): the boolean mask
    source (Tensor): the tensor to copy from

.. note::

    The :attr:`mask` operates on the :attr:`self` tensor, not on the given
    :attr:`source` tensor.
""")

add_docstr("embedding", r"""A simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    Args:
        input (LongTensor): Tensor containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): If given, pads the output with the embedding vector at :attr:`padding_idx`
                                         (initialized to zeros) whenever it encounters the index.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`nn.Embedding` for more details regarding sparse gradients.

    Shape:
        - Input: LongTensor of arbitrary shape containing the indices to extract
        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`, where V = maximum index + 1 and embedding_dim = the embedding size
        - Output: `(*, embedding_dim)`, where `*` is the input shape

    Examples::

        >>> # a batch of 2 samples of 4 indices each
        >>> input = tensor([[1,2,4,5],[4,3,2,9]])
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = rand(10, 3)
        >>> F.embedding(input, embedding_matrix)
        tensor([[[ 0.8490,  0.9625,  0.6753],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.6246,  0.9751,  0.3618],
                 [ 0.4161,  0.2419,  0.7383]],

                [[ 0.6246,  0.9751,  0.3618],
                 [ 0.0237,  0.7794,  0.0528],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.3385,  0.8612,  0.1867]]])

        >>> # example with padding_idx
        >>> weights = rand(10, 3)
        >>> weights[0, :].zero_()
        >>> embedding_matrix = weights
        >>> input = tensor([[0,2,0,5]])
        >>> F.embedding(input, embedding_matrix, padding_idx=0)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.5609,  0.5384,  0.8720],
                 [ 0.0000,  0.0000,  0.0000],
                 [ 0.6262,  0.2438,  0.7471]]])
    """)


add_docstr("dropout", r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    See :class:`~nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """)


add_docstr("conv2d", r"""

Applies a 2D convolution over an input image composed of several input
planes.

Args:
    input: input tensor of shape :math:`(\text{minibatch} , \text{in_channels} , iH , iW)`
    weight: filters of shape :math:`(\text{out_channels} , \frac{\text{in_channels}}{\text{groups}} , kH , kW)`
    bias: optional bias tensor of shape :math:`(\text{out_channels})`. Default: ``None``
    stride: the stride of the convolving kernel. Can be a single number or a
      tuple `(sH, sW)`. Default: 1
    padding: implicit paddings on both sides of the input. Can be a
      single number or a tuple `(padH, padW)`. Default: 0
    dilation: the spacing between kernel elements. Can be a single number or
      a tuple `(dH, dW)`. Default: 1
    groups: split input into groups, :math:`\text{in_channels}` should be divisible by the
      number of groups. Default: 1

Examples::

    >>> # With square kernels and equal stride
    >>> filters = randn(8,4,3,3)
    >>> inputs = randn(1,4,5,5)
    >>> F.conv2d(inputs, filters, padding=1)
""")


add_docstr("clip_grad_norm_", r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """)


add_docstr("clamp", r"""
Clamp all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]` and return
a resulting tensor:

.. math::
    y_i = \begin{cases}
        \text{min} & \text{if } x_i < \text{min} \\
        x_i & \text{if } \text{min} \leq x_i \leq \text{max} \\
        \text{max} & \text{if } x_i > \text{max}
    \end{cases}
""" + r"""
If :attr:`input` is of type `FloatTensor` or `DoubleTensor`, args :attr:`min`
and :attr:`max` must be real numbers, otherwise they should be integers.

Args:
    {input}
    min (Number): lower-bound of the range to be clamped to
    max (Number): upper-bound of the range to be clamped to

Example::

    >>> a = randn(4)
    >>> a
    tensor([-1.7120,  0.1734, -0.0478, -0.0922])
    >>> clamp(a, min=-0.5, max=0.5)
    tensor([-0.5000,  0.1734, -0.0478, -0.0922])

    >>> a = randn(4)
    >>> a
    tensor([-0.0299, -2.3184,  2.1593, -0.8883])
    >>> clamp(a, min=0.5)
    tensor([ 0.5000,  0.5000,  2.1593,  0.5000])

    >>> a = randn(4)
    >>> a
    tensor([ 0.7753, -0.4702, -0.4599,  1.1899])
    >>> clamp(a, max=0.5)
    tensor([ 0.5000, -0.4702, -0.4599,  0.5000])
""".format(**common_args))


add_docstr("cat",
           r"""Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
All tensors must either have the same shape (except in the concatenating
dimension) or be empty.

:func:`cat` can be best understood via examples.

Args:
    tensors (sequence of Tensors): any python sequence of tensors of the same type.
        Non-empty tensors provided must have the same shape, except in the
        cat dimension.
    dim (int, optional): the dimension over which the tensors are concatenated

Example::

    >>> x = randn(2, 3)
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
             -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
             -0.5790,  0.1497]])
""")

add_docstr("batch_norm",  r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    """)


add_docstr("adaptive_max_pool2d",  r"""Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
        output_size: the target output size (single integer or double-integer tuple)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    """)


add_docstr("max_pool2d",  r"""Applies a 2D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        input: input tensor :math:`(\text{minibatch} , \text{in_channels} , iH , iW)`, minibatch dim optional.
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor
    """)


add_docstr("avg_pool2d",
    r"""
    Applies a 2D average pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        input: input tensor :math:`(\text{minibatch} , \text{in_channels} , iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a
            tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
            tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
            single number or a tuple `(padH, padW)`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation. Default: ``True``
        divisor_override: if specified, it will be used as divisor, otherwise
                size of the pooling region will be used. Default: None
    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel_size}[1]}{\text{stride}[1]} + 1\right\rfloor
""")


add_docstr("adaptive_avg_pool2d", r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1})` or :math:`(C, S_{0}, S_{1})`, where
          :math:`S=\text{output_size}`.

    """)


add_docstr("addmm",
           r"""Performs a matrix multiplication of the matrices :attr:`mat1` and :attr:`mat2`.
The matrix :attr:`input` is added to the final result.

If :attr:`mat1` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
:math:`(m \times p)` tensor, then :attr:`input` must be
:ref:`broadcastable <broadcasting-semantics>` with a :math:`(n \times p)` tensor
and :attr:`out` will be a :math:`(n \times p)` tensor.

:attr:`alpha` and :attr:`beta` are scaling factors on matrix-vector product between
:attr:`mat1` and :attr:`mat2` and the added matrix :attr:`input` respectively.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)

If :attr:`beta` is 0, then :attr:`input` will be ignored, and `nan` and `inf` in
it will not be propagated.
""" + r"""
For inputs of type `FloatTensor` or `DoubleTensor`, arguments :attr:`beta` and
:attr:`alpha` must be real numbers, otherwise they should be integers.



Args:
    input (Tensor): matrix to be added
    mat1 (Tensor): the first matrix to be matrix multiplied
    mat2 (Tensor): the second matrix to be matrix multiplied

Keyword args:
    beta (Number, optional): multiplier for :attr:`input` (:math:`\beta`)
    alpha (Number, optional): multiplier for :math:`mat1 @ mat2` (:math:`\alpha`)

Example::

    >>> M = randn(2, 3)
    >>> mat1 = randn(2, 3)
    >>> mat2 = randn(3, 3)
    >>> addmm(M, mat1, mat2)
    tensor([[-4.8716,  1.4671, -1.3746],
            [ 0.7573, -3.9555, -2.8681]])
""")



add_docstr("addcmul",
           r"""
Performs the element-wise multiplication of :attr:`tensor1`
by :attr:`tensor2`, multiply the result by the scalar :attr:`value`
and add it to :attr:`input`.

.. math::
    \text{out}_i = \text{input}_i + \text{value} \times \text{tensor1}_i \times \text{tensor2}_i
""" + r"""
The shapes of :attr:`tensor`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    input (Tensor): the tensor to be added
    tensor1 (Tensor): the tensor to be multiplied
    tensor2 (Tensor): the tensor to be multiplied

Keyword args:
    value (Number, optional): multiplier for :math:`tensor1 .* tensor2`

Example::

    >>> t = randn(1, 3)
    >>> t1 = randn(3, 1)
    >>> t2 = randn(1, 3)
    >>> addcmul(t, t1, t2, value=0.1)
    tensor([[-0.8635, -0.6391,  1.6174],
            [-0.7617, -0.5879,  1.7388],
            [-0.8353, -0.6249,  1.6511]])
""")


add_docstr("addcdiv", r"""Performs the element-wise division of :attr:`tensor1` by :attr:`tensor2`,
multiply the result by the scalar :attr:`value` and add it to :attr:`input`.

.. warning::
    Integer division with addcdiv is no longer supported, and in a future
    release addcdiv will perform a true division of tensor1 and tensor2.
    The historic addcdiv behavior can be implemented as
    (input + value * trunc(tensor1 / tensor2)).to(input.dtype)
    for integer inputs and as (input + value * tensor1 / tensor2) for float inputs.
    The future addcdiv behavior is just the latter implementation:
    (input + value * tensor1 / tensor2), for all dtypes.

.. math::
    \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}
""" + r"""

The shapes of :attr:`input`, :attr:`tensor1`, and :attr:`tensor2` must be
:ref:`broadcastable <broadcasting-semantics>`.

For inputs of type `FloatTensor` or `DoubleTensor`, :attr:`value` must be
a real number, otherwise an integer.

Args:
    input (Tensor): the tensor to be added
    tensor1 (Tensor): the numerator tensor
    tensor2 (Tensor): the denominator tensor

Keyword args:
    value (Number, optional): multiplier for :math:`\text{tensor1} / \text{tensor2}`

Example::

    >>> t = randn(1, 3)
    >>> t1 = randn(3, 1)
    >>> t2 = randn(1, 3)
    >>> addcdiv(t, t1, t2, value=0.1)
    tensor([[-0.2312, -3.6496,  0.1312],
            [-1.0428,  3.4292, -0.1030],
            [-0.5369, -0.9829,  0.0430]])
""")


add_docstr("any",
           r"""

For each row of :attr:`input` in the given dimension :attr:`dim`,
returns `True` if any element in the row evaluate to `True` and `False` otherwise.

{keepdim_details}

Args:
    {input}
    {dim}
    {keepdim}

Example::

    >>> a = randn(4, 2) < 0
    >>> a
    tensor([[ True,  True],
            [False,  True],
            [ True,  True],
            [False, False]])
    >>> any(a, 1)
    tensor([ True,  True,  True, False])
    >>> any(a, 0)
    tensor([True, True])
""".format(**single_dim_common))


add_docstr("all",
           r"""
For each row of :attr:`input` in the given dimension :attr:`dim`,
returns `True` if all elements in the row evaluate to `True` and `False` otherwise.

{keepdim_details}

Args:
    {input}
    {dim}
    {keepdim}

Example::

    >>> a = rand(4, 2).bool()
    >>> a
    tensor([[True, True],
            [True, False],
            [True, True],
            [True, True]], dtype=bool)
    >>> all(a, dim=1)
    tensor([ True, False,  True,  True], dtype=bool)
    >>> all(a, dim=0)
    tensor([ True, False], dtype=bool)
""".format(**single_dim_common))


add_docstr("sum", r"""
Returns the sum of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.

{keepdim_details}

Args:
    {input}
    {dim}
    {keepdim}

Keyword args:
    {dtype}

Example::

    >>> a = randn(4, 4)
    >>> a
    tensor([[ 0.0569, -0.2475,  0.0737, -0.3429],
            [-0.2993,  0.9138,  0.9337, -1.6864],
            [ 0.1132,  0.7892, -0.1003,  0.5688],
            [ 0.3637, -0.9906, -0.4752, -1.5197]])
    >>> sum(a, 1)
    tensor([-0.4598, -0.1381,  1.3708, -2.6217])
    >>> b = arange(4 * 5 * 6).view(4, 5, 6)
    >>> sum(b, (2, 1))
    tensor([  435.,  1335.,  2235.,  3135.])
""".format(**multi_dim_common))


add_docstr("std", r"""
Calculates the standard deviation of each row of attr:`input` tensor in the given
dimension :attr:`dim`.
If :attr:`unbiased` is ``True``, Bessel's correction will be used.
Otherwise, the sample deviation is calculated, without any correction.

Args:
    {input}
    {dim}

Keyword args:
    unbiased (bool): whether to use Bessel's correction (:math:`\delta N = 1`).
    {keepdim}

Example::

    >>> a = tensor([[-0.8166, -1.3802, -0.3560]])
    >>> std(a, unbiased=False)
    tensor(0.4188)
""".format(**multi_dim_common))


add_docstr("max",
           r"""
Returns a namedtuple ``(values, indices)`` where ``values`` is the maximum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each maximum value found
(argmax).

If ``keepdim`` is ``True``, the output tensors are of the same size
as ``input`` except in the dimension ``dim`` where they are of size 1.
Otherwise, ``dim`` is squeezed (see :func:`squeeze`), resulting
in the output tensors having 1 fewer dimension than ``input``.

.. note:: If there are multiple maximal values in a reduced row then
          the indices of the first maximal value are returned.

Args:
    {input}
    {dim}
    {keepdim} Default: ``False``.

Example::

    >>> a = randn(4, 4)
    >>> a
    tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
            [ 1.1949, -1.1127, -2.2379, -0.6702],
            [ 1.5717, -0.9207,  0.1297, -1.8768],
            [-0.6172,  1.0036, -0.6060, -0.2432]])
    >>> max(a, 1)
    return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))

""".format(**single_dim_common))


add_docstr("min",
           r"""
Returns a namedtuple ``(values, indices)`` where ``values`` is the minimum
value of each row of the :attr:`input` tensor in the given dimension
:attr:`dim`. And ``indices`` is the index location of each minimum value found
(argmin).

If :attr:`keepdim` is ``True``, the output tensors are of the same size as
:attr:`input` except in the dimension :attr:`dim` where they are of size 1.
Otherwise, :attr:`dim` is squeezed (see :func:`squeeze`), resulting in
the output tensors having 1 fewer dimension than :attr:`input`.

.. note:: If there are multiple minimal values in a reduced row then
          the indices of the first minimal value are returned.

Args:
    {input}
    {dim}
    {keepdim}

Example::

    >>> a = randn(4, 4)
    >>> a
    tensor([[-0.6248,  1.1334, -1.1899, -0.2803],
            [-1.4644, -0.2635, -0.3651,  0.6134],
            [ 0.2457,  0.0384,  1.0128,  0.7015],
            [-0.1153,  2.9849,  2.1458,  0.5788]])
    >>> min(a, 1)
    return_types.min(values=tensor([-1.1899, -1.4644,  0.0384, -0.1153]), indices=tensor([2, 0, 1, 0]))

""".format(**single_dim_common))


add_docstr("mean", r"""
Returns the mean value of each row of the :attr:`input` tensor in the given
dimension :attr:`dim`. If :attr:`dim` is a list of dimensions,
reduce over all of them.

{keepdim_details}

Args:
    {input}
    {dim}
    {keepdim}

Keyword args:
    {dtype}

Example::

    >>> a = randn(4, 4)
    >>> a
    tensor([[-0.3841,  0.6320,  0.4254, -0.7384],
            [-0.9644,  1.0131, -0.6549, -1.4279],
            [-0.2951, -1.3350, -0.7694,  0.5600],
            [ 1.0842, -0.9580,  0.3623,  0.2343]])
    >>> mean(a, 1)
    tensor([-0.0163, -0.5085, -0.4599,  0.1807])
    >>> mean(a, 1, True)
    tensor([[-0.0163],
            [-0.5085],
            [-0.4599],
            [ 0.1807]])
""".format(**multi_dim_common))


add_docstr("softmax", r"""Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """)


add_docstr("log_softmax", r"""
    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Tensor. The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Args:
        input (Tensor): input
        dim (int): A dimension along which log_softmax will be computed.
        dtype (:class:`dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is cast to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input
 
    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)
    """)


add_docstr("tanh",
           r"""Returns a new tensor with the hyperbolic tangent of the elements
of :attr:`input`.

.. math::
    \text{out}_{i} = \tanh(\text{input}_{i})
""" + r"""
Args:
    {input}

Example::

    >>> a = randn(4)
    >>> a
    tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
    >>> tanh(a)
    tensor([ 0.7156, -0.6218,  0.8257,  0.2553])
""".format(**common_args))

add_docstr("gelu", r"""Applies element-wise the function
:math:`\text{GELU}(x) = x * \Phi(x)`

where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.

When the approximate argument is tanh, Gelu is estimated with:

.. math::
    \text { GRELU  }(x)=  0.5 * x * (1 + \text{Tanh}(sqrt(2 / \pi) * (x + 0.044715 * x^3)))

Shape:
    - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
    - Output: :math:`(*)`, same shape as the input.

""")

add_docstr("threshold", r"""Thresholds each element of the input Tensor.

    Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        threshold: The value to threshold at
        value: The value to replace with
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Threshold(0.1, 20)
        >>> input = randn(2)
        >>> output = m(input)
    """)

add_docstr("hardtanh", r"""Applies the HardTanh function element-wise.

    HardTanh is defined as:

    .. math::
        \text{HardTanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`min_val` and :attr:`max_val`.

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.""")

add_docstr("sigmoid",     r"""
    Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """)

add_docstr("relu",     r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        input (Tensor): input
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    """)

add_docstr("leaky_relu",     r"""Applies the element-wise function:

    .. math::
        \text{LeakyReLU}(x) = \max(0, x) + \text{negative_slope} * \min(0, x)

    or

    .. math::
        \text{LeakyRELU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative_slope} \times x, & \text{ otherwise }
        \end{cases}

    Args: 
        input (Tensor): input
        negative_slope: Controls the angle of the negative slope. Default: 1e-2
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    """)

add_docstr("nll_loss",    r"""
    The negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    If provided, the optional argument :attr:`weight` should be a 1D Tensor assigning
    weight to each of the classes. This is particularly useful when you have an
    unbalanced training set.

    The `input` given through a forward call is expected to contain
    log-probabilities of each class. `input` has to be a Tensor of size either
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case. The latter is useful for
    higher dimension inputs, such as computing NLL loss per-pixel for 2D images.

    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss` instead, if you prefer not to add an extra
    layer.

    The `target` that this loss expects should be a class index in the range :math:`[0, C-1]`
    where `C = number of classes`; if `ignore_index` is specified, this loss also accepts
    this class index (this index may not necessarily be in the class range).

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_{y_n} x_{n,y_n}, \quad
        w_{c} = \text{weight}[c] \cdot \mathbb{1}\{c \not= \text{ignore_index}\},

    where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight, and
    :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{`mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{`sum'.}
        \end{cases}


    Args:
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss. `input` is expected to be log-probabilities.
        target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    """)

add_docstr("mse_loss",     r"""Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The mean operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        input (Tensor) : the input tensor.
        target (Tensor) : the target tensor.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.

    """)

add_docstr("cross_entropy",    r"""This criterion computes the cross entropy loss between input and target.

    It is useful when training a classification problem with `C` classes.
    If provided, the optional argument :attr:`weight` should be a 1D `Tensor`
    assigning weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.

    The `input` is expected to contain raw, unnormalized scores for each class.
    `input` has to be a Tensor of size :math:`(C)` for unbatched input,
    :math:`(minibatch, C)` or :math:`(minibatch, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` for the
    `K`-dimensional case. The last being useful for higher dimension inputs, such
    as computing cross entropy loss per-pixel for 2D images.

    The `target` that this criterion expects should contain either:

    - Class indices in the range :math:`[0, C)` where :math:`C` is the number of classes; if
      `ignore_index` is specified, this loss also accepts this class index (this index
      may not necessarily be in the class range). The unreduced (i.e. with :attr:`reduction`
      set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}} l_n, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

      Note that this case is equivalent to the combination of :class:`~nn.LogSoftmax` and
      :class:`~nn.NLLLoss`.

    - Probabilities for each class; useful when labels beyond a single class per minibatch item
      are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with
      :attr:`reduction` set to ``'none'``) loss for this case can be described as:

      .. math::
          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      where :math:`x` is the input, :math:`y` is the target, :math:`w` is the weight,
      :math:`C` is the number of classes, and :math:`N` spans the minibatch dimension as well as
      :math:`d_1, ..., d_k` for the `K`-dimensional case. If
      :attr:`reduction` is not ``'none'`` (default ``'mean'``), then

      .. math::
          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
               \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}

    .. note::
        The performance of this criterion is generally better when `target` contains class
        indices, as this allows for optimized computation. Consider providing `target` as
        class probabilities only when a single class label per minibatch item is too restrictive.

    Args:
        input (Tensor) : Predicted unnormalized scores (often referred to as logits);
            see Shape section below for supported shapes.
        target (Tensor) : Ground truth class indices or class probabilities;
            see Shape section below for supported shapes.
        weight (Tensor, optional): a manual rescaling weight given to each class.
            If given, has to be a Tensor of size `C`
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Note that
            :attr:`ignore_index` is only applicable when the target contains class indices.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in
            the meantime, specifying either of those two args will override
            :attr:`reduction`. Default: ``'mean'``
        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount
            of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Shape:
        - Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with
          :math:`K \geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.
          If containing class probabilities, same shape as the input and each value should be between :math:`[0, 1]`.
        - Output: If reduction is 'none', same shape as the target. Otherwise, scalar.

        where:

        .. math::
            \begin{aligned}
                C ={} & \text{number of classes} \\
                N ={} & \text{batch size} \\
            \end{aligned}
    """)

add_docstr("binary_cross_entropy_with_logits",    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    It's possible to trade off recall and precision by adding weights to positive examples.
    In the case of multi-label classification the loss can be described as:

    .. math::
        \ell_c(x, y) = L_c = \{l_{1,c},\dots,l_{N,c}\}^\top, \quad
        l_{n,c} = - w_{n,c} \left[ p_c y_{n,c} \cdot \log \sigma(x_{n,c})
        + (1 - y_{n,c}) \cdot \log (1 - \sigma(x_{n,c})) \right],

    where :math:`c` is the class number (:math:`c > 1` for multi-label binary classification,
    :math:`c = 1` for single-label binary classification),
    :math:`n` is the number of the sample in the batch and
    :math:`p_c` is the weight of the positive answer for the class :math:`c`.

    :math:`p_c > 1` increases the recall, :math:`p_c < 1` increases the precision.

    For example, if a dataset contains 100 positive and 300 negative examples of a single class,
    then `pos_weight` for the class should be equal to :math:`\frac{300}{100}=3`.
    The loss would act as if the dataset contains :math:`3\times 100=300` positive examples.å

    Args:
        input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).
        target: Tensor of the same shape as input with values between 0 and 1.
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.

    """)

add_docstr("sub", r"""
Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.

.. math::
    \text{{out}}_i = \text{{input}}_i - \text{{alpha}} \times \text{{other}}_i

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    {input}
    other (Tensor or Number): the tensor or number to subtract from :attr:`input`.

Keyword args:
    alpha (Number): the multiplier for :attr:`other`.
    {out}

Example::

    >>> a = tensor((1, 2))
    >>> b = tensor((0, 1))
    >>> sub(a, b, alpha=2)
    tensor([1, 0])
""".format(**common_args))


add_docstr("pow",
           r"""

Takes the power of each element in :attr:`input` with :attr:`exponent` and
returns a tensor with the result.

:attr:`exponent` can be either a single ``float`` number or a `Tensor`
with the same number of elements as :attr:`input`.

When :attr:`exponent` is a scalar value, the operation applied is:

.. math::
    \text{out}_i = x_i ^ \text{exponent}

When :attr:`exponent` is a tensor, the operation applied is:

.. math::
    \text{out}_i = x_i ^ {\text{exponent}_i}
""" + r"""
When :attr:`exponent` is a tensor, the shapes of :attr:`input`
and :attr:`exponent` must be :ref:`broadcastable <broadcasting-semantics>`.

Args:
    input (float or tensor): the scalar base value for the power operation
    exponent (float or tensor): the exponent value

Example::

    >>> a = randn(4)
    >>> a
    tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
    >>> pow(a, 2)
    tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
    >>> exp = arange(1., 5.)

    >>> a = arange(1., 5.)
    >>> a
    tensor([ 1.,  2.,  3.,  4.])
    >>> exp
    tensor([ 1.,  2.,  3.,  4.])
    >>> pow(a, exp)
    tensor([   1.,    4.,   27.,  256.])

    >>> exp = arange(1., 5.)
    >>> base = 2
    >>> pow(base, exp)
    tensor([  2.,   4.,   8.,  16.])
""".format(**common_args))

add_docstr("ne", r"""
Computes :math:`\text{input} \neq \text{other}` element-wise.
""" + r"""

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Returns:
    A boolean tensor that is True where :attr:`input` is not equal to :attr:`other` and False elsewhere

Example::

    >>> ne(tensor([[1, 2], [3, 4]]), tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [True, False]])
""".format(**common_args))

add_docstr("mul", r"""
Multiplies :attr:`input` by :attr:`other`.


.. math::
    \text{out}_i = \text{input}_i \times \text{other}_i
""" + r"""

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    {input}
    other (Tensor or Number): the tensor or number to multiply input by.

Examples::

    >>> a = randn(3)
    >>> a
    tensor([ 0.2015, -0.4255,  2.6087])
    >>> mul(a, 100)
    tensor([  20.1494,  -42.5491,  260.8663])

    >>> b = randn(4, 1)
    >>> b
    tensor([[ 1.1207],
            [-0.3137],
            [ 0.0700],
            [ 0.8378]])
    >>> c = randn(1, 4)
    >>> c
    tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
    >>> mul(b, c)
    tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
            [-0.1614, -0.0382,  0.1645, -0.7021],
            [ 0.0360,  0.0085, -0.0367,  0.1567],
            [ 0.4312,  0.1019, -0.4394,  1.8753]])
""".format(**common_args))

add_docstr("matmul",
           r"""
Matrix product of two tensors.

The behavior depends on the dimensionality of the tensors as follows:

- If both tensors are 1-dimensional, the dot product (scalar) is returned.
- If both arguments are 2-dimensional, the matrix-matrix product is returned.
- If the first argument is 1-dimensional and the second argument is 2-dimensional,
  a 1 is prepended to its dimension for the purpose of the matrix multiply.
  After the matrix multiply, the prepended dimension is removed.
- If the first argument is 2-dimensional and the second argument is 1-dimensional,
  the matrix-vector product is returned.
- If both arguments are at least 1-dimensional and at least one argument is
  N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
  argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
  batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
  1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
  The non-matrix (i.e. batch) dimensions are :ref:`broadcasted <broadcasting-semantics>` (and thus
  must be broadcastable).  For example, if :attr:`input` is a
  :math:`(j \times 1 \times n \times n)` tensor and :attr:`other` is a :math:`(k \times n \times n)`
  tensor, :attr:`out` will be a :math:`(j \times k \times n \times n)` tensor.

  Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs
  are broadcastable, and not the matrix dimensions. For example, if :attr:`input` is a
  :math:`(j \times 1 \times n \times m)` tensor and :attr:`other` is a :math:`(k \times m \times p)`
  tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the
  matrix dimensions) are different. :attr:`out` will be a :math:`(j \times k \times n \times p)` tensor.



.. note::

    The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.

Arguments:
    input (Tensor): the first tensor to be multiplied
    other (Tensor): the second tensor to be multiplied

Example::

    >>> # vector x vector
    >>> tensor1 = randn(3)
    >>> tensor2 = randn(3)
    >>> matmul(tensor1, tensor2).size()
    Size([])
    >>> # matrix x vector
    >>> tensor1 = randn(3, 4)
    >>> tensor2 = randn(4)
    >>> matmul(tensor1, tensor2).size()
    Size([3])
    >>> # batched matrix x broadcasted vector
    >>> tensor1 = randn(10, 3, 4)
    >>> tensor2 = randn(4)
    >>> matmul(tensor1, tensor2).size()
    Size([10, 3])
    >>> # batched matrix x batched matrix
    >>> tensor1 = randn(10, 3, 4)
    >>> tensor2 = randn(10, 4, 5)
    >>> matmul(tensor1, tensor2).size()
    Size([10, 3, 5])
    >>> # batched matrix x broadcasted matrix
    >>> tensor1 = randn(10, 3, 4)
    >>> tensor2 = randn(4, 5)
    >>> matmul(tensor1, tensor2).size()
    Size([10, 3, 5])

""".format(**common_args))

add_docstr("logical_or",
           r"""
Computes the element-wise logical OR of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated as ``True``.

Args:
    {input}
    other (Tensor): the tensor to compute OR with

Example::

    >>> logical_or(tensor([True, False, True]), tensor([True, False, False]))
    tensor([ True, False,  True])
    >>> a = tensor([0, 1, 10, 0], dtype=int8)
    >>> b = tensor([4, 0, 1, 0], dtype=int8)
    >>> logical_or(a, b)
    tensor([ True,  True,  True, False])
    >>> logical_or(a.double(), b.double())
    tensor([ True,  True,  True, False])
    >>> logical_or(a.double(), b)
    tensor([ True,  True,  True, False])
    >>> logical_or(a, b, out=empty(4, dtype=bool))
    tensor([ True,  True,  True, False])
""".format(**common_args))

add_docstr("logical_and",
           r"""
Computes the element-wise logical AND of the given input tensors. Zeros are treated as ``False`` and nonzeros are
treated as ``True``.

Args:
    {input}
    other (Tensor): the tensor to compute AND with

Example::

    >>> logical_and(tensor([True, False, True]), tensor([True, False, False]))
    tensor([ True, False, False])
    >>> a = tensor([0, 1, 10, 0], dtype=int8)
    >>> b = tensor([4, 0, 1, 0], dtype=int8)
    >>> logical_and(a, b)
    tensor([False, False,  True, False])
    >>> logical_and(a.double(), b.double())
    tensor([False, False,  True, False])
    >>> logical_and(a.double(), b)
    tensor([False, False,  True, False])
    >>> logical_and(a, b, out=empty(4, dtype=bool))
    tensor([False, False,  True, False])
""".format(**common_args))

add_docstr("lt", r"""
Computes :math:`\text{input} < \text{other}` element-wise.
""" + r"""

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Returns:
    A boolean tensor that is True where :attr:`input` is less than :attr:`other` and False elsewhere

Example::

    >>> lt(tensor([[1, 2], [3, 4]]), tensor([[1, 1], [4, 4]]))
    tensor([[False, False], [True, False]])
""".format(**common_args))


add_docstr("le", r"""
Computes :math:`\text{input} \leq \text{other}` element-wise.
""" + r"""

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or Scalar): the tensor or value to compare

Returns:
    A boolean tensor that is True where :attr:`input` is less than or equal to
    :attr:`other` and False elsewhere

Example::

    >>> le(tensor([[1, 2], [3, 4]]), tensor([[1, 1], [4, 4]]))
    tensor([[True, False], [True, True]])
""".format(**common_args))

add_docstr("gt", r"""
Computes :math:`\text{input} > \text{other}` element-wise.
""" + r"""

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Returns:
    A boolean tensor that is True where :attr:`input` is greater than :attr:`other` and False elsewhere

Example::

    >>> gt(tensor([[1, 2], [3, 4]]), tensor([[1, 1], [4, 4]]))
    tensor([[False, True], [False, False]])
""".format(**common_args))


add_docstr("erf", r"""

Computes the error function of each element. The error function is defined as follows:

.. math::
    \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
""" + r"""
Args:
    {input}

Example::

    >>> erf(tensor([0, -1., 10.]))
    tensor([ 0.0000, -0.8427,  1.0000])
""".format(**common_args))

add_docstr("exp", r"""

Returns a new tensor with the exponential of the elements
of the input tensor :attr:`input`.

.. math::
    y_{i} = e^{x_{i}}
""" + r"""
Args:
    {input}

Example::

    >>> exp(tensor([0, math.log(2.)]))
    tensor([ 1.,  2.])
""".format(**common_args))

add_docstr("floor",
           r"""

Returns a new tensor with the floor of the elements of :attr:`input`,
the largest integer less than or equal to each element.

.. math::
    \text{out}_{i} = \left\lfloor \text{input}_{i} \right\rfloor
""" + r"""
Args:
    {input}

Example::

    >>> a = randn(4)
    >>> a
    tensor([-0.8166,  1.5308, -0.2530, -0.2091])
    >>> floor(a)
    tensor([-1.,  1., -1., -1.])
""".format(**common_args))

add_docstr("log",
           r"""

Returns a new tensor with the natural logarithm of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{e} (x_{i})
""" + r"""

Args:
    {input}

Example::

    >>> a = randn(5)
    >>> a
    tensor([-0.7168, -0.5471, -0.8933, -1.4428, -0.1190])
    >>> log(a)
    tensor([ nan,  nan,  nan,  nan,  nan])
""".format(**common_args))

add_docstr("log2",
           r"""

Returns a new tensor with the logarithm to the base 2 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{2} (x_{i})
""" + r"""

Args:
    {input}

Example::

    >>> a = rand(5)
    >>> a
    tensor([ 0.8419,  0.8003,  0.9971,  0.5287,  0.0490])


    >>> log2(a)
    tensor([-0.2483, -0.3213, -0.0042, -0.9196, -4.3504])

""".format(**common_args))

add_docstr("log10",
           r"""

Returns a new tensor with the logarithm to the base 10 of the elements
of :attr:`input`.

.. math::
    y_{i} = \log_{10} (x_{i})
""" + r"""

Args:
    {input}

Example::

    >>> a = rand(5)
    >>> a
    tensor([ 0.5224,  0.9354,  0.7257,  0.1301,  0.2251])


    >>> log10(a)
    tensor([-0.2820, -0.0290, -0.1392, -0.8857, -0.6476])

""".format(**common_args))

add_docstr("neg",
           r"""

Returns a new tensor with the negative of the elements of :attr:`input`.

.. math::
    \text{out} = -1 \times \text{input}
""" + r"""
Args:
    {input}

Example::

    >>> a = randn(5)
    >>> a
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
    >>> neg(a)
    tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])
""".format(**common_args))

add_docstr("nonzero",
           r"""

.. note::
    :func:`nonzero(..., as_tuple=False) <nonzero>` (default) returns a
    2-D tensor where each row is the index for a nonzero value.

    :func:`nonzero(..., as_tuple=True) <nonzero>` returns a tuple of 1-D
    index tensors, allowing for advanced indexing, so ``x[x.nonzero(as_tuple=True)]``
    gives all nonzero values of tensor ``x``. Of the returned tuple, each index tensor
    contains nonzero indices for a certain dimension.

    See below for more details on the two behaviors.

    When :attr:`input` is on CUDA, :func:`nonzero() <nonzero>` causes
    host-device synchronization.

**When** :attr:`as_tuple` **is ``False`` (default)**:

Returns a tensor containing the indices of all non-zero elements of
:attr:`input`.  Each row in the result contains the indices of a non-zero
element in :attr:`input`. The result is sorted lexicographically, with
the last index changing the fastest (C-style).

If :attr:`input` has :math:`n` dimensions, then the resulting indices tensor
:attr:`out` is of size :math:`(z \times n)`, where :math:`z` is the total number of
non-zero elements in the :attr:`input` tensor.

**When** :attr:`as_tuple` **is ``True``**:

Returns a tuple of 1-D tensors, one for each dimension in :attr:`input`,
each containing the indices (in that dimension) of all non-zero elements of
:attr:`input` .

If :attr:`input` has :math:`n` dimensions, then the resulting tuple contains :math:`n`
tensors of size :math:`z`, where :math:`z` is the total number of
non-zero elements in the :attr:`input` tensor.

As a special case, when :attr:`input` has zero dimensions and a nonzero scalar
value, it is treated as a one-dimensional tensor with one element.

Args:
    {input}

Keyword args:
    out (LongTensor, optional): the output tensor containing indices

Returns:
    LongTensor or tuple of LongTensor: If :attr:`as_tuple` is ``False``, the output
    tensor containing indices. If :attr:`as_tuple` is ``True``, one 1-D tensor for
    each dimension, containing the indices of each nonzero element along that
    dimension.

Example::

    >>> nonzero(tensor([1, 1, 1, 0, 1]))
    tensor([[ 0],
            [ 1],
            [ 2],
            [ 4]])
    >>> nonzero(tensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]]))
    tensor([[ 0,  0],
            [ 1,  1],
            [ 2,  2],
            [ 3,  3]])
    >>> nonzero(tensor([1, 1, 1, 0, 1]), as_tuple=True)
    (tensor([0, 1, 2, 4]),)
    >>> nonzero(tensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
    (tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]))
    >>> nonzero(tensor(5), as_tuple=True)
    (tensor([0]),)
""".format(**common_args))

add_docstr("sign",
           r"""

Returns a new tensor with the signs of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \operatorname{sgn}(\text{input}_{i})
""" + r"""
Args:
    {input}

Example::

    >>> a = tensor([0.7, -1.2, 0., 2.3])
    >>> a
    tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
    >>> sign(a)
    tensor([ 1., -1.,  0.,  1.])
""".format(**common_args))

add_docstr("sin",
           r"""

Returns a new tensor with the sine of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sin(\text{input}_{i})
""" + r"""
Args:
    {input}

Example::

    >>> a = randn(4)
    >>> a
    tensor([-0.5461,  0.1347, -2.7266, -0.2746])
    >>> sin(a)
    tensor([-0.5194,  0.1343, -0.4032, -0.2711])
""".format(**common_args))

add_docstr("sqrt",
           r"""

Returns a new tensor with the square-root of the elements of :attr:`input`.

.. math::
    \text{out}_{i} = \sqrt{\text{input}_{i}}
""" + r"""
Args:
    {input}

Example::

    >>> a = randn(4)
    >>> a
    tensor([-2.0755,  1.0226,  0.0831,  0.4806])
    >>> sqrt(a)
    tensor([    nan,  1.0112,  0.2883,  0.6933])
""".format(**common_args))

add_docstr("add", r"""

Each element of the tensor :attr:`other` is multiplied by the scalar
:attr:`alpha` and added to each element of the tensor :attr:`input`.
The resulting tensor is returned.

The shapes of :attr:`input` and :attr:`other` must be
:ref:`broadcastable <broadcasting-semantics>`.

.. math::
    \text{{out}} = \text{{input}} + \text{{alpha}} \times \text{{other}}

If :attr:`other` is of type FloatTensor or DoubleTensor, :attr:`alpha` must be
a real number, otherwise it should be an integer.

Args:
    input (Tensor): the first input tensor
    other (Tensor): the second input tensor

Keyword args:
    alpha (Number): the scalar multiplier for :attr:`other`

Example::

    >>> a = randn(4)
    >>> a
    tensor([-0.9732, -0.3497,  0.6245,  0.4022])
    >>> b = randn(4, 1)
    >>> b
    tensor([[ 0.3743],
            [-1.7724],
            [-0.5811],
            [-0.8017]])
    >>> add(a, b, alpha=10)
    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
            [-18.6971, -18.0736, -17.0994, -17.3216],
            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
""".format(**common_args))

add_docstr("bmm",
           r"""

Performs a batch matrix-matrix product of matrices stored in :attr:`input`
and :attr:`mat2`.

:attr:`input` and :attr:`mat2` must be 3-D tensors each containing
the same number of matrices.

If :attr:`input` is a :math:`(b \times n \times m)` tensor, :attr:`mat2` is a
:math:`(b \times m \times p)` tensor, :attr:`out` will be a
:math:`(b \times n \times p)` tensor.

.. math::
    \text{out}_i = \text{input}_i \mathbin{@} \text{mat2}_i
""" + r"""


.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`matmul`.

Args:
    input (Tensor): the first batch of matrices to be multiplied
    mat2 (Tensor): the second batch of matrices to be multiplied

Example::

    >>> input = randn(10, 3, 4)
    >>> mat2 = randn(10, 4, 5)
    >>> res = bmm(input, mat2)
    >>> res.size()
    Size([10, 3, 5])
""")

add_docstr("div", r"""

Divides each element of the input ``input`` by the corresponding element of
:attr:`other`.

.. math::
    \text{out}_i = \frac{{\text{input}_i}}{{\text{other}_i}}

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.
Always promotes integer types to the default scalar type.

Args:
    input (Tensor): the dividend
    other (Tensor or Number): the divisor

Examples::

    >>> a = randn(5)
    >>> a
    tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
    >>> div(a, 0.5)
    tensor([ 0.7620,  2.5548, -0.5944, -0.7439,  0.9275])
    >>> a = randn(4, 4)
    >>> a
    tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
            [ 0.1815, -1.0111,  0.9805, -1.5923],
            [ 0.1062,  1.4581,  0.7759, -1.2344],
            [-0.1830, -0.0313,  1.1908, -1.4757]])
    >>> b = randn(4)
    >>> b
    tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
    >>> div(a, b)
    tensor([[-0.4620, -6.6051,  0.5676,  1.2637],
            [ 0.2260, -3.4507, -1.2086,  6.8988],
            [ 0.1322,  4.9764, -0.9564,  5.3480],
            [-0.2278, -0.1068, -1.4678,  6.3936]])
""")

add_docstr("eq", r"""

Computes element-wise equality

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Returns:
    A boolean tensor that is True where :attr:`input` is equal to :attr:`other` and False elsewhere

Example::

    >>> eq(tensor([[1, 2], [3, 4]]), tensor([[1, 1], [4, 4]]))
    tensor([[ True, False],
            [False, True]])
""")

add_docstr('fill_',
               r"""
Fills :attr:`tensor` with the specified value.

Args:
    input (Tensor): the input tensor.
    value (Number): the value to fill tensor
""")

add_docstr("ge", r"""

Computes :math:`\text{input} \geq \text{other}` element-wise.
""" + r"""

The second argument can be a number or a tensor whose shape is
:ref:`broadcastable <broadcasting-semantics>` with the first argument.

Args:
    input (Tensor): the tensor to compare
    other (Tensor or float): the tensor or value to compare

Returns:
    A boolean tensor that is True where :attr:`input` is greater than or equal to :attr:`other` and False elsewhere

Example::

    >>> ge(tensor([[1, 2], [3, 4]]), tensor([[1, 1], [4, 4]]))
    tensor([[True, True], [False, True]])
""")

add_docstr("sigmoid_focal_loss",  r"""
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """)

add_docstr("nms",    r"""
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.
    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """)


add_docstr("roi_align",     r"""
    Performs Region of Interest (RoI) Align operator with average pooling, as described in Mask R-CNN.

    Args:
        input (Tensor[N, C, H, W]): The input tensor, i.e. a batch with ``N`` elements. Each element
            contains ``C`` feature maps of dimensions ``H x W``.
            If the tensor is quantized, we expect a batch size of ``N == 1``.
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from.
            The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            If a single Tensor is passed, then the first column should
            contain the index of the corresponding element in the batch, i.e. a number in ``[0, N - 1]``.
            If a list of Tensors is passed, then each Tensor will correspond to the boxes for an element i
            in the batch.
        output_size (int or Tuple[int, int]): the size of the output (in bins or pixels) after the pooling
            is performed, as (height, width).
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly ``sampling_ratio x sampling_ratio`` sampling points per bin are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ``ceil(roi_width / output_width)``, and likewise for height). Default: -1
        aligned (bool): If False, use the legacy implementation.
            If True, pixel shift the box coordinates it by -0.5 for a better alignment with the two
            neighboring pixel indices. This version is used in Detectron2

    Returns:
        Tensor[K, C, output_size[0], output_size[1]]: The pooled RoIs.
    """)


add_docstr("sgd", r"""
    Implements stochastic gradient descent (optionally with momentum).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},\:nesterov\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: nesterov                                                \\
            &\hspace{15mm} g_t \leftarrow g_{t-1} + \mu \textbf{b}_t                             \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                    \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """)


add_docstr("arange", r"""
    Returns a 1-D tensor of size :math:`\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil`
    with values from the interval ``[start, end)`` taken with common difference
    :attr:`step` beginning from `start`.

    Note that non-integer :attr:`step` is subject to floating point rounding errors when
    comparing against :attr:`end`; to avoid inconsistency, we advise adding a small epsilon to :attr:`end`
    in such cases.

    .. math::
        \text{out}_{{i+1}} = \text{out}_{i} + \text{step}
    """ + r"""
    Args:
        start (Number): the starting value for the set of points. Default: ``0``.
        end (Number): the ending value for the set of points
        step (Number): the gap between each pair of adjacent points. Default: ``1``.

    Example::

        >>> arange(5)
        tensor([ 0,  1,  2,  3,  4])
        >>> arange(1, 4)
        tensor([ 1,  2,  3])
        >>> arange(1, 2.5, 0.5)
        tensor([ 1.0000,  1.5000,  2.0000])
    """)


add_docstr("randperm", r"""
    Returns a random permutation of integers from ``0`` to ``n - 1``.

    Args:
        n (int): the upper bound (exclusive)

    Example::

        >>> randperm(4)
        tensor([2, 1, 0, 3])
    """)


add_docstr("uniform", r"""
    Fills :attr:`self` tensor with numbers sampled from the continuous uniform
    distribution:

    .. math::
        P(x) = \dfrac{1}{\text{end} - \text{start}}
    """)


add_docstr("random", r"""
    Fills :attr:`self` tensor with numbers sampled from the discrete uniform
    distribution over ``[from, to - 1]``. If not specified, the values are usually
    only bounded by :attr:`self` tensor's data type. However, for floating point
    types, if unspecified, range will be ``[0, 2^mantissa]`` to ensure that every
    value is representable. For example, `tensor(1, dtype=double).random_()`
    will be uniform in ``[0, 2^53]``.
    """)


add_docstr("bernoulli",     r"""
    Draws binary random numbers (0 or 1) from a Bernoulli distribution.

    The :attr:`input` tensor should be a tensor containing probabilities
    to be used for drawing the binary random number.
    Hence, all values in :attr:`input` have to be in the range:
    :math:`0 \leq \text{input}_i \leq 1`.

    The :math:`\text{i}^{th}` element of the output tensor will draw a
    value :math:`1` according to the :math:`\text{i}^{th}` probability value given
    in :attr:`input`.

    .. math::
        \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})
    """ + r"""
    The returned :attr:`out` tensor only has values 0 or 1 and is of the same
    shape as :attr:`input`.

    :attr:`out` can have integral ``dtype``, but :attr:`input` must have floating
    point ``dtype``.

    Args:
        input (Tensor): the input tensor of probability values for the Bernoulli distribution
        inplace (bool): can optionally do the operation in-place. Default: ``False``
        p (scalar): Probability scalar. When p is not empty, mask the input tensor and calculate the output according to the input scalar p; otherwise, calculate the output according to the input tensor input. The default is ` ` None`

    Example::

        >>> a = empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
        >>> a
        tensor([[ 0.1737,  0.0950,  0.3609],
                [ 0.7148,  0.0289,  0.2676],
                [ 0.9456,  0.8937,  0.7202]])
        >>> bernoulli(a)
        tensor([[ 1.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 1.,  1.,  1.]])

        >>> a = ones(3, 3) # probability of drawing "1" is 1
        >>> bernoulli(a)
        tensor([[ 1.,  1.,  1.],
                [ 1.,  1.,  1.],
                [ 1.,  1.,  1.]])
        >>> a = zeros(3, 3) # probability of drawing "1" is 0
        >>> bernoulli(a)
        tensor([[ 0.,  0.,  0.],
                [ 0.,  0.,  0.],
                [ 0.,  0.,  0.]])
    """)

 
add_docstr("masked_fill",   r"""
    Fills elements of :attr:`self` tensor with :attr:`value` where :attr:`mask` is
    True. The shape of :attr:`mask` must be
    :ref:`broadcastable <broadcasting-semantics>` with the shape of the underlying
    tensor.

    Args:
        mask (BoolTensor): the boolean mask
        value (float): the value to fill in with
    """)


add_docstr("adamw", r"""
    Implements AdamW algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)}, \:  amsgrad                       \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        param_grad (iterable) : param gradient
        exp_avg (Tensor) : the first momentum is related to the number of iterations, that is, the gradient mean value of the i th iteration
        exp_avg_sq (Tensor) : the second momentum is related to the number of iterations, that is, the mean value of the gradient square of the i iteration
        max_exp_avg_sq (Tensor) : the maximum second momentum. When the parameter 'amsgrad' is true, it will replace the second momentum to participate in the calculation
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """)


add_docstr("adam",  r"""
    Implements Adam algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: amsgrad                      \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: \widehat{v_0}^{max}\leftarrow 0\\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        param_grad (iterable) : param gradient
        exp_avg (Tensor) : the first momentum is related to the number of iterations, that is, the gradient mean value of the i th iteration
        exp_avg_sq (Tensor) : the second momentum is related to the number of iterations, that is, the mean value of the gradient square of the i iteration
        max_exp_avg_sq (Tensor) : the maximum second momentum. When the parameter 'amsgrad' is true, it will replace the second momentum to participate in the calculation
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """)


add_docstr("adadelta",  r"""
    Implements Adadelta algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `ADADELTA: An Adaptive Learning Rate Method`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6)
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _ADADELTA\: An Adaptive Learning Rate Method:
        https://arxiv.org/abs/1212.5701
    """)


add_docstr("conv_transpose2d",  r"""
    Applies a 2D transposed convolution operator over an input image
    composed of several input planes, sometimes also called "deconvolution".

    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in_channels} , iH , iW)`
        weight: filters of shape :math:`(\text{in_channels} , \frac{\text{out_channels}}{\text{groups}} , kH , kW)`
        bias: optional bias of shape :math:`(\text{out_channels})`. Default: None
        stride: the stride of the convolving kernel. Can be a single number or a
        tuple ``(sH, sW)``. Default: 1
        padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both
        sides of each dimension in the input. Can be a single number or a tuple
        ``(padH, padW)``. Default: 0
        output_padding: additional size added to one side of each dimension in the
        output shape. Can be a single number or a tuple ``(out_padH, out_padW)``.
        Default: 0
        groups: split input into groups, :math:`\text{in_channels}` should be divisible by the
        number of groups. Default: 1
        dilation: the spacing between kernel elements. Can be a single number or
        a tuple ``(dH, dW)``. Default: 1

    Examples::

        >>> # With square kernels and equal stride
        >>> inputs = randn(1, 4, 5, 5)
        >>> weights = randn(4, 8, 3, 3)
        >>> F.conv_transpose2d(inputs, weights, padding=1)
    """)
 

add_docstr("cumsum",    r"""
    Returns the cumulative sum of elements of :attr:`input` in the dimension
    :attr:`dim`.

    For example, if :attr:`input` is a vector of size N, the result will also be
    a vector of size N, with elements.

    .. math::
        y_i = x_1 + x_2 + x_3 + \dots + x_i

    Args:
        input (Tensor): the input tensor
        dim  (int): the dimension to do the operation over

    Example::

        >>> a = randn(10)
        >>> a
        tensor([-0.8286, -0.4890,  0.5155,  0.8443,  0.1865, -0.1752, -2.0595,
                0.1850, -1.1571, -0.4243])
        >>> cumsum(a, dim=0)
        tensor([-0.8286, -1.3175, -0.8020,  0.0423,  0.2289,  0.0537, -2.0058,
                -1.8209, -2.9780, -3.4022])
    """)


add_docstr("cdist",r"""
    Computes batched the p-norm distance between each pair of the two collections of row vectors.

    Args:
        x1 (Tensor): input tensor of shape :math:`B \times P \times M`.
        x2 (Tensor): input tensor of shape :math:`B \times R \times M`.
        p: p value for the p-norm distance to calculate between each vector pair
            :math:`\in [0, \infty]`.
        compute_mode:
            'use_mm_for_euclid_dist_if_necessary' - will use matrix multiplication approach to calculate
            euclidean distance (p = 2) if P > 25 or R > 25
            'use_mm_for_euclid_dist' - will always use matrix multiplication approach to calculate
            euclidean distance (p = 2)
            'donot_use_mm_for_euclid_dist' - will never use matrix multiplication approach to calculate
            euclidean distance (p = 2)
            Default: use_mm_for_euclid_dist_if_necessary.

    If x1 has shape :math:`B \times P \times M` and x2 has shape :math:`B \times R \times M` then the
    output will have shape :math:`B \times P \times R`.

    This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)`
    if :math:`p \in (0, \infty)`. When :math:`p = 0` it is equivalent to
    `scipy.spatial.distance.cdist(input, 'hamming') * M`. When :math:`p = \infty`, the closest
    scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.

    Example:

        >>> a = tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
        >>> a
        tensor([[ 0.9041,  0.0196],
                [-0.3108, -2.4423],
                [-0.4821,  1.0590]])
        >>> b = tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
        >>> b
        tensor([[-2.1763, -0.4713],
                [-0.6986,  1.3702]])
        >>> cdist(a, b, p=2)
        tensor([[3.1193, 2.0959],
                [2.7138, 3.8322],
                [2.2830, 0.3791]])
    """)


add_docstr("reciprocal",    r"""
    Returns a new tensor with the reciprocal of the elements of :attr:`input`

    .. math::
        \text{out}_{i} = \frac{1}{\text{input}_{i}}

    .. note::
        Unlike NumPy's reciprocal, reciprocal supports integral inputs. Integral
        inputs to reciprocal are automatically :ref:`promoted <type-promotion-doc>` to
        the default scalar type.
    """ + r"""
    Args:
        input (Tensor): the input tensor

    Example::

        >>> a = randn(4)
        >>> a
        tensor([-0.4595, -2.1219, -1.4314,  0.7298])
        >>> reciprocal(a)
        tensor([-2.1763, -0.4713, -0.6986,  1.3702])
    """)


add_docstr("bitwise_not",   r"""
    Computes the bitwise NOT of the given input tensor. The input tensor must be of
    integral or Boolean types. For bool tensors, it computes the logical NOT.

    Args:
        input (Tensor): the input tensor

    Example:

        >>> bitwise_not(tensor([-1, -2, 3], dtype=int8))
        tensor([ 0,  1, -4], dtype=int8)
    """)


add_docstr("argmax",    r"""
    Returns the indices of the maximum values of a tensor across a dimension.

    This is the second value returned by :meth:`max`. See its
    documentation for the exact semantics of this method.

    Args:
        input (Tensor): the input tensor
        dim (int): the dimension to reduce. If ``None``, the argmax of the flattened input is returned.
        keepdim (bool):  whether the output tensor has ``dim`` retained or not.Ignored if ``dim=None``.

    Example::

        >>> a = randn(4, 4)
        >>> a
        tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
                [-0.7401, -0.8805, -0.3402, -1.1936],
                [ 0.4907, -1.3948, -1.0691, -0.3132],
                [-1.6092,  0.5419, -0.2993,  0.3195]])
        >>> argmax(a, dim=1)
        tensor([ 0,  2,  0,  1])
    """)


add_docstr("smooth_l1_loss", r"""
    Creates a criterion that uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.
    It is less sensitive to outliers than :class:`nn.MSELoss` and in some cases
    prevents exploding gradients (e.g. see the paper `Fast R-CNN`_ by Ross Girshick).

    For a batch of size :math:`N`, the unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1, ..., l_N\}^T

    with

    .. math::
        l_n = \begin{cases}
        0.5 (x_n - y_n)^2 / beta, & \text{if } |x_n - y_n| < beta \\
        |x_n - y_n| - 0.5 * beta, & \text{otherwise }
        \end{cases}

    If `reduction` is not `none`, then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    .. note::
        Smooth L1 loss can be seen as exactly :class:`L1Loss`, but with the :math:`|x - y| < beta`
        portion replaced with a quadratic function such that its slope is 1 at :math:`|x - y| = beta`.
        The quadratic segment smooths the L1 loss near :math:`|x - y| = 0`.

    .. note::
        Smooth L1 loss is closely related to :class:`HuberLoss`, being
        equivalent to :math:`huber(x, y) / beta` (note that Smooth L1's beta hyper-parameter is
        also known as delta for Huber). This leads to the following differences:

        * As beta -> 0, Smooth L1 loss converges to :class:`L1Loss`, while :class:`HuberLoss`
          converges to a constant 0 loss.
        * As beta -> :math:`+\infty`, Smooth L1 loss converges to a constant 0 loss, while
          :class:`HuberLoss` converges to :class:`MSELoss`.
        * For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant slope of 1.
          For :class:`HuberLoss`, the slope of the L1 segment is beta.

    .. _`Fast R-CNN`: https://arxiv.org/abs/1504.08083

    Args:
        input (Tensor) : the input tensor.
        target (Tensor) : the target tensor.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        beta (float, optional): Specifies the threshold at which to change between L1 and L2 loss.
            The value must be non-negative. Default: 1.0

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same shape as the input.
    """)


add_docstr("maximum",   r"""
    Computes the element-wise maximum of :attr:`input` and :attr:`other`.

    .. note::
        If one of the elements being compared is a NaN, then that element is returned.
        :func:`maximum` is not supported for tensors with complex dtypes.

    Args:
        input (Tensor) : the input tensor.
        other (Tensor): the second input tensor

    Example::

        >>> a = tensor((1, 2, -1))
        >>> b = tensor((3, 0, 4))
        >>> maximum(a, b)
        tensor([3, 2, 4])
    """)


add_docstr("minimum", r"""
    Computes the element-wise minimum of :attr:`input` and :attr:`other`.

    .. note::
        If one of the elements being compared is a NaN, then that element is returned.
        :func:`minimum` is not supported for tensors with complex dtypes.

    Args:
        input (Tensor) : the input tensor.
        other (Tensor): the second input tensor

    Example::

        >>> a = tensor((1, 2, -1))
        >>> b = tensor((3, 0, 4))
        >>> minimum(a, b)
        tensor([1, 0, -1])
    """)


add_docstr("mm",    r"""
    Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.

    If :attr:`input` is a :math:`(n \times m)` tensor, :attr:`mat2` is a
    :math:`(m \times p)` tensor, :attr:`out` will be a :math:`(n \times p)` tensor.

    .. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
            For broadcasting matrix products, see :func:`matmul`.

    Supports strided and sparse 2-D tensors as inputs, autograd with
    respect to strided inputs.

    {tf32_note}

    Args:
        input (Tensor): the first matrix to be matrix multiplied
        mat2 (Tensor): the second matrix to be matrix multiplied

    Example::

        >>> mat1 = randn(2, 3)
        >>> mat2 = randn(3, 3)
        >>> mm(mat1, mat2)
        tensor([[ 0.4851,  0.5037, -0.3633],
                [-0.0760, -3.6705,  2.4784]])
    """)


add_docstr("conv3d",    r"""
    Applies a 3D convolution over an input image composed of several input
    planes.

    {tf32_note}

    Note:
        {cudnn_reproducibility_note}
    """ + r"""

    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in_channels} , iT , iH , iW)`
        weight: filters of shape :math:`(\text{out_channels} , \frac{\text{in_channels}}{\text{groups}} , kT , kH , kW)`
        bias: optional bias tensor of shape :math:`(\text{out_channels})`. Default: None
        stride: the stride of the convolving kernel. Can be a single number or a
        tuple `(sT, sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
        single number or a tuple `(padT, padH, padW)`. Default: 0
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

        .. warning::
            For ``padding='same'``, if the ``weight`` is even-length and
            ``dilation`` is odd in any dimension, a full :func:`pad` operation
            may be needed internally. Lowering performance.

        dilation: the spacing between kernel elements. Can be a single number or
        a tuple `(dT, dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in_channels}` should be divisible by
        the number of groups. Default: 1

    Examples::

        >>> filters = randn(33, 16, 3, 3, 3)
        >>> inputs = randn(20, 16, 50, 10, 20)
        >>> F.conv3d(inputs, filters)
    """)


add_docstr("expand",    r"""
    Returns a new view of the :attr:`self` tensor with singleton dimensions expanded
    to a larger size.

    Passing -1 as the size for a dimension means not changing the size of
    that dimension.

    Tensor can be also expanded to a larger number of dimensions, and the
    new ones will be appended at the front. For the new dimensions, the
    size cannot be set to -1.

    Expanding a tensor does not allocate new memory, but only creates a
    new view on the existing tensor where a dimension of size one is
    expanded to a larger size by setting the ``stride`` to 0. Any dimension
    of size 1 can be expanded to an arbitrary value without allocating new
    memory.

    Args:
        *sizes (Size or int...): the desired expanded size

    .. warning::

        More than one element of an expanded tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensors, please clone them first.

    Example::

        >>> x = tensor([[1], [2], [3]])
        >>> x.size()
        Size([3, 1])
        >>> x.expand(3, 4)
        tensor([[ 1,  1,  1,  1],
                [ 2,  2,  2,  2],
                [ 3,  3,  3,  3]])
        >>> x.expand(-1, 4)   # -1 means not changing the size of that dimension
        tensor([[ 1,  1,  1,  1],
                [ 2,  2,  2,  2],
                [ 3,  3,  3,  3]])
    """)


add_docstr("unfold",    r"""
    Returns a view of the original tensor which contains all slices of size :attr:`size` from
    :attr:`self` tensor in the dimension :attr:`dimension`.

    Step between two slices is given by :attr:`step`.

    If `sizedim` is the size of dimension :attr:`dimension` for :attr:`self`, the size of
    dimension :attr:`dimension` in the returned tensor will be
    `(sizedim - size) / step + 1`.

    An additional dimension of size :attr:`size` is appended in the returned tensor.

    Args:
        dimension (int): dimension in which unfolding happens
        size (int): the size of each slice that is unfolded
        step (int): the step between each slice

    Example::

        >>> x = arange(1., 8)
        >>> x
        tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
        >>> x.unfold(0, 2, 1)
        tensor([[ 1.,  2.],
                [ 2.,  3.],
                [ 3.,  4.],
                [ 4.,  5.],
                [ 5.,  6.],
                [ 6.,  7.]])
        >>> x.unfold(0, 2, 2)
        tensor([[ 1.,  2.],
                [ 3.,  4.],
                [ 5.,  6.]])
    """)


add_docstr("masked_select",     r"""
    Returns a new 1-D tensor which indexes the :attr:`input` tensor according to
    the boolean mask :attr:`mask` which is a `BoolTensor`.

    The shapes of the :attr:`mask` tensor and the :attr:`input` tensor don't need
    to match, but they must be :ref:`broadcastable <broadcasting-semantics>`.

    .. note:: The returned tensor does **not** use the same storage
            as the original tensor

    Args:
        input (Tensor) : the input tensor.
        mask  (BoolTensor): the tensor containing the binary mask to index with

    Example::

        >>> x = randn(3, 4)
        >>> x
        tensor([[ 0.3552, -2.3825, -0.8297,  0.3477],
                [-1.2035,  1.2252,  0.5002,  0.6248],
                [ 0.1307, -2.0608,  0.1244,  2.0139]])
        >>> mask = x.ge(0.5)
        >>> mask
        tensor([[False, False, False, False],
                [False, True, True, True],
                [False, False, False, True]])
        >>> masked_select(x, mask)
        tensor([ 1.2252,  0.5002,  0.6248,  2.0139])
    """)


add_docstr('index_fill',    r"""
    Fills the elements of the :attr:`self` tensor with value :attr:`value` by
    selecting the indices in the order given in :attr:`index`.

    Args:
        dim (int): dimension along which to index
        index (LongTensor): indices of :attr:`self` tensor to fill in
        value (float): the value to fill with

    Example::
        >>> x = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        >>> index = tensor([0, 2])
        >>> x.index_fill_(1, index, -1)
        tensor([[-1.,  2., -1.],
                [-1.,  5., -1.],
                [-1.,  8., -1.]])
    """)


add_docstr("linspace",      r"""
    Creates a one-dimensional tensor of size :attr:`steps` whose values are evenly
    spaced from :attr:`start` to :attr:`end`, inclusive. That is, the value are:

    .. math::
        (\text{start},
        \text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1},
        \ldots,
        \text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1},
        \text{end})
    """ + r"""

    .. warning::
        Not providing a value for :attr:`steps` is deprecated. For backwards
        compatibility, not providing a value for :attr:`steps` will create a tensor
        with 100 elements. Note that this behavior is not reflected in the
        documented function signature and should not be relied on. In a future
        PyTorch release, failing to provide a value for :attr:`steps` will throw a
        runtime error.

    Args:
        start (float): the starting value for the set of points
        end (float): the ending value for the set of points
        steps (int): size of the constructed tensor

    Example::

        >>> linspace(3, 10, steps=5)
        tensor([  3.0000,   4.7500,   6.5000,   8.2500,  10.0000])
        >>> linspace(-10, 10, steps=5)
        tensor([-10.,  -5.,   0.,   5.,  10.])
        >>> linspace(start=-10, end=10, steps=5)
        tensor([-10.,  -5.,   0.,   5.,  10.])
        >>> linspace(start=-10, end=10, steps=1)
        tensor([-10.])
    """)


add_docstr("roll",  r"""
    Roll the tensor along the given dimension(s). Elements that are shifted beyond the
    last position are re-introduced at the first position. If a dimension is not
    specified, the tensor will be flattened before rolling and then restored
    to the original shape.

    Args:
        input (Tensor) : the input tensor.
        shifts (int or tuple of ints): The number of places by which the elements
            of the tensor are shifted. If shifts is a tuple, dims must be a tuple of
            the same size, and each dimension will be rolled by the corresponding
            value
        dims (int or tuple of ints): Axis along which to roll

    Example::

        >>> x = tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
        >>> x
        tensor([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8]])
        >>> roll(x, 1, 0)
        tensor([[7, 8],
                [1, 2],
                [3, 4],
                [5, 6]])
        >>> roll(x, -1, 0)
        tensor([[3, 4],
                [5, 6],
                [7, 8],
                [1, 2]])
        >>> roll(x, shifts=(2, 1), dims=(0, 1))
        tensor([[6, 5],
                [8, 7],
                [2, 1],
                [4, 3]])
    """)


add_docstr("norm",  r"""
    Returns the matrix norm or vector norm of a given tensor.

    .. warning::

        norm is deprecated and may be removed in a future PyTorch release.
        Its documentation and behavior may be incorrect, and it is no longer
        actively maintained.

        Use :func:`linalg.norm`, instead, or :func:`linalg.vector_norm`
        when computing vector norms and :func:`linalg.matrix_norm` when
        computing matrix norms. Note, however, the signature for these functions
        is slightly different than the signature for norm.

    Args:
        input (Tensor): The input tensor. Its data type must be either a floating
            point or complex type. For complex inputs, the norm is calculated using the
            absolute value of each element. If the input is complex and neither
            :attr:`dtype` nor :attr:`out` is specified, the result's data type will
            be the corresponding floating point type (e.g. float if :attr:`input` is
            complexfloat).

        p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: ``'fro'``
            The following norms can be calculated:

            ======  ==============  ==========================
            ord     matrix norm     vector norm
            ======  ==============  ==========================
            'fro'   Frobenius norm  --
            'nuc'   nuclear norm    --
            Number  --              sum(abs(x)**ord)**(1./ord)
            ======  ==============  ==========================

            The vector norm can be calculated across any number of dimensions.
            The corresponding dimensions of :attr:`input` are flattened into
            one dimension, and the norm is calculated on the flattened
            dimension.

            Frobenius norm produces the same result as ``p=2`` in all cases
            except when :attr:`dim` is a list of three or more dims, in which
            case Frobenius norm throws an error.

            Nuclear norm can only be calculated across exactly two dimensions.

        dim (int, tuple of ints, list of ints, optional):
            Specifies which dimension or dimensions of :attr:`input` to
            calculate the norm across. If :attr:`dim` is ``None``, the norm will
            be calculated across all dimensions of :attr:`input`. If the norm
            type indicated by :attr:`p` does not support the specified number of
            dimensions, an error will occur.
        keepdim (bool, optional): whether the output tensors have :attr:`dim`
            retained or not. Ignored if :attr:`dim` = ``None`` and
            :attr:`out` = ``None``. Default: ``False``
        dtype (:class:`dtype`, optional): the desired data type of
            returned tensor. If specified, the input tensor is casted to
            :attr:`dtype` while performing the operation. Default: None.

    .. note::
        Even though ``p='fro'`` supports any number of dimensions, the true
        mathematical definition of Frobenius norm only applies to tensors with
        exactly two dimensions. :func:`linalg.norm` with ``ord='fro'`` aligns
        with the mathematical definition, since it can only be applied across
        exactly two dimensions.

    Example::

        >>> import torch
        >>> a = arange(9, dtype= float) - 4
        >>> b = a.reshape((3, 3))
        >>> norm(a)
        tensor(7.7460)
        >>> norm(b)
        tensor(7.7460)
        >>> norm(a, float('inf'))
        tensor(4.)
        >>> norm(b, float('inf'))
        tensor(4.)
        >>> c = tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= float)
        >>> norm(c, dim=0)
        tensor([1.4142, 2.2361, 5.0000])
        >>> norm(c, dim=1)
        tensor([3.7417, 4.2426])
        >>> norm(c, p=1, dim=1)
        tensor([6., 6.])
        >>> d = arange(8, dtype= float).reshape(2,2,2)
        >>> norm(d, dim=(1,2))
        tensor([ 3.7417, 11.2250])
        >>> norm(d[0, :, :]), norm(d[1, :, :])
        (tensor(3.7417), tensor(11.2250))
    """)


add_docstr("group_norm",    r"""
    Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        weight (Tensor) :  the weight required for calculate standardization, and this parameter can be learned
        bias (Tensor) :  the bias required for calculate standardization. This parameter can be learned
        eps: a value added to the denominator for numerical stability. Default: 1e-5

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """)


add_docstr("layer_norm", r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
                    \times \ldots \times \text{normalized_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)

    """)


add_docstr("adaptive_avg_pool3d", r"""
    Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    The output is of size D x H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the form D x H x W.
                     Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
                     D, H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1}, S_{2})` or :math:`(C, S_{0}, S_{1}, S_{2})`,
          where :math:`S=\text{output_size}`.

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveAvgPool3d((5,7,9))
        >>> input = randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveAvgPool3d(7)
        >>> input = randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveAvgPool3d((7, None, None))
        >>> input = randn(1, 64, 10, 9, 8)
        >>> output = m(input)

    """)


add_docstr("adaptive_max_pool3d",   r"""
    Applies a 3D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`D_{out} \times H_{out} \times W_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form :math:`D_{out} \times H_{out} \times W_{out}`.
                     Can be a tuple :math:`(D_{out}, H_{out}, W_{out})` or a single
                     :math:`D_{out}` for a cube :math:`D_{out} \times D_{out} \times D_{out}`.
                     :math:`D_{out}`, :math:`H_{out}` and :math:`W_{out}` can be either a
                     ``int``, or ``None`` which means the size will be the same as that of the input.

        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to nn.MaxUnpool3d. Default: ``False``

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where :math:`(D_{out}, H_{out}, W_{out})=\text{output_size}`.

    Examples:
        >>> # target output size of 5x7x9
        >>> m = nn.AdaptiveMaxPool3d((5,7,9))
        >>> input = randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> # target output size of 7x7x7 (cube)
        >>> m = nn.AdaptiveMaxPool3d(7)
        >>> input = randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> # target output size of 7x9x8
        >>> m = nn.AdaptiveMaxPool3d((7, None, None))
        >>> input = randn(1, 64, 10, 9, 8)
        >>> output = m(input)

    """)


add_docstr("max_pool3d",    r"""
    Applies a 3D max pooling over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on all three sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`nn.MaxUnpool3d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Examples::

        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = randn(20, 16, 50,44, 31)
        >>> output = m(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """)


add_docstr("permute",   r"""
    Returns a view of the original tensor :attr:`input` with its dimensions permuted.

    Args:
        input (Tensor) : the input tensor.
        dims (tuple of ints): The desired ordering of dimensions

    Example:
        >>> x = randn(2, 3, 5)
        >>> x.size()
        Size([2, 3, 5])
        >>> permute(x, (2, 0, 1)).size()
        Size([5, 2, 3])
    """)


add_docstr('copy_',    r"""
    Copies the elements from :attr:`other` into :attr:`input` tensor and returns
    :attr:`self`.

    The :attr:`other` tensor must be :ref:`broadcastable <broadcasting-semantics>`
    with the :attr:`self` tensor. It may be of a different data type or reside on a
    different device.

    Args:
        input (Tensor) : the target tensor to copy to.
        other (Tensor): the source tensor to copy from
    """)


add_docstr("gather",    r"""
    Gathers values along an axis specified by `dim`.

    For a 3-D tensor the output is specified by::

        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :attr:`input` and :attr:`index` must have the same number of dimensions.
    It is also required that ``index.size(d) <= input.size(d)`` for all
    dimensions ``d != dim``.  :attr:`out` will have the same shape as :attr:`index`.
    Note that ``input`` and ``index`` do not broadcast against each other.

    Args:
        input (Tensor): the source tensor
        dim (int): the axis along which to index
        index (LongTensor): the indices of elements to gather

    Keyword arguments:
        sparse_grad (bool, optional): If ``True``, gradient w.r.t. :attr:`input` will be a sparse tensor.
        out (Tensor, optional): the destination tensor

    Example::

        >>> t = tensor([[1, 2], [3, 4]])
        >>> gather(t, 1, tensor([[0, 0], [1, 0]]))
        tensor([[ 1,  1],
                [ 4,  3]])
    """)


add_docstr("remainder",   r"""
    Like :func:`fmod` this applies C++'s `std::fmod <https://en.cppreference.com/w/cpp/numeric/math/fmod>`_
    for floating point tensors and the modulus operation for integer tensors.
    Unlike :func:`fmod`, however, if the sign of the modulus is different
    than the sign of the divisor :attr:`other` then the divisor is added to the modulus.

    Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
    :ref:`type promotion <type-promotion-doc>`, and integer and float inputs.

    .. note::
        Complex inputs are not supported. In some cases, it is not mathematically
        possible to satisfy the definition of a modulo operation with complex numbers.
        See :func:`fmod` for how division by zero is handled.

    .. note::
        This op, like NumPy's `remainder <https://numpy.org/doc/stable/reference/generated/numpy.remainder.html>`_,
        is equivalent to Python's modulus operation, and different from Python's
        `math.remainder <https://docs.python.org/dev/library/math.html#math.remainder>`_ and
        C++'s `std::remainder <https://en.cppreference.com/w/cpp/numeric/math/remainder>`_ which implement
        the IEEE remainder.

    Args:
        self (Scalar): the dividend
        input (Tensor): the dividend
        other (Tensor or Scalar): the divisor

    Example::

        >>> remainder(tensor([-3., -2, -1, 1, 2, 3]), 2)
        tensor([ 1.,  0.,  1.,  1.,  0.,  1.])
        >>> remainder(tensor([1, 2, 3, 4, 5]), -1.5)
        tensor([ -0.5000, -1.0000,  0.0000, -0.5000, -1.0000 ])

    .. seealso::

        :func:`fmod` which just computes the modulus for integer inputs and
        applies C++'s `std::fmod <https://en.cppreference.com/w/cpp/numeric/math/fmod>`_
        for floating point inputs.
    """)


add_docstr("ctc_loss",  r"""
    The Connectionist Temporal Classification loss.

    Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
    probability of possible alignments of input to target, producing a loss value which is differentiable
    with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
    limits the length of the target sequence such that it must be :math:`\leq` the input length.

    Args:
        blank (int, optional): blank label. Default :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: ``'mean'``
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.

    Shape:
        - Log_probs: Tensor of size :math:`(T, N, C)` or  :math:`(T, C)` ,
          where :math:`T = \text{input length}`,
          :math:`N = \text{batch size}`, and
          :math:`C = \text{number of classes (including blank)}`.
          The logarithmized probabilities of the outputs (e.g. obtained with
          :func:`nn.functional.log_softmax`).
        - Targets: Tensor of size :math:`(N, S)` or
          :math:`(\operatorname{sum}(\text{target_lengths}))`,
          where :math:`N = \text{batch size}` and
          :math:`S = \text{max target length, if shape is } (N, S)`.
          It represent the target sequences. Each element in the target
          sequence is a class index. And the target index cannot be blank (default=0).
          In the :math:`(N, S)` form, targets are padded to the
          length of the longest sequence, and stacked.
          In the :math:`(\operatorname{sum}(\text{target_lengths}))` form,
          the targets are assumed to be un-padded and
          concatenated within 1 dimension.
        - Input_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \text{batch size}`. It represent the lengths of the
          inputs (must each be :math:`\leq T`). And the lengths are specified
          for each sequence to achieve masking under the assumption that sequences
          are padded to equal lengths.
        - Target_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \text{batch size}`. It represent lengths of the targets.
          Lengths are specified for each sequence to achieve masking under the
          assumption that sequences are padded to equal lengths. If target shape is
          :math:`(N,S)`, target_lengths are effectively the stop index
          :math:`s_n` for each target sequence, such that ``target_n = targets[n,0:s_n]`` for
          each target in a batch. Lengths must each be :math:`\leq S`
          If the targets are given as a 1d tensor that is the concatenation of individual
          targets, the target_lengths must add up to the total length of the tensor.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(N)`, where :math:`N = \text{batch size}`.

    Examples::

        >>> # Target are to be padded
        >>> T = 50      # Input sequence length
        >>> C = 20      # Number of classes (including blank)
        >>> N = 16      # Batch size
        >>> S = 30      # Target sequence length of longest target in batch (padding length)
        >>> S_min = 10  # Minimum target length, for demonstration purposes
        >>>
        >>> # Initialize random batch of input vectors, for *size = (T,N,C)
        >>> input = randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>>
        >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
        >>> target = randint(low=1, high=C, size=(N, S), dtype=long)
        >>>
        >>> input_lengths = full(size=(N,), fill_value=T, dtype=long)
        >>> target_lengths = randint(low=S_min, high=S, size=(N,), dtype=long)
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()
        >>>
        >>>
        >>> # Target are to be un-padded
        >>> T = 50      # Input sequence length
        >>> C = 20      # Number of classes (including blank)
        >>> N = 16      # Batch size
        >>>
        >>> # Initialize random batch of input vectors, for *size = (T,N,C)
        >>> input = randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>> input_lengths = full(size=(N,), fill_value=T, dtype=long)
        >>>
        >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
        >>> target_lengths = randint(low=1, high=T, size=(N,), dtype=long)
        >>> target = randint(low=1, high=C, size=(sum(target_lengths),), dtype=long)
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()

    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf

    Note:
        In order to use CuDNN, the following must be satisfied: :attr:`targets` must be
        in concatenated format, all :attr:`input_lengths` must be `T`.  :math:`blank=0`,
        :attr:`target_lengths` :math:`\leq 256`, the integer arguments must be of
        dtype :attr:`int32`.

        The regular implementation uses the (more common in PyTorch) `long` dtype.


    Note:
        In some circumstances when using the CUDA backend with CuDNN, this operator
        may select a nondeterministic algorithm to increase performance. If this is
        undesirable, you can try to make the operation deterministic (potentially at
        a performance cost) by setting ``backends.cudnn.deterministic =
        True``.
    """)


add_docstr('index_put',    r"""
    Puts values from the tensor :attr:`values` into the tensor :attr:`self` using
    the indices specified in :attr:`indices` (which is a tuple of Tensors). The
    expression ``tensor.index_put_(indices, values)`` is equivalent to
    ``tensor[indices] = values``. Returns :attr:`self`.

    If :attr:`accumulate` is ``True``, the elements in :attr:`values` are added to
    :attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
    contain duplicate elements.

    Args:
        input (Tensor) : the input tensor.
        indices1 (LongTensor): tensors used to horizontal coordinate into `self`.
        indices2 (LongTensor): tensors used to vertical coordinate into `self`.
        values (Tensor): tensor of same dtype as `self`.
        accumulate (bool): whether to accumulate into self
    """)


add_docstr('scatter',   r"""
    Writes all values from the tensor :attr:`src` into :attr:`self` at the indices
    specified in the :attr:`index` tensor. For each value in :attr:`src`, its output
    index is specified by its index in :attr:`src` for ``dimension != dim`` and by
    the corresponding value in :attr:`index` for ``dimension = dim``.

    For a 3-D tensor, :attr:`self` is updated as::

        self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
        self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
        self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    This is the reverse operation of the manner described in :meth:`~Tensor.gather`.

    :attr:`self`, :attr:`index` and :attr:`src` (if it is a Tensor) should all have
    the same number of dimensions. It is also required that
    ``index.size(d) <= src.size(d)`` for all dimensions ``d``, and that
    ``index.size(d) <= self.size(d)`` for all dimensions ``d != dim``.
    Note that ``index`` and ``src`` do not broadcast.

    Moreover, as for :meth:`~Tensor.gather`, the values of :attr:`index` must be
    between ``0`` and ``self.size(dim) - 1`` inclusive.

    .. warning::

        When indices are not unique, the behavior is non-deterministic (one of the
        values from ``src`` will be picked arbitrarily) and the gradient will be
        incorrect (it will be propagated to all locations in the source that
        correspond to the same index)!

    .. note::

        The backward pass is implemented only for ``src.shape == index.shape``.

    Additionally accepts an optional :attr:`reduce` argument that allows
    specification of an optional reduction operation, which is applied to all
    values in the tensor :attr:`src` into :attr:`self` at the indicies
    specified in the :attr:`index`. For each value in :attr:`src`, the reduction
    operation is applied to an index in :attr:`self` which is specified by
    its index in :attr:`src` for ``dimension != dim`` and by the corresponding
    value in :attr:`index` for ``dimension = dim``.

    Given a 3-D tensor and reduction using the multiplication operation, :attr:`self`
    is updated as::

        self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
        self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
        self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2

    Reducing with the addition operation is the same as using
    :meth:`~Tensor.scatter_add_`.

    Args:
        dim (int): the axis along which to index
        index (LongTensor): the indices of elements to scatter, can be either empty
            or of the same dimensionality as ``src``. When empty, the operation
            returns ``self`` unchanged.
        src (Tensor or float): the source elements to scatter.
        value (float): the source element to scatter.
        reduce (str, optional): reduction operation to apply, can be either
            ``'add'`` or ``'multiply'``.

    Example::

        >>> src = arange(1, 11).reshape((2, 5))
        >>> src
        tensor([[ 1,  2,  3,  4,  5],
                [ 6,  7,  8,  9, 10]])
        >>> index = tensor([[0, 1, 2, 0]])
        >>> zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)
        tensor([[1, 0, 0, 4, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 3, 0, 0]])
        >>> index = tensor([[0, 1, 2], [0, 1, 4]])
        >>> zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)
        tensor([[1, 2, 3, 0, 0],
                [6, 7, 0, 0, 8],
                [0, 0, 0, 0, 0]])

        >>> full((2, 4), 2.).scatter_(1, tensor([[2], [3]]),
        ...            1.23, reduce='multiply')
        tensor([[2.0000, 2.0000, 2.4600, 2.0000],
                [2.0000, 2.0000, 2.0000, 2.4600]])
        >>> full((2, 4), 2.).scatter_(1, tensor([[2], [3]]),
        ...            1.23, reduce='add')
        tensor([[2.0000, 2.0000, 3.2300, 2.0000],
                [2.0000, 2.0000, 2.0000, 3.2300]])

    """)


add_docstr("interpolate",   r"""
    Down/up samples the input to either the given :attr:`size` 

    The algorithm used for interpolation is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric sampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for resizing are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'`` . Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``
    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~nn.Upsample` for concrete examples on how this
        affects the outputs.
    """)


add_docstr("pad",   r"""
    Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(\text{padding_left}, \text{padding_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(\text{padding_left}, \text{padding_right},`
        :math:`\text{padding_top}, \text{padding_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(\text{padding_left}, \text{padding_right},`
        :math:`\text{padding_top}, \text{padding_bottom}`
        :math:`\text{padding_front}, \text{padding_back})`.

    Padding mode:
        See :class:`nn.ConstantPad2d`, :class:`nn.ReflectionPad2d`, and
        :class:`nn.ReplicationPad2d` for concrete examples on how each of the
        padding modes works. Constant padding is implemented for arbitrary dimensions.
        Replicate and reflection padding is implemented for padding the last 3
        dimensions of 5D input tensor, or the last 2 dimensions of 4D input
        tensor, or the last dimension of 3D input tensor.

    Args:
        input (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

    Examples::

        >>> t4d = empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = F.pad(t4d, p2d, "constant", 0)
        >>> print(out.size())
        Size([3, 3, 8, 4])
        >>> t4d = empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = F.pad(t4d, p3d, "constant", 0)
        >>> print(out.size())
        Size([3, 9, 7, 3])

    """)


add_docstr("unique",    r"""
    Returns the unique elements of the input tensor.

    .. note:: This function is different from :func:`unique_consecutive` in the sense that
        this function also eliminates non-consecutive duplicate values.

    .. note:: Currently in the CUDA implementation and the CPU implementation when dim is specified,
        `unique` always sort the tensor at the beginning regardless of the `sort` argument.
        Sorting could be slow, so if your input tensor is already sorted, it is recommended to use
        :func:`unique_consecutive` which avoids the sorting.

    Args:
        input (Tensor): the input tensor
        sorted (bool): Whether to sort the unique elements in ascending order
            before returning as output.
        return_inverse (bool): Whether to also return the indices for where
            elements in the original input ended up in the returned unique list.
        return_counts (bool): Whether to also return the counts for each unique
            element.
        dim (int): the dimension to apply unique. If ``None``, the unique of the
            flattened input is returned. default: ``None``

    Returns:
        (Tensor, Tensor (optional), Tensor (optional)): A tensor or a tuple of tensors containing

            - **output** (*Tensor*): the output list of unique scalar elements.
            - **inverse_indices** (*Tensor*): (optional) if
              :attr:`return_inverse` is True, there will be an additional
              returned tensor (same shape as input) representing the indices
              for where elements in the original input map to in the output;
              otherwise, this function will only return a single tensor.
            - **counts** (*Tensor*): (optional) if
              :attr:`return_counts` is True, there will be an additional
              returned tensor (same shape as output or output.size(dim),
              if dim was specified) representing the number of occurrences
              for each unique value or tensor.

    Example::

        >>> output = unique(tensor([1, 3, 2, 3], dtype=long))
        >>> output
        tensor([ 2,  3,  1])

        >>> output, inverse_indices = unique(
        ...     tensor([1, 3, 2, 3], dtype=long), sorted=True, return_inverse=True)
        >>> output
        tensor([ 1,  2,  3])
        >>> inverse_indices
        tensor([ 0,  2,  1,  2])

        >>> output, inverse_indices = unique(
        ...     tensor([[1, 3], [2, 3]], dtype=long), sorted=True, return_inverse=True)
        >>> output
        tensor([ 1,  2,  3])
        >>> inverse_indices
        tensor([[ 0,  2],
                [ 1,  2]])

    """)


add_docstr("prod",      r"""
    Returns the product of each row of the :attr:`input` tensor in the given
    dimension :attr:`dim`.

    Args:
        input (Tensor): the input tensor
        dim (int): the dimension to reduce.
        keepdim (bool): whether the output tensor has dim retained or not

    Example::

        >>> a = randn(4, 2)
        >>> a
        tensor([[ 0.5261, -0.3837],
                [ 1.1857, -0.2498],
                [-1.1646,  0.0705],
                [ 1.1131, -1.0629]])
        >>> prod(a, 1)
        tensor([-0.2018, -0.2962, -0.0821, -1.1831])
    """)


add_docstr("erfinv",   r"""
    Computes the inverse error function of :attr:`input`.
    The inverse error function is defined in the range :math:`(-1, 1)` as:

    .. math::
        \mathrm{erfinv}(\mathrm{erf}(x)) = x

    Args:
        input (Tensor): the input tensor
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """)


add_docstr("im2col",   r"""
    Extracts sliding local blocks from a batched input tensor.

    Consider a batched :attr:`input` tensor of shape :math:`(N, C, *)`,
    where :math:`N` is the batch dimension, :math:`C` is the channel dimension,
    and :math:`*` represent arbitrary spatial dimensions. This operation flattens
    each sliding :attr:`kernel_size`-sized block within the spatial dimensions
    of :attr:`input` into a column (i.e., last dimension) of a 3-D :attr:`output`
    tensor of shape :math:`(N, C \times \prod(\text{kernel\_size}), L)`, where
    :math:`C \times \prod(\text{kernel\_size})` is the total number of values
    within each block (a block has :math:`\prod(\text{kernel\_size})` spatial
    locations each containing a :math:`C`-channeled vector), and :math:`L` is
    the total number of such blocks:

    .. math::
        L = \prod_d \left\lfloor\frac{\text{spatial\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`\text{spatial\_size}` is formed by the spatial dimensions
    of :attr:`input` (:math:`*` above), and :math:`d` is over all spatial
    dimensions.

    Therefore, indexing :attr:`output` at the last dimension (column dimension)
    gives all values within a certain block.

    The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
    how the sliding blocks are retrieved.

    * :attr:`stride` controls the stride for the sliding blocks.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension before
      reshaping.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Args:
        input (Tensor): the input tensor
        kernel_size (int or tuple): the size of the sliding blocks
        stride (int or tuple, optional): the stride of the sliding blocks in the input
                                         spatial dimensions. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood. Default: 1

    * If :attr:`kernel_size`, :attr:`dilation`, :attr:`padding` or
      :attr:`stride` is an int or a tuple of length 1, their values will be
      replicated across all spatial dimensions.

    * For the case of two input spatial dimensions this operation is sometimes
      called ``im2col``.

    Shape:
        - Input: :math:`(N, C, *)`
        - Output: :math:`(N, C \times \prod(\text{kernel\_size}), L)` as described above

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """)


add_docstr("col2im",   r"""
    Combines an array of sliding local blocks into a large containing
    tensor.

    Consider a batched :attr:`input` tensor containing sliding local blocks,
    e.g., patches of images, of shape :math:`(N, C \times  \prod(\text{kernel\_size}), L)`,
    where :math:`N` is batch dimension, :math:`C \times \prod(\text{kernel\_size})`
    is the number of values within a block (a block has :math:`\prod(\text{kernel\_size})`
    spatial locations each containing a :math:`C`-channeled vector), and
    :math:`L` is the total number of blocks. (This is exactly the
    same specification as the output shape of :class:`~nn.Unfold`.) This
    operation combines these local blocks into the large :attr:`output` tensor
    of shape :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)`
    by summing the overlapping values. Similar to :class:`~nn.Unfold`, the
    arguments must satisfy

    .. math::
        L = \prod_d \left\lfloor\frac{\text{output\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`d` is over all spatial dimensions.

    * :attr:`output_size` describes the spatial shape of the large containing
      tensor of the sliding local blocks. It is useful to resolve the ambiguity
      when multiple input shapes map to same number of sliding blocks, e.g.,
      with ``stride > 0``.

    The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
    how the sliding blocks are retrieved.

    * :attr:`stride` controls the stride for the sliding blocks.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension before
      reshaping.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Args:
        input (Tensor): the input tensor
        output_size (int or tuple): the shape of the spatial dimensions of the
                                    output (i.e., ``output.sizes()[2:]``)
        kernel_size (int or tuple): the size of the sliding blocks
        stride (int or tuple): the stride of the sliding blocks in the input
                               spatial dimensions. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood. Default: 1

    * If :attr:`output_size`, :attr:`kernel_size`, :attr:`dilation`,
      :attr:`padding` or :attr:`stride` is an int or a tuple of length 1 then
      their values will be replicated across all spatial dimensions.

    * For the case of two output spatial dimensions this operation is sometimes
      called ``col2im``.

    Shape:
        - Input: :math:`(N, C \times \prod(\text{kernel\_size}), L)`
        - Output: :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)` as described above

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    """)


add_docstr("flip",   r"""
    Reverse the order of a n-D tensor along given axis in dims.

    Args:
        {input}
        dims (a list or tuple): axis to flip on

    Example::

        >>> x = arange(8).view(2, 2, 2)
        >>> x
        tensor([[[ 0,  1],
                [ 2,  3]],

                [[ 4,  5],
                [ 6,  7]]])
        >>> flip(x, [0, 1])
        tensor([[[ 6,  7],
                [ 4,  5]],

                [[ 2,  3],
                [ 0,  1]]])
    """.format(**common_args))


add_docstr("cholesky_ex",   r"""
    Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.

    Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
    the **Cholesky decomposition** of a complex Hermitian or real symmetric positive-definite matrix
    :math:`A \in \mathbb{K}^{n \times n}` is defined as

    .. math::

        A = LL^{\text{H}}\mathrlap{\qquad L \in \mathbb{K}^{n \times n}}

    where :math:`L` is a lower triangular matrix and
    :math:`L^{\text{H}}` is the conjugate transpose when :math:`L` is complex, and the transpose when :math:`L` is real-valued.

    Supports input of float, double, cfloat and cdouble dtypes.
    Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
    the output has the same batch dimensions.

    Keyword args:
        input (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                    consisting of symmetric or Hermitian positive-definite matrices.
        upper (bool, optional): whether to return an upper triangular matrix.
            The tensor returned with upper=True is the conjugate transpose of the tensor
            returned with upper=False.
        check_errors (bool, optional): controls whether to check the content of ``infos``. Default: `False`.

    Raises:
        RuntimeError: if the :attr:`A` matrix or any matrix in a batched :attr:`A` is not Hermitian
                    (resp. symmetric) positive-definite. If :attr:`A` is a batch of matrices,
                    the error message will include the batch index of the first matrix that fails
                    to meet this condition.
    Examples::
        >>> A = randn(2, 2, dtype=torch.complex128)
        >>> A = A @ A.t().conj()  # creates a Hermitian positive-definite matrix
        >>> L, info = cholesky_ex(A)
        >>> A
        tensor([[ 2.3792+0.0000j, -0.9023+0.9831j],
                [-0.9023-0.9831j,  0.8757+0.0000j]], dtype=torch.complex128)
        >>> L
        tensor([[ 1.5425+0.0000j,  0.0000+0.0000j],
                [-0.5850-0.6374j,  0.3567+0.0000j]], dtype=torch.complex128)
        >>> info
        tensor(0, dtype=torch.int32)
    """)


add_docstr("triangular_solve",   r"""
    Solves a system of equations with a triangular coefficient matrix :math:`A`
    and multiple right-hand sides :math:`b`.

    In particular, solves :math:`AX = b` and assumes :math:`A` is upper-triangular
    with the default keyword arguments.

    `triangular_solve(b, A)` can take in 2D inputs `b, A` or inputs that are
    batches of 2D matrices. If the inputs are batches, then returns
    batched outputs `X`

    If the diagonal of :attr:`A` contains zeros or elements that are very close to zero and
    :attr:`unitriangular`\ `= False` (default) or if the input matrix is badly conditioned,
    the result may contain `NaN` s.

    Supports input of float, double, cfloat and cdouble data types.

    Args:
        input (Tensor): multiple right-hand sides of size :math:`(*, m, k)` where
                    :math:`*` is zero of more batch dimensions
        A (Tensor): the input triangular coefficient matrix of size :math:`(*, m, m)`
                    where :math:`*` is zero or more batch dimensions
        upper (bool, optional): whether to solve the upper-triangular system
            of equations (default) or the lower-triangular system of equations. Default: ``True``.
        transpose (bool, optional): whether :math:`A` should be transposed before
            being sent into the solver. Default: ``False``.
        unitriangular (bool, optional): whether :math:`A` is unit triangular.
            If True, the diagonal elements of :math:`A` are assumed to be
            1 and not referenced from :math:`A`. Default: ``False``.

    Returns:
        A namedtuple `(solution, cloned_coefficient)` where `cloned_coefficient`
        is a clone of :math:`A` and `solution` is the solution :math:`X` to :math:`AX = b`
        (or whatever variant of the system of equations, depending on the keyword arguments.)
    """)
