DIOPI Functions
===================
.. currentmodule:: docs.EN_doc

Function List
------------------------
* Unary Functions(15)
     - abs_
     - cos_
     - erf_
     - erfinv_
     - exp_
     - floor_
     - log_
     - log2_
     - log10_
     - neg_
     - nonzero_
     - sign_
     - sin_
     - sqrt_
     - reciprocal_

* Binary Functions(18)
     - add_
     - bmm_
     - div_
     - eq_
     - fill_\_
     - ge_
     - gt_
     - le_
     - lt_
     - logical_and_
     - logical_or_
     - matmul_
     - mul_
     - ne_
     - pow_
     - sub_
     - bitwise_not_
     - remainder_

* Loss Functions(8)
     - binary_cross_entropy_with_logits_
     - cross_entropy_
     - mse_loss_
     - nll_loss_
     - sigmoid_focal_loss_
     - smooth_l1_loss_
     - ctc_loss_

* Activation Functions(9)
     - leaky_relu_
     - relu_
     - sigmoid_
     - hardtanh_
     - threshold_
     - gelu_
     - tanh_
     - softmax_
     - log_softmax_

* Reduce Functions(8)
     - mean_
     - min_
     - max_
     - std_
     - sum_
     - all_
     - any_
     - argmax_

* Optimizer Functions(4)
     - sgd_
     - adam_
     - adamw_
     - adadelta_

* Other Functions(67)
     - addcdiv_
     - addcmul_
     - addmm_
     - adaptive_avg_pool2d_
     - avg_pool2d_
     - max_pool2d_
     - adaptive_max_pool2d_
     - batch_norm_
     - cat_
     - clamp_
     - clip_grad_norm_\_
     - conv2d_
     - dropout_
     - embedding_
     - index_select_
     - masked_scatter_
     - linear_
     - one_hot_
     - select_
     - sort_
     - split_
     - stack_
     - topk_
     - transpose_
     - tril_
     - where_
     - nms_
     - roi_align_
     - arange_
     - randperm_
     - uniform_
     - random_
     - bernoulli_
     - masked_fill_
     - conv_transpose2d_
     - cumsum_
     - cdist_
     - maximum_
     - minimum_
     - mm_
     - conv3d_
     - expand_
     - unfold_
     - masked_select_
     - index_fill_
     - linspace_
     - roll_
     - norm_
     - group_norm_
     - layer_norm_
     - adaptive_avg_pool3d_
     - adaptive_max_pool3d_
     - max_pool3d_
     - permute_
     - copy_\_
     - gather_
     - index_put_
     - scatter_
     - interpolate_
     - pad_
     - unique_
     - prod_
     - im2col_
     - col2im_
     - flip_
     - cholesky_
     - triangular_solve_


1. Unary Functions
------------------------

abs
~~~~~~~~
.. autofunction:: abs

cos
~~~~~~~~
.. autofunction:: cos

erf
~~~~~~~~
.. autofunction:: erf

erfinv
~~~~~~~~
.. autofunction:: erfinv

exp
~~~~~~~~
.. autofunction:: exp

floor
~~~~~~~~
.. autofunction:: floor

log
~~~~~~~~
.. autofunction:: log

log2
~~~~~~~~
.. autofunction:: log2

log10
~~~~~~~~
.. autofunction:: log10

neg
~~~~~~~~
.. autofunction:: neg

nonzero
~~~~~~~~
.. autofunction:: nonzero

sign
~~~~~~~~
.. autofunction:: sign

sin
~~~~~~~~
.. autofunction:: sin

sqrt
~~~~~~~~
.. autofunction:: sqrt

reciprocal
~~~~~~~~~~~~~~~
.. autofunction:: reciprocal

2. Binary Functions
------------------------

add
~~~~~~~~
.. autofunction:: add

bmm
~~~~~~~~
.. autofunction:: bmm

div
~~~~~~~~
.. autofunction:: div

eq
~~~~~~~~
.. autofunction:: eq

fill_
~~~~~~~~
.. autofunction:: fill_

ge
~~~~~~~~
.. autofunction:: ge

gt
~~~~~~~~
.. autofunction:: gt

le
~~~~~~~~
.. autofunction:: le

lt
~~~~~~~~
.. autofunction:: lt

logical_and
~~~~~~~~~~~
.. autofunction:: logical_and

logical_or
~~~~~~~~~~~
.. autofunction:: logical_or

matmul
~~~~~~~~~~~
.. autofunction:: matmul

mul
~~~~~~~~~~~
.. autofunction:: mul

ne
~~~~~~~~~~~
.. autofunction:: ne

pow
~~~~~~~~~~~
.. autofunction:: pow

sub
~~~~~~~~
.. autofunction:: sub

bitwise_not
~~~~~~~~~~~~~~~
.. autofunction:: bitwise_not

remainder
~~~~~~~~~~~~~~~
.. autofunction:: remainder

3. Loss Functions
------------------------

binary_cross_entropy_with_logits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: binary_cross_entropy_with_logits

cross_entropy
~~~~~~~~~~~~~~~~
.. autofunction:: cross_entropy

mse_loss
~~~~~~~~~~
.. autofunction:: mse_loss

nll_loss
~~~~~~~~~~
.. autofunction:: nll_loss

sigmoid_focal_loss
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: sigmoid_focal_loss

smooth_l1_loss
~~~~~~~~~~~~~~~
.. autofunction:: smooth_l1_loss

ctc_loss
~~~~~~~~~~~~~~~
.. autofunction:: ctc_loss

4. Activation Functions
------------------------

leaky_relu
~~~~~~~~~~
.. autofunction:: leaky_relu

relu
~~~~~~~~~~
.. autofunction:: relu

sigmoid
~~~~~~~~~~
.. autofunction:: sigmoid

hardtanh
~~~~~~~~~~
.. autofunction:: hardtanh

threshold
~~~~~~~~~~
.. autofunction:: threshold

gelu
~~~~~~~~~~
.. autofunction:: gelu

tanh
~~~~~~~~
.. autofunction:: tanh

softmax
~~~~~~~~~~
.. autofunction:: softmax

log_softmax
~~~~~~~~~~~~
.. autofunction:: log_softmax


5. Reduce Functions
------------------------

mean
~~~~~~~~~~
.. autofunction:: mean

min
~~~~~~~~~~
.. autofunction:: min

max
~~~~~~~~~~
.. autofunction:: max

std
~~~~~~~~~~
.. autofunction:: std

sum
~~~~~~~~~~
.. autofunction:: sum

any
~~~~~~~~~~
.. autofunction:: any

all
~~~~~~~~~~
.. autofunction:: all

argmax
~~~~~~~~
.. autofunction:: argmax

6. Optimizer Functions
------------------------

sgd
~~~~~~~~~~
.. autofunction:: sgd

adamw
~~~~~~~~~~
.. autofunction:: adamw

adam
~~~~~~~~~~
.. autofunction:: adam

adadelta
~~~~~~~~~~
.. autofunction:: adadelta

7. Other Functions
------------------------

addcdiv
~~~~~~~~
.. autofunction:: addcdiv

addcmul
~~~~~~~~
.. autofunction:: addcmul

addmm
~~~~~~~~
.. autofunction:: addmm

adaptive_avg_pool2d
~~~~~~~~~~~~~~~~~~~
.. autofunction:: adaptive_avg_pool2d

avg_pool2d
~~~~~~~~~~
.. autofunction:: avg_pool2d

max_pool2d
~~~~~~~~~~
.. autofunction:: max_pool2d

adaptive_max_pool2d
~~~~~~~~~~~~~~~~~~~
.. autofunction:: adaptive_max_pool2d

batch_norm
~~~~~~~~~~
.. autofunction:: batch_norm

cat
~~~~~~~~
.. autofunction:: cat

clamp
~~~~~~~~
.. autofunction:: clamp

clip_grad_norm_
~~~~~~~~~~~~~~~~~
.. autofunction:: clip_grad_norm_

conv2d
~~~~~~~~
.. autofunction:: conv2d

dropout
~~~~~~~~~~~~~~~
.. autofunction:: dropout

embedding
~~~~~~~~~~~~~~~
.. autofunction:: embedding

index_select
~~~~~~~~~~~~~~~
.. autofunction:: index_select

masked_scatter
~~~~~~~~~~~~~~~
.. autofunction:: masked_scatter

linear
~~~~~~~~~~~~~~~
.. autofunction:: linear

one_hot
~~~~~~~~~~
.. autofunction:: one_hot

select
~~~~~~~~~~
.. autofunction:: select

sort
~~~~~~~~~~
.. autofunction:: sort

split
~~~~~~~~~~
.. autofunction:: split

stack
~~~~~~~~~~
.. autofunction:: stack

topk
~~~~~~~~~~
.. autofunction:: topk

transpose
~~~~~~~~~~
.. autofunction:: transpose

tril
~~~~~~~~~~
.. autofunction:: tril

where
~~~~~~~~~~
.. autofunction:: where

nms
~~~~~~~~~~
.. autofunction:: nms

roi_align
~~~~~~~~~~
.. autofunction:: roi_align

arange
~~~~~~~~~~
.. autofunction:: arange

randperm
~~~~~~~~~~
.. autofunction:: randperm

uniform   
~~~~~~~~~~
.. autofunction:: uniform

random   
~~~~~~~~~~
.. autofunction:: random

bernoulli
~~~~~~~~~~
.. autofunction:: bernoulli

masked_fill
~~~~~~~~~~~~~~~~~
.. autofunction:: masked_fill

conv_transpose2d
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: conv_transpose2d

cumsum
~~~~~~~~
.. autofunction:: cumsum

cdist
~~~~~~~~
.. autofunction:: cdist

maximum
~~~~~~~~
.. autofunction:: maximum

minimum
~~~~~~~~
.. autofunction:: minimum

mm
~~~~~~~~
.. autofunction:: mm

conv3d
~~~~~~~~
.. autofunction:: conv3d

expand
~~~~~~~~
.. autofunction:: expand

unfold
~~~~~~~~
.. autofunction:: unfold

masked_select
~~~~~~~~~~~~~~~
.. autofunction:: masked_select

index_fill
~~~~~~~~~~~~~~~
.. autofunction:: index_fill

linspace
~~~~~~~~~~~~~~~
.. autofunction:: linspace

roll
~~~~~~~~
.. autofunction:: roll

norm
~~~~~~~~
.. autofunction:: norm

group_norm
~~~~~~~~~~~~~~~
.. autofunction:: group_norm

layer_norm
~~~~~~~~~~~~~~~
.. autofunction:: layer_norm

adaptive_avg_pool3d
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: adaptive_avg_pool3d

adaptive_max_pool3d
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: adaptive_max_pool3d

max_pool3d
~~~~~~~~~~~~
.. autofunction:: max_pool3d

permute
~~~~~~~~~~~~
.. autofunction:: permute

copy_
~~~~~~~~~~~~
.. autofunction:: copy_

gather
~~~~~~~~~~~~
.. autofunction:: gather

index_put
~~~~~~~~~~~~
.. autofunction:: index_put

scatter
~~~~~~~~~~~~
.. autofunction:: scatter

interpolate
~~~~~~~~~~~~
.. autofunction:: interpolate

pad
~~~~~~~~~~~~
.. autofunction:: pad

unique
~~~~~~~~~~~~
.. autofunction:: unique

prod
~~~~~~~~~~~~
.. autofunction:: prod

im2col
~~~~~~~~~~~~
.. autofunction:: im2col

col2im
~~~~~~~~~~~~
.. autofunction:: col2im

flip
~~~~~~~~~~~~
.. autofunction:: flip

cholesky
~~~~~~~~~~~~
.. autofunction:: cholesky_ex

triangular_solve
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: triangular_solve


.. _broadcasting-semantics:

Broadcasting semantics
------------------------

Many operations support NumPy's broadcasting semantics.
See https://numpy.org/doc/stable/user/basics.broadcasting.html for details.

In short, if an operation supports broadcast, then its Tensor arguments can be
automatically expanded to be of equal sizes (without making copies of the data).


.. _type-promotion-doc:

Type promotion
------------------------

When the dtypes of inputs to an arithmetic operation (`add`, `sub`, `div`, `mul`) differ, we promote
by finding the minimum dtype that satisfies the following rules:

* If the type of a scalar operand is of a higher category than tensor operands
  (where complex > floating > integral > boolean), we promote to a type with sufficient size to hold
  all scalar operands of that category.
* If a zero-dimension tensor operand has a higher category than dimensioned operands,
  we promote to a type with sufficient size and category to hold all zero-dim tensor operands of
  that category.
* If there are no higher-category zero-dim operands, we promote to a type with sufficient size
  and category to hold all dimensioned operands.

A floating point scalar operand has dtype `get_default_dtype()` and an integral
non-boolean scalar operand has dtype `int64`. Unlike numpy, we do not inspect
values when determining the minimum `dtypes` of an operand.  Quantized and complex types
are not yet supported.

Promotion Examples::

    >>> float_tensor = ones(1, dtype=float)
    >>> double_tensor = ones(1, dtype=double)
    >>> complex_float_tensor = ones(1, dtype=complex64)
    >>> complex_double_tensor = ones(1, dtype=complex128)
    >>> int_tensor = ones(1, dtype=int)
    >>> long_tensor = ones(1, dtype=long)
    >>> uint_tensor = ones(1, dtype=uint8)
    >>> double_tensor = ones(1, dtype=double)
    >>> bool_tensor = ones(1, dtype=bool)
    # zero-dim tensors
    >>> long_zerodim = tensor(1, dtype=long)
    >>> int_zerodim = tensor(1, dtype=int)

    >>> add(5, 5).dtype
    int64
    # 5 is an int64, but does not have higher category than int_tensor so is not considered.
    >>> (int_tensor + 5).dtype
    int32
    >>> (int_tensor + long_zerodim).dtype
    int32
    >>> (long_tensor + int_tensor).dtype
    int64
    >>> (bool_tensor + long_tensor).dtype
    int64
    >>> (bool_tensor + uint_tensor).dtype
    uint8
    >>> (float_tensor + double_tensor).dtype
    float64
    >>> (complex_float_tensor + complex_double_tensor).dtype
    complex128
    >>> (bool_tensor + int_tensor).dtype
    int32
    # Since long is a different kind than float, result dtype only needs to be large enough
    # to hold the float.
    >>> add(long_tensor, float_tensor).dtype
    float32

When the output tensor of an arithmetic operation is specified, we allow casting to its `dtype` except that:
  * An integral output tensor cannot accept a floating point tensor.
  * A boolean output tensor cannot accept a non-boolean tensor.
  * A non-complex output tensor cannot accept a complex tensor

Casting Examples::

    # allowed:
    >>> float_tensor *= float_tensor
    >>> float_tensor *= int_tensor
    >>> float_tensor *= uint_tensor
    >>> float_tensor *= bool_tensor
    >>> float_tensor *= double_tensor
    >>> int_tensor *= long_tensor
    >>> int_tensor *= uint_tensor
    >>> uint_tensor *= int_tensor

    # disallowed (RuntimeError: result type can't be cast to the desired output type):
    >>> int_tensor *= float_tensor
    >>> bool_tensor *= int_tensor
    >>> bool_tensor *= uint_tensor
    >>> float_tensor *= complex_float_tensor

