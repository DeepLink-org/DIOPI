DIOPI 函数
===================
.. currentmodule:: docs.CN_doc

Function List
------------------------
* Unary Functions(14)
     - abs_
     - cos_
     - erf_
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

* Other Functions(62)
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
