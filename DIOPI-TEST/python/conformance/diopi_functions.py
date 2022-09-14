# -*- coding: UTF-8 -*-
import math

from ctypes import c_float, c_int64, c_int32, c_void_p, byref
from .diopi_rt import Sizes, Scalar, Tensor, device_impl_lib
from .utils import check_returncode, check_function, squeeze
from . import Dtype, raw_like
from collections import namedtuple


def broadcast_out_size(size1, size2):
    sizeO = size1 if len(size1) > len(size2) else size2
    length = len(size2) if len(size1) > len(size2) else len(size1)
    idx = -1
    while length > 0:
        assert size1[idx] == size2[idx] or size1[idx] == 1 or size2[idx] == 1,\
            "size1 and size2 must be broadcastable"
        sizeO[idx] = size1[idx] if size2[idx] == 1 else size2[idx]
        idx -= 1
        length -= 1

    return sizeO


def reduce_op_process(input, dim=None, keepdim=False, dtype=None):
    sizeI = list(input.size())
    if dim is None:
        for i in range(0, len(sizeI)):
            sizeI[i] = 1
        dim = []
    elif isinstance(dim, list):
        for i in dim:
            sizeI[i] = 1
    else:
        sizeI[dim] = 1
        dim = [dim]

    if dtype is None:
        dtype = input.get_dtype()

    out = Tensor(sizeI, dtype)
    if ~keepdim:
        squeeze(out)
    return dim, out


def fill(tensor, value):
    r"""
    释义
        使用指定值填充 *tensor* 张量。
    参数
        - *tensor* ( **Tensor** ) : 待填充张量
        - *value* ( **number** ) : 填充值
    C API
        :guilabel:`diopiFill`
    """
    func = check_function("diopiFill")
    ret = func(tensor.context_handle, tensor.tensor_handle, c_float(value))
    check_returncode(ret)
    return tensor


def ones_like(tensor):
    new_tensor = raw_like(tensor)
    fill(new_tensor, 1)
    return new_tensor


def unary_op(input, inplace, call) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle)

    check_returncode(ret)
    return out


def binary_op(input, other, inplace, call) -> Tensor:
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle,
                   other.tensor_handle)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, other.tensor_handle)

    check_returncode(ret)
    return out


def binary_op_scalar(input, other, inplace, call, alpha=None) -> Tensor:
    args = "input.context_handle, "
    if inplace:
        out = input
    else:
        if not isinstance(other, Tensor):
            out = raw_like(input)
        else:
            sizeI = input.size()
            sizeO = other.size()
            outsize = broadcast_out_size(list(sizeI), list(sizeO))
            out = Tensor(outsize, input.get_dtype())
        args = args + "out.tensor_handle, "

    if not isinstance(other, Tensor):
        call = call + "Scalar"
        other = Scalar(input.get_dtype(), other)
        args = args + "input.tensor_handle, byref(other)"
    else:
        args = args + "input.tensor_handle, other.tensor_handle"\

    if alpha is not None:
        alpha = Scalar(input.get_dtype(), alpha)
        args = args + ", byref(alpha)"

    func = check_function(call)
    ret = eval(f'func({args})')

    check_returncode(ret)
    return out


def softmax(input, dim, dtype=None):
    r"""
    释义
        对输入张量应用 *softmax* 函数，使得输出张量中元素值在范围 [0,1] 之间, 且元素总和为 1。
        相应公式如下:

        .. math::
            \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **number** ) : 对输入张量应用 *softmax* 函数的维度 (使得沿着该维度的元素和为 **1**)
        - *dtype* ( **Dtype**, 可选) : 期望的返回值数据类型，如果指定，输入张量将会提前转换为 *dtype* 类型以防止数值溢出。
    C API
        :guilabel:`diopiSoftmax`
    """
    if dim is None:
        dim = 0
    if input.numel() == 0:
        return input
    if dtype is None:
        dtype = input.get_dtype()
    out = raw_like(input)

    func = check_function('diopiSoftmax')
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(dim), c_int32(dtype.value))
    check_returncode(ret)
    return out


def relu(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入 *input* 张量逐元素做 *relu* 整流线性变换:

            :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    参数
        - *input* ( **Tensor** ): 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiRelu` :guilabel:`diopiReluInp`
    """
    return unary_op(input, inplace, 'diopiRelu')


def abs(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量逐元素计算绝对值:

        .. math::
            \text { out }_{i}=\mid \text { input }_{i} \mid
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiAbs` :guilabel:`diopiAbsInp`
    """
    return unary_op(input, inplace, 'diopiAbs')


def floor(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量逐元素做向下取整:

        .. math::
            \text { out }_{i}=\left\lfloor\text { input }_{i}\right\rfloor
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiFloor` :guilabel:`diopiFloorInp`
    """
    return unary_op(input, inplace, 'diopiFloor')


def sign(input) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素计算符号函数 *Sgn* 值:

        .. math::
            \text { out }_{i}=\operatorname{sgn}\left(\text { input }_{i}\right)
    参数
        - *input* ( **Tensor** ) : 输入张量
    C API
        :guilabel:`diopiSign`
    """
    return unary_op(input, False, 'diopiSign')


def sigmoid(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素做 *sigmoid* 变换:

        .. math::
            \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiSigmoid` :guilabel:`diopiSigmoidInp`
    """
    return unary_op(input, inplace, 'diopiSigmoid')


def sqrt(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素开方:

        .. math::
            \text { out }_{i}=\sqrt{\text { input }_{i}}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiSqrt` :guilabel:`diopiSqrtInp`
    """
    return unary_op(input, inplace, 'diopiSqrt')


def neg(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素取相反数:

        .. math::
            \text { out }=-1 \times \text { input }
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiNeg` :guilabel:`diopiNegInp`
    """
    return unary_op(input, inplace, 'diopiNeg')


def sin(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素计算三角函数sin值:

        .. math::
            \text { out }_{i}=\sin \left(\text { input }_{i}\right)
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiSin` :guilabel:`diopiSinInp`
    """
    return unary_op(input, inplace, 'diopiSin')


def cos(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素计算其三角函数cos值:

        .. math::
            \text { out }_{i}=\cos \left(\text { input }_{i}\right)
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiCos` :guilabel:`diopiCosInp`
    """
    return unary_op(input, inplace, 'diopiCos')


def tanh(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素计算其双曲正切函数值:

        .. math::
            \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiTanh` :guilabel:`diopiTanhInp`
    """
    return unary_op(input, inplace, 'diopiTanh')


def exp(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素计算其以 *e* 为底的指数函数值:

        .. math::
            y_{i}=e^{x_{i}}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiExp` :guilabel:`diopiExpInp`
    """
    return unary_op(input, inplace, 'diopiExp')


def log(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素计算其自然对数:

        .. math::
            y_{i}=\log _{e}\left(x_{i}\right)
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiLog` :guilabel:`diopiLogInp`
    """
    return unary_op(input, inplace, 'diopiLog')


def log2(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素计算其以2为底的对数值:

        .. math::
            y_{i}=\log _{2}\left(x_{i}\right)
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiLog2` :guilabel:`diopiLog2Inp`
    """
    return unary_op(input, inplace, 'diopiLog2')


def log10(input, inplace=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 逐元素计算其以10为底的对数值:

        .. math::
            y_{i}=\log _{10}\left(x_{i}\right)
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiLog10` :guilabel:`diopiLog10Inp`
    """
    return unary_op(input, inplace, 'diopiLog10')


def erf(input, inplace=False) -> Tensor:
    r"""
    释义
        计算输入张量 *input* 的误差函数, 误差函数如下:

        .. math::
            \operatorname{erf}(x)=\frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} dt
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiErf` :guilabel:`diopiErfInp`
    """
    return unary_op(input, inplace, 'diopiErf')


def add(input, other, alpha=1) -> Tensor:
    # todo: 需要解释广播机制，类型提升
    r"""
    释义
        将 *other* 乘以 *alpha* 后再加至张量 *input* 上:

        .. math::
            \text { out }_{i}=\text { input }_{i}+\text { alpha } \times \text { other }_{i}

        支持广播、类型提升以及整数、浮点数和复数输入。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **number** ) : 与输入张量相加
        - *alpha* ( **Tensor**  , 可选 ) : *other* 的的乘数
    C API
        :guilabel:`diopiAdd` :guilabel:`diopiAddScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiAdd', alpha=alpha)


def sub(input, other, alpha=1.0) -> Tensor:
    r"""
    释义
        张量 *input* 减去 *other*, 减数通过与 *alpha* 相乘进行缩放:

        .. math::
            \text { out }_{i}=\text { input }_{i}-\text { alpha } \times \text { other }_{i}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **number** ) : 与输入张量相加
        - *alpha* ( **number** ) : 减数缩放因子
    C API
        :guilabel:`diopiSub` :guilabel:`diopiSubScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiSub', alpha=alpha)


def eq(input, other) -> Tensor:
    r"""
    释义
        将张量 *input* 与 *other* 比较，逐元素判断是否相等。
        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 比较值, 可以是张量或数值
    返回值
        返回一个布尔类型的张量，如果 *input* 等于 *other*， 该处值为 True，反之为 False
    C API
        :guilabel:`diopiEq` :guilabel:`diopiEqScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiEq')


def ne(input, other) -> Tensor:
    r"""
    释义
        逐元素计算 :math:`\text{input} \neq \text{other}`。
        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 比较值, 可以是张量或数值
    返回值
        返回一个布尔类型的张量，如果 *input* 不等于 *other*， 该处值为 True，反之为 False
    C API
        :guilabel:`diopiNe` :guilabel:`diopiNeScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiNe')


def ge(input, other) -> Tensor:
    r"""
    释义
        张量比较, 逐元素计算 :math:`\text{input} \geq \text{other}`。
        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 比较值, 可以是张量或数值
    C API
        :guilabel:`diopiGe` :guilabel:`diopiGeScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiGe')


def gt(input, other) -> Tensor:
    r"""
    释义
        张量比较, 逐元素计算 :math:`\text{input} \ge \text{other}`。
        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 比较值, 可以是张量或数值
    C API
        :guilabel:`diopiGt` :guilabel:`diopiGtScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiGt')


def le(input, other) -> Tensor:
    r"""
    释义
        张量比较, 逐元素计算 :math:`\text{input} \leq \text{other}`。
        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 比较值, 可以是张量或数值
    C API
        :guilabel:`diopiLe` :guilabel:`diopiLeScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiLe')


def lt(input, other) -> Tensor:
    r"""
    释义
        张量比较, 逐元素计算 :math:`\text{input} < \text{other}`。
        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 比较值, 可以是张量或数值
    C API
        :guilabel:`diopiLt` :guilabel:`diopiLtScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiLt')


def mul(input, other) -> Tensor:
    r"""
    释义
        张量 *input* 与 *other* 相乘(矩阵乘):

        .. math::
            \text { out }_{i}=\operatorname{input}_{i} \times \text { other }_{i}

        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 乘数, 可以是张量或数值
    C API
        :guilabel:`diopiMul` :guilabel:`diopiMulScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiMul')


def div(input, other) -> Tensor:
    r"""
    释义
        张量除, 输入张量 *input* 每个元素都除以 *other* 中与之对应的元素:

        .. math::
            \text { out }_{i}=\frac{\text { input }_{i}}{\text { other }_{i}}
    参数
        - *input* ( **Tensor** ) : 被除数
        - *other* ( **Tensor**  或者 **number** ) : 除数
    C API
        :guilabel:`diopiDiv` :guilabel:`diopiDivScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiDiv')


def logical_and(input, other) -> Tensor:
    r"""
    释义
        张量逻辑与, 对应元素进行逻辑与操作, 对于张量中的每个元素, 零元素视为False, 非零元素视为True。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor** ) : 用于计算逻辑与的张量
    C API
        :guilabel:`diopiBitwiseAnd` :guilabel:`diopiBitwiseAndScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiBitwiseAnd')


def logical_or(input, other) -> Tensor:
    r"""
    释义
        张量逻辑或, 对应元素进行逻辑或操作, 对于张量中的每个元素, 零元素视为False, 非零元素视为True。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor** ) : 用于计算逻辑与的张量
    C API
        :guilabel:`diopiBitwiseOr` :guilabel:`diopiBitwiseOrScalar`
    """
    return binary_op_scalar(input, other, False, 'diopiBitwiseOr')


def leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor:
    r"""
    释义
        对张量 *input* 逐元素做 *leaky_relu* :

        .. math::
            \text { LeakyReLU }(x)=\max (0, x)+\text { negative_slope } * \min (0, x)
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *negative_slope* ( **float** ) : 负斜率控制因子
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiLeakyReLu` :guilabel:`diopiLeakyReLuInp`
    """
    negative_slope = byref(Scalar(Dtype.float64, negative_slope))
    if inplace:
        out = input
        func = check_function("diopiLeakyReluInp")
        ret = func(input.context_handle,
                   out.tensor_handle, input.tensor_handle, negative_slope)
    else:
        out = raw_like(input)
        func = check_function("diopiLeakyRelu")
        ret = func(input.context_handle,
                   out.tensor_handle, input.tensor_handle, negative_slope)

    check_returncode(ret)
    return out


def bmm(input, mat2) -> Tensor:
    r"""
    释义
        批处理张量乘法(矩阵乘), 其中输入张量 *input* 与 *mat2* 张量均为三维张量:

        .. math::
            \text { out }_{i}=\text { input }_{i} @ \operatorname{mat}_{i}

        如果 *input* 为一个 :math:`(b \times n \times m)` 的张量, *mat2* 为一个 :math:`(b \times m \times p)` 的张量,
        则结果为一个 :math:`(b \times n \times p)` 的张量。

    参数
        - *input* ( **Tensor** ) : 输入张量
        - *mat2* ( **Tensor** ) : 与输入张量做矩阵乘的张量
    C API
        :guilabel:`diopiBmm`
    """
    size1 = list(input.size())
    assert(len(size1) == 3), 'input must be 3d tensor'
    size2 = mat2.size()
    assert(len(size2) == 3), 'mat2 must be 3d tensor'
    assert(size1[0] == size2[0]), 'invalid args'
    assert(size1[2] == size2[1]), 'invalid args'

    size_out = size1
    size_out[2] = size2[2]
    out = Tensor(size_out, input.get_dtype())

    func = check_function("diopiBmm")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, mat2.tensor_handle)
    check_returncode(ret)
    return out


def addcmul(input, tensor1, tensor2, value=1) -> Tensor:
    r"""
    释义
        执行 *tensor1* 与 *tensor2* 的逐元素乘法，将结果乘以标量值 *value* 后再加至输入张量 *input* :

        .. math::
            \text { out }_{i}=\text { input }_{i}+\text { value } \times \text { tensor }_{i} \times \operatorname{tensor}_{i}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *tensor1* ( **Tensor** ) : 用来做乘法的张量 **1**
        - *tensor2* ( **Tensor** ) : 用来做乘法的张量 **2**
        - *value* ( **number** ) : 张量相乘结果的缩放因子，默认值为 1
    C API
        :guilabel:`diopiAddcmul`
    """
    size1 = list(tensor1.size())
    size2 = list(tensor2.size())
    sizeI = list(input.size())
    sizeO = broadcast_out_size(size1, size2)
    sizeO = broadcast_out_size(sizeI, sizeO)
    out = Tensor(sizeO, input.get_dtype())
    value = byref(Scalar(input.get_dtype(), value))

    func = check_function("diopiAddcmul")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               tensor1.tensor_handle, tensor2.tensor_handle, value)
    check_returncode(ret)
    return out


def matmul(input, other) -> Tensor:
    r"""
    释义
        张量乘法，依据输入张量维度的不同，乘法规则不同:
            - 如果两个张量都是一维的，则返回点积（标量）。
            - 如果两个张量都是二维的，则返回矩阵相乘结果。
            - 如果第一个参数是一维的, 第二个参数是二维的, 为了矩阵乘法的目的, 给第一个参数的前面增加 1 个维度。在矩阵相乘之后, 前置维度被移除。
            - 如果第一个参数是二维的, 第二个参数是一维的, 则返回矩阵向量积。
            - 如果两个参数至少为一维且至少一个参数为 N 维(其中 N > 2), 则返回批处理矩阵乘法。
              如果第一个参数是一维的, 则将 1 添加到其维度之前, 在批量矩阵相乘之后移除。
              如果第二个参数是一维的, 则将 1 添加到其维度之后，在批量矩阵相乘之后移除。
              非矩阵（即批处理矩阵) 维度是必须可广播的, 例如 : 若输入张量 *input* 形状为 :math:`(j, 1, n, n)`, *other* 形状为 :math:`(k, n, n)`,
              则结果 *out* 形状为 :math:`(j, k, n, n)`。

        .. note::
            广播逻辑在确定输入是否可广播时仅查看批处理的维度, 而不是矩阵的维度。 例如:
            如果输入张量 *input* 形状为 :math:`(j, 1, n, m)`, *other* 形状为 :math:`(k, m, p)`, 即使 *input* 与 *other* 后两个维度不同，
            但依旧满足广播条件, 结果 *out* 形状为 :math:`(j, k, n, p)`。
    参数
        - *input* ( **Tensor** ) : 乘法的第一个输入张量
        - *other* ( **Tensor** ) : 乘法的第二个输入张量
    C API
        :guilabel:`diopiMatmul`
    """
    # tocheck: the shape of out tensor
    out = raw_like(input)
    sizeI = list(input.size())
    sizeO = list(other.size())

    # vector x vector
    if len(sizeI) == 1 and len(sizeO) == 1:
        out = Tensor((), input.get_dtype())
    # (batched) matrix x vector
    elif len(sizeO) == 1:
        sizeI[-1] = 1
        out = Tensor(sizeI,  input.get_dtype())
    # pretended matrix x (batched) matrix
    elif len(sizeI) == 1:
        sizeO[-2] = 1
        out = Tensor(sizeO, input.get_dtype())
    # (batched) matrix x (batched) matrix
    else:
        sizeI[-1] = sizeO[-1]
        if len(sizeI) > 3 and len(sizeO) > 2:
            assert sizeI[-3] == sizeO[-3] or sizeI[-3] == 1 or sizeO[-3] == 1,\
                'input and other should be broadcastable'
            sizeI[-3] = sizeI[-3] if sizeI[-3] == 1 else sizeO[-3]
        out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiMatmul")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, other.tensor_handle)
    check_returncode(ret)

    # out.squeeze()
    return out


def clamp(input, min=None, max=None, inplace=False) -> Tensor:
    r"""
    释义
        将 *input* 中的所有元素限制在 [ min, max ] 范围内。返回值如下:

        .. math::
            y_i = \begin{cases}
                \text{min}_{i} & \text{if } x_i < \text{min}_{i}  \\
                x_i & \text{if } \text{min}_{i}  \leq x_i \leq \text{max}_{i}  \\
                \text{max}_{i}  & \text{if } x_i > \text{max}_{i}
            \end{cases}

        如果 *min* 为 *None*, 则无下界。若 *max* 为 *None*, 则无上界。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *min* ( **number** 或者 **Tensor** , 可选) : 取值下界
        - *max* ( **number** 或者 **Tensor** , 可选) : 取值上界
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiClamp` :guilabel:`diopiClampMax` :guilabel:`diopiClampMin`
    """
    assert min is not None or max is not None,\
        "min and max can not be None in the meantime"
    if max is None:
        return clamp_min(input, min, inplace)
    if min is None:
        return clamp_max(input, max, inplace)

    call = "diopiClamp"
    args = "input.context_handle, "
    if inplace:
        out = input
        call = call + "Inp"
    else:
        out = raw_like(input)
        args = args + "out.tensor_handle, "

    if isinstance(min, Tensor):
        assert(isinstance(max, Tensor)), 'min and max must have same type'
        args += "input.tensor_handle, min.tensor_handle, max.tensor_handle"
    else:
        assert(~isinstance(max, Tensor)), 'min and max must have same type'
        call = call + 'Scalar'
        min = byref(Scalar(input.get_dtype(), min))
        max = byref(Scalar(input.get_dtype(), max))
        args = args + "input.tensor_handle, min, max"

    func = check_function(call)
    ret = func(eval(f'{args}'))
    check_returncode(ret)
    return out


def clamp_min(input, min, inplace=False) -> Tensor:
    call = "diopiClampMin"
    args = "input.context_handle, "
    if inplace:
        out = input
        call = call + "Inp"
    else:
        out = raw_like(input)
        args = args + "out.tensor_handle, "

    if isinstance(min, Tensor):
        args = args + "input.tensor_handle, min.tensor_handle"
    else:
        call = call + 'Scalar'
        min = byref(Scalar(input.get_dtype(), min))
        args = args + "input.tensor_handle, min"

    func = check_function(call)
    ret = func(eval(f'{args}'))
    check_returncode(ret)
    return out


def clamp_max(input, max, inplace=False) -> Tensor:
    call = "ClampMax"
    args = "input.context_handle, "
    if inplace:
        out = input
        call = call + "Inp"
    else:
        out = raw_like(input)
        args = args + "out.tensor_handle, "

    if isinstance(max, Tensor):
        args = args + "input.tensor_handle, max.tensor_handle"
    else:
        call = call + 'Scalar'
        max = byref(Scalar(input.get_dtype(), max))
        args = args + "input.tensor_handle, max"

    func = check_function(call)
    ret = func(eval(f'{args}'))
    check_returncode(ret)
    return out


def mean(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    r"""
    释义
        返回给定维度 *dim* 中输入张量的每一行的平均值。 如果 *dim* 是维度列表，则对列表中所有维度进行归约。
        如果 *dim* 等于 ``None``，将对所有元素计算平均值。

        如果 *keepdim* 为 ``True``, 则除了在维度 *dim* 上，输出张量大小与输入张量相同。输出张量在维度 *dim* 上大小为1。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** 或者 **list(int)** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
        - *dtype* ( **Dtype**, 可选) : 输出的数据类型
    C API
        :guilabel:`diopiMean`
    """
    assert isinstance(dim, (int, list)) or dim is None,\
        "dim should be int or list or None"

    dim, out = reduce_op_process(input, dim, keepdim, dtype)
    func = check_function("diopiMean")
    dim1 = Sizes(tuple(dim))
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim1, c_int32(dtype.value))
    check_returncode(ret)
    return out


def std(input, unbiased=False, dim=None, keepdim=False) -> Tensor:
    r"""
    释义
        如果 *unbiased* 为 ``True``，则将使用 *Bessel* 校正。 否则，将直接计算样本偏差，而不进行任何校正。
        如果 *dim* 等于 ``None``，将对所有元素计算标准差。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *unbiased* ( **bool** ) : 是否使用Bessel校正
        - *dim* ( **int** 或者 **list(int)** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiStd`
    """
    assert isinstance(dim, (int, list)) or dim is None,\
        "dim should be int or list or None"

    dim, out = reduce_op_process(input, dim, keepdim)
    dim1 = Sizes(tuple(dim))
    func = check_function("diopiStd")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim1, unbiased)
    check_returncode(ret)
    return out


def min(input, dim=0, keepdim=False) -> Tensor:
    r"""
    释义
        返回给定维度 *dim* 中输入张量的每一行的最小值。 如果 *dim* 是维度列表, 则对所有维度进行归约。

        如果 *keepdim* 为 ``True``, 则除了在维度 *dim* 上，输出张量大小与输入张量相同。输出张量在维度 *dim* 上大小为1。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiMin`
    """
    assert isinstance(dim, int), "dim should be int"

    dim, out = reduce_op_process(input, dim, keepdim)
    indices = Tensor(out.size(), Dtype.int64)
    func = check_function("diopiMin")

    ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
               input.tensor_handle, dim)
    check_returncode(ret)
    Res = namedtuple('Res', ['values', 'indices'])
    output = Res(out, indices)
    return output


def convert_reduction(name):
    if name == 'none':
        return 0
    if name == 'mean':
        return 1
    if name == "sum":
        return 2
    return 3


def binary_cross_entropy_with_logits(input, target, weight=None,
                                     reduction='mean', pos_weight=None):
    r"""
    释义
        计算目标 *target* 和输入 *input* 之间的二值交叉函数:
        这种损失将 *Sigmoid* 层和 *BCELoss* 组合在一个函数中, 若 *reduction* 为 **None** :

        .. math::
            \ell(x, y)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\prime}, \quad l_{n}=-w_{n}\left[y_{n}
            \cdot \log \sigma\left(x_{n}\right)+\left(1-y_{n}\right) \cdot
            \log \left(1-\sigma\left(x_{n}\right)\right)\right]

        其中 *N* 表示 *batch_size* 。此外, 若 *reduction* 不为 **None**, 则:

        .. math::
            \ell(x, y)=\left\{\begin{array}{ll}\operatorname{mean}(L), & \text { if reduction = 'mean' }
            \\\operatorname{sum}(L), & \text { if reduction }=\text { 'sum' }\end{array}\right.

        .. hint:: 可以通过向正例添加权重来权衡召回率和精度。 在多标签分类的情况下, 损失可以描述为:

            .. math::
                \ell_c(x, y) = L_c = \{l_{1,c},\dots,l_{N,c}\}^\top, \quad

            .. math::
                l_{n,c} = - w_{n,c} \left[ p_c y_{n,c} \cdot \log \sigma(x_{n,c})
                + (1 - y_{n,c}) \cdot \log (1 - \sigma(x_{n,c})) \right]

            其中, *c* 表示类别数量( *c* >1 : 多分类, *c* =1 : 单标签二分类)，:math:`p_c` 表示类别 *c* 的权重。

            :math:`p_c` > 1 增加召回率，:math:`p_c` < 1 增加精度。

            例如，一个数据集在某单个类别中有100个正样本，300个负样本，那么该类别的 *pos_weight* 应该等于
            :math:`\frac{300}{100}=3`。损失函数将表现得好像数据集中包含 :math:`3\times 100=300` 正样本一样。
    参数
        - *input* ( **Tensor** ) : 任意形状的输入张量, 表示未归一化分数， 通常也称作 *logits*
        - *target* ( **Tensor** ) : 与输入张量维度相同, 且其取值范围在[0,1]
        - *weight* ( **Tensor** , 可选): 手动设置的调整权重, 可自动扩展以适配输入张量的形状
        - *reduction* ( **string** , 可选) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean* 。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum``: 输出求和。其默认值为 *mean*
        - *pos_weight* ( **Tensor** , 可选 ) : 正样本的权重, 其长度必须与类别数量相同
    C API
        :guilabel:`diopiBCEWithLogits`
    """
    assert input.size() == target.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'
    if pos_weight is not None:
        assert isinstance(pos_weight, Tensor), \
            'pos_weigth must be a Tensor'
        pos_weight = pos_weight.tensor_handle
    else:
        # represent pos_weight = None by pass a nullptr
        pos_weight = c_void_p()

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiBCEWithLogits")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, pos_weight, reduction_mode)
    check_returncode(ret)
    return out


def cross_entropy(input, target, weight=None, ignore_index=- 100,
                  reduction='mean', label_smoothing=0.0):
    r"""
    释义
        计算目标和输入 *logits* 之间的交叉熵损失, 由于该方法能够手动设置各个类别的权重, 常用于具有 *C* 类的多类别任务的训练。

        其中输入张量 *input* 必须为原始的, 未被正则化的类别分数。 且必须至少有一个维度为 *C* 。

        目标张量 *target* 也必须满足如下条件 **之一** :

            - *target* 为类别索引，取值范围为[0, *C* ), 其中 *C* 表示类别的数量, 当 *reduction* 为 *none* 时, 其损失计算方式如下:

            .. math::
                \ell(x, y)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top}, \quad

            .. math::
                l_{n}=-w_{y_{n}} \log \frac{\exp \left(x_{n, y_{n}}\right)}{\sum_{c=1}^{C} \exp \left(x_{n, c}\right)}
                \cdot 1\left\{y_{n} \neq \text { ignore_index }\right\}

            其中 :math:`x` 表示输入, :math:`y` 表示目标，:math:`w` 是权重, :math:`C` 是类别数量，:math:`N` 等于输入大小除以类别总数。
            此外, 若reduction不为 *none* , 则:

            .. math::
                \ell(x, y)=\left\{\begin{array}{ll}\sum_{n=1}^{N} \frac{1}{\sum_{n=1}^{N} w_{y_{n}}
                \cdot 1\left\{y_{n} \neq \text { ignore_index }\right\}} l_{n}, & \text { if reduction }=\text { 'mean' }
                \\\sum_{n=1}^{N} l_{n}, & \text { if reduction }=\text { 'sum' }\end{array}\right.

            - *target* 为类别概率, 常用于多标签分类任务如混合标签，标签平滑等情况。当 *reduction* 为 *none* 时, 其损失计算方式如下:

            .. math::
                \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
                l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

            其中 :math:`x` 表示输入, :math:`y` 表示目标，:math:`w` 是权重, :math:`C` 是类别数量，:math:`N` 等于输入大小除以类别总数。
            此外, 若 *reduction* 不为 *none* , 则:

            .. math::
                \ell(x, y) = \begin{cases}\frac{\sum_{n=1}^N l_n}{N}, &\text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &\text{if reduction} = \text{`sum'.}\end{cases}
    参数
        - *input* ( **Tensor** ) : 输入张量, 表示未归一化分数，通常也称作 *logits*
        - *target* ( **Tensor** ) : 目标张量, 表示真值类别索引或者类别概率
        - *weight* ( **Tensor**, 可选) : 对每个类别手动设置的调整权重, 若非空则其大小为 *C*
        - *ignore_index* ( **int**, 可选) : 指定一个被忽略且不影响输入梯度的目标值, 当目标包含类别索引时才能使用该参数, 其默认值为 -100
        - *reduction* ( **string** , 可选) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean*。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum`` : 输出求和。 其默认值为 *mean*
        - *label_smoothing* ( **float** , 可选) : 其取值范围为 [0.0, 1.0] 的浮点数, 指定计算损失时的平滑量，其中 0.0 表示不平滑。其默认值为 0.0
    形状
        - *input* : 形状为 :math:`(C)`, :math:`(N, C)` 或者 :math:`(N, C, d_1, d_2, ..., d_K)` 其中 :math:`K \geq 1`
          表示 K-维损失
        - *target* : 如果为类别索引, 形状为 :math:`()`, :math:`(N)` 或者 :math:`(N, d_1, d_2, ..., d_K)` 其中 :math:`K \geq 1`
          表示 K-维损失。值范围为 :math:`[0, C)`。如果为类别概率，形状和 *input* 相同，值范围为 :math:`[0, 1]`
        - 输出 : 如果 *reduction* 为 *none*, 和 *target* 形状相同。否则为标量

        其中, N 表示批大小， C 表示类别数量
    C API
        :guilabel:`diopiCrossEntropyLoss`
    """
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    if reduction == 'none':
        out = raw_like(target)
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiCrossEntropyLoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, reduction_mode,
               ignore_index, label_smoothing)
    check_returncode(ret)
    return out


def mse_loss(input, target, reduction='mean'):
    r"""
    释义
        计算输入张量 *input* 与 目标张量 *target* 之间每个对应元素的均方误差。
        当 *reduction* 为 *none* 时，损失函数描述为:

        .. math::
            \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
            l_n = \left( x_n - y_n \right)^2,

        其中 *N* 表示 *batch_size* 。当 *reduction* 不为 *none* 时，损失函数描述为:

        .. math::
            \ell(x, y) =
            \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
            \end{cases}

        其中 :math:`x` 与 :math:`y` 为任意维度的张量。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *target* ( **Tensor** ) : 目标张量
        - *reduction* ( **string** , 可选) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean*。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum`` : 输出求和。 默认值为 mean
    C API
        :guilabel:`diopiMSELoss`
    """
    assert input.shape() == target.shape(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if reduction == 'none':
        out = raw_like(input)
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiMSELoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, reduction_mode)
    check_returncode(ret)
    return out


def conv2d(input, weight, bias=None, stride=1,
           padding=0, dilation=1, groups=1) -> Tensor:
    r"""
    释义
        对输入张量 *input* 应用2D卷积操作。该操作定义为:

        .. math::
            \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
            \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

        其中 :math:`\star` 表示2D卷积操作， N表示批大小，C表示通道数， H和W分别表示输入像素的宽度和高度
    参数
        - *input* ( **Tensor** ) : 输入张量，其形状为 :math:`(\text{minibatch}, \text{in_channels}, iH, iW)`
        - *weight* ( **Tensor** ) : 卷积核，其形状为 :math:`(\text{out_channels}, \text{in_channels/groups}, kH, kW)`
        - *bias* ( **Tensor** ) : 偏置项，其形状为 :math:`(\text{out_channels})`，默认为 *None*
        - *stride* ( **number** 或者 **tuple** ) : 卷积核的步长，默认为 1
        - *padding* ( **string** 或者 **number** 或者 **tuple** ) : 输入的每一侧的隐式填充，其值可以为 "valid"、"same"、数字
          或者元组 :math:`(padH, padW)` 。当其值为 ``"valid"`` 时等同于无填充，当其值为 ``"same"`` 时，卷积输出维度与输入相同。默认值为 0
        - *dilation* ( **number** 或者 **tuple** ) : 卷积核元素之间的步长，可以为单个数字或者 :math:`(sH, sW)` 元组，默认值为 1
        - *groups* ( **number** ) : 输入张量 *input* 被分组的组数，该值必须被 *in_channel* 整除，默认值为 1
    C API
        :guilabel:`diopiConvolution2d`
    """
    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
        bias = bias.tensor_handle
    else:
        bias = c_void_p()

    sizeI = input.size()
    sizeW = list(weight.size())
    assert len(sizeI) == 4 and len(sizeW) == 4,\
        'input and weight must be 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    sizeO.append(sizeW[0])

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    for i in range(-2, 0):
        # equivalent kernel size
        sizeW[i] += (sizeW[i] - 1) * (dilation[i] - 1)
        sizeO.append(int((sizeI[i] - sizeW[i] + 2*padding[i])/stride[i]) + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    dilation = Sizes(tuple(dilation))

    out = Tensor(sizeO, input.get_dtype())
    func = check_function("diopiConvolution2d")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               weight.tensor_handle, bias, stride, padding, dilation, groups)
    check_returncode(ret)
    return out


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True, divisor_override=None) -> Tensor:
    r"""
    释义
        在 :math:`(kH, kW)` 区域中按步长 :math:`(sH, sW)` 步长应用 2D 平均池化操作。 输出张量的通道数与输入张量的通道数相同。
        其操作描述为:

        .. math::
            out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
            input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)
    参数
        - *input* ( **Tensor** ) : 输入张量，其形状为 :math:`(\text{minibatch} , \text{in_channels} , iH , iW)`
        - *kernel_size* ( **number** 或者 **tuple** ) : 池化区域的大小。 可以是单个数字或元组 :math:`(kH, kW)`
        - *stride* ( **number** 或者 **tuple** ) : 池化操作的步长。 可以是单个数字或元组 :math:`(sH, sW)`。 默认值为 kernel_size
        - *padding* ( **number** 或者 **tuple** ) : 对输入张量 *input* 四周进行隐式零填充。 可以是单个数字或元组 :math:`(padH, padW)`。 默认值为 0
        - *ceil_mode* ( **bool** ) : 当为 ``True`` 时，将在公式中使用 *ceil* 而不是 *floor* 来计算输出形状。 默认值为 False
        - *count_include_pad* ( **bool** ) : 当为 ``True`` 时，将在均值计算中包含零填充。 默认值为 True
        - *divisor_override* ( **number** ) : 如果指定，它将用作在计算平均池化时的除数，否则默认除以池化区元素总数。 默认值为 *None*
    C API
        :guilabel:`diopiAvgPool2d`
    """
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    for i in range(-2, 0):
        if ceil_mode:
            sizeO.append(math.ceil((sizeI[i] - kernel_size[i] + 2*padding[i])/stride[i]) + 1)
        else:
            sizeO.append(math.floor((sizeI[i] - kernel_size[i] + 2*padding[i])/stride[i]) + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    out = Tensor(sizeO, input.get_dtype())

    func = check_function("diopiAvgPool2d")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               kernel_size, stride, padding, ceil_mode, count_include_pad,
               byref(divisor_override))
    check_returncode(ret)
    return out


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False) -> Tensor:
    r"""
    释义
        对输入张量 *input* 应用2D 最大值池化。
        若输入张量的维度为 :math:`(N, C, H, W)` ，输出张量的维度为 :math:`(N, C, H_{out}, W_{out})` ，
        且池化区域的大小为 :math:`(kH, kW)` 池化操作定义如下:

        .. math::
            \begin{aligned}
                out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                        & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                    \text{stride[1]} \times w + n)
            \end{aligned}

        其中，若 *padding* 不为零， 则输入张量四周将用负无穷进行填充。 *dilation* 表示滑动窗口中元素之间的步长。
    参数
        - *input* ( **Tensor** ) : 输入张量，其维度为 :math:`(\text{minibatch} , \text{in_channels} , iH , iW)`,
          minibatch这个维度是可选的
        - *kernel_size* ( **number** 或者 **tuple** ) : 池化区域的大小， 可以是单个数字或元组 :math:`(kH, kW)`
        - *stride* ( **number** 或者 **tuple** ) : 池化操作的步长。 可以是单个数字或元组 :math:`(sH, sW)`。 默认值为 kernel_size
        - *padding* ( **number** ) : 在输入张量两侧隐式负无穷大填充，其值必须 >= 0 且 <= kernel_size / 2
        - *dilation* ( **number** ) : 滑动窗口内元素之间的步长，其值必须大于 0
        - *ceil_mode* ( **bool** ) : 如果为 ``True`` ，将使用向上取整而不是默认的向下取整来计算输出形状。 这确保了输入张量中的每个元素都被滑动窗口覆盖。
        - *return_indices* ( **bool** ) : 如果为 ``True`` ，将返回所有滑动窗口产生的最大值的位置索引，该结果将后续会被应用于反池化。默认值为 False
    C API
        :guilabel:`diopiMaxPool2d` :guilabel:`diopiMaxPool2dWithIndices`
    """
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    for i in range(-2, 0):
        if ceil_mode:
            sizeO.append(math.ceil((sizeI[i] - kernel_size[i] + 2*padding[i])/stride[i]) + 1)
        else:
            sizeO.append(math.floor((sizeI[i] - kernel_size[i] + 2*padding[i])/stride[i]) + 1)

    stride = Sizes(tuple(stride))
    padding = Sizes(tuple(padding))
    kernel_size = Sizes(tuple(kernel_size))
    dilation = Sizes(tuple(dilation))
    out = Tensor(sizeO, input.get_dtype())
    if not return_indices:
        func = check_function("diopiMaxPool2d")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, kernel_size,
                   stride, padding, dilation, ceil_mode)
        check_returncode(ret)
        return out
    else:
        func = check_function("diopiMaxPool2dWithIndices")
        indices = Tensor(sizeO, Dtype.int64)
        ret = func(input.context_handle, out.tensor_handle,
                   indices.tensor_handle, input.tensor_handle,
                   kernel_size, stride, padding, dilation, ceil_mode)
        check_returncode(ret)
        return out, indices


def adaptive_avg_pool2d(input, output_size):
    r"""
    释义
        对输入张量 *input* 做 2D 自适应平均池化。对于任意大小的 *input* ，
        其输出张量大小为 *output_size* 。此外，输出张量的通道数与输入张量相同。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *output_size* ( **number** 或者 **tuple** ) : 输出张量的大小，当为单个数字 :math:`H_{out}` 时，
          表示输出张量大小为 :math:`H_{out} \times H_{out}`
    形状
        - *input*: :math:`(N, C, H_{in}, W_{in})` 或者 :math:`(C, H_{in}, W_{in})`
        - *输出*: :math:`(N, C, H_{out}, W_{out})` 或者 :math:`(C, H_{out}, W_{out})`, 其中
          :math:`(H_{out}, W_{out})=\text{output_size}`
    C API
        :guilabel:`diopiAaptiveAvgPool2d`
    """
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    for i in range(-2, 0):
        if output_size[i] is None:
            sizeO.append(sizeI[i])

    out = Tensor(sizeO, input.get_dtype())
    output_size = Sizes((sizeO[-2], sizeO[-1]))

    func = check_function("diopiAaptiveAvgPool2d")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, output_size)
    check_returncode(ret)
    return out


def adaptive_max_pool2d(input, output_size, return_indices=False):
    r"""
    释义
        对输入张量 *input* 做 2D 自适应最大值池化。对于任意大小的 *input* ，
        其输出张量大小为 *output_size* 。此外，输出张量的通道数与输入张量相同。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *output_size* ( **number** 或者 **tuple** ) : 输出张量的大小，当为单个数字 :math:`H_{out}` 时，
          表示输出张量大小为 :math:`H_{out} \times H_{out}`
        - *return_indices* ( **bool** ) : 如果为 ``True`` ，将返回所有滑动窗口产生的最大值的位置索引。默认值为 False
    形状
        - *input*: :math:`(N, C, H_{in}, W_{in})` 或者 :math:`(C, H_{in}, W_{in})`
        - *输出*: :math:`(N, C, H_{out}, W_{out})` 或者 :math:`(C, H_{out}, W_{out})`, 其中
          :math:`(H_{out}, W_{out})=\text{output_size}`
    C API
        :guilabel:`diopiAdaptiveMaxPool2d` :guilabel:`diopiAdaptiveMaxPool2dWithIndices`
    """
    sizeI = input.size()
    assert len(sizeI) == 4 or len(sizeI) == 3,\
        'input must be 3d or 4d tensors'

    sizeO = []
    sizeO.append(sizeI[0])
    if len(sizeI) == 4:
        sizeO.append(sizeI[1])

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    for i in range(-2, 0):
        sizeO.append(output_size[i])

    out = Tensor(sizeO, input.get_dtype())
    output_size = Sizes(tuple(output_size))

    if return_indices:
        func = check_function("diopiAaptiveMaxPool2dWithIndices")
        indices = Tensor(sizeO, Dtype.int64)
        ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
                   input.tensor_handle, output_size)
        check_returncode(ret)
        return out, indices
    else:
        func = check_function("diopiAaptiveMaxPool2d")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, output_size)
    check_returncode(ret)
    return out


def dropout(input, p=0.5, training=True, inplace=False):
    r"""
    释义
        在训练模式下， 基于伯努利分布抽样，以概率 p 对输入张量 *input* 的值随机置零。
        此外在训练过程中，输出张量将以因子 :math:`\frac{1}{1-p}` 进行缩放。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *p* ( **float** ) : 输入张量元素被置零的概率，默认值为 0.5
        - *training* ( **bool** ) : 是否为训练模式，默认为: True 。当为 ``False`` 时，*dropout* 将不会执行
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiDropout` :guilabel:`diopiDropoutInp`
    """
    call = "Dropout"
    args = 'input.context_handle, '
    if inplace:
        out = input
        call = call + 'Inp'
    else:
        out = raw_like(input)
        args = args + 'out.tensor_handle, '

    args = args + "input.tensor_handle, p, train"
    func = check_function(call)
    ret = func(eval(f'{args}'))
    check_returncode(ret)
    return out


def index_select(input, dim, index) -> Tensor:
    r"""
    释义
        使用索引 *index* 中的条目，沿着维度 *dim* 的方向，对输入张量 *input* 进行数据索引，
        将索引到的数据作为一个新的 *tensor* 进行返回。其中，返回张量与输入张量有相同的维数，
        在 *dim* 方向，输出张量维度大小与索引长度相同，其他维度大小与输入张量相同。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 取索引数据所在的维度
        - *index* ( **Tensor** ) : 包含索引下标的一维张量
    C API
        :guilabel:`diopiIndexSelect`
    """
    sizeI = list(input.size())
    sizeI[dim] = index.numel()
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiIndexSelect")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, dim, index.tensor_handle)
    check_returncode(ret)
    return out


def select(input, dim, index) -> Tensor:
    r"""
    释义
        在给定索引 *index* 处沿选定维度 *dim* 对输入张量 *input* 进行切片，并返回切片到的数据。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 取索引数据所在的维度
        - *index* ( **int** ) : 索引下标
    C API
        :guilabel:`diopiSelect` :guilabel:`diopiSelectCopy`
    """
    sizeI = list(input.size())
    sizeI[dim] = 1
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiSelectCopy")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, dim, index)
    check_returncode(ret)
    return out


def masked_scatter(input, mask, source) -> Tensor:
    r"""
    释义
        在掩码 *mask* 为 True 的位置将元素从 *source* 复制到 *self* 。
        掩码的形状与张量 *self* 必须是可广播的。*source* 中的元素至少应与掩码中的元素数量一样多。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *mask* ( **BoolTensor** ) : 布尔类型的掩码
        - *source* （ **Tensor** ) : 被复制的张量
    C API
        :guilabel:`diopiMaskedScatter`
    """
    assert mask.get_dtype() == Dtype.bool, \
        "mask must be bool tensor"
    out = raw_like(input)

    func = check_function("MaskedScatter")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               mask.tensor_handle, source.tensor_handle)
    check_returncode(ret)
    return out


def nonzero(input):
    # note: pytorch(1.12) has argument 'as_tuple' to return multiple 1d tensor
    r"""
    释义
        返回一个二维张量，其中每一行代表 *input* 中一个非零元素的索引坐标。
        如果 *input* 有 :math:`n` 维, 那么输出索引张量的形状为
        :math:`(z \times n)`, 其中 :math:`z` 等于 *input* 中非零元素的总数。
    参数
        - *input* ( **Tensor** ) : 输入张量
    C API
        :guilabel:`diopiNonzero`
    """
    out = Tensor((), Dtype.int64)
    func = check_function("diopiNonzero")
    ret = func(input.context_handle, byref(out.tensor_handle),
               input.tensor_handle)
    check_returncode(ret)
    return out


def linear(input, weight, bias=None) -> Tensor:
    r"""
    释义
        对 *input* 应用如下线性变换。:math:`y = x A^T + b`。
    参数
        - *input* ( **Tensor** ) : 输入张量 :math:`x`，形状为 *( # , in_features)* ， #表示任意数量的维度，包括可以为 *None*
        - *weight* ( **Tensor** ) : 权重项 :math:`A`，形状为 *(out_features, in_features)* 或者 *(in_feature)*
        - *bias* （ **Tensor** ) : 偏置项 :math:`b`，形状为 *(out_features)* 或者 *()*
    返回值
        输出张量形状为 *( #, out_features)* 或者 *( #)*， 取决于权重的形状
    C API
        :guilabel:`diopiLinear`
    """
    if bias is not None:
        assert isinstance(bias, Tensor), \
            'bias must be a Tensor'
        bias = bias.tensor_handle
    else:
        bias = c_void_p()

    sizeI = list(input.size())
    sizeW = list(weight.size())
    sizeI[-1] = sizeW[-2] if len(sizeW) == 2 else 1
    out = Tensor(sizeI, input.get_dtype())
    func = check_function("diopiLinear")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               weight.tensor_handle, bias)
    check_returncode(ret)
    return out


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    # todo： padding_idx和scale_grad_by_freq用于autograd时保存的变量 ？
    r"""
    释义
        一个简单的查找表，用于在固定的字典查找固定大小的嵌入向量表示。

        该功能通常用于使用索引检索词嵌入。输入是索引列表和嵌入矩阵，输出是相应的词嵌入。
    参数
        - *input* ( **LongTensor** ) : 包含嵌入矩阵索引的张量
        - *weight* ( **Tensor** ) :  行数等于最大可能索引 + 1 的嵌入矩阵，列数等于嵌入向量大小
        - *padding_idx* ( **int**，可选) :  如果指定，则 *padding_idx* 处的条目不会影响梯度。
          因此，*padding_idx* 处的嵌入向量在训练期间不会更新，即它保持是一个固定的 “*pad*”
        - *max_norm* ( **float**，可选) : 如果给定，则范数大于 *max_norm* 的嵌入向量被重新归一化为具有范数 *max_norm*。
          注意: 这将就地修改权重 *weight*
        - *norm_type* ( **float**，可选) : 为 *max_norm* 选项计算 *p-norm* 时的 *p*，默认为2
        - *scale_grad_by_freq* ( **bool**，可选) : 如果给定，这将通过小批量中单词频率的倒数来缩放梯度，默认为 *False*
        - *sparse* ( **bool**，可选) : 如果为 ``True``，权重将是一个稀疏张量
    形状
        - *input* : 包含要提取的索引的任意形状的 *LongTensor*
        - *weight* :  形状为 *(V, embedding_dim)* 的浮点型嵌入矩阵，其中 V = 最大索引 + 1，*embedding_dim* = 嵌入向量大小
        - 输出 : 形状为 *(#, embedding_dim)* , 其中 # 是输入的形状
    C API
        :guilabel:`diopiEmbedding`
    """
    sizeI = list(input.size())
    sizeW = weight.size()
    sizeI.append(sizeW[-1])
    out = Tensor(sizeI, weight.get_dtype())

    # note: scale_grad_by_freq and sparse are useless during forward phase
    func = check_function("diopiEmbedding")
    ret = func(input.context_handle, out.tensor_handle, weight.tensor_handle,
               input.tensor_handle, padding_idx, scale_grad_by_freq, sparse)
    check_returncode(ret)
    return out


def tril(input, diagonal=0) -> Tensor:
    r"""
    释义
        返回输入矩阵（二维张量）或批处理矩阵 ( *batch of matrices*) 的下三角部分，输出的结果张量其他元素设置为 0。

        矩阵的下三角部分定义为对角线上及对角线以下的元素。

        参数 *diagonal* 控制要考虑的对角线。如果 *diagonal* 等于 ``0``，则保留主对角线上及以下的所有元素。
        正值的 *diagonal* 包括在主对角线上相同数量的对角线，同样，负值排除在主对角线下方相同数量的对角线。
        主对角线是以下索引的集合 :math:`\lbrace (i, i) \rbrace` for :math:`i \in [0, \min\{d_{1}, d_{2}\} - 1]` 其中
        :math:`d_{1}, d_{2}` 是矩阵的维度。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *diagonal* ( **int**， 可选) :  考虑的对角线
    C API
        :guilabel:`diopiETril`
    """
    out = raw_like(input)
    func = check_function("diopiTril")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, diagonal)
    check_returncode(ret)
    return out


def cat(tensors, dim=0) -> Tensor:
    r"""
    释义
        沿给定的维度 *dim* 拼接序列 *tensors* 中的张量。

        所有的张量必须有相同的形状 (在拼接维度 *dim* 上例外）或者为空。
    参数
        - *tensors* ( **多个tensor组成的序列** ) : 相同数据类型张量的任何 *python* 序列
        - *dim* ( **int** ) :  插入的维度, 值必须在 0 和 张量的维度数量之间
    C API
        :guilabel:`diopiCat`
    """
    assert isinstance(tensors, (list, tuple)),\
        "tensors must be a list or tuple"
    insNum = len(tensors)
    sum = 0
    c_tensors = []
    for tensor in tensors:
        sizeI = list(tensor.size())
        sum += sizeI[dim]
        c_tensors.append(tensor.tensor_handle)

    sizeI[dim] = sum
    out = Tensor(sizeI, tensors[0].get_dtype())
    func = check_function("diopiCat")
    ret = func(input.context_handle, out.tensor_handle,
               byref(c_tensors), insNum, dim)
    check_returncode(ret)
    return out


def stack(tensors, dim=0) -> Tensor:
    r"""
    释义
        沿给定的维度 *dim* 拼接序列 *tensors* 中的张量。

        所有的张量必须有相同的形状。
    参数
        - *tensors* ( **多个tensor组成的序列** ) : 将被拼接的张量序列
        - *dim* ( **int** ) :  插入的维度。值必须在 0 和 张量的维度数量之间
    C API
        :guilabel:`diopiStack`
    """
    assert isinstance(tensors, (list, tuple)),\
        "tensors must be a list or tuple"
    insNum = len(tensors)
    sizeI = list(tensors[0].size())
    sum = insNum * sizeI[dim]

    c_tensors = []
    for tensor in tensors:
        c_tensors.append(tensor.tensor_handle)

    sizeI[dim] = sum
    out = Tensor(sizeI, tensors[0].get_dtype())
    func = check_function("diopiStack")
    ret = func(input.context_handle, out.tensor_handle,
               byref(c_tensors), insNum, dim)
    check_returncode(ret)
    return out


def sort(input, dim=- 1, descending=False, stable=False):
    r"""
    释义
        对 *input* 沿给定的维度 *dim* 上的元素进行升序排序。

        如果 *dim* 没有给定，默认选择 *input* 的最后一个维度。

        如果 *descending* 设为 ``True``，元素将以降序排序。

        如果布尔选项 *stable* 设为 ``True``， 排序算法是稳定的， 对相同的元素维持原有顺序。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int**，可选) :  进行排序的维度 *dim*
        - *descending* ( **bool**，可选) :  用以决定排序顺序（升序或降序）
        - *stable* ( **bool**，可选) :  用以选择稳定的排序算法，稳定的排序算法保证相同元素的顺序维持不变
    返回值
        返回值是一个 *（Tensor， LongTensor）* 的元组，其含义为 *（values， indices）*，
        其中 *values* 是排序后的值，*indices* 是在维度 *dim* 上的位置索引。
    C API
        :guilabel:`diopiSort`
    """
    vals = raw_like(input)
    sizeI = input.size()
    indices = Tensor(sizeI, Dtype.int64)

    func = check_function("diopiSort")
    ret = func(input.context_handle, vals.tensor_handle, indices.tensor_handle,
               input.tensor_handle, dim, descending, byref(stable))
    check_returncode(ret)
    return vals, indices


def topk(input, k, dim=-1, largest=True, sorted=True):
    r"""
    释义
        返回 *input* 沿给定的维度 *dim* 的前 *k* 个最大的值。

        如果 *dim* 没有给定，默认选择 *input* 的最后一个维度。

        如果 *largest* 设为 ``False``，将会返回前 *k* 个最小的值。

        如果布尔选项 *sorted* 设为 ``True``， 将会返回的排序后的前 *k* 个值。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *k* ( **int** ) :  *”top-k“* 中的k
        - *dim* ( **int**，可选) :  进行排序的维度 *dim*
        - *largest* ( **bool**，可选) :  用以决定返回前 *k* 个最大的 或者 最小的元素
        - *sorted* ( **bool**，可选) :  用以决定返回的前 *k* 个值是否有序
    返回值
        返回值是一个 （Tensor， LongTensor） 的元组，其含义为 （values， indices），
        其中 *values* 是前 *k* 个值，*indices* 是在维度 *dim* 上的位置索引。
    C API
        :guilabel:`diopiTopk`
    """
    sizeI = list(input.size())
    sizeI[dim] = k
    values = Tensor(sizeI, input.get_dtype())
    indices = Tensor(sizeI, Dtype.int64)

    func = check_function("diopiTopk")
    ret = func(input.context_handle, values.tensor_handle,
               indices.tensor_handle, input.tensor_handle,
               k, dim, largest, sorted)
    check_returncode(ret)
    return values, indices


def transpose(input, dim0, dim1) -> Tensor:
    r"""
    释义
        返回 *input* 的转置版本，在给定的维度 *dim0* 和 *dim1* 上交换。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim0* ( **int** ) :  将被转置的第一个维度
        - *dim1* ( **int** ) :  将被转置的第二个维度
    返回值
        *input* 的形状为 *(#, dim0, #, dim1, #)*, 返回张量形状为 *(#, dim1, #, dim0, #)*。
    C API
        :guilabel:`diopiTranspose`
    """
    sizeI = list(input.size())
    tmp = sizeI[dim0]
    sizeI[dim0] = sizeI[dim1]
    sizeI[dim1] = tmp
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiTranspose")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, dim0, dim1)
    check_returncode(ret)
    return out


def one_hot(input, num_classes=- 1):
    r"""
    释义
        对于输入形状为 *(\#)* 的长整型张量，将返回形状为 (\#, num_classes) 的张量。
        返回张量在最后一个维度上和输入类别值对应位置的值为1, 其余均为0。
    参数
        - *input* ( **LongTensor** ) : 任何形状的类别值
        - *num_classes* ( **int** ) :  类别的总数。如果设为 ``-1``, 类别的总数将会被推断为输入张量的最大类别值加上1
    返回值
        比输入多一个维度的长整型张量, 在输入类别值对应的位置值为1, 其余为0。
    C API
        :guilabel:`diopiOneHot`
    """
    assert num_classes == -1 or num_classes > 0,\
        "num_classes must be -1 or >0"
    sizeI = input.size()
    # todo: can not have the shape of output, out should be a pointer
    if num_classes == -1:
        out = Tensor((1, ), Dtype.int64)
    else:
        sizeI += (num_classes, )
        out = Tensor(sizeI, Dtype.int64)

    func = check_function("diopiOneHot")
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, num_classes)
    check_returncode(ret)
    return out


def split(tensor, split_size_or_sections, dim=0):
    r"""
    释义
        将张量拆分为块。每个块都是原始张量的拷贝。

        如果 *split_size_or_sections* 是整型, *tensor* 将会尽可能被分为相等大小的块。
        如果输入张量在维度 *dim* 的大小不能被 *split_size* 整除, 最后的块可能会比其他块更小。

        如果 *split_size_or_sections* 是整数列表, *tensor* 将被分割成数量为
        ``len(split_size_or_sections)`` 的块, 每个块在维度 *tensor* 上的大小将通过 *split_size_or_sections* 指定。
    参数
        - *tensor* ( **Tensor** ) : 将被分割的张量
        - *split_size_or_sections* ( **int** 或者 **list(int)** ) : 每个块的大小 或者 每个块大小的列表
        - *dim* ( **int** ) : 沿着该维度分割张量
    C API
        :guilabel:`diopiSplitWithSizes`
    """
    assert isinstance(split_size_or_sections, (int, list)),\
        "split_size_or_sections must be int or list"
    sizeI = list(tensor.size())
    sum = sizeI[dim]
    outs = []
    idx = 0
    splitSizes = ()
    is_int = isinstance(split_size_or_sections, int)

    while sum > 0:
        sizeI[dim] = split_size_or_sections if is_int else\
                     split_size_or_sections[idx]
        sizeI[dim] = sizeI[dim] if sum > sizeI[dim] else sum
        idx += 1
        sum -= sizeI[dim]
        splitSizes += (sizeI[dim], )
        out = Tensor(sizeI, Dtype.int64)
        outs.append(out)

    splitSizes = Sizes(splitSizes)
    assert sum == 0,\
        "split_size_or_sections should be compatible with tensor shape"
    func = check_function("diopiSplitWithSizes")
    ret = func(input.context_handle, byref(outs), idx,
               tensor.tensor_handle, byref(splitSizes), dim)
    check_returncode(ret)
    return outs


def pow(input, exponent) -> Tensor:
    r"""
    释义
        对 *input* 中每一个元素进行指数为 *exponent* 的幂运算。
        *exponent* 可以是一个浮点数或者是一个与 *input* 拥有相同元素个数的张量。

        如果 *exponent* 是一个标量, pow实现为:

        .. math::
           \text{out}_i = x_i ^ \text{exponent}

        如果 *exponent* 是一个张量, pow实现为:

        .. math::
            \text{out}_i = x_i ^ {\text{exponent}_i}

        如果 *exponent* 是一个张量, *input* 和 *exponent* 的形状必须是可以 **广播** 的。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *exponent* ( **float** 或者 **Tensor**  ) : 指数值
    C API
        :guilabel:`diopiPowScalar` :guilabel:`diopiPow` :guilabel:`diopiPowTensor`
    """
    if not isinstance(input, Tensor):
        assert isinstance(exponent, Tensor),\
            "exponent must be tensor when input is scalar"
        func = check_function("diopiPowScalar")
        # todo: return type = input type or float ?
        out = raw_like(exponent)
        if isinstance(input, int):
            input = byref(Scalar(Dtype.int64, input))
        else:
            input = byref(Scalar(Dtype.float64, input))
        ret = func(exponent.context_handle, out.tensor_handle, input, exponent.tensor_handle)
    elif not isinstance(exponent, Tensor):
        assert isinstance(input, Tensor),\
            "input must be tensor when exponent is scalar"
        func = check_function("diopiPow")
        out = raw_like(input)
        if isinstance(exponent, int):
            exponent = byref(Scalar(Dtype.int64, exponent))
        else:
            exponent = byref(Scalar(Dtype.float64, exponent))
        ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, exponent)
    else:
        sizeI = list(input.size())
        sizeE = list(exponent.size())
        sizeO = broadcast_out_size(sizeI, sizeE)
        out = Tensor(sizeO, input.get_dtype())

        func = check_function("diopiPowTensor")
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, exponent.tensor_handle)
    check_returncode(ret)
    return out


def where(condition, input, other) -> Tensor:
    # todo: add scalar version for pytorch 1.12
    r"""
    释义
        依据条件 *condition* 返回包含 *input* 或者 *other* 元素的张量 *output* ，其大小由 *condition* 、 *input* 及 *other* 广播得到:

        .. math::
            \text { out }_{i}=\left\{\begin{array}{ll}
            \mathbf{input}_{i} & \text { if condition } \\
            \mathbf{other}_{i} & \text { otherwise }
            \end{array}\right.
    参数
        - *condition* ( **Tensor** ) : bool 类型张量，对应位置为 true，则返回 x 的元素或值，否则返回 y 的元素或值
        - *input* ( **Tensor** 或者 **Scalar** ) : 备选张量
        - *other* ( **Tensor** 或者 **Scalar** ) : 备选张量
    C API
        :guilabel:`diopiWhere`
    """
    assert(condition.get_dtype() in (Dtype.bool, Dtype.uint8)),\
        "condition must be a bool tensor"
    sizeX = list(input.size())
    sizeY = list(other.size())
    sizeC = list(condition.size())
    sizeO = broadcast_out_size(sizeX, sizeY)
    sizeO = broadcast_out_size(sizeC, sizeO)
    assert (input.get_dtype() == other.get_dtype()),\
        " input and other shoule be the same type "
    out = Tensor(sizeO, input.get_dtype())

    func = check_function("diopiWhere")
    ret = func(input.context_handle, out.tensor_handle, condition.tensor_handle,
               input.tensor_handle, other.tensor_handle)
    check_returncode(ret)
    return out


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    r"""
    释义
        裁剪可迭代参数 *parameters* 中的梯度范数。在所有的梯度上计算范数，如同将所有 *parameters* 中的梯度拼接成单个向量来计算。
        原梯度将被覆盖。
    参数
        - *parameters* ( **Tensor** ) : 将被梯度归一化的可迭代(或单个)张量
        - *max_norm* ( **float** 或者 **int** ) : 梯度最大范数
        - *norm_type* ( **float** 或者 **int** ) : 所使用的范数，默认为 2.0 ，范数可以为无穷大范数 *inf*
        - *error_if_nonfinite* ( **bool** ) : 如果为 ``True`` ，则当参数的梯度的总范数为 *nan*、*inf*
          或 *-inf* 时，会抛出错误。默认为 False
    C API
        :guilabel:`diopiClipGradNorm`
    """
    assert(isinstance(max_norm, (int, float))),\
        "max_norm must be a int or float"
    assert(isinstance(norm_type, (int, float))),\
        "norm_type must be a int or float"

    if isinstance(parameters, Tensor):
        input = parameters
        parameters = [parameters]
        parametersNum = 1
    else:
        input = parameters[0]
        parametersNum = len(parameters)
    out = 0.0

    func = check_function("diopiClipGradNorm")
    ret = func(input.context_handle, byref(out), byref(parameters), parametersNum,
               max_norm, norm_type, error_if_nonfinite)
    check_returncode(ret)
    return out


def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-05) -> Tensor:
    # todo: momentum is useless in C API
    r"""
    释义
        对输入张量 *input* 的每个特征通道 *channel* 做批量标准化，其操作描述如下:

        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

        其中，均值和方差为每个通道下小批量数据的均值和方差，
        :math:`\gamma` 和 :math:`\beta` 为长度为 C(通道数)的可学习张量。
        默认情况下，在运行期间，该层会对其计算的均值和方差进行估计，在估计时默认动量 *momentum* 为 0.1。

        若 *training* 被置为 ``True`` ，该层则不会追踪运行时统计数据 ( *running_mean* 和 *running_var* ) 来进行均值和方差的估计，
        而是直接使用当前 *batch* 的数据进行估计。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *training* ( **bool** ) : 判断是否在训练阶段
        - *running_mean* ( **Tensor** , 可选) : 推理过程中加权统计后的均值
        - *running_val* ( **Tensor** , 可选) : 推理过程中加权统计后的方差
        - *weight* ( **Tensor** , 可选) : 权重项 :math:`\gamma`
        - *bias* ( **Tensor** , 可选) : 偏置项 :math:`\beta`
        - *momentum* ( **float** ) : 用于计算运行时均值和方差，可以设置为 *None* ，默认值为 0.1
        - *eps* ( **float** ) : 批量归一化时，加在分母上的值，以此保证数据稳定性。默认值为 1e-5
    C API
        :guilabel:`diopiBatchNorm`
    """
    save_mean = mean(input, 1)
    tmp = sqrt(std(input, 1) + eps)
    tmp_1 = Tensor((1,), input.get_dtype())
    fill(tmp_1, 1)
    save_invstd = div(tmp_1, tmp)

    if weight is None:
        weight = c_void_p()
    else:
        weight = weight.tensor_handle

    if bias is None:
        bias = c_void_p()
    else:
        bias = bias.tensor_handle

    func = check_function("diopiBatchNorm")
    if training:
        assert(running_mean is None and running_var is None),\
            "if trainging, running_mean and running_var are useless"
        running_mean = c_void_p()
        running_var = c_void_p()
    else:
        running_mean = running_mean.tensor_handle
        running_var = running_var.tensor_handle

    out = raw_like(input)
    ret = func(input.context_handle, save_mean.tensor_handle, save_invstd.tensor_handle,
               input.tensor_handle, weight, bias, running_mean, running_var, training,
               momentum, eps)
    check_returncode(ret)
    return out


def log_softmax(input, dim, dtype=None):
    r"""
    释义
        对输入张量逐元素进行 *softmax* 操作之后再计算其对数值。相应公式如下:

        .. math::
            \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

        使用 *log_softmax* 函数比分别使用 *log* 和 *softmax* 更快更稳定。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **number** ) : 对输入张量应用 *log_softmax* 函数的维度
        - *dtype* ( **Dtype**, 可选) : 期望的返回值数据类型，如果指定，输入张量将会提前转换为 *dtype* 类型以防止数值溢出。
    C API
        :guilabel:`diopiLogSoftmax`
    """
    if dim is None:
        dim = 0
    if input.numel() == 0:
        return input
    if dtype is None:
        dtype = input.get_dtype()
    out = raw_like(input)

    func = check_function('diopiLogSoftmax')
    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, c_int64(dim), c_int32(dtype.value))
    check_returncode(ret)
    return out


def hardtanh(input, min_val=- 1.0, max_val=1.0, inplace=False) -> Tensor:
    r"""
    释义
        对输入 *input* 张量逐元素做如下变换:

        .. math::
            \text{HardTanh}(x) = \begin{cases}
                \text{max_val} & \text{ if } x > \text{ max_val } \\
                \text{min_val} & \text{ if } x < \text{ min_val } \\
                x & \text{ otherwise } \\
            \end{cases}
    参数
        - *input* ( **Tensor** ): 输入张量
        - *min_val* ( **number** ): 线性范围的下限，默认值为 -1
        - *max_val* ( **number** ): 线性范围的上限，默认值为 1
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiHardtanh` :guilabel:`diopiHardtanhInp`
    """
    call = "diopiHardtanh"
    min_val = byref(Scalar(input.get_dtype(), min_val))
    max_val = byref(Scalar(input.get_dtype(), max_val))
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle, min_val, max_val)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, min_val, max_val)

    check_returncode(ret)
    return out


def threshold(input, threshold, value, inplace=False) -> Tensor:
    r"""
    释义
        对输入 *input* 张量逐元素做如下变换:

        .. math::
            y =\begin{cases}
            x, &\text{ if } x > \text{threshold} \\
            \text{value}, &\text{ otherwise }
            \end{cases}
    参数
        - *input* ( **Tensor** ): 输入张量
        - *threshold* ( **number** ): 阈值
        - *value* ( **number** ): 填充值
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiThreshold` :guilabel:`diopiThresholdInp`
    """
    call = "diopiThreshold"
    threshold = byref(Scalar(input.get_dtype(), threshold))
    value = byref(Scalar(input.get_dtype(), value))
    if inplace:
        out = input
        call = call + "Inp"
        func = check_function(call)
        ret = func(input.context_handle, input.tensor_handle, threshold, value)
    else:
        out = raw_like(input)
        func = check_function(call)
        ret = func(input.context_handle, out.tensor_handle,
                   input.tensor_handle, threshold, value)

    check_returncode(ret)
    return out


def gelu(input, approximate='none') -> Tensor:
    r"""
    释义
        对输入张量逐元素应用如下变换:

        如果 *approximate* 等于 ``none``:

        .. math::
            \text { GRELU  }(x)= x \times \Phi(x)

        其中 :math:`\Phi(x)` 是高斯分布的累积分布函数。

        如果 *approximate* 等于 ``tanh``, 将做以下近似估计:

        .. math::
            \text { GRELU  }(x)=  0.5 * x * (1 + \text{Tanh}(sqrt(2 / \pi) * (x + 0.044715 * x^3)))
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *approximate* ( **string** ) : 是否采用近似估计
    C API
        :guilabel:`diopiGelu`
    """
    assert isinstance(approximate, str),\
        "approximate must be a string."
    out = raw_like(input)
    func = check_function("diopiGelu")

    ret = func(input.context_handle, out.tensor_handle,
               input.tensor_handle, byref(approximate))

    check_returncode(ret)
    return out


def addcdiv(input, tensor1, tensor2, value=1) -> Tensor:
    r"""
    释义
        执行 *tensor1* 与 *tensor2* 的逐元素除法，将结果乘以标量值 *value* 后再加至输入张量 *input* :

        .. math::
            \text{out}_i = \text{input}_i + \text{value} \times \frac{\text{tensor1}_i}{\text{tensor2}_i}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *tensor1* ( **Tensor** ) : 用来做分子的张量
        - *tensor2* ( **Tensor** ) : 用来做分母的张量
        - *value* ( **number** ) : 张量相除结果的缩放因子，默认值为 1
    C API
        :guilabel:`diopiAddcdiv`
    """
    size1 = list(tensor1.size())
    size2 = list(tensor2.size())
    sizeI = list(input.size())
    sizeO = broadcast_out_size(size1, size2)
    sizeO = broadcast_out_size(sizeI, sizeO)
    out = Tensor(sizeO, input.get_dtype())
    value = byref(Scalar(input.get_dtype(), value))

    func = check_function("diopiAddcdiv")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               tensor1.tensor_handle, tensor2.tensor_handle, value)
    check_returncode(ret)
    return out


def addmm(input, mat1, mat2, beta=1, alpha=1) -> Tensor:
    r"""
    释义
        执行 *mat1* 与 *mat2* 的矩阵乘法，将结果乘以标量值 *alpha* 后再加至输入张量 *beta* x *input*。

        如果 *mat1* 形状为 :math:`(n \times m)`, *mat2* 形状为 :math:`(m \times p)`, 那么 *input* 必须能和一个
        形状为 :math:`(n \times p)` 的张量可广播，输出张量形状为 :math:`(n \times p)`。

        .. math::
            \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)

        如果输入张量为浮点型数据， 缩放因子 *alpha* 和 *beta* 必须是实数，否则，应为整数。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *mat1* ( **Tensor** ) : 矩阵乘法的第一个张量
        - *mat2* ( **Tensor** ) : 矩阵乘法的第二个张量
        - *alpha* ( **number**，可选 ) : *input* 的缩放因子，默认值为 1
        - *beta* ( **number**，可选 ) : 张量相乘结果的缩放因子，默认值为 1
    C API
        :guilabel:`diopiAddmm`
    """
    size1 = list(mat1.size())
    size2 = mat2.size()
    size1[-1] = size2[-1]
    sizeI = list(input.size())
    sizeO = broadcast_out_size(sizeI, size1)
    out = Tensor(sizeO, input.get_dtype())
    alpha = byref(Scalar(input.get_dtype(), alpha))
    beta = byref(Scalar(input.get_dtype(), beta))

    func = check_function("diopiAddmm")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               mat1.tensor_handle, mat2.tensor_handle, beta, alpha)
    check_returncode(ret)
    return out


def sum(input, dim=None, keepdim=False, dtype=None) -> Tensor:
    r"""
    释义
        返回给定维度 *dim* 中输入张量的每一行的和。 如果 *dim* 是维度列表，则对所有维度进行归约。

        如果 *keepdim* 为 ``True``, 则除了在维度 *dim* 上，输出张量大小与输入张量相同。输出张量在维度 *dim* 上大小为1。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
        - *dtype* ( **Dtype**, 可选) : 输出数据类型
    C API
        :guilabel:`diopiSum`
    """
    assert isinstance(dim, (int, list)) or dim is None,\
        "dim should be int or list"
    func = check_function("diopiSum")
    dim, out = reduce_op_process(input, dim, keepdim, dtype)
    dim1 = Sizes(tuple(dim))
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim1, c_int32(dtype.value))
    check_returncode(ret)
    return out


def max(input, dim, keepdim=False):
    r"""
    释义
        返回给定维度 *dim* 中输入张量的每一行的最大值。 如果 *dim* 是维度列表, 则对所有维度进行归约。

        如果 *keepdim* 为 ``True``, 则除了在维度 *dim* 上，输出张量大小与输入张量相同。输出张量在维度 *dim* 上大小为1。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiMax`
    """
    assert isinstance(dim, int), "dim should be int"
    dim, out = reduce_op_process(input, dim, keepdim)
    indices = Tensor(out.size(), Dtype.int64)
    func = check_function("diopiMax")
    ret = func(input.context_handle, out.tensor_handle, indices.tensor_handle,
               input.tensor_handle, dim)
    check_returncode(ret)
    Res = namedtuple('Res', ['values', 'indices'])
    output = Res(out, indices)
    return output


def any(input, dim, keepdim=False) -> Tensor:
    r"""
    释义
        判定输入张量在给定维度 *dim* 上的每一行是否有任一元素为 True。

        如果 *keepdim* 为 ``True``, 则除了在维度 *dim* 上，输出张量大小与输入张量相同。输出张量在维度 *dim* 上大小为1。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiAny`
    """
    assert isinstance(dim, int), "dim should be int"
    dim, out = reduce_op_process(input, dim, keepdim, Dtype.bool)
    func = check_function("diopiAny")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dim)
    check_returncode(ret)
    return out


def all(input, dim, keepdim=False) -> Tensor:
    r"""
    释义
        判定输入张量在给定维度 *dim* 上的每一行是否所有元素均为 True。

        如果 *keepdim* 为 ``True``, 则除了在维度 *dim* 上，输出张量大小与输入张量相同。输出张量在维度 *dim* 上大小为1。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiAll`
    """
    assert isinstance(dim, int), "dim should be int"
    dim, out = reduce_op_process(input, dim, keepdim, Dtype.bool)
    func = check_function("diopiAll")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle, dim)
    check_returncode(ret)
    return out


def nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
    r"""
    释义
        负对数似然损失。常用于 *C* 类训练分类任务。

        当 *reduction* 为 *none* 时, 其损失计算方式如下:

        .. math::
            \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad

        .. math::
            l_n = - w_{y_n} x_{n,y_n}, \quad
            w_{c} = \text{weight}[c] \cdot 1\{c \not= \text{ignore_index}\},

        其中 :math:`x` 表示输入, :math:`y` 表示目标，:math:`w` 是权重, :math:`C` 是类别数量，:math:`N` 等于输入大小除以类别总数。

        此外, 若 *reduction* 不为 *none* , 则:

        .. math::
            \ell(x, y) = \begin{cases}
                \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
                \text{if reduction} = \text{`mean';}\\
                \sum_{n=1}^N l_n,  &
                \text{if reduction} = \text{`sum'.}
            \end{cases}
    参数
        - *input* ( **Tensor** ) : 输入张量, 一般为对数概率
        - *target* ( **Tensor** ) : 目标张量, 表示类别索引，值范围为 :math:`[0, C)`
        - *weight* ( **Tensor**, 可选) : 对每个类别手动设置的调整权重, 若非空则其大小为 *C*
        - *ignore_index* ( **int**, 可选) : 指定一个被忽略且不影响输入梯度的目标值, 当目标包含类别索引时才能使用该参数, 默认值为 -100
        - *reduction* ( **string** , 可选) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean*。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum`` : 输出求和。 其默认值为 *mean*
    形状
        - *input* : 形状为 :math:`(N, C)` 或者 :math:`(N, C, d_1, d_2, ..., d_K)` 其中 :math:`K \geq 1`
          表示 K-维损失
        - *target* : 形状为 :math:`(N)` 或者 :math:`(N, d_1, d_2, ..., d_K)` 其中 :math:`K \geq 1`
          表示 K-维损失。值范围为 :math:`[0, C)`
        - 输出 : 如果 *reduction* 为 *none*, 和 *target* 形状相同。否则为标量

        其中, N 表示批大小， C 表示类别数量
    C API
        :guilabel:`diopiNLLLoss`
    """
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if weight is not None:
        assert isinstance(weight, Tensor), \
            'weigth must be a Tensor'
        weight = weight.tensor_handle
    else:
        weight = c_void_p()

    if reduction == 'none':
        out = raw_like(target)
    else:
        out = Tensor((1,), input.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiNLLLoss")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               target.tensor_handle, weight, reduction_mode, ignore_index)
    check_returncode(ret)
    return out


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction='none') -> Tensor:
    r"""
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
    """
    assert inputs.size() == targets.size(), \
        'target shape must be the same as input shape'
    assert reduction in ['mean', 'sum', 'none'], \
        'reduction must be one of (mean, sum, none)'

    if reduction == 'none':
        out = raw_like(inputs)
    else:
        out = Tensor((1,), inputs.get_dtype())

    reduction_mode = convert_reduction(reduction)
    func = check_function("diopiSigmoidFocalLoss")
    ret = func(inputs.context_handle, out.tensor_handle, inputs.tensor_handle,
               targets.tensor_handle, alpha, gamma, reduction_mode)
    check_returncode(ret)
    return out


def nms(boxes, scores, iou_threshold) -> Tensor:
    r"""
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
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    # todo: can not have the shape of output, out should be a pointer
    size_boxes = boxes.size()
    assert len(size_boxes) == 2 and size_boxes[1] == 4,\
        "boxes must be a tensor of shape (N,4)"

    size_scores = scores.size()
    assert len(size_scores) == 1 and size_scores[0] == size_boxes[0],\
        "boxes must be a tensor of shape (N)"

    out = Tensor((1,), Dtype.int64)
    func = check_function("diopiNms")
    ret = func(boxes.context_handle, byref(out.tensor_handle), boxes.tensor_handle,
               scores.tensor_handle, iou_threshold)
    check_returncode(ret)
    return out


def roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False) -> Tensor:
    r"""
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
    """
    # todo: boxes can only be a tensor due to functions.h
    if isinstance(boxes, Tensor):
        size_boxes = boxes.size()
        assert len(size_boxes) == 2 and size_boxes[1] == 5,\
            "boxes should be a tensor of shape (N,5)"
    elif isinstance(boxes, list):
        size_boxes = boxes[0].size()
        assert len(size_boxes) == 2 and size_boxes[1] == 4,\
            "boxes should be a list of tensor of shape (N,4)"

    sizeI = list(input.size())
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    sizeI[-1] = output_size[-1]
    sizeI[-2] = output_size[-2]

    out = Tensor(sizeI, input.get_dtype())
    func = check_function("diopiRoiAlign")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               boxes.tensor_handle, spatial_scale, output_size[-2],
               output_size[-1], sampling_ratio, aligned)
    check_returncode(ret)
    return out


def slice_op(input, dim, index) -> Tensor:
    sizeI = list(input.size())
    num = int((index.stop - index.start)/index.step)
    sizeI[dim] = num
    out = Tensor(sizeI, input.get_dtype())

    func = check_function("diopiSlice")
    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               dim, index.start, index.stop, index.step)

    check_returncode(ret)
    return out


def index(input, **kwargs) -> Tensor:
    new_args = []
    hasEllipsis = False
    once = True
    for ele in kwargs.values():
        if ele is None:
            hasEllipsis = True
        else:
            if hasEllipsis and once:
                once = False
                sizeI = input.size()
                sizeE = ele.size()
                length = len(sizeI) - len(sizeE) - len(new_args)
                for i in range(length):
                    tmp = Tensor((), ele.get_dtype())
                    new_args.append(tmp.tensor_handle)
            new_args.append(ele.tensor_handle)

    out = Tensor((1, ), input.get_dtype())
    func = check_function("diopiIndex")

    ret = func(input.context_handle, out.tensor_handle, input.tensor_handle,
               byref(new_args), len(new_args))
    check_returncode(ret)
    return out


def sgd(param, param_grad, buf, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    # buf, param_grad are mutable
    func = check_function("diopiSGD")
    ret = func(param.context_handle, param.tensor_handle, param_grad.tensor_handle,
               buf.tensor_handle, lr, momentum, dampening, weight_decay, nesterov)
    check_returncode(ret)
    return param, buf
