import conformance.diopi_functions as F
from conformance.diopi_functions import abs, cos, erf, exp, floor, log, log2, log10, neg, nonzero,\
                                        sign, sin, sqrt, add, bmm, div, eq, fill_, ge, gt, le, lt,\
                                        logical_and, logical_or, matmul, mul, ne, pow, sub, binary_cross_entropy_with_logits,\
                                        cross_entropy, mse_loss, nll_loss, leaky_relu, relu, sigmoid, hardtanh, threshold,\
                                        gelu, tanh, softmax, log_softmax, mean, min, max, std, sum, all, any, addcdiv, addcmul,\
                                        addmm, adaptive_avg_pool2d, avg_pool2d, max_pool2d, adaptive_max_pool2d, batch_norm,\
                                        cat, clamp, clip_grad_norm_, conv2d, dropout, embedding, index_select, masked_scatter,\
                                        linear, one_hot, select, sort, split, stack, topk, transpose, tril, where
from conformance.diopi_functions import sigmoid_focal_loss, nms, slice_op, index, sgd, roi_align
from conformance.diopi_functions import arange, randperm, uniform, random, bernoulli, masked_fill, adamw, adam, adadelta, conv_transpose2d, \
                                        cumsum, cdist, reciprocal, bitwise_not, argmax, smooth_l1_loss, maximum, minimum, mm, conv3d, \
                                        expand, unfold, masked_select, index_fill, linspace, roll, norm, group_norm, layer_norm,\
                                        adaptive_avg_pool3d, adaptive_max_pool3d, max_pool3d, permute, copy_, gather, remainder,\
                                        ctc_loss, index_put, scatter, interpolate, pad, unique, prod

def add_docstr(attr, docstr):
    assert hasattr(F, attr)
    fuc = getattr(F, attr)
    fuc.__doc__ = docstr

add_docstr("fill_",
    r"""
    释义
        使用指定值填充 *tensor* 张量。
    参数
        - *tensor* ( **Tensor** ) : 待填充张量
        - *value* ( **number** ) : 填充值
    C API
        :guilabel:`diopiFill`
    """)

add_docstr("softmax",
    r"""
    释义
        对输入张量应用 *softmax* 函数，使得输出张量中元素值在范围 [0,1] 之间, 且元素总和为 1。
        相应公式如下:

        .. math::
            \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **number** ) : 对输入张量应用 *softmax* 函数的维度 (使得沿着该维度的元素和为 **1**)
        - *dtype* ( **Dtype**, 可选 ) : 期望的返回值数据类型，如果指定，输入张量将会提前转换为 *dtype* 类型以防止数值溢出。
    C API
        :guilabel:`diopiSoftmax`
    """)


add_docstr("relu",
    r"""
    释义
        对输入 *input* 张量逐元素做 *relu* 整流线性变换:

            :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`
    参数
        - *input* ( **Tensor** ): 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiRelu` :guilabel:`diopiReluInp`
    """)


add_docstr("abs",
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
    """)


add_docstr("floor",
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
    """)


add_docstr("sign",
    r"""
    释义
        对输入张量 *input* 逐元素计算符号函数 *Sgn* 值:

        .. math::
            \text { out }_{i}=\operatorname{sgn}\left(\text { input }_{i}\right)
    参数
        - *input* ( **Tensor** ) : 输入张量
    C API
        :guilabel:`diopiSign`
    """)


add_docstr("sigmoid",
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
    """)


add_docstr("sqrt",
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
    """)

add_docstr("neg",
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
    """)


add_docstr("sin",
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
    """)


add_docstr("cos",
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
    """)


add_docstr("tanh",
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
    """)


add_docstr("exp",
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
    """)


add_docstr("log",
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
    """)


add_docstr("log2",
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
    """)


add_docstr("log10",
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
    """)


add_docstr("erf",
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
    """)


add_docstr("add",
    r"""
    释义
        将 *other* 乘以 *alpha* 后再加至张量 *input* 上:

        .. math::
            \text { out }_{i}=\text { input }_{i}+\text { alpha } \times \text { other }_{i}

        支持广播、类型提升以及整数、浮点数输入。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **number** ) : 与输入张量相加
        - *alpha* ( **Tensor**  , 可选 ) : *other* 的的乘数
    C API
        :guilabel:`diopiAdd` :guilabel:`diopiAddScalar`
    """)


add_docstr("sub",
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
    """)


add_docstr("eq",
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
    """)


add_docstr("ne",
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
    """)


add_docstr("ge",
    r"""
    释义
        张量比较, 逐元素计算 :math:`\text{input} \geq \text{other}`。
        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 比较值, 可以是张量或数值
    C API
        :guilabel:`diopiGe` :guilabel:`diopiGeScalar`
    """)


add_docstr("gt",
    r"""
    释义
        张量比较, 逐元素计算 :math:`\text{input} > \text{other}`。
        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 比较值, 可以是张量或数值
    C API
        :guilabel:`diopiGt` :guilabel:`diopiGtScalar`
    """)


add_docstr("le",
    r"""
    释义
        张量比较, 逐元素计算 :math:`\text{input} \leq \text{other}`。
        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 比较值, 可以是张量或数值
    C API
        :guilabel:`diopiLe` :guilabel:`diopiLeScalar`
    """)


add_docstr("lt",
    r"""
    释义
        张量比较, 逐元素计算 :math:`\text{input} < \text{other}`。
        *other* 可以是一个数字或张量，如为张量，其形状和 *input* 须可广播。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor**  或者 **float** ) : 比较值, 可以是张量或数值
    C API
        :guilabel:`diopiLt` :guilabel:`diopiLtScalar`
    """)


add_docstr("mul",
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
    """)

add_docstr("div",
    r"""
    释义
        张量除, 输入张量 *input* 每个元素都除以 *other* 中与之对应的元素:

        .. math::
            \text { out }_{i}=\frac{\text { input }_{i}}{\text { other }_{i}}
    参数
        - *input* ( **Tensor** ) : 被除数
        - *other* ( **Tensor**  或者 **number** ) : 除数
        - *rounding_mode* ( **str**, 可选 ): 应用于结果的舍入类型, ``None`` : 默认行为，不执行舍入，如果 *input* 和 *other* 都是整数类型，
          则将输入提升为默认标量类型; ``trunc`` : 将除法结果向零舍入; ``floor`` : 向下舍入除法的结果
    C API
        :guilabel:`diopiDiv` :guilabel:`diopiDivScalar`
    """)


add_docstr("logical_and",
    r"""
    释义
        张量逻辑与, 对应元素进行逻辑与操作, 对于张量中的每个元素, 零元素视为False, 非零元素视为True。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor** ) : 用于计算逻辑与的张量
    C API
        :guilabel:`diopiBitwiseAnd` :guilabel:`diopiBitwiseAndScalar`
    """)


add_docstr("logical_or",
    r"""
    释义
        张量逻辑或, 对应元素进行逻辑或操作, 对于张量中的每个元素, 零元素视为False, 非零元素视为True。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor** ) : 用于计算逻辑或的张量
    C API
        :guilabel:`diopiBitwiseOr` :guilabel:`diopiBitwiseOrScalar`
    """)


add_docstr("leaky_relu",
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
        :guilabel:`diopiLeakyRelu` :guilabel:`diopiLeakyReluInp`
    """)


add_docstr("bmm",
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
    """)


add_docstr("addcmul",
    r"""
    释义
        执行 *tensor1* 与 *tensor2* 的逐元素乘法，将结果乘以标量值 *value* 后再加至输入张量 *input* :

        .. math::
            \text { out }_{i}=\text { input }_{i}+\text { value } \times \text { tensor1 }_{i} \times \operatorname{tensor2}_{i}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *tensor1* ( **Tensor** ) : 用来做乘法的第一个张量
        - *tensor2* ( **Tensor** ) : 用来做乘法的第二个张量
        - *value* ( **number** ) : 张量相乘结果的缩放因子，默认值为 1
    C API
        :guilabel:`diopiAddcmul`
    """)


add_docstr("matmul",
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
    """)


add_docstr("clamp",
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
        - *min* ( **number** 或者 **Tensor** , 可选 ) : 取值下界
        - *max* ( **number** 或者 **Tensor** , 可选 ) : 取值上界
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiClamp` :guilabel:`diopiClampMax` :guilabel:`diopiClampMin`
        :guilabel:`diopiClampInp` :guilabel:`diopiClampMaxInp` :guilabel:`diopiClampMinInp`
        :guilabel:`diopiClampInpScalar` :guilabel:`diopiClampMaxInpScalar` :guilabel:`diopiClampMinInpScalar`
        :guilabel:`diopiClampScalar` :guilabel:`diopiClampMaxScalar` :guilabel:`diopiClampMinScalar`
    """)


add_docstr("mean",
    r"""
    释义
        返回给定维度 *dim* 中输入张量的每一行的平均值。 如果 *dim* 是维度列表，则对列表中所有维度进行归约。
        如果 dim 等于 ``None``, 将对所有元素计算均值。

        如果 *keepdim* 为 ``True``, 输出张量在维度 *dim* 上大小为1，其他维度上大小与输入张量相同。
        如果 *keepdim* 为 ``False``, 输出张量将被 *squeeze*，即移除所有大小为1的维度。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** 或者 **list(int)** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
        - *dtype* ( **Dtype**, 可选 ) : 输出的数据类型
    C API
        :guilabel:`diopiMean`
    """)


add_docstr("std",
    r"""
    释义
        如果 *unbiased* 为 ``True``，则将使用 *Bessel* 校正。 否则，将直接计算样本偏差，而不进行任何校正。
        如果 *dim* 等于 ``None``, 将对所有元素计算标准差。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *unbiased* ( **bool** ) : 是否使用Bessel校正
        - *dim* ( **int** 或者 **list(int)** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiStd`
    """)


add_docstr("min",
    r"""
    释义
        返回给定维度 *dim* 中输入张量的每一行的最小值。

        如果 *keepdim* 为 ``True``, 输出张量在维度 *dim* 上大小为1，其他维度上大小与输入张量相同。
        如果 *keepdim* 为 ``False``, 输出张量将被 *squeeze*，即移除所有大小为1的维度。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiMin`
    """)


add_docstr("binary_cross_entropy_with_logits",
    r"""
    释义
        计算目标 *target* 和输入 *input* 之间的二值交叉函数:
        这种损失将 *Sigmoid* 层和 *BCELoss* 组合在一个函数中, 若 *reduction* 为 *none* :

        .. math::
            \ell(x, y)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\prime}, \quad l_{n}=-w_{n}\left[y_{n}
            \cdot \log \sigma\left(x_{n}\right)+\left(1-y_{n}\right) \cdot
            \log \left(1-\sigma\left(x_{n}\right)\right)\right]

        其中 *N* 表示 *batch_size* 。此外, 若 *reduction* 不为 *none*, 则:

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
        - *weight* ( **Tensor** , 可选 ): 手动设置的调整权重, 可自动扩展以适配输入张量的形状
        - *reduction* ( **string** , 可选 ) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean* 。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum``: 输出求和。其默认值为 *mean*
        - *pos_weight* ( **Tensor** , 可选 ) : 正样本的权重, 其长度必须与类别数量相同
    C API
        :guilabel:`diopiBCEWithLogits`
    """)


add_docstr("cross_entropy",
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
        - *weight* ( **Tensor**, 可选 ) : 对每个类别手动设置的调整权重, 若非空则其大小为 *C*
        - *ignore_index* ( **int**, 可选 ) : 指定一个被忽略且不影响输入梯度的目标值, 当目标包含类别索引时才能使用该参数, 其默认值为 -100
        - *reduction* ( **string** , 可选 ) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean*。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum`` : 输出求和。 其默认值为 *mean*
        - *label_smoothing* ( **float** , 可选 ) : 其取值范围为 [0.0, 1.0] 的浮点数, 指定计算损失时的平滑量，其中 0.0 表示不平滑。其默认值为 0.0
    形状
        - *input* : 形状为 :math:`(C)`, :math:`(N, C)` 或者 :math:`(N, C, d_1, d_2, ..., d_K)` 其中 :math:`K \geq 1`
          表示 K-维损失
        - *target* : 如果为类别索引, 形状为 :math:`()`, :math:`(N)` 或者 :math:`(N, d_1, d_2, ..., d_K)` 其中 :math:`K \geq 1`
          表示 K-维损失。值范围为 :math:`[0, C)`。如果为类别概率，形状和 *input* 相同，值范围为 :math:`[0, 1]`
        - 输出 : 如果 *reduction* 为 *none*, 和 *target* 形状相同。否则为标量

        其中, N 表示批大小， C 表示类别数量
    C API
        :guilabel:`diopiCrossEntropyLoss`
    """)


add_docstr("mse_loss",
    r"""
    释义
        计算输入张量 *input* 与 目标张量 *target* 之间每个对应元素的均方误差。
        当 *reduction* 为 *none* 时，损失函数描述为:

        .. math::
            \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
            l_n = \left( x_n - y_n \right)^2,

        其中 *N* 表示批大小 。当 *reduction* 不为 *none* 时，损失函数描述为:

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
        - *reduction* ( **string** , 可选 ) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean*。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum`` : 输出求和。 默认值为 mean
    C API
        :guilabel:`diopiMSELoss`
    """)


add_docstr("conv2d",
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
    """)

add_docstr("avg_pool2d",
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
    """)


add_docstr("max_pool2d",
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
        - *dilation* ( **number** ) : 滑动窗口内元素之间的间隔，其值必须大于 0
        - *ceil_mode* ( **bool** ) : 如果为 ``True`` ，将使用向上取整而不是默认的向下取整来计算输出形状。 这确保了输入张量中的每个元素都被滑动窗口覆盖。
        - *return_indices* ( **bool** ) : 如果为 ``True`` ，将返回所有滑动窗口产生的最大值的位置索引，该结果将后续会被应用于反池化。默认值为 False
    C API
        :guilabel:`diopiMaxPool2d` :guilabel:`diopiMaxPool2dWithIndices`
    """)


add_docstr("adaptive_avg_pool2d",
    r"""
    释义
        对输入张量 *input* 做 2D 自适应平均池化。对于任意大小的 *input*,
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
    """)


add_docstr("adaptive_max_pool2d",
    r"""
    释义
        对输入张量 *input* 做 2D 自适应最大值池化。对于任意大小的 *input*,
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
    """)


add_docstr("dropout",
    r"""
    释义
        在训练模式下， 基于伯努利分布抽样，以概率 p 对输入张量 *input* 的值随机置零。
        此外在训练过程中，输出张量将以因子 :math:`\frac{1}{1-p}` 进行缩放。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *p* ( **float** ) : 输入张量元素被置零的概率，默认值为 0.5
        - *training* ( **bool** ) : 是否为训练模式，默认为 True。当为 ``False`` 时，*dropout* 将不会执行
        - *inplace* ( **bool** ) : 是否覆盖原数据
    C API
        :guilabel:`diopiDropout` :guilabel:`diopiDropoutInp`
    """)


add_docstr("index_select",
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
    """)


add_docstr("select",
    r"""
    释义
        在给定索引 *index* 处沿选定维度 *dim* 对输入张量 *input* 进行切片，并返回切片到的数据。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 取索引数据所在的维度
        - *index* ( **int** ) : 索引下标
    C API
        :guilabel:`diopiSelect`
    """)


add_docstr("masked_scatter",
    r"""
    释义
        在掩码 *mask* 为 True 的位置将元素从 *source* 复制到 *input* 。
        掩码的形状与张量 *input* 必须是可广播的。*source* 中的元素至少应与掩码中的元素数量一样多。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *mask* ( **BoolTensor** ) : 布尔类型的掩码
        - *source* （ **Tensor** ) : 被复制的张量
    C API
        :guilabel:`diopiMaskedScatter`
    """)


add_docstr("nonzero",
    r"""
    释义
        返回一个二维张量，其中每一行代表 *input* 中一个非零元素的索引坐标。
        如果 *input* 有 :math:`n` 维, 那么输出索引张量的形状为
        :math:`(z \times n)`, 其中 :math:`z` 等于 *input* 中非零元素的总数。
    参数
        - *input* ( **Tensor** ) : 输入张量
    C API
        :guilabel:`diopiNonzero`
    """)


add_docstr("linear",
    r"""
    释义
        对 *input* 应用如下线性变换。:math:`y = x A^T + b`。
    参数
        - *input* ( **Tensor** ) : 输入张量 :math:`x`，形状为 *( \# , in_features)* ， \# 表示任意数量的维度，包括可以为 *None*
        - *weight* ( **Tensor** ) : 权重项 :math:`A`，形状为 *(out_features, in_features)* 或者 *(in_features)*
        - *bias*（ **Tensor** ) : 偏置项 :math:`b`，形状为 *(out_features)* 或者 *()*
    返回值
        输出张量形状为 *( \#, out_features)* 或者 *( \#)*， 取决于权重的形状
    C API
        :guilabel:`diopiLinear`
    """)


add_docstr("embedding",
    r"""
    释义
        一个简单的查找表，用于在固定的字典查找固定大小的嵌入向量表示。

        该功能通常用于使用索引检索词嵌入。输入是索引列表和嵌入矩阵，输出是相应的词嵌入。
    参数
        - *input* ( **LongTensor** ) : 包含嵌入矩阵索引的张量
        - *weight* ( **Tensor** ) :  行数等于最大可能索引 + 1 的嵌入矩阵，列数等于嵌入向量大小
        - *padding_idx* ( **int**，可选 ) :  如果指定，则 *padding_idx* 处的条目不会影响梯度。
          因此，*padding_idx* 处的嵌入向量在训练期间不会更新，即它保持是一个固定的 “*pad*”
        - *max_norm* ( **float**，可选 ) : 如果给定，则范数大于 *max_norm* 的嵌入向量被重新归一化为具有范数 *max_norm*。
          注意: 这将就地修改权重 *weight*
        - *norm_type* ( **float**，可选 ) : 为 *max_norm* 选项计算 *p-norm* 时的 *p*，默认为2
        - *scale_grad_by_freq* ( **bool**，可选 ) : 如果给定，这将通过小批量中单词频率的倒数来缩放梯度，默认为 *False*
        - *sparse* ( **bool**，可选) : 如果为 ``True``, 权重将是一个稀疏张量
    形状
        - *input* : 包含要提取的索引的任意形状的 *LongTensor*
        - *weight* :  形状为 *(V, embedding_dim)* 的浮点型嵌入矩阵，其中 V = 最大索引 + 1，*embedding_dim* = 嵌入向量大小
        - 输出 : 形状为 *(#, embedding_dim)* , 其中 # 是输入的形状
    C API
        :guilabel:`diopiEmbedding` :guilabel:`diopiEmbeddingRenorm_`
    """)


add_docstr("tril",
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
    """)

add_docstr("cat",
    r"""
    释义
        沿给定的维度 *dim* 拼接序列 *tensors* 中的张量。

        所有的张量必须有相同的形状 (在拼接维度 *dim* 上例外）或者为空。
    参数
        - *tensors* ( **多个tensor组成的序列** ) : 相同数据类型张量的任何 *python* 序列
        - *dim* ( **int** ) :  插入的维度, 值必须在 0 和 张量的维度数量之间
    C API
        :guilabel:`diopiCat`
    """)


add_docstr("stack",
    r"""
    释义
        沿给定的维度 *dim* 拼接序列 *tensors* 中的张量。

        所有的张量必须有相同的形状。
    参数
        - *tensors* ( **多个tensor组成的序列** ) : 将被拼接的张量序列
        - *dim* ( **int** ) :  插入的维度。值必须在 0 和 张量的维度数量之间
    C API
        :guilabel:`diopiStack`
    """)


add_docstr("sort",
    r"""
    释义
        对 *input* 沿给定的维度 *dim* 上的元素进行升序排序。

        如果 *dim* 没有给定，默认选择 *input* 的最后一个维度。

        如果 *descending* 设为 ``True``，元素将以降序排序。

        如果布尔选项 *stable* 设为 ``True``， 排序算法是稳定的， 对相同的元素维持原有顺序。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int**，可选 ) :  进行排序的维度 *dim*
        - *descending* ( **bool**，可选 ) :  用以决定排序顺序（升序或降序）
        - *stable* ( **bool**，可选 ) :  用以选择稳定的排序算法，稳定的排序算法保证相同元素的顺序维持不变
    返回值
        返回值是一个（Tensor， LongTensor）的元组，其含义为（values， indices），
        其中 *values* 是排序后的值，*indices* 是在维度 *dim* 上的位置索引。
    C API
        :guilabel:`diopiSort`
    """)


add_docstr("topk",
    r"""
    释义
        返回 *input* 沿给定的维度 *dim* 的前 *k* 个最大的值。

        如果 *dim* 没有给定，默认选择 *input* 的最后一个维度。

        如果 *largest* 设为 ``False``，将会返回前 *k* 个最小的值。

        如果布尔选项 *sorted* 设为 ``True``， 将会返回的排序后的前 *k* 个值。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *k* ( **int** ) :  *”top-k“* 中的k
        - *dim* ( **int**，可选 ) :  进行排序的维度 *dim*
        - *largest* ( **bool**，可选 ) :  用以决定返回前 *k* 个最大的 或者 最小的元素
        - *sorted* ( **bool**，可选 ) :  用以决定返回的前 *k* 个值是否有序
    返回值
        返回值是一个（Tensor， LongTensor）的元组，其含义为（values， indices），
        其中 *values* 是前 *k* 个值，*indices* 是在维度 *dim* 上的位置索引。
    C API
        :guilabel:`diopiTopk`
    """)


add_docstr("transpose",
    r"""
    释义
        返回 *input* 的转置版本，在给定的维度 *dim0* 和 *dim1* 上交换。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim0* ( **int** ) :  将被转置的第一个维度
        - *dim1* ( **int** ) :  将被转置的第二个维度
    返回值
        *input* 的形状为 *(\#, dim0, \#, dim1, \#)*, 返回张量形状为 *(\#, dim1, \#, dim0, \#)*。
    C API
        :guilabel:`diopiTranspose`
    """)


add_docstr("one_hot",
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
    """)


add_docstr("split",
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
    """) 


add_docstr("pow",
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
    """)


add_docstr("where",
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
    """)


add_docstr("clip_grad_norm_",
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
    """)


add_docstr("batch_norm",
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
        - *running_mean* ( **Tensor** , 可选 ) : 加权统计后的均值
        - *running_var* ( **Tensor** , 可选 ) : 加权统计后的方差
        - *weight* ( **Tensor** , 可选 ) : 权重项 :math:`\gamma`
        - *bias* ( **Tensor** , 可选 ) : 偏置项 :math:`\beta`
        - *momentum* ( **float** ) : 用于计算运行时均值和方差，可以设置为 *None*, 默认值为 0.1
        - *eps* ( **float** ) : 批量归一化时，加在分母上的值，以此保证数据稳定性, 默认值为 1e-5
    C API
        :guilabel:`diopiBatchNorm`
    """)


add_docstr("log_softmax",
    r"""
    释义
        对输入张量逐元素进行 *softmax* 操作之后再计算其对数值。相应公式如下:

        .. math::
            \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

        使用 *log_softmax* 函数比分别使用 *log* 和 *softmax* 更快更稳定。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **number** ) : 对输入张量应用 *log_softmax* 函数的维度
        - *dtype* ( **Dtype**, 可选 ) : 期望的返回值数据类型，如果指定，输入张量将会提前转换为 *dtype* 类型以防止数值溢出。
    C API
        :guilabel:`diopiLogSoftmax`
    """)


add_docstr("hardtanh",
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
    """)


add_docstr("threshold",
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
    """)


add_docstr("gelu",
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
    """)


add_docstr("addcdiv",
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
    """)


add_docstr("addmm",
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
        - *alpha* ( **number**，可选 ) : 张量相乘结果的缩放因子，默认值为 1
        - *beta* ( **number**，可选 ) : *input* 的缩放因子，默认值为 1
    C API
        :guilabel:`diopiAddmm`
    """)


add_docstr("sum",
    r"""
    释义
        返回给定维度 *dim* 中输入张量的每一行的和。 如果 *dim* 是维度列表，则对所有维度进行归约。
        如果 dim 等于 ``None``, 将对所有元素求和。

        如果 *keepdim* 为 ``True``, 输出张量在维度 *dim* 上大小为1，其他维度上大小与输入张量相同。
        如果 *keepdim* 为 ``False``, 输出张量将被 *squeeze*，即移除所有大小为1的维度。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** 或者 **list(int)** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
        - *dtype* ( **Dtype**, 可选) : 输出数据类型
    C API
        :guilabel:`diopiSum`
    """)


add_docstr("max",
    r"""
    释义
        返回给定维度 *dim* 中输入张量的每一行的最大值。

        如果 *keepdim* 为 ``True``, 输出张量在维度 *dim* 上大小为1，其他维度上大小与输入张量相同。
        如果 *keepdim* 为 ``False``, 输出张量将被 *squeeze*，即移除所有大小为1的维度。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiMax`
    """)


add_docstr("any",
    r"""
    释义
        判定输入张量在给定维度 *dim* 上的每一行是否有任一元素为 True。

        如果 *keepdim* 为 ``True``, 输出张量在维度 *dim* 上大小为1，其他维度上大小与输入张量相同。
        如果 *keepdim* 为 ``False``, 输出张量将被 *squeeze*，即移除所有大小为1的维度。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiAny`
    """)


add_docstr("all",
    r"""
    释义
        判定输入张量在给定维度 *dim* 上的每一行是否所有元素均为 True。

        如果 *keepdim* 为 ``True``, 输出张量在维度 *dim* 上大小为1，其他维度上大小与输入张量相同。
        如果 *keepdim* 为 ``False``, 输出张量将被 *squeeze*，即移除所有大小为1的维度。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 进行归约的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiAll`
    """)


add_docstr("nll_loss",
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
        - *weight* ( **Tensor**, 可选 ) : 对每个类别手动设置的调整权重, 若非空则其大小为 *C*
        - *ignore_index* ( **int**, 可选 ) : 指定一个被忽略且不影响输入梯度的目标值, 当目标包含类别索引时才能使用该参数, 默认值为 -100
        - *reduction* ( **string** , 可选 ) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean*。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum`` : 输出求和。 其默认值为 *mean*
    形状
        - *input* : 形状为 :math:`(N, C)` 或者 :math:`(N, C, d_1, d_2, ..., d_K)` 其中 :math:`K \geq 1`
          表示 K-维损失
        - *target* : 形状为 :math:`(N)` 或者 :math:`(N, d_1, d_2, ..., d_K)` 其中 :math:`K \geq 1`
          表示 K-维损失。值范围为 :math:`[0, C)`
        - 输出 : 如果 *reduction* 为 *none*, 和 *target* 形状相同。否则为标量

        其中, N 表示批大小， C 表示类别数量
    C API
        :guilabel:`diopiCrossNLLLoss`
    """)


add_docstr("sigmoid_focal_loss",  r"""
    释义
        Focal loss 用于解决分类任务中的前景类-背景类数量不均衡的问题。
        该算子通过下式计算focal loss:

        .. math::
           Out = -targets * alpha * {(1 - \sigma(inputs))}^{gamma}\log(\sigma(inputs)) - (1 - targets) * (1 - alpha) * {\sigma(inputs)}^{gamma}\log(1 - \sigma(inputs))

        其中 :math:`\sigma(inputs) = \frac{1}{1 + \exp(-inputs)}`。
        当 *reduction* 为 ``none`` 时，直接返回最原始的输出结果。当 *reduction* 为 ``mean`` 时，返回输出的均值。
        当 *reduction* 为 ``sum`` 时，返回输出的求和。
    参数
        - *inputs* ( **FloatTensor** ) : 输入张量, 表示未归一化分数，通常也称作 *logits*
        - *targets* ( **FloatTensor** ) : 形状同 *inputs*, 为 *inputs* 中每个元素所对应的二分类标签值。对于负样本，值为0， 对于正样本，值为1
        - *alpha* ( **number**, 可选 ): 用于平衡正样本和负样本的超参数，取值范围 (0, 1)。默认值为0.25
        - *gamma* ( **number**, 可选 ) : 用于平衡易分样本和难分样本的超参数，默认值设置为2.0
        - *reduction* ( **string** , 可选 ) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean*。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum`` : 输出求和。 其默认值为 *mean*
    C API
        :guilabel:`diopiSigmoidFocalLoss`
    """)


add_docstr("nms", r"""
    释义
        非极大抑制(non-maximum suppression, NMS)用于在目标检测应用对检测边界框(bounding box)中搜索局部最大值，
        即只保留处于同一检测目标位置处重叠的框中分数最大的一个框。IoU(Intersection Over Union) 被用于判断两个框是否重叠，该值大于门限值(iou_threshold)则被认为两个框重叠。
        其计算公式如下:

        ..  math::
            IoU = \frac{intersection_area(box1, box2)}{union_area(box1, box2)}

    参数
        - *boxes* ( **Tensor[N, 4]** ) : 待进行计算的框坐标，它应当是一个形状为[num_boxes, 4]的2-D Tensor，以[[x1, y1, x2, y2], ...]的形式给出, 其中(x1, y1)是左上角的坐标值，(x2, y2)是右下角的坐标值，其关系应符合 ``0 <= x1 < x2 && 0 <= y1 < y2``
        - *scores* ( **Tensor[N]** ) : *boxes* 中每个框的相应分数
        - *iou_threshold* ( **float** ): 用于判断两个框是否重叠的IoU门限值。 如果IoU(box1, box2) > threshold，box1和box2将被认为是重叠框

    返回值
        被NMS保留的检测边界框的索引，按分数降序排序, 数据类型为int64

    C API
        :guilabel:`diopiNms`
    """)


add_docstr("roi_align", r"""
    释义
        RoI Align是在指定输入的感兴趣区域上执行双线性插值以获得固定大小的特征图，如 Mask R-CNN论文中所述。
    参数
        - *input* ( **Tensor[N, C, H, W]** ) : 输入张量
        - *boxes* ( **Tensor[K, 5]** 或者 **List[Tensor[L, 4]]** ) : 待执行池化的RoIs(Regions of Interest)的框坐标。如果传递了一个张量，那么第一列应该
          包含 *batch* 中相应元素的索引，即 [0, N - 1] 中的数字。如果传递了张量列表，则每个张量将对应于 *batch* 中元素 i 的框
        - *output_size* ( **int** 或者 **Tuple[int, int]** ) : 池化后输出的尺寸(H, W)。如果为单个int型整数，则H和W都与其相等
        - *spatial_scale* ( **float**, 可选 ) : 空间比例因子，将输入坐标映射到 *boxes* 中坐标的比例因子, 默认为1
        - *sampling_ratio* ( **int** ) : 用于计算每个池化输出 bin 的输出值。如果 > 0，
          然后恰好使用每个 bin 的 ``sampling_ratio x sampling_ratio`` 采样点。如果
          <= 0，则使用自适应数量的网格点（计算为 ``ceil（roi_width / output_width）``，同样适用于高度）。默认值为-1 
        - *aligned* ( **bool** , 可选 ) : 如果为 ``True``，表示像素移动框将其坐标移动-0.5，以便与两个相邻像素索更好地对齐。如果为 ``False``，则是使用默认实现
    返回值
        池化后的RoIs，为一个形状是(RoI数量，C, output_size[0], output_size[1]）的 4-D Tensor
    C API
        :guilabel:`diopiRoiAlign`
""")

add_docstr("slice_op", r"""
    释义
        在维度 *dim* 上对输入张量 *input* 按索引 *index* 进行切片
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 进行切片的维度
        - *index* ( **slice** ) : 索引值，为 python 的 *slice(start, end, step)* 对象
    C API
        :guilabel:`diopiSlice`
""")


add_docstr("index", r"""
    释义
        对输入张量 *input* 按键值对中的索引进行切片，索引规则参考: https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *kwargs* ( **dict** ) : 可以传入整型或者布尔型张量序列
    C API
        :guilabel:`diopiIndex`
""")


add_docstr("sgd", r"""
    释义
        实现梯度下降（可选动量），梯度更新流程如下：

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

    其中Nesterov动量基于：`On the importance of initialization and momentum in deep learning`__.

    参数
        - *param* ( **Tensor** 或者 **dict**) : 迭代梯度优化的参数或dict参数组
        - *param_grad* ( **Tensor** ) : 参数梯度
        - *buf* ( **Tensor** ) : 动量缓存
        - *lr* ( **float** ) : 学习率
        - *momentum* ( **float** ，可选) : 动量因子，默认值为 0
        - *dampending* ( **float** ，可选) : 动量阻尼，默认值为 0
        - *weight_decay* ( **float** ，可选) : 权重衰减系数，默认值为 0
        - *nesterov* ( **bool** ，可选) : 是否使用Nesterov动量，默认值为 false
    C API
        :guilabel:`diopiSgd`
""")

add_docstr("arange", r"""
    释义
        返回从 :attr:`start` 开始，以步长 :attr:`step` 到 :attr:`end` 结束的一维张量，
        其中张量大小为 :math:`\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil`

        .. math::
            \text{out}_{{i+1}} = \text{out}_{i} + \text{step}
        .. note:: 非整型的 :attr:`step` 在与 :attr:`end` 比较时可能会受到浮点舍入误差的影响，导致不一致性 
    参数
        - *start* ( **Number** ) : 结果张量起始值，默认为 0.
        - *end* ( **Number** ) : 结果张量终止值的上界（结果张量中不包含该值）
        - *step* ( **Number** ) : 结果张量相邻元素之间的差值，默认值为 1.
    C API
        :guilabel:`diopiArange`
""")

add_docstr("randperm", r"""
    释义
        返回从0到n-1的整数的随机排列
    参数
        - *n* ( **int** ) : 结果张量数值上界（不包含）
    C API
        :guilabel:`diopiRandperm`
""")

add_docstr("uniform", r"""
    释义
        用从连续均匀分布中采样的数字填充张量 :attr:`input`

        .. math::
            P(x) = \dfrac{1}{\text{end} - \text{start}}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *start* ( **Number** ) : 均匀分布采样下界，默认值为0
        - *end* ( **Number** ) : 均匀分布采样上界，默认值为1
    C API
        :guilabel:`diopiUniformInp`
""")

add_docstr("random", r"""
    释义
        用从离散均匀分布中采样的值填充 :attr:`input` 张量，数值分布在[start，end-1]范围内。如果未指定，值通常为
        仅以 :attr:`input` 张量自身的数据类型为界。对于浮点类型，如果未指定，范围将为[0，2^m]，以确保
        值是可表示的
        
        例如， :attr:`torch.tensor(1, dtype=torch.double).random_()` 将在[0，2^53]中保持一致

    参数
        - *input* ( **Tensor** ) : 输入张量
        - *start* ( **Number** ) : 均匀分布采样下界
        - *end* ( **Number** ) : 均匀分布采样上界
    C API
        :guilabel:`diopiRandomInp`
""")

add_docstr("bernoulli", r"""
    释义
       依据输入张量计算其对应的Bernoulli分布值，因此输入张量的每个值取值范围必须为
       :math:`0 \leq \text{input}_i \leq 1`.
       
        .. math::
            \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{input}_{i})

       输出张量的数据类型可为 ``integral`` ，但输入张量必须为 ``float``

    参数
        - *input* ( **Tensor** ) : 输入张量
        - *inplace* ( **bool** ) : 是否覆盖原数据
        - *p* ( **float** ) : 概率标量，当 p 不为空时，屏蔽input输入张量并按输入标量 p 计算输出，否则按输入张量input计算，默认为 :attr:`None`
    C API
        :guilabel:`diopiBernoulli` :guilabel:`diopiBernoulliInp` :guilabel:`diopiBernoulliScalar`
""")

add_docstr("adamw", r"""
    释义
        adamw梯度优化算子实现

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
    
    参数
        - *param* ( **Tensor** 或者 **dict** ) : 迭代梯度优化的参数或dict参数组
        - *param_grad* ( **Tensor** 或者 **dict** ) : 参数梯度
        - *exp_avg* ( **Tensor** ) : 第一动量，与迭代次数相关，也即第 `i` 次迭代的梯度均值
        - *exp_avg_sq* ( **Tensor** ) : 第二动量，与迭代次数相关，也即第 `i` 次迭代的梯度平方的均值
        - *max_exp_avg_sq* ( **Tensor** ) : 最大第二动量，当参数 `amsgrad` 为True时，代替第二动量参与计算
        - *lr* ( **float** ) : 学习率，默认值为 1e-3
        - *beta1* ( **float** ) : 用于计算梯度均值的权重系数，默认值为 0.9
        - *beta2* ( **float** ) : 用于计算梯度平方的均值的权重系数，默认值为 0.999
        - *eps* ( **float** ，可选) : 梯度更新稳定系数，与分母相加，默认值为 1e-8
        - *weight_decay* ( **float** ，可选) : 权重衰减系数，默认值为 1e-2
        - *step* ( **int** ) : 迭代次数
        - *amsgrad* ( **bool** ) : 是否使用 `On the Convergence of Adam and Beyond` _算法变体，默认为 False
    C API
        :guilabel:`diopiAdamW`
""")

add_docstr("adam", r"""
    释义
        adam梯度优化算子实现

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
    
    参数
        - *param* ( **Tensor** 或者 **dict** ) : 迭代梯度优化的参数或dict参数组
        - *param_grad* ( **Tensor** 或者 **dict** ) : 参数梯度
        - *exp_avg* ( **Tensor** ) : 第一动量，与迭代次数相关，也即第 `i` 次迭代的梯度均值
        - *exp_avg_sq* ( **Tensor** ) : 第二动量，与迭代次数相关，也即第 `i` 次迭代的梯度平方的均值
        - *max_exp_avg_sq* ( **Tensor** ) : 最大第二动量，当参数`amsgrad`为True时，代替第二动量参与计算
        - *lr* ( **float** ) : 学习率，默认值为 1e-3
        - *beta1* ( **float** ) : 用于计算梯度均值的权重系数，默认值为 0.9
        - *beta2* ( **float** ) : 用于计算梯度平方的均值的权重系数，默认值为 0.999
        - *eps* ( **float** ，可选) : 梯度更新稳定系数，与分母相加，默认值为 1e-8
        - *weight_decay* ( **float** ，可选) : 权重衰减系数，默认值为 0
        - *step* ( **int** ) : 迭代次数
        - *amsgrad* ( **bool** ) : 是否使用 `On the Convergence of Adam and Beyond` _算法变体，默认为 False
    C API
        :guilabel:`diopiAdam`
""")

add_docstr("adadelta", r"""
    释义
        adadelta梯度优化算子实现

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
    
    参数
        - *param* ( **Tensor** 或者 **dict** ) : 迭代梯度优化的参数或dict参数组
        - *param_grad* ( **Tensor** 或者 **dict** ) : 参数梯度
        - *square_avg* ( **Tensor** ) : 动量，梯度平方的均值
        - *acc_delta* ( **Tensor** ) : 累计变量，用于更新计算梯度
        - *lr* ( **float** ) : 学习率，默认值为 1.0
        - *rho* ( **float** ) : 用于计算梯度平方的均值的权重系数，默认值为 0.9
        - *eps* ( **float** ，可选) : 梯度更新稳定系数，与分母相加，默认值为 1e-6
        - *weight_decay* ( **float** ，可选) : 权重衰减系数，默认值为 0
    C API
        :guilabel:`diopiAdadelta`
""")

add_docstr("masked_fill", r"""
    释义
        依据掩码 :attr:`mask` 将输入张量 :attr:`input` 中对应掩码值为True的元素置为  :attr:`value` 
        ，其中 :attr:`mask` 的形状是可广播的
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *mask* ( **Tensor** ) : 掩码张量
        - *value* ( **float** ) : 填充值
    C API
        :guilabel:`diopiMaskedFill` :guilabel:`diopiMaskedFillScalar` :guilabel:`diopiMaskedFillInp` :guilabel:`diopiMaskedFillInpScalar`
""")

add_docstr("conv_transpose2d", r"""
    释义
        对输入张量 :attr:`input` 应用2D转置卷积算子，有时也称为“反卷积”。
    参数
        - *input* ( **Tensor** ) : 输入张量，其形状为 :math:`(\text{minibatch} , \text{in_channels} , iH , iW)` 
        - *weight* ( **Tensor** ) : 反卷积权重，其形状为 :math:`(\text{in_channels} , \frac{\text{out_channels}}{\text{groups}} , kH , kW)`.
        - *bias* ( **Tensor** ) : 反卷积偏置量，其形状为 :math:`(\text{out_channels})` 默认值为 None
        - *stride* ( **Number** 或者 **tuple** ) : 卷积核步长，默认值为 1
        - *padding* ( **Number** 或者 **tuple** ) : 输入张量零值填充拓展大小，默认值为 0
        - *output_padding* ( **Number** 或者 **tuple** ) : 输出张量零值填充拓展大小，默认值为 0
        - *groups* ( **Number** 或者 **tuple** ) : 将输入张量进行分组，其中 :math:`\text{in_channels}` 必须能够整除该值，默认值为 1
        - *dilation* ( **Number** 或者 **tuple** ) : 卷积核元素之间的间距，默认值为 1
    C API
        :guilabel:`diopiConvTranspose2d`
""")

add_docstr("cumsum", r"""
    释义
        返回输入张量 :attr:`input` 在维度 :attr:`dim` 上的元素累计和

        .. math::
            y_i = x_1 + x_2 + x_3 + \dots + x_i
        .. note:: 如果输入张量 :attr:`input` 大小为 `N` ，则返回张量的大小也为 `N` ，即输出张量为输入张量的前缀和
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 计算累计和实施维度
    C API
        :guilabel:`diopiCumsum`
""")

add_docstr("cdist", r"""
    释义
        计算张量 :attr:`x1` 与张量 :attr:`x2` 每对行向量之间的 :attr:`p` 范数距离 
    参数
        - *x1* ( **Tensor** ) : 输入张量1，形状为 :math:`B \times P \times M`.
        - *x2* ( **Tensor** ) : 输入张量2，形状为 :math:`B \times R \times M`.
        - *p* ( **int** ) : 范数距离类型，取值范围为 :math:`\in [0, \infty]`
        - *compute_mode* ( **string** ) : 当取值为 ``use_mm_for_euclid_dist_if_necessary`` ，
          如果参与计算张量维度大小满足P>25或R>25，将使用矩阵乘法计算欧几里得距离。当取值为 ``use_mm_for_euclid_dist``  ，
          将始终使用矩阵乘法来计算欧几里得距离。当取值为 ``donot_use_mm_for_euclid_dist`` ，
          将永远不会使用矩阵乘积法来计算欧氏距离。默认值： ``use_mm_for_euclid_dist_if_necessary``
    C API
        :guilabel:`diopiCdist`
""")

add_docstr("reciprocal", r"""
    释义
        返回输入张量 :attr:`input` 的倒数

        .. math::
            \text{out}_{i} = \frac{1}{\text{input}_{i}}
    参数
        - *input* ( **Tensor** ) : 输入张量
    C API
        :guilabel:`diopiReciprocal` :guilabel:`diopiReciprocalInp`
""")

add_docstr("bitwise_not", r"""
    释义
        返回输入张量的元素按位( ``bit`` )取反的新张量，输入张量必须为整型或者bool型，当为bool型时，对元素直接逻辑取反
    参数
        - *input* ( **Tensor** ) : 输入张量
    C API
        :guilabel:`diopiBitwiseNot`
""")

add_docstr("argmax", r"""
    释义
        返回维度 :attr:`dim` 上输入张量 :attr:`input` 最大值的下标索引。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 查找最大值的维度，如果为None，则在整个张量上查找
        - *keepdim* ( **bool** ) : 输出张量维度是否与输入张量保持相同，如果dim为None，则忽略该参数
    C API
        :guilabel:`diopiArgmax`
""")

add_docstr("smooth_l1_loss", r"""
    释义
        平滑的L1范数损失函数，相比L1范数损失函数能够防止梯度爆炸

        当reduction为`none`时，该损失函数定义为：

        .. math::
            \ell(x, y) = L = \{l_1, ..., l_N\}^T

        其中

        .. math::
            l_n = \begin{cases}
            0.5 (x_n - y_n)^2 / beta, & \text{if } |x_n - y_n| < beta \\
            |x_n - y_n| - 0.5 * beta, & \text{otherwise }
            \end{cases}

        若reduction不为`none`，则：

        .. math::
            \ell(x, y) =
            \begin{cases}
                \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
                \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
            \end{cases}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *target* ( **Tensor** ) : 目标张量
        - *reduction* ( **string** ，可选 ) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean*。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum`` : 输出求和。 其默认值为 *mean*
        - *beta* ( **float** ，可选 ) : 用于约束损失变化的阈值，必须为非负数，默认值为 1.0
    C API
        :guilabel:`diopiSmoothL1Loss`
""")

add_docstr("maximum", r"""
    释义
        逐元素比较张量 :attr:`input` 与张量 :attr:`other` ，返回较大元素

        .. note:: 当元素为 ``NaN`` 时，直接返回该元素
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor** ) : 比较张量
    C API
        :guilabel:`diopiMaximum`
""")

add_docstr("minimum", r"""
    释义
        逐元素比较张量 :attr:`input` 与 :attr:`other` ，返回较小元素

        .. note:: 当元素为 ``NaN`` 时，直接返回该元素
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *other* ( **Tensor** ) : 比较张量
    C API
        :guilabel:`diopiMinimum`
""")

add_docstr("mm", r"""
    释义
        对张量 :attr:`input` 与 :attr:`mat2` 进行矩阵乘法运算

        若张量 :attr:`input` 形状为 :math:`(n \times m)` ,张量 :attr:`mat2` 形状为
        :math:`(m \times p)` ,则输出 :attr:`out` 形状为 :math:`(n \times p)`
    参数
        - *input* ( **Tensor** ) : 被乘张量
        - *other* ( **Tensor** ) : 乘张量
    C API
        :guilabel:`diopiMm`
""")

add_docstr("conv3d", r"""
    释义
        对三维输入张量 :attr:`input` 进行3D卷积操作
    参数
        - *input* ( **Tensor** ) : 输入张量，形状为 :math:`(\text{minibatch} , \text{in_channels} , iT , iH , iW)`
        - *weight* ( **Tensor** ) : 卷积核权重张量，形状为  :math:`(\text{out_channels} , \frac{\text{in_channels}}{\text{groups}} , kT , kH , kW)`
        - *bias* ( **Tensor** ) : 偏置张量，形状为 :math:`(\text{out_channels})`，默认为None
        - *stride* ( **Number** 或者 **tuple** ) : 卷积核步长，默认值为 1
        - *padding* ( **Number** 或者 **tuple** ) : 输入张量拓展零值填充大小，默认值为 0
        - *groups* ( **Number** 或者 **tuple** ) : 将输入张量进行分组，其中 :math:`\text{in_channels}` 必须能够整除该值，默认值为 1
        - *dilation* ( **Number** 或者 **tuple** ) : 卷积核元素之间的间距，默认值为 1
    C API
        :guilabel:`diopiConvolution3d`
""")

add_docstr("expand", r"""
    释义
        返回输入张量 :attr:`input` 扩展对应维度到 :attr:`size` 大小的新张量

        当 :attr:`size` 中元素值为 -1 时，表示该维度不被拓展。其中张量 :attr:`input` 也可以
        拓展到更高维，更高维度将被置于第一维之前，此时更高维对应  :attr:`size` 的值不能为 -1
        
        .. note:: 拓展张量不会分配新的内存，只是通过设置步幅为 0的方式来改变视图
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *size* ( **torch.Size** 或者 **int** ) : 各个维度拓展大小
    C API
        :guilabel:`diopiExpand`
""")

add_docstr("unfold", r"""
    释义
        在维度 :attr:`dimension` 上对输入张量 :attr:`input` 以 :attr:`size` 为大小
        以 :attr:`step` 为步幅进行切片，将切片结果作为新张量返回
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dimension* ( **int** ) : 切片实施的维度
        - *size* ( **int**) : 切片大小
        - *step* (**int**) : 切片步幅
    C API
        :guilabel:`diopiUnfold`
""")

add_docstr("masked_select", r"""
    释义
        依据掩码 :attr:`mask` 选择输入张量 :attr:`input` 对应元素的值，并返回由这些值组成的一维张量

        .. note:: 新张量与原张量存储位置不同
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *mask* ( **Tensor** ) : 掩码张量
    C API
        :guilabel:`diopiMaskedSelect`
""")

add_docstr("index_fill", r"""
    释义
        在维度 :attr:`dim` ，依据索引 :attr:`index` 对输入张量 :attr:`input` 填充  :attr:`value` 
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 对输入张量实施填充的维度
        - *index* ( **LongTensor** ) : 所需填充的索引下标组成的张量
        - *value* ( **float** 或者 **0-dimension-Tensor** ) : 填充值
    C API
        :guilabel:`diopiIndexFill` :guilabel:`diopiIndexFillScalar` :guilabel:`diopiIndexFillInp` :guilabel:`diopiIndexFillInpScalar`
""")

add_docstr("linspace", r"""
    释义
        创建并返回一个长度为 :attr:`step` 大小，从 :attr:`start` 到 :attr:`end` 的一维张量，该张量取值如下：

        .. math::
            (\text{start},
            \text{start} + \frac{\text{end} - \text{start}}{\text{steps} - 1},
            \ldots,
            \text{start} + (\text{steps} - 2) * \frac{\text{end} - \text{start}}{\text{steps} - 1},
            \text{end})
    参数
        - *start* ( **float** ) : 张量起始点（包括）
        - *end* ( **float** ) : 张量终止点（包括）
        - *step* ( **int** ) : 张量大小
    C API
        :guilabel:`diopiLinspace`
""")

add_docstr("roll", r"""
    释义
        沿着维度 :attr:`dim` 循环滚动输入张量 :attr:`input` 的元素。其中元素偏移大小为 :attr:`shifts` 
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *shifts* ( **int** 或者 **tuple** ) : 张量元素移动的位置数，如果为元组，则大小必须与dim元组相同
        - *dim* ( **int** 或者 **tuple** ) : 对输入张量实施循环滚动的维度
    C API
        :guilabel:`diopiRoll`
""")

add_docstr("norm", r"""
    释义
        返回给定张量的  :attr:`p` 矩阵范数或向量范数。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *p* ( **int** 或者 **float** 或者 **inf** 或者 **-inf** 或者 **'fro'** 或者 **'nuc'** ) : 范数类型
          以下范数类型可被计算：

          ======  ==============  ==========================
          ord     matrix norm     vector norm
          ======  ==============  ==========================
          'fro'   Frobenius norm  --
          'nuc'   nuclear norm    --
          Number  --              sum(abs(x)**ord)**(1./ord)
          ======  ==============  ==========================

        - *dim* ( **int** 或者 **tuple** ，可选) : 对输入张量计算范数的维度，当为None时对整个张量计算范数
        - *keepdim* ( **bool** ，可选) : 结果是否保留原有维度，若dim为None，则忽略该值。默认值为False
    C API
        :guilabel:`diopiNorm`
""")

add_docstr("group_norm", r"""
    释义
        对输入张量 :attr:`input` 在通道维度上分组进行标准化

        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

        将输入张量 :attr:`input` 的通道维度(第二维)划分为 :attr:`num_groups` 组，每组包含 :math:`num\_channels/num\_groups` 个通道数，
        对每组分别求均值和标准差进而标准化。
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *num_groups* ( **int** ) : 输入张量通道被分组的个数，必须能被通道数整除
        - *weight* ( **Tensor** ) : 计算标准化所需的权重，该参数可被学习
        - *bias* ( **Tensor** ) : 计算标准化所需的偏置项，该参数可被学习
        - *eps* ( **float** ) : 计算标准化数值稳定系数，与分母相加，默认值为 1e-5
    形状
        - *input* : 形状为 :math:`(N, C, *)` ，其中 C = num_channels
        - *output* : 形状为  :math:`(N, C, *)` ，与输入张量相同
    C API
        :guilabel:`diopiGroupNorm`
""")

add_docstr("layer_norm", r"""
    释义
        对输入张量 :attr:`input` 进行层标准化

        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
        
        其中均值和方差从输入张量的后 D 维计算得到，D为 :attr:`normalized_shape` 的大小
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *normalized_shape* ( **int** 或者 **list** 或者 **torch.Size** ) : 进行层标准化的形状
          
          .. math::
                [* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
                    \times \ldots \times \text{normalized_shape}[-1]]
        - *weight* ( **Tensor** ) : 计算标准化所需的权重，该参数可被学习
        - *bias* ( **Tensor** ) : 计算标准化所需的偏置项，该参数可被学习
        - *eps* ( **float** ) : 计算标准化数值稳定系数，与分母相加，默认值为 1e-5
    形状
        - *input* : 形状为 :math:`(N, *)` 
        - *output* : 形状为  :math:`(N, *)` ，与输入张量相同
    C API
        :guilabel:`diopiLayerNorm`
""")

add_docstr("adaptive_avg_pool3d", r"""
    释义
        对输入张量 :attr:`input` 进行自适应3D平均值池化

        对任意输入张量 :attr:`input` ，其输出张量后三维的维度大小总为 :attr:`output_size` ，输出张量的通道特征与输入张量相同
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *output_size* ( **Union[int, None, Tuple[Optional[int], Optional[int], Optional[int]]]** ) : 输出张量的大小，其形式为
          `DxHxW`  ，可以为三元组，也可以为单个数值 `D` 。当为单个数值时，自动扩充为 `DxDxD` 。其中 `D` , `H` 以及 `W` 可以为 ``int``, or ``None`` ，当
          为 ``None`` 时，其值与对应输入张量大小相同
    形状
        - *input* : 形状为 :math:`(N, C, H_{in}, W_{in})` 或者 :math:`(C, H_{in}, W_{in})`
        - *output* : 形状为  :math:`(N, C, S_{0}, S_{1}, S_{2})` 或者 :math:`(C, S_{0}, S_{1}, S_{2})`,
          其中 :math:`S=\text{output_size}`
    C API
        :guilabel:`diopiAdaptiveAvgPool3d`
""")

add_docstr("adaptive_max_pool3d", r"""
    释义
        对输入张量 :attr:`input` 进行自适应3D最大值池化

        对任意输入张量 :attr:`input` ，其输出张量后三维的维度大小总为 :attr:`output_size` ，输出张量的通道特征与输入张量相同
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *output_size* ( **Union[int, None, Tuple[Optional[int], Optional[int], Optional[int]]]** ) : 输出张量的大小，其形式为
          `DxHxW`  ，可以为三元组，也可以为单个数值 `D` 。当为单个数值时，自动扩充为 `DxDxD` 。其中 `D` , `H` 以及 `W` 可以为 ``int``, or ``None`` ，当
          为 ``None`` 时，其值与对应输入张量大小相同
        - *return_indices* ( **bool** ) : 是否返回池化结果值对应输入的索引下标，默认为 False
    形状
        - *input* : 形状为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 或者 :math:`(C, D_{in}, H_{in}, W_{in})`
        - *output* : 形状为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 或者 :math:`(C, D_{out}, H_{out}, W_{out})`,
          其中 :math:`(D_{out}, H_{out}, W_{out})=\text{output_size}`
    C API
        :guilabel:`diopiAdaptiveMaxPool3d` :guilabel:`diopiAdaptiveMaxPool3dWithIndices`
""")

add_docstr("max_pool3d", r"""
    释义
        对输入张量 :attr:`input` 进行3D最大值池化

        若输入张量 :attr:`input` ，其输出大小为  :math:`(N, C, D, H, W)` ，
        滑动窗口 :attr:`kernel_size` 大小为 :math:`(kD, kH, kW)` ，输出张量 :attr:`out` 的大小 :math:`(N, C, D_{out}, H_{out}, W_{out})` 计算方式如下：

        .. math::
            \begin{aligned}
                \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                                & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                                \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
            \end{aligned}
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *kernel_size* ( **Union[int, Tuple[int, int, int]]** ) : 池化区域的大小， 可以是单个数字或三元组 :math:`(kD, kH, kW)` ，当为单个数值时表示三个维度值相同
        - *stride* ( **Union[int, Tuple[int, int, int]]** ) : 池化操作的步长。 可以是单个数字或三元组 :math:`(sD, sH, sW)` ，当为单个数值时表示三个维度值相同。 默认值为 kernel_size
        - *padding* ( **Union[int, Tuple[int, int, int]]** ) : 在输入张量三个维度上隐式负无穷大填充，可以是单个数字或三元组 :math:`(pD, pH, pW)` ，当为单个数值时表示三个维度值相同
        - *dilation* ( **Union[int, Tuple[int, int, int]]** ) : 滑动窗口内元素之间的间隔，其值必须大于 0 ，可以是单个数字或三元组 :math:`(dD, dH, dW)` ，当为单个数值时表示三个维度值相同
        - *ceil_mode* ( **bool** ) : 如果为 ``True`` ，将使用向上取整而不是默认的向下取整来计算输出形状。 这确保了输入张量中的每个元素都被滑动窗口覆盖。
        - *return_indices* ( **bool** ) : 如果为 ``True`` ，将返回所有滑动窗口产生的最大值的位置索引，该结果将后续会被应用于反池化。默认值为 False
    形状
        - *input* : 形状为 :math:`(N, C, D_{in}, H_{in}, W_{in})` 或者 :math:`(C, D_{in}, H_{in}, W_{in})`
        - *output* : 形状为 :math:`(N, C, D_{out}, H_{out}, W_{out})` 或者 :math:`(C, D_{out}, H_{out}, W_{out})`  其中
        
          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor
    C API
        :guilabel:`diopiMaxPool3d` :guilabel:`diopiMaxPool3dWithIndices`
""")

add_docstr("permute", r"""
    释义
        返回输入张量 :attr:`input` 按 :attr:`dims` 位置顺序置换维度后的新张量
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dims* ( **tuple** ) : 置换时张量维度的位置顺序
    C API
        :guilabel:`diopiPermute`
""")

add_docstr("copy_", r"""
    释义
        从张量 :attr:`other` 拷贝数据到张量 :attr:`input` 并返回

        .. note:: 张量 :attr:`other` 必须是可广播的，其数据类型及所在设备可不受限制
    参数
        - *input* ( **Tensor** ) : 被赋值数据的张量
        - *other* ( **Tensor** ) : 数据源张量
    C API
        :guilabel:`diopiCopyInp`
""")

add_docstr("gather", r"""
    释义
        从输入张量 :attr:`input` 沿着维度 :attr:`dim` 按照 :attr:`index` 收集值

        对于一个3D张量，其收集数据方式可为 ::

            out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
            out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
            out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

        .. note:: :attr:`input` 与 :attr:`index` 必须有相同大小的维度，而且对任意维度 ``d != dim`` 有 ``index.size(d) <= input.size(d)``
        
        输出张量 :attr:`out` 形状与 :attr:`index` 相同
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 收集数据的维度
        - *index* ( **LongTensor** ) : 收集元素的索引下标
    C API
        :guilabel:`diopiGather`
""")

add_docstr("remainder", r"""
    释义
        返回输入张量 :attr:`input` 或输入标量 :attr:`self` 除以 :attr:`other` 的模，其中 :attr:`other` 可为张量或者标量
    参数
        - *self* ( **Scalar** ) : 被除数，可为 None ，当input 为 None时， 按照self计算
        - *input* ( **Tensor** ) : 被除张量， 可为 None， 当为None时 self 不为None
        - *other* ( **Tensor** 或者 **Scalar** ) : 除数或除张量
    C API
        :guilabel:`diopiRemainder` :guilabel:`diopiRemainderScalar` :guilabel:`diopiRemainderTensor`
""")

add_docstr("ctc_loss", r"""
    释义
        计算输入连续时序张量 :attr:`log_probs` 与目标张量 :attr:`tragets` 之间的CTC连续时序损失，用于
        解决label与target不对齐的问题

        该损失函数通过对齐输入连续时序张量 :attr:`log_probs` 与目标张量 :attr:`tragets` 的概率并求和得到损失值
    参数
        - *log_probs* ( **Scalar** ) : 连续时序预测值，形状为 :math:`(T, N, C)` 或者 :math:`(T, C)` ，其中 T 为时序长度、
          N 为batch_size、C为类别个数（包括空label）
        - *targets* ( **Tensor** ) : 序列目标值，形状为 :math:`(N, S)` or
          :math:`(\operatorname{sum}(\text{target_lengths}))` ，其中N为batch_size， S为目标序列的最大长度，其中目标序列值
          为类别索引下标。当形状为 :math:`(N, S)` ，目标序列已被填补至最大序列长度。当形状为 :math:`(\operatorname{sum}(\text{target_lengths}))`
          表示目标序列为被填补，并被拼接在一个维度上
        - *input_lengths* ( **Tensor** 或者 **tuple** ) : 表示每个输入的长度，其大小为 N，即batch_size大小，其每个元素的值必须 :math:`\leq T` 
        - *target_lengths* ( **Tensor** 或者 **tuple** ) : 表示每个目标序列的长度，其大小为 N，即batch_size大小，其每个元素的值必须 :math:`\leq S`
          当target的形状为 :math:`(N, S)` 时，可通过该值目标时序中标签的真实长度，即 ``target_n = targets[n,0:s_n]`` 。
          当target的形状为 :math:`(\operatorname{sum}(\text{target_lengths}))` ，其和必须与 traget 张量长度相同
        - *blank* ( **int** ，可选) : 空白标签，默认为0
        - *reduction* ( **string** ，可选) : 损失归约方式, 可以为 *none* , *sum* 或者 *mean* 。
          其中 ``none`` : 不使用任何归约, ``mean`` : 输出的和除以输出的元素个数, 即求均值, ``sum``: 输出求和。其默认值为 *mean*
        - *zero_infinity* ( **bool** ，可选) : 是否将无穷损失（当输入序列太短无法对其目标序列时会出现）和相关梯度归零，默认为False
    C API
        :guilabel:`diopiCTCLoss`
""")

add_docstr("index_put", r"""
    释义
        依据下标索引 :attr:`indices1` 与 :attr:`indices2` 将 :attr:`values` 赋值或添加到 :attr:`input` 上

        当 :attr:`accumulate` 为 False时： ``tensor[indices1][indices2] = values`` 
        
        .. note:: 其中当赋值过程中索引重复时，赋值结果不唯一确定

        当 :attr:`accumulate` 为 True时： ``tensor[indices1][indices2] += values``
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *indices1* ( **LongTensor** ) : 索引横坐标
        - *indices2* ( **LongTensor** ) : 索引纵坐标
        - *values* ( **LongTensor** ) : 与输入张量数据类型相同
        - *accumulate* ( **bool** ) : 是否将元素值累加
    C API
        :guilabel:`diopiIndexPut` :guilabel:`diopiIndexPutInp`
""")

add_docstr("scatter", r"""
    释义
        将张量 :attr:`src` 或 :attr:`value` 按照 :attr:`index` 全部赋值给输入张量 :attr:`input` ，
        其中在赋值时，输入张量在维度  :attr:`dim` 取值为  :attr:`index` 对应索引的值，例如： 

        对于一个 `3D` 张量
        
        当 :attr:`reduce` 为 :attr:`None` 时， 其数据赋值方式可为 ::

            self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
            self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
            self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

        当 :attr:`reduce` 为 'add' 或 'mutiply' 时， 其数据赋值方式为加或者乘

        .. note:: 其中 :attr:`input` 与 :attr:`index` 及 :attr:`src` 必须有相同大小的维度，而且对任意维度 ``d != dim`` 有 ``index.size(d) <= src.size(d)`` 和 ``index.size(d) <= input.size(d)`` 
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 取索引值的维度
        - *index* ( **LongTensor** ) : 索引下标
        - *src* ( **Tensor** ) : 数据源张量
        - *value* ( **float** ) : 数据源标量，当src为None时，采用该值对输入张量赋值
        - *reduce* ( **string** ) : 数据赋值方式，可为‘add’或者 ‘multiply’ ，默认为None
    C API
        :guilabel:`diopiScatter` :guilabel:`diopiScatterInp` :guilabel:`diopiScatterInpScalar` :guilabel:`diopiScatterScalar`
""")

add_docstr("interpolate", r"""
    释义
        按照指定的 :attr:`size` 对输入张量 :attr:`input` 上采样，其中采样方式由 :attr:`mode` 指定 
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *size* ( **int** 或者 **tuple** ) : 采样后张量的大小
        - *mode* ( **string** ) : 采样方式，其值可为 'bilinear', 'nearest', 'bicubic', 'trilinear', 'linear'， 默认值为 'nearest'
        - *align_corners* ( **bool** ，可选) : 是否采用中心像素点进行输入输出对齐，如果为True，则使用中心像素点来对齐输入输出的。如果为False，则
          使用角像素点进行对齐，并使用差值来填充超出边界的部分
    C API
        :guilabel:`diopiUpsampleNearest` :guilabel:`diopiUpsampleLinear`
""")

add_docstr("pad", r"""
    释义
        对输入张量 :attr:`input` 的后 :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` 维度以方式 :attr:`mode` 进行扩展填补，其中
        填补维度对应方式为从最后一维度开始向前， :attr:`pad` 前两个值用于最后一个维度的填补，依次向前填补 

        每个维度填补大小由  :attr:`pad` 指定，例如：

        当pad为： :math:`(\text{padding_left}, \text{padding_right})` 
        对输入张量的最后一个维度填补

        当pad为： :math:`(\text{padding_left}, \text{padding_right},`
        :math:`\text{padding_top}, \text{padding_bottom}`
        :math:`\text{padding_front}, \text{padding_back})`
        对输入张量的后三个维度进行填补

        .. note:: 当 :attr:`mode` 为'const'时可以对任意维度张量进行填补。当 :attr:`mode` 为 'replicate' 或者 'reflect'
            时，可对4维或5维张量的后3维填补，或者对3维或者4维张量的后2维填补，或者对2维或者3维张量的最后一维进行填补
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *pad* ( **tuple** ) : 指定对张量填补的大小，若其其长度为m，则m必须为偶数且 :math:`\frac{m}{2} \leq` 输入张量的维度
        - *mode* ( **string** ) : 填补方式，其值可为 ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'`` ， 默认值为 ``'constant'``
        - *value* ( **bool** ，可选) : 当填补方式为 ``'constant'`` ，所需的填补值
    C API
        :guilabel:`diopiPad`
""")

add_docstr("unique", r"""
    释义
        去除输入张量 :attr:`input` 中重复值，并返回该新张量
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *sorted* ( **bool** ) : 是否对结果进行升序排序
        - *return_inverse* ( **bool** ) : 是否返回索引张量，其中索引张量形状与输入张量相同，表示输入张量中每个元素在输出张量中的新下标
        - *return_counts* ( **bool** ) : 是否返回计数张量，其中计数张量形状与输出张量或者输入张量在维度 ``dim`` 的大小相同，表示输出张量中元素的计数个数
        - *dim* ( **int** ) : 指定实施去重的维度，可为None，当为None时对整个输入张量去重，默认值为None
    C API
        :guilabel:`diopiUnique`
""")

add_docstr("prod", r"""
    释义
        沿着维度  :attr:`dim` 对输入张量 :attr:`input` 的元素求乘积
    参数
        - *input* ( **Tensor** ) : 输入张量
        - *dim* ( **int** ) : 做乘积操作的维度
        - *keepdim* ( **bool** ) : 结果是否保留原有维度
    C API
        :guilabel:`diopiProd`
""")