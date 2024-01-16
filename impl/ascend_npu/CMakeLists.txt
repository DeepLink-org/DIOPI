cmake_minimum_required(VERSION 3.14)
project(ascend_impl)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)
message(STATUS "downloading op-plugin")
FetchContent_Declare(op_plugin
  GIT_REPOSITORY https://gitee.com/ascend/op-plugin
  GIT_TAG 186f4b87e2db4791a6ed697cea78d58a0ae92db1
)
FetchContent_MakeAvailable(op_plugin)
message(STATUS "op-plugin download done")

add_custom_target(op_plugin_gen
  COMMAND cd ${op_plugin_SOURCE_DIR} && bash ./gencode.sh 2.0 python
)
add_subdirectory(third_party/acl)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/acl/inc)

# find python3
find_package(Python COMPONENTS Interpreter Development)
MESSAGE(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
MESSAGE(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
include_directories(SYSTEM ${Python_INCLUDE_DIRS})
link_directories(${Python_LIBRARIES})

# find torch
if(NOT DEFINED PYTORCH_DIR)
  execute_process(
    COMMAND sh -c "python -c 'import torch;print(torch.__path__[0])'"
    OUTPUT_VARIABLE PYTORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_package(Torch REQUIRED
    PATHS  ${PYTORCH_DIR}
)
message(STATUS "TORCH_INSTALL_PREFIX:" ${TORCH_INSTALL_PREFIX})
message(STATUS "TORCH_CXX_FLAGS:" ${TORCH_CXX_FLAGS})   # include -D_GLIBCXX_USE_CXX11_ABI
message(STATUS "TORCH_INCLUDE_DIRS:" ${TORCH_INCLUDE_DIRS})

include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_INSTALL_PREFIX}/lib)

# Because many interfaces have not yet been implemented, -Wreturn-type=0 option is temporarily added.
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wno-return-type -Wno-unused-function -Wno-unused-but-set-variable -Wno-unused-variable -g -O0 "
)

set(OP_PLUGIN_DIR ${op_plugin_SOURCE_DIR}/op_plugin/ops/base_ops/aclops/)
set(OP_PLUGIN_UTILS_DIR ${op_plugin_SOURCE_DIR}/op_plugin/utils/)
file(GLOB_RECURSE OP_PLUGIN_UTILS_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${OP_PLUGIN_UTILS_DIR}/*.cpp)

set(OP_PLUGIN_SRC
    ${OP_PLUGIN_SRC}
    #${CMAKE_CURRENT_SOURCE_DIR}/op-plugin/op_plugin/OpInterface.cpp
    ${OP_PLUGIN_DIR}/AbsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AcoshKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AcosKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveAvgPool1dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveAvgPool2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveAvgPool2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveAvgPool3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveAvgPool3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveMaxPool2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveMaxPool2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddbmmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddcdivKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddcmulKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddmmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddmvKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddReluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddrKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AffineGridGeneratorBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AffineGridGeneratorKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AllKernelNpu.cpp
    ${OP_PLUGIN_DIR}/_AminmaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AmpForeachNonFiniteCheckAndUnscaleKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AnchorResponseFlagsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AnyKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ApplyAdamKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ApplyAdamWKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ArangeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ArgmaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ArgminKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ArgsortKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AsinhKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AsinKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AsStridedKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Atan2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/AtanhKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AtanKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AvgPool2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AvgPool2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AvgPool3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AvgPool3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BaddbmmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNMSKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormBackwardElemtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormBackwardReduceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormElemtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormGatherStatsUpdateKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormGatherStatsWithCountsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormReduceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormStatsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BernoulliKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BertApplyAdamKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BinaryCrossEntropyBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BinaryCrossEntropyKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BinaryCrossEntropyWithLogitsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BincountKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BitwiseAndKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BitwiseNotKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BitwiseOrKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BitwiseXorKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BmmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BmmV2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/BoundingBoxDecodeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BoundingBoxEncodeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BroadcastKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CastKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CatKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CdistBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CdistKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CeilKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CeluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ChannelShuffleKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CiouKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ClampKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Col2imKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ComplexKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConfusionTransposeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConstantPadNdKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Conv2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Conv2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Conv3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Conv3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvDepthwise2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvDepthWise2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvTbcBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvtbcKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvTranspose2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvTranspose2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvTranspose3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CoshKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CosKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CountNonZeroKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CropAndResizeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CtcLossBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CtcLossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CummaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CumminKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CumprodKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CumsumKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DecodeJpegKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DeformableConv2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DiagKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DiouKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DivKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DotKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DropoutKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DropoutWithAddSoftmaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DropoutWithByteMaskKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EluBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EmbeddingBagDenseBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EmbeddingBagKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EmbeddingDenseBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EmbeddingRenormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EqKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EqualKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ErfcKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ErfinvKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ErfKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Exp2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/ExpKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Expm1KernelNpu.cpp
    ${OP_PLUGIN_DIR}/EyeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FastGeluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FillDiagonalKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FillKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FlipKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FloatStatusKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FloorDivideKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FloorKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FmodKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FracKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FusedAttentionLnQKV.cpp
    ${OP_PLUGIN_DIR}/FusedAttentionQKVGradKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FusedAttentionScoreKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GatherKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GcdKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GerKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GiouKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GluBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GridAssignPositiveKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GridSampler2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GridSampler2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GridSampler3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GroupNormBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GroupNormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GruKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardShrinkBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardShrinkKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardsigmoidBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardsigmoidKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardSwishBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardSwishKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardtanhBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardtanhKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IfmrKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__iLshift__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/Im2colKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ImageNormalizeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ImgToTensorKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IndexAddKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IndexFillKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IndexingKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IndexPutKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IndexSelectKernelNpu.cpp
    ${OP_PLUGIN_DIR}/InverseKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__Ior__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/IouKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__iRshift__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/IscloseKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IsfiniteKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IsInKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IsInTensorScalarKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IsNegInfKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IsPosInfKernelNpu.cpp
    ${OP_PLUGIN_DIR}/KthvalueKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LayerNormBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LayerNormEvalKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LayerNormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LeakyReluBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LeakyReluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LerpKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LinalgCrossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LinalgQrKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LinearKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LinspaceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Log10KernelNpu.cpp
    ${OP_PLUGIN_DIR}/Log1pKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Log2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogAddExp2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogAddExpKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogicalAndKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogicalNotKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogicalOrKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogicalXorkernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSigmoidBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSigmoidKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSoftmaxBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSoftmaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSpaceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSumExpKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__Lshift__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/LstmCellKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LstmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaskedFillKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaskedFillRangeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaskedScatterKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaskedSelectKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MatmulBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MatmulKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxPool2dWithIndicesBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxPool2dWithIndicesKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxPool3dWithIndicesBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxPool3dWithIndicesKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxUnpool2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxUnpool3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxV1KernelNpu.cpp
    ${OP_PLUGIN_DIR}/MeanKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MedianKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MinKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MinV1KernelNpu.cpp
    ${OP_PLUGIN_DIR}/MishBackwardV2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/MishKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MishV2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/MmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MseLossBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MseLossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MulKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MultiHeadAttentionKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MultilabelMarginLossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MultinomialKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MvKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NanToNumKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NativeDropoutKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NegKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NLLLoss2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NLLLoss2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NLLLossBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NLLLossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NmsRotatedKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NmsV4KernelNpu.cpp
    ${OP_PLUGIN_DIR}/NmsWithMaskKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NnpackSpatialConvolutionKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NonzeroKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NormalizationKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NormalizeBatchKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NormalKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/OneHotKernelNpu.cpp
    ${OP_PLUGIN_DIR}/OnehotNpu.cpp
    ${OP_PLUGIN_DIR}/OnesKernelNpu.cpp
    ${OP_PLUGIN_DIR}/OnesLikeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PackPaddedSequenceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PadKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PadPackedSequenceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PdistKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PowKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ProdKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PsRoiPoolingKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PutKernelNpu.cpp
    ${OP_PLUGIN_DIR}/QuantizePerChannelKernelNpu.cpp
    ${OP_PLUGIN_DIR}/QuantizePerTensorKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RandomChoiceWithMaskKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RandomKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RandpermKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RangeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RealKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReciprocalKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReflectionPad1dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReflectionPad1dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReflectionPad2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReflectionPad2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RemainderKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RenormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RepeatKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReplicationPad1dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReplicationPad1dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReplicationPad2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReplicationPad2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReshapeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReverseKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RoiAlignBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RoiAlignKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RollKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RotaryMulKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RotatedBoxDecodeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RotatedBoxEncodeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RotatedIouKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RotatedOverlapsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RoundDecimalsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RoundKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__Rshift__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/RsqrtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RsubKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ScaledMaskedSoftmaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ScatterAddKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ScatterKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ScatterV1KernelNpu.cpp
    ${OP_PLUGIN_DIR}/SearchsortedKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SeluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SigmoidBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SigmoidKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SignBitsPackKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SignBitsUnpackKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SignKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SiluBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SiluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SincKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SinhKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SinKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SliceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SlogdetKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SlowConvDilated2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SlowConvDilated2DKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SlowConvTranspose2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SlowConvTranspose2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SmoothL1LossBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SmoothL1LossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftMarginLossBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftMarginLossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftmaxBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftmaxCrossEntropyWithLogitsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftmaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftplusBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftplusKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftShrinkBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftShrinkKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SortKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SortWithoutIndicesKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SqrtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/StackKernelNpu.cpp
    ${OP_PLUGIN_DIR}/StrideAddKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SubKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SubSampleKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SumKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TakeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TanhBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TanhKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TanKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ThresholdBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ThresholdKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TopKKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TransposeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TriangularSolveKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TrilKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TriuKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TrueDivideKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TruncKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UniformKernelNpu.cpp
    ${OP_PLUGIN_DIR}/_Unique2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/UniqueConsecutiveKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpSampleBicubic2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleBicubic2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleBilinear2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleBilinear2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleLinear1dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleLinear1dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleNearest1dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleNearest1dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleNearest2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleNearest2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleNearest3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpSampleNearest3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleTrilinear3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleTrilinear3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/VarKernelNpu.cpp
    ${OP_PLUGIN_DIR}/VDotKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ViewcopyKernelNpu.cpp
    ${OP_PLUGIN_DIR}/WhereKernelNpu.cpp
    ${OP_PLUGIN_DIR}/XlogyKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__Xor__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/YoloBoxesEncodeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ZerosKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ZerosLikeKernelNpu.cpp
)

set(OP_PLUGIN_SRC_V_DIR  ${op_plugin_SOURCE_DIR}/op_plugin/ops/v2r0/aclops/)
set(OP_PLUGIN_SRC
    ${OP_PLUGIN_SRC}
    ${OP_PLUGIN_SRC_V_DIR}/SumKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/BatchNormKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/BmmV2KernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/CatKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/ConfusionTransposeKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/ConvolutionKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/CumprodKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/CumsumKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/EmbeddingBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/EmbeddingBagBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/EmbeddingKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/GeluBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/GeluKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/GridSampler3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/Im2colKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/IndexAddKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/IndexCopyKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/IndexKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/KlDivKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/L1LossKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/LinalgSvdKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/MaxUnpool2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/MaxUnpool3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/MaxV1KernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/MeanKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/MinV1KernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/PreluBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/PreluKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/PsRoiPoolingKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/RepeatInterLeaveKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/RreluWithNoiseKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/ScatterKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/SoftplusBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/StdKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/SumKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/SymeigKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/TriangularSolveKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/VarKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/WeightNormKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/WhereKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/ZerosKernelNpu.cpp
)

set(OP_PLUGIN_API_DIR ${op_plugin_SOURCE_DIR}/op_plugin/ops/base_ops/opapi/)
set(OP_PLUGIN_API_SRC1
    ${OP_PLUGIN_API_DIR}/AddKernelNpuOpApi.cpp
)

set(OP_PLUGIN_UTILS_SRC
    #${OP_PLUGIN_UTILS_DIR}/AdvancedIndex.cpp
    ${OP_PLUGIN_UTILS_DIR}/KernelNpuOutputSize.cpp
    ${OP_PLUGIN_UTILS_DIR}/op_api_common.cpp
    ${OP_PLUGIN_UTILS_DIR}/OpUtils.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/EmbeddingKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/GeluBackwardKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/GeluKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/GridSampler3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/IndexCopyKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/IndexKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/LeakyReluBackwardKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/LinalgSvdKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/LogSoftmaxKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/MeanKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/PreluBackwardKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/PreluKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/RepeatInterLeaveKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/ScatterKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/SoftplusBackwardKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/SumKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/TriangularSolveKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/VarKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/WhereKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/ZerosKernelNpu.cpp
)




set(ASCEND_IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/torch_npu/csrc/impl/)
set(ASCEND_IMPL_SRC

)

set(OP_PLUGIN_API_DIR ${op_plugin_SOURCE_DIR}/op_plugin/ops/v2r0/opapi/)
set(OP_PLUGIN_API_SRC
    ${OP_PLUGIN_API_SRC}
)

set(DIOPI_IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/diopi_impl/)
file(GLOB DIOPI_IMPL_SRC "${DIOPI_IMPL_DIR}/*.cpp" "${DIOPI_IMPL_DIR}/functions_ext/*.cpp")

add_definitions(-DBUILD_LIBTORCH)


include_directories(SYSTEM ${op_plugin_SOURCE_DIR} torch_npu ${CMAKE_CURRENT_SOURCE_DIR})

set(IMPL_SRC
  torch_npu/csrc/DIOPIAdapter.cpp
  torch_npu/csrc/AclOpCompileInterface.cpp
  torch_npu/csrc/NpuVariables.cpp
  torch_npu/csrc/AdvancedIndex.cpp
  torch_npu/csrc/NPUNativeFunctions.cpp
  torch_npu/csrc/CopyKernel.cpp
)

set(OLD_IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ascend/)
set(OLD_IMPL_SRC
    ${OLD_IMPL_DIR}/ascend_tensor.cpp
    ${OLD_IMPL_DIR}/functions_mmcv/roi_align_npu.cpp
    ${OLD_IMPL_DIR}/functions/bmm.cpp
    ${OLD_IMPL_DIR}/functions/embedding.cpp
    ${OLD_IMPL_DIR}/functions/cumsum.cpp
    ${OLD_IMPL_DIR}/functions/fill.cpp
    ${OLD_IMPL_DIR}/functions/cat.cpp
    ${OLD_IMPL_DIR}/functions/pow.cpp
    ${OLD_IMPL_DIR}/functions/masked_fill.cpp
    ${OLD_IMPL_DIR}/functions/topk.cpp
    ${OLD_IMPL_DIR}/functions/baddbmm.cpp
    ${OLD_IMPL_DIR}/functions/conv2d.cpp
    ${OLD_IMPL_DIR}/functions/transpose.cpp
    ${OLD_IMPL_DIR}/functions/copy.cpp
    ${OLD_IMPL_DIR}/functions/normal.cpp
    ${OLD_IMPL_DIR}/functions/sort.cpp
    ${OLD_IMPL_DIR}/functions/isnan.cpp
    ${OLD_IMPL_DIR}/functions/activation.cpp
    ${OLD_IMPL_DIR}/functions/loss.cpp
    ${OLD_IMPL_DIR}/functions/contiguous.cpp
    ${OLD_IMPL_DIR}/functions/bitwise.cpp
    ${OLD_IMPL_DIR}/functions/upsample.cpp
    ${OLD_IMPL_DIR}/functions/tril.cpp
    ${OLD_IMPL_DIR}/functions/floor.cpp
    ${OLD_IMPL_DIR}/functions/mm.cpp
    ${OLD_IMPL_DIR}/functions/addcdiv.cpp
    ${OLD_IMPL_DIR}/functions/logic.cpp
    ${OLD_IMPL_DIR}/functions/linear.cpp
    ${OLD_IMPL_DIR}/functions/error.cpp
    ${OLD_IMPL_DIR}/functions/scatter.cpp
    ${OLD_IMPL_DIR}/functions/sgn.cpp
    ${OLD_IMPL_DIR}/functions/cast.cpp
    ${OLD_IMPL_DIR}/functions/sin.cpp
    ${OLD_IMPL_DIR}/functions/one_hot.cpp
    ${OLD_IMPL_DIR}/functions/triu.cpp
    ${OLD_IMPL_DIR}/functions/binary.cpp
    ${OLD_IMPL_DIR}/functions/cos.cpp
    ${OLD_IMPL_DIR}/functions/addcmul.cpp
    ${OLD_IMPL_DIR}/functions/interpolate.cpp
    ${OLD_IMPL_DIR}/functions/threshold.cpp
    ${OLD_IMPL_DIR}/functions/nonzero.cpp
    ${OLD_IMPL_DIR}/functions/stack.cpp
    ${OLD_IMPL_DIR}/functions/slice.cpp
    ${OLD_IMPL_DIR}/functions/unary.cpp
    ${OLD_IMPL_DIR}/functions/argmax.cpp
    ${OLD_IMPL_DIR}/functions/clamp.cpp
    ${OLD_IMPL_DIR}/functions/dropout.cpp
    ${OLD_IMPL_DIR}/functions/arange.cpp
    ${OLD_IMPL_DIR}/functions/gather.cpp
    ${OLD_IMPL_DIR}/functions/layer_norm.cpp
    ${OLD_IMPL_DIR}/functions/index_select.cpp
    ${OLD_IMPL_DIR}/functions/repeat.cpp
    ${OLD_IMPL_DIR}/functions/group_norm.cpp
    ${OLD_IMPL_DIR}/functions/flip.cpp
    ${OLD_IMPL_DIR}/functions/pool.cpp
    ${OLD_IMPL_DIR}/functions/uniform.cpp
    ${OLD_IMPL_DIR}/functions/batch_norm.cpp
    ${OLD_IMPL_DIR}/functions/minmax.cpp
    ${OLD_IMPL_DIR}/functions/norm.cpp
    ${OLD_IMPL_DIR}/functions/reduce.cpp
    ${OLD_IMPL_DIR}/functions/where.cpp
    ${OLD_IMPL_DIR}/functions/multinomial.cpp
    ${OLD_IMPL_DIR}/functions/addmm.cpp
    ${OLD_IMPL_DIR}/functions/expand.cpp
    ${OLD_IMPL_DIR}/functions/linspace.cpp
    ${OLD_IMPL_DIR}/functions/split.cpp
    ${OLD_IMPL_DIR}/functions_ext/rms_norm.cpp
    #${OLD_IMPL_DIR}/test/export_functions.cpp
    #${OLD_IMPL_DIR}/test/conform_test.cpp
    ${OLD_IMPL_DIR}/common/utils.cpp
    ${OLD_IMPL_DIR}/common/debug.cpp
    ${OLD_IMPL_DIR}/common/generator_helper.cpp
    ${OLD_IMPL_DIR}/common/stream_lock.cpp
    ${OLD_IMPL_DIR}/env_vars.cpp
    ${OLD_IMPL_DIR}/init.cpp
)

list(APPEND IMPL_SRC ${OP_PLUGIN_UTILS_SRC} ${OP_PLUGIN_SRC} ${OP_PLUGIN_API_SRC} ${ASCEND_IMPL_SRC} ${DIOPI_IMPL_SRC} ${OLD_IMPL_SRC})

# adaptor
set(USE_ADAPTOR ON)
if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(USE_ADAPTOR ON)
endif()

if(USE_ADAPTOR)
    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/../ascend/ ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})

    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")
    set(GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} --config_device=ascend --impl_plugin=True
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency)
    list(APPEND IMPL_SRC ${GEN_FILES} ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp)
    add_definitions(-DTEST_USE_ADAPTOR)
endif()

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
  set(ASCEND_DIR /usr/local/Ascend)
endif()

if(EXISTS ${ASCEND_DIR}/ascend-toolkit/latest/)
  message(STATUS "ascend-toolkit exists:" ${ASCEND_DIR}/ascend-toolkit/latest/)
  message(STATUS "ASCEND_DIR:" ${ASCEND_DIR})
  #include_directories(${ASCEND_DIR}/ascend-toolkit/latest/include/)
  link_directories(${ASCEND_DIR}/ascend-toolkit/latest/lib64)
else()
    message(FATAL_ERROR "No ascend-toolkit found.")
endif()


# third_party include
set(THIRD_PARTY_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/half/include)

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
add_dependencies(${DEVICEIMPL} op_plugin_gen)
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_include_directories(${DEVICEIMPL} SYSTEM PUBLIC ${THIRD_PARTY_INCLUDE_DIRS})
target_link_libraries(${DEVICEIMPL} ascendcl acl_op_compiler graph c10 torch_cpu ${Python_LIBRARIES})

if(USE_ADAPTOR)
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
endif()

if (TEST)
    add_subdirectory(test)
endif()
