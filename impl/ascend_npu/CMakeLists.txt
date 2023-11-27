cmake_minimum_required(VERSION 3.4)
project(ascend_impl)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(ASCEND_DIR $ENV{ASCEND_CUSTOM_PATH})
else()
  set(ASCEND_DIR /usr/local/Ascend)
endif()

execute_process(
    COMMAND sh -c "bash  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/acl/libs/build_stub.sh"
    COMMAND sh -c "echo 'download op-plugin ...'; cd ${CMAKE_CURRENT_SOURCE_DIR}/; git clone  https://gitee.com/ascend/op-plugin/ op-plugin || cd op-plugin && git checkout 186f4b87e2db4791a6ed697cea78d58a0ae92db1 && bash  ./gencode.sh 2.0 python "
)
add_subdirectory(third_party/acl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/acl/inc)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/acl/libs/)

#if(EXISTS ${ASCEND_DIR}/ascend-toolkit/latest/)
#  message(STATUS "ascend-toolkit exists:" ${ASCEND_DIR}/ascend-toolkit/latest/)
#  message(STATUS "ASCEND_DIR:" ${ASCEND_DIR})
#  #include_directories(${ASCEND_DIR}/ascend-toolkit/latest/include/)
#  link_directories(${ASCEND_DIR}/ascend-toolkit/latest/lib64)
#else()
#    message(FATAL_ERROR "No ascend-toolkit found.")
#endif()

if(NOT DEFINED PYTORCH_DIR)
  execute_process(
    COMMAND sh -c "dirname $(python -c 'import torch;print(torch.__path__[0])')"
    OUTPUT_VARIABLE PYTORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
set(TORCH_LIBRARY_DIR "${PYTORCH_DIR}/torch/lib")
link_directories(${TORCH_LIBRARY_DIR})
list(APPEND TORCH_INCLUDE_DIRS ${PYTORCH_DIR}/torch/include/
     ${PYTORCH_DIR}/torch/include/torch/csrc/api/include/)
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
message(STATUS "Torch TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")



if(NOT DEFINED DIPU_ABI_V)
  execute_process(
    COMMAND
      sh -x -c
      "python -c 'import torch, builtins; print(next(item[-4:-2] for item in dir(builtins)      \
          if \"__pybind11_internals_v4_gcc_libstdcpp_cxxabi10\" in item))'"
    OUTPUT_VARIABLE DIPU_ABI_V)
endif()

if(NOT DEFINED DIPU_COMPILED_WITH_CXX11_ABI)
execute_process(
    COMMAND
      sh -x -c
      "python -c 'import torch;print(1 if torch.compiled_with_cxx11_abi() else 0)'"
    OUTPUT_VARIABLE DIPU_COMPILED_WITH_CXX11_ABI)
endif()

if(DIPU_COMPILED_WITH_CXX11_ABI GREATER 0)
  set(DIPU_COMPILED_WITH_CXX11_ABI 1)
else()
  set(DIPU_COMPILED_WITH_CXX11_ABI 0)
endif()

add_definitions(-fabi-version=${DIPU_ABI_V})
message(STATUS "DIPU_ABI_V: ${DIPU_ABI_V}")

# Because many interfaces have not yet been implemented, -Wreturn-type=0 option is temporarily added.
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${DIPU_COMPILED_WITH_CXX11_ABI} -Wno-return-type -g -O0 "
)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=${DIPU_COMPILED_WITH_CXX11_ABI})
message(STATUS "DIPU_COMPILED_WITH_CXX11_ABI:" ${DIPU_COMPILED_WITH_CXX11_ABI})


set(OP_PLUGIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op-plugin/op_plugin/ops/base_ops/aclops/)
set(OP_PLUGIN_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op-plugin/op_plugin/utils/)
#file(GLOB_RECURSE OP_PLUGIN_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${OP_PLUGIN_DIR}/*.cpp)
#file(GLOB_RECURSE OP_PLUGIN_UTILS_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${OP_PLUGIN_UTILS_DIR}/*.cpp)
#file(GLOB_RECURSE OP_PLUGIN_API_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/op-plugin/op_plugin/ops/base_ops/opapi/*.cpp)

set(OP_PLUGIN_SRC
    #${OP_PLUGIN_UTILS_DIR}/../OpInterface.cpp
    ${OP_PLUGIN_DIR}/AddKernelNpu.cpp
    ${OP_PLUGIN_DIR}/StrideAddKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormElemtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormBackwardElemtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormBackwardKernelNpu.cpp
    #${OP_PLUGIN_DIR}/BatchNormReduceKernelNpu.cpp
    #${OP_PLUGIN_DIR}/BatchNormBackwardReduceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvDepthwise2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SlowConvTranspose2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SlowConvTranspose2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Conv2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Conv2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SumKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CastKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BroadcastKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxPool2dWithIndicesBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxPool2dWithIndicesKernelNpu.cpp
)
set(debug
    ${OP_PLUGIN_DIR}/TransposeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxPool3dWithIndicesKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SubSampleKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SubKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IsInKernelNpu.cpp
    ${OP_PLUGIN_DIR}/OnesKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpSampleBicubic2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/XlogyKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NmsV4KernelNpu.cpp
    ${OP_PLUGIN_DIR}/LerpKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GridAssignPositiveKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SinKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogAddExpKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__Ior__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/AcosKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MishBackwardV2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/AffineGridGeneratorBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NLLLoss2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MvKernelNpu.cpp
    ${OP_PLUGIN_DIR}/OnehotNpu.cpp
    ${OP_PLUGIN_DIR}/ReplicationPad1dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ApplyAdamWKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NonzeroKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IscloseKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EqKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EluBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RoundKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FloorKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MinKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TrilKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LstmCellKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvTranspose2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReflectionPad1dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SliceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BinaryCrossEntropyKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LeakyReluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleNearest2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EmbeddingRenormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleLinear1dBackwardKernelNpu.cpp
    #${OP_PLUGIN_DIR}/RenormKernelNpu.cpp
    #${OP_PLUGIN_DIR}/DropoutWithAddSoftmaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleNearest2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RotatedBoxDecodeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CropAndResizeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CumsumKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReverseKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RollKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GridSampler2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MseLossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MultilabelMarginLossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleLinear1dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogicalAndKernelNpu.cpp
    ${OP_PLUGIN_DIR}/_AminmaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardShrinkKernelNpu.cpp
    #${OP_PLUGIN_DIR}/ReshapeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IouKernelNpu.cpp
    #${OP_PLUGIN_DIR}/UniformKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GroupNormBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SqrtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NmsWithMaskKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FusedAttentionLnQKV.cpp
    ${OP_PLUGIN_DIR}/UniqueConsecutiveKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FracKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReflectionPad2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaskedFillRangeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveAvgPool2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PutKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CummaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveMaxPool2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BoundingBoxDecodeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AnchorResponseFlagsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DivKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxPool3dWithIndicesBackwardKernelNpu.cpp

    ${OP_PLUGIN_DIR}/SoftmaxKernelNpu.cpp
    #${OP_PLUGIN_DIR}/DropoutWithByteMaskKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GerKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GridSampler2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PackPaddedSequenceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IndexingKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReflectionPad1dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ImageNormalizeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CdistBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/KthvalueKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleBilinear2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardtanhBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConfusionTransposeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogAddExp2KernelNpu.cpp
    #${OP_PLUGIN_DIR}/BmmV2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/ProdKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__Rshift__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/GridSampler3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RsubKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SortWithoutIndicesKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AsinKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Conv3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CiouKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ScatterV1KernelNpu.cpp
    ${OP_PLUGIN_DIR}/ArgminKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SigmoidKernelNpu.cpp
    #${OP_PLUGIN_DIR}/ScatterKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SiluBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AcoshKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GroupNormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvtbcKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReplicationPad1dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MishV2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/Col2imKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SignBitsPackKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxV1KernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxUnpool3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftmaxBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNMSKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SinhKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SlowConvDilated2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IfmrKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CoshKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ComplexKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaskedSelectKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BertApplyAdamKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CeilKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LayerNormBackwardKernelNpu.cpp
    #${OP_PLUGIN_DIR}/NormalKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleBicubic2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveAvgPool3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FlipKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ArgmaxKernelNpu.cpp
    #${OP_PLUGIN_DIR}/LinearKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CeluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftMarginLossKernelNpu.cpp
    #${OP_PLUGIN_DIR}/ConvTranspose3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SiluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ErfKernelNpu.cpp
    #${OP_PLUGIN_DIR}/MmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ExpKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AnyKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LinalgQrKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PdistKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RsqrtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AvgPool3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TanhKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleNearest1dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PsRoiPoolingKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveAvgPool2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConstantPadNdKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AvgPool2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LinalgCrossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaskedScatterKernelNpu.cpp
    ${OP_PLUGIN_DIR}/QuantizePerChannelKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RepeatKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SignBitsUnpackKernelNpu.cpp
    #${OP_PLUGIN_DIR}/MatmulBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NegKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RoundDecimalsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TanhBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NLLLossBackwardKernelNpu.cpp
    #${OP_PLUGIN_DIR}/AsStridedKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TruncKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RotaryMulKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BitwiseNotKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LtKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RotatedBoxEncodeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardSwishBackwardKernelNpu.cpp

    ${OP_PLUGIN_DIR}/AvgPool2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SincKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LayerNormEvalKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ErfcKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IsPosInfKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EmbeddingBagKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSigmoidBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardShrinkBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DecodeJpegKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MedianKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSpaceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SlogdetKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DiouKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SigmoidBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FloorDivideKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NmsRotatedKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Log2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReplicationPad2dKernelNpu.cpp
    #${OP_PLUGIN_DIR}/IndexPutKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TriangularSolveKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RangeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CatKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpSampleNearest3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardSwishKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Atan2KernelNpu.cpp
    #${OP_PLUGIN_DIR}/RandomKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SignKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BincountKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReciprocalKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AbsKernelNpu.cpp
    #${OP_PLUGIN_DIR}/MultinomialKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SearchsortedKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NormalizationKernelNpu.cpp
    #${OP_PLUGIN_DIR}/DropoutKernelNpu.cpp
    #${OP_PLUGIN_DIR}/FusedAttentionScoreKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ZerosKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BitwiseOrKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddcmulKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PowKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/VarKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardsigmoidBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ChannelShuffleKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PadKernelNpu.cpp
    ${OP_PLUGIN_DIR}/VDotKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Log10KernelNpu.cpp
    ${OP_PLUGIN_DIR}/EmbeddingDenseBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DotKernelNpu.cpp
    #${OP_PLUGIN_DIR}/FloatStatusKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FmodKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSigmoidKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AtanhKernelNpu.cpp
    ${OP_PLUGIN_DIR}/OnesLikeKernelNpu.cpp
    #${OP_PLUGIN_DIR}/LstmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/PadPackedSequenceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TrueDivideKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RemainderKernelNpu.cpp

    ${OP_PLUGIN_DIR}/IsNegInfKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TakeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IsfiniteKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TanKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Expm1KernelNpu.cpp
    ${OP_PLUGIN_DIR}/EmbeddingBagDenseBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IndexAddKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxUnpool2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LeakyReluBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveAvgPool1dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EyeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReplicationPad2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ScaledMaskedSoftmaxKernelNpu.cpp
    #${OP_PLUGIN_DIR}/BmmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddReluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SortKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GcdKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogicalXorkernelNpu.cpp
    ${OP_PLUGIN_DIR}/SmoothL1LossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddrKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ThresholdKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BinaryCrossEntropyBackwardKernelNpu.cpp
    #${OP_PLUGIN_DIR}/RandomChoiceWithMaskKernelNpu.cpp
    ${OP_PLUGIN_DIR}/OneHotKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RoiAlignKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Conv3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddmvKernelNpu.cpp
    #${OP_PLUGIN_DIR}/DeformableConv2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NormalizeBatchKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TopKKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogicalNotKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MseLossBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RotatedOverlapsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RotatedIouKernelNpu.cpp
    #${OP_PLUGIN_DIR}/AmpForeachNonFiniteCheckAndUnscaleKernelNpu.cpp
    #${OP_PLUGIN_DIR}/BernoulliKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardtanhKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleNearest3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ArgsortKernelNpu.cpp
    #${OP_PLUGIN_DIR}/MeanKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Im2colKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormGatherStatsWithCountsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleNearest1dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSoftmaxBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FillKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Exp2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftShrinkKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GatherKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MultiHeadAttentionKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSoftmaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AvgPool3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BoundingBoxEncodeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveMaxPool2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvDepthWise2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__Lshift__KernelNpu.cpp
    #${OP_PLUGIN_DIR}/MatmulKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FastGeluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CtcLossKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogicalOrKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ArangeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CtcLossBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__iLshift__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/ScatterAddKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NanToNumKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IndexFillKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AsinhKernelNpu.cpp
    #${OP_PLUGIN_DIR}/StackKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CdistKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FillDiagonalKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MulKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BitwiseXorKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RealKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ClampKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AtanKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CumprodKernelNpu.cpp
    #${OP_PLUGIN_DIR}/ZerosLikeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SmoothL1LossBackwardKernelNpu.cpp
    #${OP_PLUGIN_DIR}/ViewcopyKernelNpu.cpp
    #${OP_PLUGIN_DIR}/BaddbmmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/EqualKernelNpu.cpp
    #${OP_PLUGIN_DIR}/RandpermKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BinaryCrossEntropyWithLogitsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftplusKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MaxKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleTrilinear3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleBilinear2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BitwiseAndKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IndexSelectKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftMarginLossBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/QuantizePerTensorKernelNpu.cpp
    ${OP_PLUGIN_DIR}/UpsampleTrilinear3dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SeluKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CumminKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NLLLoss2dKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LogSumExpKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftShrinkBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GiouKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddbmmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GeKernelNpu.cpp
    #${OP_PLUGIN_DIR}/MaskedFillKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ConvTbcBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__iRshift__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/ApplyAdamKernelNpu.cpp
    ${OP_PLUGIN_DIR}/TriuKernelNpu.cpp
    #${OP_PLUGIN_DIR}/CountNonZeroKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LayerNormKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AddcdivKernelNpu.cpp
    ${OP_PLUGIN_DIR}/MishKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ErfinvKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ReflectionPad2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/DiagKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ImgToTensorKernelNpu.cpp
    ${OP_PLUGIN_DIR}/CosKernelNpu.cpp
    ${OP_PLUGIN_DIR}/LinspaceKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftplusBackwardKernelNpu.cpp
    #${OP_PLUGIN_DIR}/MinV1KernelNpu.cpp
    ${OP_PLUGIN_DIR}/InverseKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SlowConvDilated2DKernelNpu.cpp
    ${OP_PLUGIN_DIR}/ThresholdBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/GluBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/_Unique2KernelNpu.cpp
    ${OP_PLUGIN_DIR}/AllKernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormGatherStatsUpdateKernelNpu.cpp
    ${OP_PLUGIN_DIR}/RoiAlignBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NLLLossKernelNpu.cpp
    #${OP_PLUGIN_DIR}/NativeDropoutKernelNpu.cpp
    #${OP_PLUGIN_DIR}/AddmmKernelNpu.cpp
    ${OP_PLUGIN_DIR}/WhereKernelNpu.cpp
    ${OP_PLUGIN_DIR}/FusedAttentionQKVGradKernelNpu.cpp
    ${OP_PLUGIN_DIR}/HardsigmoidKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AdaptiveAvgPool3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_DIR}/__Xor__KernelNpu.cpp
    ${OP_PLUGIN_DIR}/BatchNormStatsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/AffineGridGeneratorKernelNpu.cpp
    ${OP_PLUGIN_DIR}/YoloBoxesEncodeKernelNpu.cpp
    ${OP_PLUGIN_DIR}/IsInTensorScalarKernelNpu.cpp
    ${OP_PLUGIN_DIR}/Log1pKernelNpu.cpp
    ${OP_PLUGIN_DIR}/SoftmaxCrossEntropyWithLogitsKernelNpu.cpp
    ${OP_PLUGIN_DIR}/NnpackSpatialConvolutionKernelNpu.cpp
    #${OP_PLUGIN_DIR}/GruKernelNpu.cpp
)

set(OP_PLUGIN_SRC_V_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/op-plugin/op_plugin/ops/v2r0/aclops/)
set(OP_PLUGIN_SRC
    ${OP_PLUGIN_SRC}
    ${OP_PLUGIN_SRC_V_DIR}/SumKernelNpu.cpp
)
set(debug
    ${OP_PLUGIN_SRC_V_DIR}/BatchNormKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/IndexKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/CumsumKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/PreluBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/EmbeddingBagBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/StdKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/LinalgSvdKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/ConfusionTransposeKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/BmmV2KernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/WeightNormKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/ConvolutionKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/ScatterKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/MaxV1KernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/PsRoiPoolingKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/EmbeddingKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/PreluKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/TriangularSolveKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/CatKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/GeluBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/ZerosKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/MaxUnpool2dBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/KlDivKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/VarKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/IndexAddKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/L1LossKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/MeanKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/Im2colKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/SymeigKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/MaxUnpool3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/GridSampler3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/GeluKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/RreluWithNoiseKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/CumprodKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/EmbeddingBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/RepeatInterLeaveKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/SoftplusBackwardKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/MinV1KernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/WhereKernelNpu.cpp
    ${OP_PLUGIN_SRC_V_DIR}/IndexCopyKernelNpu.cpp
)

set(OP_PLUGIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op-plugin/op_plugin/ops/base_ops/opapi/)
set(OP_PLUGIN_API_SRC
    ${OP_PLUGIN_API_DIR}/AddKernelNpuOpApi.cpp
)
set(OP_PLUGIN_API_SRC1
    ${OP_PLUGIN_API_DIR}/CosKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ReflectionPad1dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/Log2KernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SoftmaxBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ConvtbcKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AddrKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LeakyReluKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/NeKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SoftmaxKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/EyeKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/IsNegInfKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SqrtKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/CummaxKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MulKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/CatKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/FloorDivideKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SigmoidBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/GerKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ErfcKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ErfinvKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ForeachSubScalarInplaceKernelOpApi.cpp
    ${OP_PLUGIN_API_DIR}/RsqrtKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/NormalKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/RandomKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MaxKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/DivKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MishKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LeKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/IndexFillKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AvgPool2dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/EmbeddingRenormKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/CeluKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogicalNotKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ConvDepthwise2dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SoftplusBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/Col2imKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AddcdivKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/RollKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/Log1pKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MmKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/NLLLoss2dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/IsPosInfKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/IsFiniteKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ConvolutionKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ConstantPadNdKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MaskedScatterKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/EmbeddingDenseBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AcoshKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/IndexAddKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BinaryCrossEntropyKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/FlipKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AmaxKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AsinKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/FillDiagonalKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/NLLLossKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/CumsumKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/__Ior__KernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/IsInKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleLinear1dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AddKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/TakeKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/IsInTensorScalarKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogSoftmaxBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AminKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SiluBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/InverseKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SoftShrinkKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/Exp2KernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/EluKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BitwiseAndKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleNearest1dBackwardKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/DropoutKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BatchNormGatherStatsWithCountsKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogicalAndKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ReplicationPad1dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ConvolutionBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MishBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MvKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleTrilinear3dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ComplexKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AddmvKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ScatterAddKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AllKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BitwiseXorKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ErfKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/HardSigmoidKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SigmoidKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/XlogyKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AddcmulKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/HardSwishBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LerpKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/RangeKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/NLLLossBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/GatherKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AdaptiveMaxPool2dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/StackKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AffineGridGeneratorKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LinalgQrKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ReflectionPad1dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BinaryCrossEntropyBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/CoshKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/CtcLossKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/FracKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/RReluWithNoiseKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/EqualKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LayerNormKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogSumExpKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/DotKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/RealKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/HardsigmoidBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BatchNormBackwardReduceKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SeluKernelOpApi.cpp
    ${OP_PLUGIN_API_DIR}/CtcLossBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/CeilKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LayerNormBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/HardSwishKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/NegKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ForeachMulScalarInplaceKernelOpApi.cpp
    ${OP_PLUGIN_API_DIR}/HardtanhKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/IsCloseKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AsinhKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MultilabelMarginLossKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/EqKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleNearest1dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MatmulBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogicalOrKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BatchNormElemtKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/HardtanhBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleNearest2dKernelOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogSigmoidBackwardKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/RandpermKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/BernoulliKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AnyKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/RsubKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AtanhKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SiluKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LeakyReluBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/GluKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BatchNormKernelOpApi.cpp
    ${OP_PLUGIN_API_DIR}/Expm1KernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleTrilinear3dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/PdistKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SmoothL1LossKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/OnesLikeKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleNearest2dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ZerosKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleLinear1dKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/AmpForeachNonFiniteCheckAndUnscaleKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AddbmmKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/UniformKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/TrilKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ProdKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogSigmoidNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AdaptiveAvgPool2dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/CountNonZeroKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MedianKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SincKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/TanhKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogicalXorKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleNearest3dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MinKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/Log10KernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MseLossBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/NLLLoss2dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/GeKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/FloorKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SearchsortedKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MaskedFillKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/TopkKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/HardShrinkKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/OneHotKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/PutKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SortKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/KlDivKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MatmulKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SmoothL1LossBackwrdKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ReluKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AcosKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SoftMarginLossKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SoftMarginLossBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogSoftmaxKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/GcdKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/CumminKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SoftshrinkBackwardNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ReciprocalKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/Atan2KernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/PowKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AdaptiveAvgPool2dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MaxUnpool3dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ThresholdBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogAddExpKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ReflectionPad2dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/GridSampler2dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/TriuKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/TanKernelOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UniqueConsecutiveKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/RepeatKernelOpApi.cpp
    ${OP_PLUGIN_API_DIR}/TruncKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/TanhBackwardNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BmmKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ClampKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/TrueDivideKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ArgSortKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BincountKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SlogdetKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ArgmaxKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/NativeDropoutKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/FillKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/GtKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SoftplusKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/VDotKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ExpKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SignbitKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AdaptiveAvgPool3dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/RenormKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/EluBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/CastKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogAddExp2KernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/GridSampler2dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LinalgCrossKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/IndexSelectKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MaskedSelectKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/MaxUnpool2dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ChannelShuffleKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BatchNormBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ArgminKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ThresholdKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleBilinear2dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BatchNormBackwardElemtKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SubKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/HardshrinkBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/RemainderKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/RoundKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/FmodKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SumKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LogKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BaddbmmKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/LtKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SinKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleBilinear2dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SinhKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/RoundDecimalsKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/GluGradKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AbsKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/DiagKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ReplicationPad2dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AtanKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ReplicationPad2dBackwardKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/UpsampleNearest3dKernelOpApi.cpp
    ${OP_PLUGIN_API_DIR}/NormKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/KthvalueKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/IndexPutKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/SignKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/OnesKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AminmaxKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BatchNormStatsKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ReplicationPad1dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ReflectionPad2dKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BitwiseNotKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/NonzeroKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/BitwiseOrKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ArangeKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/HistcKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ZerosLikeKernelNpuOpApi.cpp
    #${OP_PLUGIN_API_DIR}/MultinomialKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/AddmmKernelNpuOpApi.cpp
    ${OP_PLUGIN_API_DIR}/ScatterKernelNpuOpApi.cpp
)

set(OP_PLUGIN_UTILS_SRC
    ${OP_PLUGIN_UTILS_DIR}/KernelNpuOutputSize.cpp
    ${OP_PLUGIN_UTILS_DIR}/OpUtils.cpp
    ${OP_PLUGIN_UTILS_DIR}/op_api_common.cpp
    #${OP_PLUGIN_UTILS_DIR}/AdvancedIndex.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/SumKernelNpu.cpp
)
set(OP_PLUGIN_UTILS_SRC1
    ${OP_PLUGIN_UTILS_SRC}
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/IndexKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/PreluBackwardKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/LinalgSvdKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/ScatterKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/EmbeddingKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/PreluKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/TriangularSolveKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/GeluBackwardKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/ZerosKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/VarKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/LeakyReluBackwardKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/MeanKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/LogSoftmaxKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/GridSampler3dBackwardKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/GeluKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/RepeatInterLeaveKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/SoftplusBackwardKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/WhereKernelNpu.cpp
    ${OP_PLUGIN_UTILS_DIR}/custom_functions/aclops/IndexCopyKernelNpu.cpp
)

set(ASCEND_IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/torch_npu/csrc/impl/)
set(ASCEND_IMPL_SRC

)

set(OP_PLUGIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op-plugin/op_plugin/ops/v2r0/opapi/)
set(OP_PLUGIN_API_SRC
    ${OP_PLUGIN_API_SRC}
)

set(DIOPI_IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/diopi_impl/)
set(DIOPI_IMPL_SRC
    ${DIOPI_IMPL_DIR}/error.cpp
    ${DIOPI_IMPL_DIR}/binary.cpp
    ${DIOPI_IMPL_DIR}/conv2d.cpp
    ${DIOPI_IMPL_DIR}/batch_norm.cpp
)

add_definitions(-DBUILD_LIBTORCH)

include_directories(op-plugin torch_npu ${CMAKE_CURRENT_SOURCE_DIR})
set(IMPL_SRC
  torch_npu/csrc/DIOPIAdapter.cpp
  torch_npu/csrc/AclOpCompileInterface.cpp
)

#list(APPEND IMPL_SRC ${OP_PLUGIN_UTILS_SRC} ${OP_PLUGIN_SRC} ${OP_PLUGIN_API_SRC} ${ASCEND_IMPL_SRC} ${DIOPI_IMPL_SRC})
list(APPEND IMPL_SRC ${OP_PLUGIN_UTILS_SRC} ${OP_PLUGIN_SRC} ${OP_PLUGIN_API_SRC} ${ASCEND_IMPL_SRC} ${DIOPI_IMPL_SRC})

# adaptor
set(USE_ADAPTOR OFF)
if(EXISTS "${PROJECT_SOURCE_DIR}/convert_config.yaml")
    set(USE_ADAPTOR ON)
endif()

if(USE_ADAPTOR)

    file(GLOB ADAPTOR_TEMPLATE_CODE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${ADAPTOR_DIR}/codegen/*.py)
    add_custom_target(adaptor_gen_dependency DEPENDS ${ADAPTOR_TEMPLATE_CODE})

    set(ADAPTOR_CSRC_PATH "${ADAPTOR_DIR}/csrc")
    set(GEN_FILES ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp ${ADAPTOR_CSRC_PATH}/impl_functions.hpp)
    add_custom_target(adaptor_code_gen
        COMMAND python3 ${ADAPTOR_DIR}/codegen/gen.py --diopi_dir=${CMAKE_SOURCE_DIR}/../ --output_dir=${ADAPTOR_CSRC_PATH} --config_device=ascend
        BYPRODUCTS ${GEN_FILES}
        DEPENDS adaptor_gen_dependency)
    list(APPEND IMPL_SRC ${GEN_FILES} ${ADAPTOR_CSRC_PATH}/convert.cpp ${ADAPTOR_CSRC_PATH}/diopi_adaptor.cpp)
    add_definitions(-DTEST_USE_ADAPTOR)
endif()

add_library(${DEVICEIMPL} SHARED ${IMPL_SRC})
set_target_properties(${DEVICEIMPL} PROPERTIES SUFFIX ".so")
target_link_libraries(${DEVICEIMPL} ascendcl acl_op_compiler c10 torch_cpu)

if(USE_ADAPTOR)
    add_dependencies(${DEVICEIMPL} adaptor_code_gen)
endif()

if (TEST)
    add_subdirectory(test)
endif()
