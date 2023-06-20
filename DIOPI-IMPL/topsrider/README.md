# TopsRider Impl for DIOPI

## 环境依赖

### 软件环境
- TopsDNN: Tops Deep Neural Network Library，燧原性能深度优化的 AI 算子实现库
- TopsRuntime：燧原运行时库
- Driver：燧原GCU驱动

### 硬件环境
- i20: 燧原第二代推理芯片


## 更新历史

**2022.12.13**
1. 新增支持diopiBatchNorm,diopiAddmm,diopiAdaptiveAvgPool2d
2. 限制：
+ diopiBatchNorm仅仅支持training=false
+ diopiAdaptiveAvgPool2d只支持4D tensor且输出size最好是（1,1）
+ diopiAddmm只支持2D tensor

**2022.11.24**
1. 新增支持diopiAdd, diopiAvgPool2d，已完成fp32算子的一致性测试。

**2022.11.22**
1. 新增对vgg16模型相关算子的支持，包括diopiConvolution2d、diopiMaxPool2d、diopiLinear、diopiRelu、diopiSoftmax，
2. 针对vgg16模型的算子形状和fp32的数据类型，已完成算子的一致性测试。