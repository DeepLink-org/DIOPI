# DIOPI

## 简介
设备无关算子接口（Device Independent Operator Interface, DIOPI）旨在训练框架和人工智能芯片之间定义一套良好的计算交互，对两者之间的工程实施和技术依赖进行有效的解耦，有利于两者共同构建训练生态。
<div align=center><img src='image/diopi.png' width='70%'></img></div>

DIOPI 提供了一套 C-API [算子函数声明](include/diopi/functions.h)。对于训练框架来说，接入 DIOPI 算子；对于芯片来说，实现 DIOPI 的函数声明，接入芯片算子库，完成算子的调用逻辑。DIOPI 在框架和芯片计算库之间定义了统一的**标准接口**，使得两者可以独立开发，且计算库可以无缝移植到其他支持 DIOPI 的训练框架。

为实现以上所述，[一致性测试套件](https://github.com/ParrotsDL/ConformanceTest-DIOPI)提供了一套完整的测试框架和算子测例集合，方便芯片厂商在没有框架的情况下实现 DIOPI 算子函数，并对正确性进行验证。