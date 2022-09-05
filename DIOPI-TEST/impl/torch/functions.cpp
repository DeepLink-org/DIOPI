/**
 * @file functions.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-09-05
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <diopi/functions.h>
#include <iostream>

#include <torch/torch.h>

extern "C" diopiError_t diopiRelu(diopiContextHandle_t ctx, diopiTensorHandle_t out, const diopiTensorHandle_t input)
{
    torch::Tensor tensor = torch::rand({2, 3});
    namespace F = torch::nn::functional;
    F::relu(tensor, F::ReLUFuncOptions().inplace(true));
    std::cout << tensor << std::endl;
}
