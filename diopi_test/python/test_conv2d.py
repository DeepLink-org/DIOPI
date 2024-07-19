import torch
import torch_dipu
import torch.nn as nn
import os

def add_forward(x):
    return x + x

def linear_forward(x):
    torch.manual_seed(0)
    device = x.device
    dim = x.shape[-1]
    
    linear = nn.Linear(dim, dim)
    linear.to(device)
    
    y = linear(x)
    return y

def conv_forward(x):
    torch.manual_seed(0)
    device = x.device
    batch, in_ch, w, h = x.shape
    
    conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
    conv1.to(device)
    
    y = conv1(x)
    return y
    

def batch_norm_forward(x):
    device = x.device
    batch, in_ch, w, h = x.shape
    
    conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
    conv1.to(device)
    
    y = conv1(x)
    return y
    
    
def test_func_acc(func, x_cpu):
    x_dipu = x_cpu.cuda()
    # x_dipu = x_cpu
    
    y_cpu = func(x_cpu)
    y_dipu = func(x_dipu)
    
    return torch.sum(torch.abs(y_dipu.cpu() - y_cpu)) / x_cpu.numel()

def print_diff(fun_name, diff):
    print(f"{fun_name} mean diff: {diff}")

if __name__ == "__main__":
    
    torch.cuda.set_device(0)
    os.environ['DIPU_DUMP_OP_ARGS'] = '1'
    os.environ['DIPU_AUTOCOMPARE_OPS_LIST'] = '.*'
    print_diff("add_forward" ,test_func_acc(add_forward, x_cpu=torch.randn(100,100)))
    print_diff("linear_forward", test_func_acc(linear_forward, x_cpu=torch.randn(1000,1000)))
    print_diff("conv_forward", test_func_acc(conv_forward, x_cpu=torch.randn(2, 32, 100,100)))