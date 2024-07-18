import torch
import torch_dipu
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

x = torch.ones([4, 64, 128], dtype=torch.int32).cuda()

unique_x, inverse_indices  , counts = torch.unique(x, dim=1, sorted=False, return_inverse=True, return_counts=True)

print(unique_x)

print(inverse_indices)

print(counts)
