import torch
import torch_dipu

# # 创建一个张量
# x = torch.tensor([1, 2, 3, 2, 3, 4, 4, 5, 6, 6]).cuda()
# # 使用unique算子
# unique_x = torch.unique(x)

# print(unique_x)


# 创建一个二维张量
x = torch.tensor([[4, 5, 6,1],[1, 2, 3,1], [2, 3, 4,1],  [1, 2, 3,1]])

# 在第一维度上使用unique算子
unique_x, inverse_indices , counts  = torch.unique(x, dim=0, sorted=False, return_inverse=True, return_counts=True)

print(unique_x)

# 创建一个二维张量
x = torch.tensor([[4, 5, 6,1],[1, 2, 3,1], [2, 3, 4,1],  [1, 2, 3,1]]).cuda()

# 在第一维度上使用unique算子
unique_x = torch.unique(x, dim=0)

print(unique_x)

print(inverse_indices)

print(counts)