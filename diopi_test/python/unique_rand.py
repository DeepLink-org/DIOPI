import torch
import torch_dipu

x = torch.randn([252], dtype=torch.float32)
# x = torch.tensor([[4, 5, 6,1],[1, 2, 3,1], [2, 3, 4,1],  [1, 2, 3,1]])

# 在第一维度上使用unique算子
# unique_x = torch.unique(x, dim=1,)
unique_x, inverse_indices   = torch.unique(x, dim=-1, sorted=True, return_inverse=True, return_counts=False)

# print(unique_x)

# # 创建一个二维张量
# x = torch.tensor([[4, 5, 6,1],[1, 2, 3,1], [2, 3, 4,1],  [1, 2, 3,1]]).cuda()

# # 在第一维度上使用unique算子
# unique_x = torch.unique(x, dim=0)

# print(unique_x)

print(unique_x)

print(inverse_indices)

# print(counts)