import torch
a = torch.rand(1,2,2,2)
b = torch.rand(1,2,2,2)
cos = torch.rand(2,1,2)
sin = torch.rand(2,1,2)

res = apply_rotary_emb(a,b,cos,sin)

print(res)