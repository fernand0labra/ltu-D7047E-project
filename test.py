import torch
a=torch.randn(2, 1, 3)
print(a)
print(a.dim())
a = a.view(1, *a.shape)
print(a.shape)
print(a)