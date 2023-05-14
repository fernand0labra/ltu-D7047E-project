import torch

x = torch.randn(3, 3, requires_grad=True)
y = torch.clamp(x, 0, 1)
z = y.sum()

z.backward() # compute gradients

print(x)
print(x.grad) # prints gradients of x
