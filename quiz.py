import torch
import numpy as np

# Quiz 1

# 1. Make two tensor A, B which its shape is [2, 3]
A = torch.randint(1, 10, size=(2,3))
B = torch.randint(1, 10, size=(2,3))
print(A)
print(B)

# 2. Sum A and B
C = A + B
print(C)

# 3. Subtract A and B
C = A - B
print(C)

# 4. Sum all elemnts of each A and B
A_sum = A.sum()
B_sum = B.sum()
print(A_sum.item())
print(B_sum.item())

# Quiz2
# 1. Make tensor that shape is [1, 5, 3, 3]
A = torch.arange(0, 45).view(1, 5, 3, 3)

# 2. Transpose the tensor [1, 3, 3, 5]
At = torch.transpose(A, 1, 3)
print("A transpose \n {}".format(At))

# 3. print [0, 2, 2, all elements]
print(At[0, 2, 2, :])

# Quiz3
# 1. Make tensor A, B which its shape is [2, 3]
A = torch.randint(1, 10, (2, 3))
B = torch.randint(1, 10, (2, 3))
print("tensors : \n {} \n {}".format(A, B))
AB_concnat = torch.cat([A, B], dim=1)
print("concat A and B : \n {}".format(AB_concnat))
AB_stack = torch.stack([A, B], dim=1)
print("stack A and B : \n {}".format(AB_stack))