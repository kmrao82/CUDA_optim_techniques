from torch.utils.cpp_extension import load

vector_add=load(
	name = "vector_add",
	sources = ["vector_add.cpp", "vector_add_kernel.cu"],
	verbose = True
	)

import torch
a = torch.rand(1000, device='cuda')
b = torch.rand(1000, device='cuda')
c = vector_add.vector_add(a,b)
print(c)