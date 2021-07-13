import torch
m = torch.tensor([[[1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2]]])
print(torch.sum(m, dim = (1, 2)))
