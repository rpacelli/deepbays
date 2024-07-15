import numpy as np
import torch
from scipy.integrate import dblquad, quad

### VERSION OF KERNELS THAT WORK WITH TORCH. NEEDED WHEN GRADIENT OF THE ACTION CANNOT BE COMPUTED EXACTLY
def computeKmatrixTorch(C, kernel):
    diagonal = torch.diag(C).clone() #.detach().requires_grad_(True)
    diagonal_column = diagonal[:, None]  # Shape (n, 1) where n is the length of the diagonal
    diagonal_row = diagonal[None, :]     # Shape (1, n)
    k = kernel(diagonal_column, C.clone(), diagonal_row )#.detach().requires_grad_(True), diagonal_row)
    return k

def computeKmatrixMultipleCTorch(C1,C12,C2,  kernel):
    diagonal1 = torch.diag(C1).clone() #.detach().requires_grad_(True)
    diagonal2 = torch.diag(C2).clone()
    diagonal_column = diagonal1[:, None]  # Shape (n, 1) where n is the length of the diagonal
    diagonal_row = diagonal2[None, :]     # Shape (1, n)
    k = kernel(diagonal_column, C12.clone(), diagonal_row )#.detach().requires_grad_(True), diagonal_row)
    return k

def kernel_erf_torch(cxx,cxy,cyy):
    return (2 / torch.pi) * torch.arcsin((2 * cxy)/torch.sqrt((1 + 2 * cxx)*(1 + 2 * cyy)))

def kernel_relu_torch(cxx, cxy, cyy):
    u = cxy / torch.sqrt(cxx * cyy)
    kappa = (1 / (2 * torch.pi)) * (u * (torch.pi - torch.arccos(u)) + torch.sqrt(1 - u**2))
    return torch.sqrt(cxx * cyy) * kappa

def divide2dImage(array, k):
    n = array.shape[1]
    chunks = []
    for i in range(0, n, k):
        for j in range(0, n, k):
            chunk = array[i:i+k, j:j+k]
            chunks.append(chunk)
    return chunks