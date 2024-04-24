import numpy as np
import torch

### VERSION OF KERNELS THAT WORK WITH NUMPY
def computeKmatrix(C, kernel):
    diagonal = np.diag(C)
    K = kernel(diagonal[:, None], C, diagonal[None, :]) # diagonal[:, None] makes it a column vector, diagonal[None, :] makes it a row vector
    return K

def kernel_erf(cxx,cxy,cyy):
    return (2 / np.pi)*np.arcsin((2 * cxy) / np.sqrt((1 + 2 * cxx)*(1 + 2 * cyy)))

def kernel_relu(cxx, cxy, cyy):
    u = cxy / np.sqrt(cxx * cyy)
    kappa = (1 / (2 * np.pi)) * (u * (np.pi - np.arccos(u)) + np.sqrt(1 - u**2))
    return np.sqrt(cxx * cyy) * kappa

def kernel_id(cxx, cxy, cyy):
    return cxy    

### VERSION OF KERNELS THAT WORK WITH TORCH. NEEDED WHEN GRADIENT OF THE ACTION CANNOT BE COMPUTED EXACTLY
def computeKmatrixTorch(C, kernel):
    diagonal = torch.diag(C).clone() #.detach().requires_grad_(True)
    diagonal_column = diagonal[:, None]  # Shape (n, 1) where n is the length of the diagonal
    diagonal_row = diagonal[None, :]     # Shape (1, n)
    k = kernel(diagonal_column, C.clone(), diagonal_row )#.detach().requires_grad_(True), diagonal_row)
    return k

def kernel_erf_torch(cxx,cxy,cyy):
    return (2 / torch.pi) * torch.arcsin((2 * cxy)/torch.sqrt((1 + 2 * cxx)*(1 + 2 * cyy)))

#def kmatrix(C,kernel):
#    P = len(C)
#    K = np.zeros_like(C)
#    for i in range(P): 
#        for j in range(i,P):         
#            K[i][j] = kernel(C[i][i], C[i][j], C[j][j])
#            K[j][i] = K[i][j]
#    return K