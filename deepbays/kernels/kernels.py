import numpy as np
import torch
from scipy.integrate import dblquad, quad
from functools import partial
import math

### VERSION OF KERNELS THAT WORK WITH NUMPY
def computeKmatrix(C, kernel):
    diagonal = np.diag(C)
    K = kernel(diagonal[:, None], C, diagonal[None, :]) # diagonal[:, None] makes it a column vector, diagonal[None, :] makes it a row vector
    return K

def kernel_erf(cxx,cxy,cyy):
    return (2 / np.pi)*np.arcsin((2 * cxy) / np.sqrt((1 + 2 * cxx)*(1 + 2 * cyy)))

def kernel_tanh(cxx: float, cxy: float, cyy: float):
    limit = 200
    output_shape = np.broadcast(cxx, cxy, cyy).shape
    #print(output_shape)
    # Initialize arrays to store results and errors
    result = np.empty(output_shape)
    error = np.empty(output_shape)
    # Perform integration for each element of the output shape
    for i in np.ndindex(output_shape):
        if len(i) == 1:
            cxx_val = cxx[i]
            cxy_val = cxy[i]
            cyy_val = cyy[i]
        else:
            cxx_val = cxx[i[0],0]
            cxy_val = cxy[i]
            cyy_val = cyy[0,i[1]]
        #cyy_val = cyy[indices]
        tildeC = np.array([[cxx_val, cxy_val], [cxy_val, cyy_val]])
        if cxx_val == cxy_val:
            def integralTanhKernelSingle(t1):
                #vec_t = np.array([t1, t2])
                gaussian_part = math.exp(-0.5 * t1**2 /cxx_val)
                tanh_part = math.tanh(t1)**2
                return gaussian_part * tanh_part
            result[i], error[i] = quad(integralTanhKernelSingle, -limit, limit)
            print(result[i], error[i])
            norm = 1/(2 * np.pi * np.sqrt(cxx_val))
        else:
            invTildeC = np.linalg.inv(tildeC)
            detTildeC = np.linalg.det(tildeC)
            def integralTanhKernel(t1, t2):
                vec_t = np.array([t1, t2])
                gaussian_part = math.exp(-0.5 * np.dot(vec_t.T, np.dot(invTildeC, vec_t)))
                tanh_part = math.tanh(t1) * math.tanh(t2)
                return gaussian_part * tanh_part
            result[i], error[i] = dblquad(integralTanhKernel, -limit, limit, -limit, limit)
            norm = 1 /(2 * np.pi *np.sqrt( detTildeC))
    print(result, error)
    return result * norm

#def integralTanhKernel(t1, t2, invTildeC):
#    vec_t = np.array([t1, t2])
#    gaussian_part = math.exp(-0.5 * np.dot(vec_t.T, np.dot(invTildeC, vec_t)))
#    tanh_part = math.tanh(t1) * math.tanh(t2)
#    return gaussian_part * tanh_part
#
#def integralTanhKernelSingle(t1, invTildeC):
#    #vec_t = np.array([t1, t2])
#    gaussian_part = math.exp(-0.5 * t1**2 * invTildeC)
#    tanh_part = math.tanh(t1)
#    return gaussian_part * tanh_part

def kernel_relu(cxx, cxy, cyy):
    u = cxy / np.sqrt(cxx * cyy)
    kappa = (1 / (2 * np.pi)) * (u * (np.pi - np.arccos(u)) + np.sqrt(1 - u**2))
    return np.sqrt(cxx * cyy) * kappa

def kernel_relu_bias(cxx, cxy, cyy):
    u = cxy / np.sqrt(cxx * cyy)
    kappa = (1 / (2 * np.pi)) * (u * (np.pi - np.arccos(u)) + np.sqrt(1 - u**2))
    return np.sqrt(cxx * cyy) * kappa 

def mean_relu(c):
    return np.sqrt(c/2*np.pi)

def kernel_id(cxx, cxy, cyy):
    return cxy    

#def kmatrix(C,kernel):
#    P = len(C)
#    K = np.zeros_like(C)
#    for i in range(P): 
#        for j in range(i,P):         
#            K[i][j] = kernel(C[i][i], C[i][j], C[j][j])
#            K[j][i] = K[i][j]
#    return K

def divide2dImage(array, k):
    n = array.shape[1]
    chunks = []
    for i in range(0, n, k):
        for j in range(0, n, k):
            chunk = array[i:i+k, j:j+k]
            chunks.append(chunk)
    return chunks