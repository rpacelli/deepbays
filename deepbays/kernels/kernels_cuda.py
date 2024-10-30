import cupy as cp

### VERSION OF KERNELS THAT WORK WITH CUPY

def kernel_erf_cuda(cxx,cxy,cyy):
    return (2 / cp.pi)*cp.arcsin((2 * cxy) / cp.sqrt((1 + 2 * cxx)*(1 + 2 * cyy)))

def kernel_relu_cuda(cxx, cxy, cyy):
    u = cxy / cp.sqrt(cxx * cyy)
    kappa = (1 / (2 * cp.pi)) * (u * (cp.pi - cp.arccos(u)) + cp.sqrt(1 - u**2))
    return cp.sqrt(cxx * cyy) * kappa

def kernel_relu_bias_cuda(cxx, cxy, cyy):
    u = cxy / cp.sqrt(cxx * cyy)
    kappa = (1 / (2 * cp.pi)) * (u * (cp.pi - cp.arccos(u)) + cp.sqrt(1 - u**2))
    return cp.sqrt(cxx * cyy) * kappa 

def mean_relu_cuda(c):
    return cp.sqrt(c/2*cp.pi)

def kernel_id_cuda(cxx, cxy, cyy):
    return cxy    
