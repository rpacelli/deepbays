import os, os.path,time,argparse
import numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F
from ..conv_geometry import convolution_geometry, layer_precisions, positive_int, spatial_pair

class Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.erf(x)
    
class Quad(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x**2

class Id(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class Square(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.square(x)

class Norm(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
    def forward(self, x):
        return x/self.norm
    
def relu(x):  # CK: deprecated I think
    print("Warning: this relu function which is not a nn.Module, and may be removed soon")
    return x if x > 0 else 0

def make_act_module(actfunc_string):
    """Takes 'relu', 'erf', 'id', 'square' and returns a corresponding torch.nn.Module instance.
       (This func fixes a silent nonlinearity module cloning 'bug', 
        but is also useful for importing elsewhere.) """
    if actfunc_string == "relu":
        return nn.ReLU() 
    elif actfunc_string == "erf":
        return Erf()
    elif actfunc_string == "id":
        return Id() 
    elif actfunc_string == "square":
        return Square()
    elif actfunc_string in ("quad", "quadratic"):
        return Quad()
    else:
        raise ValueError(f"nonlinearity {actfunc_string} appears to not implemented so far!")

class FCNet:
    """Dense network with D outputs; D=1 preserves the scalar architecture.

    Each readout is divided by sqrt(N1)*gamma, independently of D.
    D is keyword-only so existing positional calls retain their meaning.
    """

    def __init__(self, 
                 N0 : int, #input dimension
                 N1 : int, #number of hidden layer units
                 L : int, #number of hidden layers. depth = L+1
                 bias : bool = False, #if True, add bias for each layer
                 act : str = "erf", #activation function
                 precisions = None,  # list of layer-wise precisions if deviating from all 1. 
                 gamma : float = 1.0, # feature learning parameter, for MF use sqrt(N1) and for SP use 1.0
                                     # Note: Biases are not affected by gamma here. Recheck if this is desired (no biases in Yang'22 or most other MF papers). 
                 *, D : int = 1
                 ):
        self.D = positive_int(D, "D")
        self.N0, self.N1, self.L, self.act, self.bias, self.precisions, self.gamma = N0, N1, L, act, bias, precisions, gamma
        if precisions is not None:
            assert len(precisions) == L+1, "Precisions must be a list of one precision value per layer"
        else:
            self.precisions = [1.] * (L+1) 
            
        
    def Sequential(self):
        modules = []
        # build input layer
        first_layer = nn.Linear(self.N0, self.N1, bias=self.bias)
        init.normal_(first_layer.weight, std = 1. / np.sqrt(self.precisions[0]) )
        if self.bias:
            init.constant_(first_layer.bias,0)  # CK: why is bias init with constant_ here, but with normal_ an other layers? maybe harmonize this.
        modules.append(first_layer)
        modules.append(Norm(np.sqrt(self.N0)))
        # build intermediary layers
        for l in range(self.L-1): 
            modules.append(make_act_module(self.act))
            layer = nn.Linear(self.N1, self.N1, bias = self.bias)
            init.normal_(layer.weight, std = 1. / np.sqrt(self.precisions[l+1]) )
            if self.bias:
                init.normal_(layer.bias,std = 1)
            modules.append(layer)
            modules.append(Norm(np.sqrt(self.N1)))
        # build output layer
        modules.append(make_act_module(self.act))
        last_layer = nn.Linear(self.N1, self.D, bias=self.bias)
        init.normal_(last_layer.weight, std = 1. / np.sqrt(self.precisions[-1]) ) 
        if self.bias:
                init.normal_(last_layer.bias,std = 1)
        modules.append(last_layer)
        modules.append(Norm(np.sqrt(self.N1) * self.gamma)) # amounts to 1/sqrt(N1) for SP and 1/N for full muP with gamma = sqrt(N1) 
        sequential = nn.Sequential(*modules)
        print(f'\nThe network has {self.L} dense hidden layer(s) of size {self.N1} with {self.act} actviation function and feature learning param gamma {self.gamma} \n', sequential)
        return sequential
    
class ConvNet:
    """L convolutional layers and D linear outputs (default D=1).

    N0 is a (height, width) tuple or the pixel count of a square image (per
    input channel). Inputs have shape (batch, inputChannels, height, width).
    Nc is a common channel count, or a list of L counts;
    CNN_deep theory currently requires equal channel counts. mask/stride/padding use
    tuples for spatial pairs and lists for per-layer settings. 'same' padding
    supports any stride and puts extra padding on the right/bottom.

    Each weight has prior variance 1/precision; Norm divides convolution
    outputs by sqrt(input_channels * filter_area), and the scalar readout by
    sqrt(final_channels * final_patches) * gamma. With pooling='avg', hidden
    features are spatially averaged before readout, normalized by
    sqrt(final_channels)*gamma. D never changes the normalization. Biases
    are unsupported. The default agrees with rkgp.CNN_deep as before.
    """

    def __init__(self, N0, Nc, L, mask=3, stride=1, bias=False, act="erf",
                 inputChannels=1, precisions=None, gamma=1., padding="valid",
                 *, D=1, pooling=None):
        self.D = positive_int(D, "D")
        if pooling not in (None, 'avg'):
            raise ValueError("pooling must be None or 'avg'")
        self.pooling = pooling
        self.L = positive_int(L, "L")
        self.inputChannels = positive_int(inputChannels, "inputChannels")
        if isinstance(N0, (int, np.integer)):
            pixels = positive_int(N0, "N0")
            side = int(np.sqrt(pixels))
            if side * side != pixels:
                raise ValueError("integer N0 must be a square pixel count; use (height, width) for a rectangle")
            self.image_shape = (side, side)
        else:
            self.image_shape = spatial_pair(N0, "N0")
        if bias:
            raise ValueError("the matching CNN theory has no biases; use bias=False")
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be positive and finite")
        if act == "quadratic":
            act = "quad"
        make_act_module(act)  # validate before building a partial network
        counts = Nc if isinstance(Nc, list) else [Nc] * self.L
        if len(counts) != self.L:
            raise ValueError("Nc must be a positive integer or a list with L entries")
        self.channels = tuple(positive_int(n, "Nc") for n in counts)
        self.precisions = layer_precisions(precisions, self.L)
        self.geometry = convolution_geometry(self.image_shape, self.L, mask, stride, padding)
        self.N0, self.Nc = N0, Nc
        self.mask, self.stride, self.padding = mask, stride, padding
        self.bias, self.act, self.gamma = False, act, float(gamma)
        self.patch_shapes = tuple(layer.output_shape for layer in self.geometry)
        self.final_patches = self.geometry[-1].patches

    def Sequential(self):
        modules = []
        input_channels = self.inputChannels
        for l, (channels, geometry) in enumerate(zip(self.channels, self.geometry)):
            if any(geometry.padding):
                modules.append(nn.ZeroPad2d(geometry.padding))
            layer = nn.Conv2d(input_channels, channels, geometry.kernel_size,
                              stride=geometry.stride, padding=0, bias=False)
            init.normal_(layer.weight, std=1. / np.sqrt(self.precisions[l]))
            modules.extend([layer, Norm(np.sqrt(input_channels * geometry.area)),
                            make_act_module(self.act)])
            input_channels = channels
        if self.pooling == 'avg':
            modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.Flatten())
        features = self.channels[-1] * (1 if self.pooling == 'avg' else self.final_patches)
        readout = nn.Linear(features, self.D, bias=False)
        init.normal_(readout.weight, std=1. / np.sqrt(self.precisions[-1]))
        modules.extend([readout, Norm(np.sqrt(features) * self.gamma)])
        return nn.Sequential(*modules)


class old_ConvNet:
    def __init__(self, N0 : int,  #input size, necessary to compute normalization
                 Nc : int,  # number of channels in internal layers
                 mask : int, #mask
                 stride : int, #striding of convolution
                 bias : bool = False, #if True, add bias for each layer
                 act :str = "erf", #activation function
                 inputChannels : int = 1 #number of inputs channel, defaults for black and white image
                 ):
        self.Nc, self.N0, self.inputChannels, self.mask, self.stride, self.bias, self.act = Nc, N0, inputChannels, mask, stride, bias, act

    def Sequential(self):
        if self.act == "relu":
            act = nn.ReLU()
        elif self.act == "erf":
            act = Erf()
        modules = []
        # First convolutional layer
        first_layer = nn.Conv2d(self.inputChannels, self.Nc, self.mask, stride = self.stride, bias=self.bias)
        init.normal_(first_layer.weight, std=1)
        if self.bias:
            init.constant_(first_layer.bias, 0)
        modules.extend([Norm(self.mask), first_layer, act])
        # Flatten the tensor before the fully connected layer
        modules.append(nn.Flatten())
        with torch.no_grad(): 
            dummy_net_sequential = nn.Sequential(*modules)
            out = dummy_net_sequential(torch.randn(self.inputChannels,int(np.sqrt(self.N0)),int(np.sqrt(self.N0))).unsqueeze(0))
            FC_params = len(out.flatten())  # number of parameters in the last layer   
        # Fully connected layer
        last_layer = nn.Linear(FC_params, 1, bias=self.bias)
        init.normal_(last_layer.weight, std=1)
        if self.bias:
            init.constant_(last_layer.bias, 0)
        modules.append(Norm(np.sqrt(FC_params)))
        modules.append(last_layer)
        sequential = nn.Sequential(*modules)
        print(f'\nThe network has 1 convolutional hidden layer with {self.Nc} kernels of size {self.mask} and {self.act} activation function', sequential)
        return sequential
