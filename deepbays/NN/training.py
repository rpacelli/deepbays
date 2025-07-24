"""
Implements Langevin dynamics to sample from a model weight posterior.

Standard usage: 
    model = ...
    opt   = LangevinOpt(model, lr, temp, priors)
    for step in range(sample_steps):
        currentloss = train(model, Xtrain, ytrain, regLoss, opt) # one Langevin step with quadritic likelihood (regLoss)
    
"""
# import numpy as np
import torch 


class LangevinOpt(torch.optim.Optimizer):
    
    def __init__(self, model: torch.nn.Module, lr, temperature, priors):
        """ 
        Langevin dynamics on a NN weight posterior at temperature T, including weight decay (weight prior).
        
        The eq. implemented is 
           delta_p = - lr * ( grad_p(L) +  temp * prior * p ) + sqrt( 2 * lr * temp ) * xi 
        where 
        p are the parameters, 
        grad_p(L) is the gradient of the likelihood wrt. p,
        prior the prior precisions of the parameters, 
        xi is iid. standard normal noise.
        
        Note that the gradient of the likelihood must be accumulated in .grad of 
        the model weights before calling step(). The weight decay due to the 
        gradient of the weight prior, however is done inside step() (together with the addition of the noise),
        and should therefore not be accumulated in the weight .grad before.
        In other words, use with a loss (likelihood) such as 0.5 * sum(y - model(X))**2 that does not include the weight prior.
        
        Arguments:
            model : torch.nn.Module
            lr    : float
                Learning rate.
            temperature : float
                The temperature of the Langevin dynamics.
            priors : array-like of shape (L,)
                Prior precisions of the weights. 
                Must be array-like containing one float per layer of weights.
        
        """
        defaults = {'lr': lr, 'temperature': temperature}
        param_groups = []
        for layer in model.children():
            if isinstance(layer, (torch.nn.Linear)):
                param_groups.append({'params': layer.parameters()})

        super().__init__(param_groups, defaults)

        for group, lambda_j in zip(self.param_groups, priors):
            group['lambda'] = lambda_j
            group['noise_std'] = (2 * group['lr'] * group['temperature']) ** 0.5

        assert len(priors) == len(self.param_groups), "Lenght mismatch"

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    grad = param.grad.detach()
                    # First: weight decay from priors. Warning: must be first since this depends on current state of param itself!!
                    param.mul_(1. - group['lr'] * group['temperature'] * group['lambda']) 
                    # gradient term
                    param.add_(grad, alpha=-group['lr'])
                    # noise
                    param.add_(torch.randn_like(param, memory_format=torch.preserve_format), alpha=group['noise_std'])
                    
    # end def LangevinOpt


    
def test(net, test_data, test_labels, criterion):
        net.eval()
        with torch.no_grad():
                outputs = net(test_data)
                loss = criterion(outputs, test_labels) 
        return loss.item()

def computeNormsAndOverlaps(Snet,Tnet):
    i = len(Snet)-1
    Snorm = torch.linalg.matrix_norm(Snet[i].weight)
    Tnorm = torch.linalg.matrix_norm(Tnet[i].weight)
    overlap = torch.dot(Snet[i].weight.flatten(),Tnet[i].weight.flatten())
    return Snorm, Tnorm, overlap

def train(net, data, labels, criterion, optimizer):
    optimizer.zero_grad()
    loss = criterion(net(data), labels)
    loss.backward(retain_graph=False)
    optimizer.step()
    return loss.item()

def regLoss(output, target):
    return 0.5 * torch.sum((output - target)**2)
    