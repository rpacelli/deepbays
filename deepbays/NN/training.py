"""
Implements Langevin dynamics to sample from a model weight posterior.

Standard usage: 
    model = ...
    opt   = LangevinOpt(model, lr, temp, priors)
    for step in range(sample_steps):
        currentloss = train(model, Xtrain, ytrain, regLoss, opt) # one Langevin step with quadritic likelihood (regLoss)

Implements also minibatch SGD with L2 regularization.

Standard usage:
    model = ...
    opt   = SGDOpt(model, lr, batch_size, regularizations)
    for step in range(train_steps):
        currentloss = trainSGD(model, Xtrain, ytrain, regLoss, opt) # one SGD step on a random minibatch with quadratic loss (regLoss)

Implements also Adam (on minibatches, or full batch if batch_size >= P) with L2 regularization.

Standard usage:
    model = ...
    opt   = AdamOpt(model, lr, batch_size, regularizations)
    for step in range(train_steps):
        currentloss = trainAdam(model, Xtrain, ytrain, regLoss, opt) # one Adam step on a random minibatch with quadratic loss (regLoss)

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


class SGDOpt(torch.optim.Optimizer):

    def __init__(self, model: torch.nn.Module, lr, batch_size, regularizations):
        """
        Stochastic gradient descent (GD on minibatches) on the NN weights, including weight decay (L2 regularization).

        The eq. implemented is
           delta_p = - lr * ( grad_p(L_B) +  reg * p )
        where
        p are the parameters,
        grad_p(L_B) is the gradient of the loss wrt. p, estimated on a random minibatch B of size batch_size,
        reg the L2 regularization strengths of the parameters.

        Note that the gradient of the loss must be accumulated in .grad of
        the model weights before calling step(). The weight decay due to the
        gradient of the regularization, however is done inside step(),
        and should therefore not be accumulated in the weight .grad before.
        In other words, use with a loss such as 0.5 * sum(y - model(X))**2 that does not include the regularization.
        The minibatch is drawn with sample_batch() (see trainSGD), which uses the batch_size stored in the optimizer.

        Arguments:
            model : torch.nn.Module
            lr    : float
                Learning rate.
            batch_size : int
                Size of the minibatch used to estimate the gradient at each step.
            regularizations : array-like of shape (L,)
                L2 regularization strengths of the weights.
                Must be array-like containing one float per layer of weights.

        """
        defaults = {'lr': lr, 'batch_size': batch_size}
        param_groups = []
        for layer in model.children():
            if isinstance(layer, (torch.nn.Linear)):
                param_groups.append({'params': layer.parameters()})

        super().__init__(param_groups, defaults)

        for group, lambda_j in zip(self.param_groups, regularizations):
            group['lambda'] = lambda_j

        assert len(regularizations) == len(self.param_groups), "Lenght mismatch"

        self.batch_size = batch_size

    def sample_batch(self, data, labels):
        """ Draws a random minibatch (without replacement) of size batch_size from (data, labels). """
        P = data.shape[0]
        idx = torch.randperm(P, device=data.device)[:min(self.batch_size, P)]
        return data[idx], labels[idx]

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    grad = param.grad.detach()
                    # First: weight decay from regularization. Warning: must be first since this depends on current state of param itself!!
                    param.mul_(1. - group['lr'] * group['lambda'])
                    # gradient term
                    param.add_(grad, alpha=-group['lr'])

    # end def SGDOpt


class AdamOpt(torch.optim.Optimizer):

    def __init__(self, model: torch.nn.Module, lr, batch_size, regularizations,
                 betas=(0.9, 0.999), eps=1e-8, decoupled=False):
        """
        Adam on the NN weights (on minibatches), including weight decay (L2 regularization).

        The eq. implemented is (per parameter, with g = grad_p(L_B) + reg * p)
           m     <- beta1 * m + (1 - beta1) * g
           v     <- beta2 * v + (1 - beta2) * g**2
           mhat  =  m / (1 - beta1**t) ,  vhat = v / (1 - beta2**t)
           delta_p = - lr * mhat / ( sqrt(vhat) + eps )
        where
        p are the parameters,
        grad_p(L_B) is the gradient of the loss wrt. p, estimated on a random minibatch B of size batch_size,
        reg the L2 regularization strengths of the parameters,
        t the number of steps taken so far by that parameter (used for the bias correction).

        Set batch_size >= P (number of training points) to recover full-batch GD with Adam.

        Note that the gradient of the loss must be accumulated in .grad of
        the model weights before calling step(). The gradient of the regularization,
        however, is added inside step(), and should therefore not be accumulated in the weight .grad before.
        In other words, use with a loss such as 0.5 * sum(y - model(X))**2 that does not include the regularization.
        The minibatch is drawn with sample_batch() (see trainSGD), which uses the batch_size stored in the optimizer.

        Arguments:
            model : torch.nn.Module
            lr    : float
                Learning rate (step size of the normalized update, not of the raw gradient).
            batch_size : int
                Size of the minibatch used to estimate the gradient at each step.
            regularizations : array-like of shape (L,)
                L2 regularization strengths of the weights.
                Must be array-like containing one float per layer of weights.
            betas : (float, float)
                Decay rates of the first and second moment running averages.
            eps : float
                Term added to the denominator for numerical stability.
            decoupled : bool
                If False (default), the regularization enters the gradient g, so that the
                fixed point of the dynamics is the minimum of L + 0.5 * reg * |p|**2,
                the same as for SGDOpt/LangevinOpt.
                If True, the weight decay is applied directly to the parameters (AdamW style),
                p <- (1 - lr * reg) * p, which is NOT the minimizer of the regularized loss,
                since the decay is then not rescaled by sqrt(vhat).

        """
        defaults = {'lr': lr, 'batch_size': batch_size, 'betas': betas, 'eps': eps}
        param_groups = []
        for layer in model.children():
            if isinstance(layer, (torch.nn.Linear)):
                param_groups.append({'params': layer.parameters()})

        super().__init__(param_groups, defaults)

        for group, lambda_j in zip(self.param_groups, regularizations):
            group['lambda'] = lambda_j

        assert len(regularizations) == len(self.param_groups), "Lenght mismatch"

        self.batch_size = batch_size
        self.decoupled = decoupled

    def sample_batch(self, data, labels):
        """ Draws a random minibatch (without replacement) of size batch_size from (data, labels). """
        P = data.shape[0]
        idx = torch.randperm(P, device=data.device)[:min(self.batch_size, P)]
        return data[idx], labels[idx]

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, eps, lambda_j = group['lr'], group['eps'], group['lambda']
            beta1, beta2 = group['betas']
            for param in group['params']:
                if param.grad is not None:
                    grad = param.grad.detach()

                    state = self.state[param]
                    if len(state) == 0:  # lazy initialization of the moments
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

                    if self.decoupled:
                        # AdamW style: weight decay straight on the params, not seen by the moments.
                        # Warning: must be first since this depends on current state of param itself!!
                        param.mul_(1. - lr * lambda_j)
                    else:
                        # regularization enters the gradient, so it gets preconditioned like the rest
                        grad = grad.add(param, alpha=lambda_j)

                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    # running averages of the gradient and of its square
                    exp_avg.mul_(beta1).add_(grad, alpha=1. - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1. - beta2)
                    # bias corrections
                    bias_corr1 = 1. - beta1 ** state['step']
                    bias_corr2 = 1. - beta2 ** state['step']
                    # normalized gradient term
                    denom = (exp_avg_sq / bias_corr2).sqrt_().add_(eps)
                    param.addcdiv_(exp_avg, denom, value=-lr / bias_corr1)

    # end def AdamOpt


    
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

def trainSGD(net, data, labels, criterion, optimizer):
    """
    One SGD step on a random minibatch of size optimizer.batch_size.
    The minibatch loss is rescaled by P / batch_size, so that its gradient is an unbiased
    estimate of the gradient of the loss on the full dataset (same scale as in train()).
    """
    P = data.shape[0]
    batch_data, batch_labels = optimizer.sample_batch(data, labels)
    optimizer.zero_grad()
    loss = (P / batch_data.shape[0]) * criterion(net(batch_data), batch_labels)
    loss.backward(retain_graph=False)
    optimizer.step()
    return loss.item()

def trainAdam(net, data, labels, criterion, optimizer, rescale=True):
    """
    One Adam step on a random minibatch of size optimizer.batch_size (full batch if batch_size >= P).
    Same structure as trainSGD(): the minibatch loss is rescaled by P / batch_size, so that its
    gradient is an unbiased estimate of the gradient of the loss on the full dataset
    (same scale as in train()) and the returned loss is comparable with the one of train()/trainSGD().

    Note that, unlike in trainSGD(), this rescaling does not change the trajectory: the Adam update
    -lr * mhat / (sqrt(vhat) + eps) is invariant under a global rescaling of the gradient, up to eps.
    Set rescale=False to feed the bare minibatch loss (useful when the summed loss is so large
    that eps becomes negligible, or to log the un-rescaled minibatch loss).
    """
    P = data.shape[0]
    batch_data, batch_labels = optimizer.sample_batch(data, labels)
    optimizer.zero_grad()
    loss = criterion(net(batch_data), batch_labels)
    if rescale:
        loss = (P / batch_data.shape[0]) * loss
    loss.backward(retain_graph=False)
    optimizer.step()
    return loss.item()

def regLoss(output, target):
    return 0.5 * torch.sum((output - target)**2)
    