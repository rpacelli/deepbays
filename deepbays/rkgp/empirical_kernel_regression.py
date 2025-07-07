import numpy as np
import copy
from scipy.optimize import fsolve
from .. import kernels
import torch


def behead_network(net : torch.nn.Sequential, featureloc : int = -2):
    """ 
    Takes a sequential model and return a (copied) feature_extractor model 
    where the top readout (and normalization) layer have been removed.
    
    featureloc: Location of the last-layer features in the model, specifies number of children modules to chop off. default -2  (Linear() and Norm())
    """
    device = next(net.parameters()).device  # assumes all params of net on same device, but well...
    copynet = copy.deepcopy(net)
    feature_extractor = torch.nn.Sequential(*list(copynet.children())[:featureloc]).to(device) # we slice off the last Linear() and the last Norm() blocks to get the feature layer 
    return feature_extractor


class empirical_kernel_regressor():
    # todo: how to set the regularization for the muP case?
    def __init__(self, 
                 feature_extractor  : torch.nn.Sequential, 
                 Xtrain     : torch.Tensor, 
                 Ytrain      : torch.Tensor,
                 reg : float = 1e-8,
                 device = 'cpu',
                 ):
        self.feature_extractor = feature_extractor.to(device) 
        self.Xtrain = Xtrain.to(device)
        self.P = len(Xtrain)
        # while the truncated net and Xtrain can be on device, we will do the rest with numpy arrays
        if len(Ytrain.shape) == 1:
            self.Ytrain = Ytrain.to('cpu').numpy()
        elif Ytrain.shape[1] == 1: # catch shape (P, 1) and squeeze
            self.Ytrain = Ytrain.squeeze(1).to('cpu').numpy()  
        else: raise ValueError(f"Ytrain has wrong shape {Ytrain.shape} (multiple outputs currently not implemented)")
        self.reg = reg
        self.device = device
        
        print('\nInit empirical kernel regression and inverting kernel...')
        with torch.no_grad():
            self.feats = self.feature_extractor(self.Xtrain).to('cpu').numpy()
            
        self.N = self.feats.shape[1]
        self.KXX = self.feats.dot(self.feats.T) #/ self.N  # uses np.dot already, not torch.dot !
        self.invK = np.linalg.inv(self.KXX + reg * np.eye(self.P))
        self.optweights = self.feats.T.dot(self.invK.dot(self.Ytrain)) # ridge regression readout vector
        print('... done.\n')


    def predict(self, Xtest):
        self.Xtest = Xtest.to(self.device)
        with torch.no_grad():
            self.testfeats = self.feature_extractor(self.Xtest).to('cpu').numpy()
            
        self.Ypred = self.testfeats.dot(self.optweights)
        return self.Ypred
    
    def averageGeneralization(self, Xtest, Ytest):
        """ average generalization error of optimal (regularized) readout from empirical features"""
        self.predict(Xtest)
        self.empGenerror = np.mean((Ytest.squeeze(1).to('cpu').numpy() - self.Ypred)**2) 
        return self.empGenerror        
    
    
    #### below are functions assuming the empirical kernel KXX is the kernel of a GP regression. 
    #### This is not the case in practice since we are here using the deterministic, empirical test features and the output is thus deterministic (gen error corresponds to bias of the GP regression)
    
    def computeTestsetKernelsGP(self, Xtest):
        self.Ptest = len(Xtest)
        with torch.no_grad():
            self.testfeats = self.feature_extractor(Xtest.to(self.device)).to('cpu').numpy()
        self.K0 = np.dot(self.testfeats, self.testfeats.T).diagonal() #/ self.N
        self.K0X = np.dot(self.testfeats, self.feats.T) #/ self.N
    
    def predictGP(self, Xtest):
        #todo: check if normalizing by internal dimension is currently correct
        self.computeTestsetKernelsGP(Xtest) # computes self.K0 and self.K0X
        K0X_invK = self.K0X.dot(self.invK)
        
        self.Y_meanpred = K0X_invK.dot(self.Ytrain)
        self.Y_varpred = self.K0 - (K0X_invK * self.K0X).sum(axis=1)  # np.sum gives trace of K0X_invK.dot(self.K0X.T) without computing the offdiag elements
        return self.Y_meanpred, self.Y_varpred
    
    def averageGeneralizationGP(self, Xtest, Ytest):
        print("Warning: This generror assumes the empirical train and test kernels as GP kernels, but in practice the net has no stochasticity so the variance computed here is not present in the network.")
        self.predictGP(Xtest)
        biases = Ytest.squeeze(1).to('cpu').numpy() - self.Y_meanpred   # here first un-squared convention
        self.Bias = (biases**2).mean()  # here Bias refers to the squared version
        self.Variance = self.Y_varpred.mean()
        self.Generror = self.Bias + self.Variance
        return self.Generror, self.Bias, self.Variance
    