import scipy.linalg
import torch
import numpy as np
import utils
import scipy

class CONV_1HL():
    def __init__(self, 
                 Nc : int,
                 T : float,
                 M : int,
                 S : int,
                 l0 : float = 1.0,
                 l1 : float = 1.0,
                 act : str = "erf",
                 bias : bool = False):

        self.Nc, self.T, self.M, self.S, self.l0, self.l1 = Nc, T, M, S, l0, l1
        self.act = act
        if bias: raise SystemExit("Bias not available, aborting.")

    def preprocess(self, Xtrain, Ytrain):
        self.x = Xtrain
        self.y = Ytrain.squeeze().float()
        self.P, self.N0 = Xtrain.shape
        self.patches, self.remainder = divmod(self.N0, self.S)

        self.C = torch.zeros(self.P, self.P, self.patches, self.patches)

        for i in range(self.patches):
            for j in range(self.patches):
                self.C[:, :, i, j] = torch.einsum("ij,kj->ik", self.x[ :, self.S*i:self.S*i+self.M], self.x[ :, self.S*j:self.S*j+self.M] ) / (self.l0 * self.M)

        if self.act == "lin":
            self.K = self.C / (self.l1 * self.patches)
            self.Kmu = self.Cmu / (self.l1 * self.patches)
            self.K0  = self.C0  / (self.l1 * self.patches)
        elif self.act == "erf":
            Cden = torch.einsum( "mmii,nnjj->mnij", torch.sqrt(1+2*self.C), torch.sqrt(1+2*self.C) )
            self.K = (2 / torch.pi) * torch.arcsin( (2 * self.C) /  Cden ) / (self.l1 * self.patches)
        else: raise SystemExit("Act not available, aborting.")

    def optimize(self, Q0 = 1, lr=0.0005, tolerance = 1e-6, max_epochs = 1000, maxCheck = 5):
        if torch.sum(Q0)==1: Q0 = torch.eye(self.patches)
        Q0vec = self.toVector(Q0)
        Qvec = Q0vec.clone().detach().requires_grad_(True)
        self.alpha = self.P / self.Nc
        def S(Qvec):
            Q = self.toMatrix(Qvec)
            KR = torch.einsum("ij,mnij->mn", Q, self.K)
            term1 = torch.trace(Q)
            term2 = - torch.log( torch.det(Q) )
            gpKernel = self.T*torch.eye(self.P) + KR
            term3 = (self.alpha / self.P) * torch.logdet( gpKernel )
            term4 = (self.alpha / self.P) * torch.matmul( self.y, torch.matmul( torch.linalg.inv(self.T*torch.eye(self.P) + KR) , self.y ) )
            return term1 + term2 + term3 + term4
        optimizer = torch.optim.Adam([Qvec], lr)
        previous_loss = 100.
        self.optState = False
        self.optEpochs = 0
        check = 0
        for i in range(max_epochs):
            optimizer.zero_grad()
            loss = S(Qvec)
            loss.backward()
            optimizer.step()
            loss_change = abs(loss.item() - previous_loss)
            if (loss_change < tolerance):
                check += 1
            if (loss_change < tolerance) and (check > maxCheck):
                self.optState = True
                break
            previous_loss = loss.item()
            self.optEpochs +=1
        self.optQvec = Qvec
        self.optQ = self.toMatrix(self.optQvec)

    def setIW(self):
        self.optQ = torch.eye(self.patches)
        self.optState = True

    def setKR(self):
        self.KR = torch.einsum('ij,mnij->mn', self.optQ, self.K)

    def predict(self, xt):

        self.xt = xt
        self.Pt, self.N0t = xt.shape

        self.Cmu = torch.zeros(self.Pt, self.P, self.patches, self.patches)
        self.C0  = torch.zeros(self.Pt, self.patches, self.patches)

        for i in range(self.patches):
            for j in range(self.patches):
                self.Cmu[:, :, i, j] = torch.einsum("ij,kj->ik", self.xt[:, self.S*i:self.S*i+self.M], self.x[ :, self.S*j:self.S*j+self.M] ) / (self.l0 * self.M)
                self.C0[:, i, j]     = torch.einsum("ij,ij->i" , self.xt[:, self.S*i:self.S*i+self.M], self.xt[:, self.S*j:self.S*j+self.M] ) / (self.l0 * self.M)

        if self.act == "lin":
            self.K = self.C / (self.l1 * self.patches)
            self.Kmu = self.Cmu / (self.l1 * self.patches)
            self.K0  = self.C0  / (self.l1 * self.patches)
        if self.act == "erf":
            Cmuden = torch.einsum( "mii,nnjj->mnij", torch.sqrt(1+2*self.C0), torch.sqrt(1+2*self.C) )
            C0den = torch.einsum( "mii,mjj->mij", torch.sqrt(1+2*self.C0), torch.sqrt(1+2*self.C0) )
            self.Kmu = (2 / torch.pi) * torch.arcsin( (2 * self.Cmu) /  Cmuden ) / (self.l1 * self.patches)
            self.K0 = (2 / torch.pi) * torch.arcsin( (2 * self.C0) /  C0den ) / (self.l1 * self.patches)

        self.KRmu = torch.einsum('ij,mnij->mn', self.optQ, self.Kmu)
        self.KR0  = torch.einsum('ij,mij->m', self.optQ, self.K0)
        self.setKR()

        self.gpKernel = self.T * torch.eye(self.P) + self.KR
        self.gpKernel_inv = torch.linalg.inv(self.gpKernel)

        self.yhat = torch.einsum('ij,j->i', self.KRmu, torch.einsum('ij,j->i', self.gpKernel_inv, self.y) )
        return self.yhat
    
    def averageLoss(self, Ytest):
        self.yt = Ytest.squeeze().float()
        self.Sigma = self.KR0 - torch.einsum('ki,ik->k', self.KRmu, torch.einsum('ij,kj->ik', self.gpKernel_inv, self.KRmu))
        bias = torch.sum( torch.square(self.yt - self.yhat) ) / self.Pt
        variance = torch.sum(self.Sigma) / self.Pt
        predLoss = bias + variance
        return predLoss.item(), bias.item(), variance.item()
    

    def toVector(self, Q):
        patches = Q.shape[0]
        dof = patches * (patches+1) /2
        dof = int(dof)
        Qvec = torch.zeros(dof)
        count = 0
        for i in range(patches):
            for j in range(i, patches):
                Qvec[count] = Q[i,j]
                count +=1
        return Qvec

    def toMatrix(self, Qvec):
        dof = Qvec.shape[0]
        patches = ( -1 + np.sqrt(1+8*dof) ) / 2
        patches = int(patches)
        Q = torch.zeros(patches, patches)
        count = 0
        for i in range(patches):
            for j in range(i, patches):
                Q[i,j] = Qvec[count]
                Q[j,i] = Qvec[count]
                count +=1
        return Q
