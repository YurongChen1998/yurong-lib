import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class robust_PCA(nn.Module):
    def __init__(self, M, gamma=None, mu=None, tol=None, max_iter=None):
        super(robust_PCA, self).__init__()
        self.num_pixel, self.band = M.shape
        self.M_norm = torch.norm(M, p='fro')
        self.L = torch.zeros([self.num_pixel, self.band])
        self.S = torch.zeros([self.num_pixel, self.band])
        self.Y = torch.zeros([self.num_pixel, self.band])
        
        ######  Define Hyper-parameteres #####
        if gamma:
            self.gamma = gamma
        else:
            self.gamma = 1 / np.sqrt(max(self.num_pixel, self.band))
            
        if mu:
            self.mu = mu
        else:
            self.mu = 10 * self.gamma
            
        if tol:
            self.tol = tol
        else:
            self.tol = 1e-6
        
        if max_iter:
            self.max_iter = max_iter
        else:
            self.max_iter = 1000
        ######  Define Hyper-parameteres #####
        
    def shrinkage(self, tau, X):
        sing_X = torch.sign(X)
        X = torch.abs(X) - tau
        X[X < 0] = 0
        return sing_X * X
    
    def svd_decom(self, tau, X):
        U, S, Vh = torch.svd(X, some=True)
        S = torch.diag(S)
        S = self.shrinkage(tau, S)
        return torch.mm(torch.mm(U, S), torch.transpose(Vh, 0, 1))
        
    def forward(self, M):
        for i in range(self.max_iter):
            self.L = self.svd_decom(1/self.mu, M - self.S + (1/self.mu)*self.Y)
            self.S = self.shrinkage(self.gamma/self.mu, M - self.L + (1/self.mu)*self.Y)
            Z = M - self.L - self.S
            self.Y = self.Y + self.mu * Z
            
            error = torch.norm(Z, p='fro')/self.M_norm
            
            if i % 10 == 0:
                print('iter:', i, 'error:', error, 'rank:', torch.matrix_rank(self.L).item())
                
        return self.L, self.S        
  
if __name__ == '__main__':     
    num_case = 200
    band = 50
    rank = 5
    
    rand_row = torch.rand(rank, num_case)
    toy_data = torch.zeros([band, num_case])
    for i in range(band):
        idx = int(torch.rand(1) * rank)
        toy_data[i, :] = rand_row[idx, :]
    
    rand_noise = torch.rand(band, num_case)
    anomaly = torch.sign(rand_noise - 0.5)
    rand_noise = torch.where(rand_noise>0.2, 0, 1)
    anomaly = anomaly * rand_noise
    data = toy_data + anomaly
    
    rpca = robust_PCA(data)
    L, S = rpca(data)
    
    ##### Plot Figure #####
    fig, axes = plt.subplots(2, 3)
    axes[0, 0].imshow(toy_data, cmap='gray')
    axes[0, 0].set_title('Low Rank Matrix')
    axes[0, 1].imshow(anomaly, cmap='gray')
    axes[0, 1].set_title('Anomaly Matrix')
    axes[0, 2].imshow(data, cmap='gray')
    axes[0, 2].set_title('Data Matrix')
    
    axes[1, 0].imshow(L, cmap='gray')
    axes[1, 0].set_title('Reconstructed Low Rank Matrix')
    axes[1, 1].imshow(S, cmap='gray')
    axes[1, 1].set_title('Reconstructed Anomaly Matrix')
    axes[1, 2].imshow(L + S, cmap='gray')
    axes[1, 2].set_title('Reconstructed Data Matrix')
    plt.show()
    
    
    print("..............Data Matrix..............")
    print(data)
    print("..............Reconstructed Data Matrix..............")
    print(L + S)
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    print("..............Low Rank Matrix..............")
    print(toy_data)
    print("..............Reconstructed Low Rank Matrix..............")
    print(L)
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    print("..............Anomaly Matrix..............")
    print(anomaly)
    print("..............Reconstructed Anomaly Matrix..............")
    print(S)
    
