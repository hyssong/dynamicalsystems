import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression

class mindy(nn.Module):
    def __init__(self, nparcel, ninput, device='cpu'):
        super(mindy, self).__init__()

        self.nparcel = nparcel
        self.ninput = ninput
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.lambda1, self.lambda2, self.lambda3 = 0.075, 0.2, 0.05
        self.low_rank = int(self.nparcel/3)

        # Parameters
        self.b = 20 / 3
        self.scale = .01
        self.W_S = nn.Parameter(torch.randn((self.nparcel, self.nparcel), dtype=torch.float32) * self.scale)
        self.W_1 = nn.Parameter(torch.randn((self.nparcel, self.low_rank), dtype=torch.float32) * self.scale)
        self.W_2 = nn.Parameter(torch.randn((self.nparcel, self.low_rank), dtype=torch.float32) * self.scale)
        self.Decay = nn.Parameter(5 + torch.randn((self.nparcel,), dtype=torch.float32) * self.scale * 50)
        self.alpha = nn.Parameter(5 + torch.randn((self.nparcel,), dtype=torch.float32) * self.scale * 5)
        if self.ninput>0: self.beta = nn.Parameter(torch.randn((self.nparcel, self.ninput), dtype=torch.float32) * self.scale)
        self.to(self.device)

        # Optimizer
        self.lr = 2.5e-5
        self.optimizer = optim.NAdam(self.parameters(), lr=self.lr)

    def to_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def to_numpy(self, data):
        return data.cpu().detach().numpy()

    def update_weights(self, loss, retain_graph=True):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

    def forward(self, X0, mu0):
        self.X0 = self.to_tensor(X0)
        if self.ninput>0: self.mu0 = self.to_tensor(mu0)
        self.nT = self.X0.shape[1]

        self.W_L = self.W_1 @ self.W_2.T
        self.W = self.W_S + self.W_L
        self.psi_x = torch.sqrt(self.alpha.unsqueeze(1).repeat(1, self.nT-1)**2 + (self.b * self.X0[:,:-1] + 0.5)**2) - \
                     torch.sqrt(self.alpha.unsqueeze(1).repeat(1, self.nT-1)**2 + (self.b * self.X0[:,:-1] - 0.5)**2)

        self.lambdas = self.lambda1 * torch.norm(self.W_S, p=1) + \
                       self.lambda2 * torch.trace(torch.abs(self.W_S)) + \
                       self.lambda3 * (torch.norm(self.W_1, p=1) + torch.norm(self.W_2, p=1))

        if self.ninput > 0:
            self.Xpred = self.X0[:,:-1] + self.W @ self.psi_x - self.Decay.unsqueeze(1).repeat(1, self.nT-1) * self.X0[:,:-1] + self.beta @ self.mu0[:,:-1]
            self.J = 0.5 * torch.norm(self.X0[:,1:]-self.Xpred, p=2) ** 2 + self.lambdas
        else:
            self.Xpred = self.X0[:,:-1] + self.W @ self.psi_x - self.Decay.unsqueeze(1).repeat(1, self.nT-1) * self.X0[:,:-1]
            self.J = 0.5 * torch.norm(self.X0[:,1:]-self.Xpred, p=2) ** 2 + self.lambdas

        return self.X0[:, 1:], self.Xpred, self.J

    @torch.no_grad()
    def forward_nograd(self, X0, mu0):
        return self.forward(X0, mu0)

    @torch.no_grad()
    def linearregression(self):
        Y = self.X0[:, 1:] - self.X0[:, :-1]
        self.psi_x = torch.sqrt(self.alpha.unsqueeze(1).repeat(1, self.nT-1)**2 + (self.b * self.X0[:,:-1] + 0.5)**2) - \
                     torch.sqrt(self.alpha.unsqueeze(1).repeat(1, self.nT-1)**2 + (self.b * self.X0[:,:-1] - 0.5)**2)
        X1 = self.W @ self.psi_x
        X2 = (-1) * self.Decay.unsqueeze(1).repeat(1, self.nT - 1) * self.X0[:, :-1]

        Y_vec = Y.reshape(-1, 1)
        X1_vec = X1.reshape(-1, 1)
        X2_vec = X2.reshape(-1, 1)

        if self.ninput > 0:
            X3 = self.beta @ self.mu0[:, :-1]
            X3_vec = X3.reshape(-1, 1)
            X = torch.cat((X1_vec, X2_vec, X3_vec), dim=1)
        else:
            X = torch.cat((X1_vec, X2_vec), dim=1)

        XtX = torch.matmul(X.T, X)
        XtY = torch.matmul(X.T, Y_vec)
        theta = torch.linalg.solve(XtX, XtY)

        if self.ninput > 0:
            return theta[0, 0].item(), theta[1, 0].item(), theta[2, 0].item()
        else:
            return theta[0, 0].item(), theta[1, 0].item()

    @torch.no_grad()
    def predict(self, X0, mu0):
        self.X0 = self.to_tensor(X0)
        if self.ninput>0: self.mu0 = self.to_tensor(mu0)
        self.nT = self.X0.shape[1]

        self.psi_x = torch.sqrt(self.alpha.unsqueeze(1).repeat(1, self.nT-1)**2 + (self.b * self.X0[:,:-1] + 0.5)**2) - \
                     torch.sqrt(self.alpha.unsqueeze(1).repeat(1, self.nT-1)**2 + (self.b * self.X0[:,:-1] - 0.5)**2)

        if self.ninput>0: self.pW, self.pD, self.pB = self.linearregression()
        else: self.pW, self.pD = self.linearregression()

        if self.ninput > 0:
            self.Xpred = self.X0[:,:-1] + self.pW * self.W @ self.psi_x - self.pD * self.Decay.unsqueeze(1).repeat(1, self.nT-1) * self.X0[:,:-1] + self.pB * self.beta @ self.mu0[:,:-1]
        else:
            self.Xpred = self.X0[:,:-1] + self.pW * self.W @ self.psi_x - self.pD * self.Decay.unsqueeze(1).repeat(1, self.nT-1) * self.X0[:,:-1]

        if self.ninput>0: return self.Xpred, self.pW, self.pD, self.pB
        else: return self.Xpred, self.pW, self.pD


