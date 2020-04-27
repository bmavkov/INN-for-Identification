# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:28:09 2019

@author: bojan.mavkov
"""


import torch
import torch.nn as nn




import numpy as np



class NeuralStateSpaceModel(nn.Module):
   
    def __init__(self, n_x, n_u, n_feat, init_small=False):
        super(NeuralStateSpaceModel, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
        self.net = nn.Sequential(
            nn.Linear(n_x+n_u, n_feat[0]),  # 2 states, 1 input
            nn.Sigmoid(),
            nn.Linear(n_feat[0], n_feat[1]),
            nn.Sigmoid(),
            nn.Linear(n_feat[1], n_x),
        )

        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, X, U):
        XU = torch.cat((X, U), -1)
        DX = self.net(XU)
        return DX
    
    
class INN:
    
    def __init__(self, nn_model):
        self.nn_model = nn_model

    def INN_est(self, X_hat, U, dt):
        X_dot = self.nn_model(X_hat, U)
        DX = dt*torch.cumsum(X_dot, dim=0)
        x0 = X_hat[0, :]
        X_hat_I = x0 + DX

        return X_hat_I

class NeuralStateSpaceModel_y(nn.Module):
   
    def __init__(self, n_x, n_y, n_feat=64):
        super(NeuralStateSpaceModel_y, self).__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.n_feat = n_feat
        self.net = nn.Sequential(
            nn.Linear(n_x, n_feat),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(n_feat, n_y)
        )

    def forward(self, X):
        Y = self.net(X)
        return Y