# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:28:09 2019

@author: bojan.mavkov
"""


import torch
import torch.nn as nn




import numpy as np



class NeuralStateSpaceModel(nn.Module):
   
    def __init__(self, n_x, n_u, n_feat=64):
        super(NeuralStateSpaceModel, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
        self.net = nn.Sequential(
            nn.Linear(n_x+n_u, n_feat[0]),  # 2 states, 1 input
            #nn.Dropout(0.2),
            #nn.ReLU6(),
            nn.Sigmoid(),
           # nn.SELU(),
            nn.Linear(n_feat[0], n_feat[1]),
         #  #  nn.Dropout(0.2),
           # nn.ReLU6(),
            nn.Sigmoid(),
           # nn.ReLU(),
         #    #nn.SELU(),
            nn.Linear(n_feat[1], n_feat[2]),
         # #   nn.Dropout(0.2),
            nn.Sigmoid(),
           # nn.ReLU(),
         #    nn.Linear(n_feat, n_feat),
         #    nn.Sigmoid(),
         #    nn.Linear(n_feat, n_feat),
         #    nn.Sigmoid(),
         #    nn.Linear(n_feat, n_feat),
         #    nn.Sigmoid(),
            nn.Linear(n_feat[2], n_x),
         #   nn.ReLU()
        )

       
    
    def forward(self, X,U):
        XU = torch.cat((X,U),-1)
        DX = self.net(XU)
        return DX
    
    
class INN:
    
    def __init__(self, nn_model):
        self.nn_model = nn_model

    def INN_est(self, X_est,U,dt):
        X_est_torch = self.nn_model(X_est.float(),U.float())
        X_sum = torch.cumsum(X_est_torch, dim=0)
        x0=X_est[0,:]
        xdot_int=torch.add(x0,dt*X_sum)
 

        return xdot_int

    
    
class NeuralStateSpaceModel_y(nn.Module):
   
    def __init__(self, n_x, n_y, n_feat=64):
        super(NeuralStateSpaceModel_y, self).__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.n_feat = n_feat
        self.net = nn.Sequential(
            nn.Linear(n_x, n_feat),  # 2 states, 1 input
           # nn.Dropout(0.2),
            nn.ReLU(),
          #  nn.Sigmoid(),
           # nn.SELU(),
         #   nn.Linear(n_feat, n_feat),
          #  nn.Dropout(0.2),
            #nn.ReLU(),
         #   nn.Sigmoid(),
            #nn.SELU(),
        #    nn.Linear(n_feat, n_feat),
         #   nn.Dropout(0.2),
          #  nn.ReLU(),
       #     nn.Sigmoid(),
         #   nn.Linear(n_feat, n_feat),
          #  nn.Sigmoid(),
            nn.Linear(n_feat, n_y)
        )

       
    
    def forward(self, X):
        DX = self.net(X)
        return DX
    
  