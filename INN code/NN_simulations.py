# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:44:10 2019

@author: bojan.mavkov
"""

import numpy as np
import torch


def f_sim(ss_model, x0, n_x, U, dt):
    N = np.shape(U)[0]
    X_sim = torch.zeros((N, n_x), dtype=U.dtype)

    x_tmp = x0
    for i in range(N):
            X_sim[i, :] = x_tmp
            ustep = U[i]
            x_dot = ss_model(x_tmp.float(), ustep.float())
            x_tmp = dt*x_dot+x_tmp
    return X_sim


def f_onestep(ss_model, X, U, dt, n_x, X0):
    
    N = np.shape(U)[0]
    X_list=X0
    X_sim=np.zeros((N,n_x))
    X_sim=torch.from_numpy(X_sim)
    x_sum=X_list
    for i in range(N):
            ustep = U[i]
            x_p1 = ss_model(X_list.float(),ustep.float())
            x_sum = dt*x_p1+x_sum
            X_list=X[i]
            X_sim[i,:]=x_sum
    X_sim=X_sim.data.numpy()
    return X_sim
    