# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:44:10 2019

@author: bojan.mavkov
"""



import numpy as np
import torch







def f_sim_incr(ss_model,X0,n_x,U,dt):
    
    
    N = np.shape(U)[0]
    X_list=X0
    X_sim=np.zeros((N,n_x))
    X_sim=torch.from_numpy(X_sim)
    x_sum=X_list
    for i in range(N):
            ustep = U[i]
            x_p1 = ss_model(X_list.float(),ustep.float())
            x_sum = dt*x_p1+x_sum
            X_list=x_sum
            X_sim[i,:]=X_list
    X_sim=X_sim.data.numpy()
     
    return X_sim

def f_sim_y_incr(ss_model,X0,n_x,U,dt):
    
    
    N = np.shape(U)[0]
    X_list=X0
    X_sim=np.zeros((N,n_x))
    X_sim=torch.from_numpy(X_sim)
    x_sum=X_list
    for i in range(N):
            ustep = U[i]
            x_p1 = ss_model(X_list.float(),ustep.float())
            x_sum = dt*x_p1+x_sum
            X_list=x_sum
            X_sim[i,:]=X_list
    
     
    return X_sim



def f_sim_os(ss_model,X,U,dt,n_x,X0):
    
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
    