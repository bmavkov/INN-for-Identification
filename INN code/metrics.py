# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 13:16:16 2020

@author: install
"""
import numpy as np


def metric(y_sim, Outputs, type_metric):
    if type_metric == 'ERMS':
         err_train=y_sim-Outputs
         N = np.shape(y_sim)[0]
         Erms_train= ((1/N)*np.sum(err_train**2))
         norm=np.sqrt(Erms_train)
    elif type_metric == 'R2':
         SSE = np.sum((y_sim-Outputs)**2)
         y_mean = np.mean(Outputs)
         SST = np.sum((Outputs - y_mean)**2)
         norm = 1 - SSE/SST
    else:
         print("Wrong Norm")
         norm = np.nan

    return norm
