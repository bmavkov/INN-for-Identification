
"""
Created on Tue Jan 21 14:15:39 2020

@author: bojan.mavkov
"""

import torch 
import torch.optim as optim
import time
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Models
from NNmodels import NeuralStateSpaceModel
from NNmodels import NeuralStateSpaceModel_y
from NNmodels import INN

# Simulation functions
from NN_simulations import f_sim_os
from NN_simulations import f_sim_incr
from NN_simulations import f_sim_y_incr

# Metrics
from metrics import metric


if __name__ == '__main__':

    # In[Settings]
    lr = 1e-4  # learning rate
    num_iter = 400000  # gradient-based optimization steps
    test_freq = 100  # print message every test_freq iterations
    add_noise = False  # Add additional noise to the data

    n_feat = [40, 40]  # number of neurons per layer of the state mapping function

    fix_om = False  # Fix the output mapping function
    n_feat_y = 5  # number of neurons of the output mapping function

    onestep_sim = False  # one-step simulations (Only applicable to the training data)
    do_mean = True  # remove the mean values of the data
    save_model = False  # Save the identified model parameters and simulations

    n_x = 2  # Number of states
    n_y = 1  # Number of outputs
    alpha = 1/8  # Scaling factor

    # In[Load data]

    data_ident = pd.read_csv('data/dataBenchmark.csv')
    dt = data_ident[['Ts']].values[0].item()

    Inputs = data_ident[['uEst']].values.astype(np.float32)
    Outputs = data_ident[['yEst']].values.astype(np.float32)

    Inputs_val = data_ident[['uVal']].values.astype(np.float32)
    Outputs_val = data_ident[['yVal']].values.astype(np.float32)

    # Number of data
    N = np.shape(Inputs)[0]

    if add_noise:
        lvl_out = abs(np.mean(Outputs, axis=0))
        noise_out = np.random.normal(0, lvl_out, Outputs.shape)
        lvl_in = abs(np.mean(Inputs, axis=0))
        noise_in = np.random.normal(0, lvl_in, Inputs.shape)
        Inputs = Inputs
        Outputs = Outputs+0.7*noise_out

    n_u = Inputs.shape[1]
    Ident_inputs = Inputs
    Ident_outputs = Outputs

    Ident_inputs_val = Inputs_val
    Ident_outputs_val = Outputs_val

    # In[Normalize input and outputs]
    if do_mean:
        # Remove the mean values and divide with the standard deviation
        mean_in_train = np.mean(Ident_inputs, axis=0)
        mean_out_train = np.mean(Ident_outputs, axis=0)
        standard_inp = np.std(Ident_inputs, axis=0)
        standard_outputs = np.std(Ident_outputs, axis=0)
        Ident_inputs = Ident_inputs-mean_in_train
        Ident_outputs = Ident_outputs-mean_out_train
        Ident_inputs = Ident_inputs/standard_inp
        Ident_outputs = Ident_outputs/standard_outputs

        Ident_inputs_val = Ident_inputs_val-mean_in_train
        Ident_outputs_val = Ident_outputs_val-mean_out_train
        Ident_inputs_val = Ident_inputs_val/standard_inp
        Ident_outputs_val = Ident_outputs_val/standard_outputs

    # In[Convert data to PyTorch tensors]
    In_tor = torch.from_numpy(Ident_inputs)
    Out_tor = torch.from_numpy(Ident_outputs)

    U_val = Ident_inputs_val
    U_val = torch.from_numpy(U_val)

    U = Ident_inputs
    U = torch.from_numpy(U)

    X = np.zeros((N, n_x), dtype=np.float32)
    X[:, 0] = Ident_outputs[:, 0]
    X_hat = torch.tensor(X, requires_grad=True)
    X = torch.from_numpy(X)

    # In[INN model setup]
    ss_model = NeuralStateSpaceModel(n_x=n_x, n_u=n_u, n_feat=n_feat)
    INN_model = INN(ss_model)
    if not fix_om:
        y_model = NeuralStateSpaceModel_y(n_x=n_x, n_y=n_y, n_feat=n_feat_y)

    # In[Setup Optimizer]
    if fix_om:
        params_net = list(INN_model.nn_model.parameters())
        params_states = [X_hat]
        optimizer = optim.Adam([
            {'params': params_net,    'lr': lr},
            {'params': params_states, 'lr': lr}], lr=lr)
    else:
        params_net = list(INN_model.nn_model.parameters())
        params_states = [X_hat]
        params_y = list(y_model.parameters())
        optimizer = optim.Adam([
            {'params': params_net,    'lr': lr},
            {'params': params_states, 'lr': lr},
            {'params': params_y, 'lr': lr}], lr=lr)

    # In[Training loop]
    J_tot = np.zeros((num_iter + 1))
    J_x = np.zeros((num_iter + 1))
    J_y = np.zeros((num_iter + 1))

    start_time = time.time()
    for itr in range(1, num_iter + 1):

        optimizer.zero_grad()

        # Pass X_hat through the INN to obtain X_hat_I
        X_hat_I = INN_model.INN_est(X_hat, U, dt)

        if fix_om:
          Y_hat = X_hat[:, [0]]
        else:
          Y_hat = y_model(X_hat)

        # Compute fit loss
        err_x = X_hat_I - X_hat

        if fix_om:
           err_y = Y_hat - Out_tor
        else:
           err_y = Y_hat - Out_tor

        loss_x = (1/alpha)*torch.mean(err_x**2)*dt
        loss_y = torch.mean(err_y**2)
        loss = loss_x + loss_y

        # Statistics
        J_tot[itr] = loss.data.numpy()
        J_x[itr] = loss_x.data.numpy()
        J_y[itr] = loss_y.data.numpy()

        if itr % test_freq == 0:
           print('Iter {:04d} | Total Loss {:.4f} | Loss X {:.4f} | Loss Y {:.4f}'.format(itr, loss.item(), loss_x.item(), loss_y.item()))

        # Optimization step
        loss.backward()
        optimizer.step()

    train_time = time.time() - start_time
    print(f"\nTrain time: {train_time:.2f}")

    # In[Figures]
    f1 = plt.figure()
    plt.plot(Outputs)
    plt.title('outputs (Training)')

    f2 = plt.figure()
    plt.plot(Inputs)
    plt.title('Inputs (Training)')

    # Plot the cost functions
    f3 = plt.figure()
    plt.plot(J_tot, 'k')
    plt.title('Loss Function')

    f4 = plt.figure()
    plt.plot(J_x, 'k')
    plt.title('Loss Jx Function')

    f5 = plt.figure()
    plt.plot(J_y, 'k')
    plt.title('Loss Jy Function')

    # In[Simulate on the training set]
    X0 = X_hat[0, :]
    if onestep_sim:
       X_sim = f_sim_os(INN_model.nn_model, X_hat.detach(), U, dt, n_x, X0)
       y_ss = X_sim[:, 0]
    else:
       if fix_om:
         X_sim = f_sim_incr(INN_model.nn_model, X0, n_x, U, dt)
         y_ss = X_sim[:, 0]
         y_ss = np.transpose([y_ss])
       else:
         X_sim = f_sim_y_incr(ss_model, X0, n_x, U, dt)
         y_ss = y_model(X_sim.float())
         y_ss = y_ss.data.numpy()

    # In[De-normalize output]
    if do_mean:
     y_sim = (y_ss*standard_outputs)+mean_out_train
    else:
     y_sim = y_ss

    Erms_train = metric(y_sim, Outputs, 'ERMS')
    print('ERMS training data: {:.3f}'.format(Erms_train))

    f9 = plt.figure()
    plt.plot(y_sim, '-b', label='Output sim')
    plt.plot(Outputs, 'k', label='Output ')
    plt.title('Train Data')
    leg = f9.legend()

    # In[Simulate on the validation set]
    if fix_om:
         X_sim_val = f_sim_incr(INN_model.nn_model, X0, n_x, U_val, dt)
         y_ss_val = X_sim_val[:, 0]
         y_ss_val = np.transpose([y_ss_val])
    else:
         X_sim_val = f_sim_y_incr(INN_model.nn_model, X0, n_x, U_val, dt)
         y_ss_val = y_model(X_sim_val)
         y_ss_val = y_ss_val.data.numpy()


    if do_mean:
     y_sim_val=(y_ss_val*standard_outputs)+mean_out_train
    else:
     y_sim_val=y_ss_val

    Erms_val = metric(y_sim_val, Outputs_val, 'ERMS')
    print('ERMS validation data: {:.3f}'.format(Erms_val))



    f11 = plt.figure()
    plt.plot(y_sim_val, '-b', label='Output sim')
    plt.plot(Outputs_val, 'k', label='Output ')
    plt.title('Test Data')
    leg = f11.legend()

    # In[Save model]
    if save_model:
        torch.save(ss_model, 'saved_model_output')
        torch.save( mean_in_train, 'mean_in_train_output')
        torch.save(mean_out_train, 'mean_out_train_output')
        torch.save(standard_inp, 'standard_inp_output')
        torch.save(standard_outputs, 'standard_outputs_output')
        sio.savemat('arrdata.mat', mdict={'X_sim_val': X_sim_val,'y_sim_val': y_sim_val,'Outputs_val': Outputs_val,'X_sim_rs': X_sim,'y_sim': y_sim,'Outputs': Outputs,'Erms_val': Erms_val,'Erms_train': Erms_train})
