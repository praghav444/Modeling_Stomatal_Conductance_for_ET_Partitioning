# -*- coding: utf-8 -*-
"""
@author: Pushpendra Raghav
"""
import gc
import mpl_scatter_density
import math
import h5py
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
print(tf.__version__)
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy as sp
from scipy import optimize
from scipy.optimize import fsolve
import pandas as pd
import argparse
import datetime
import matplotlib.dates as mdate

def rmse(y_test, y):
    ''' Calculates Root Mean Square Error (RMSE)
    
        Arguments
        ----------
        y_test: float
                Observations
        y: float
           Predictions
    '''
    return sp.sqrt(sp.mean((y_test - y) ** 2))

def f_1(x, A, B):
    ''' Calculates output from the function first order equation y = ax + b
    
        Arguments
        ----------
        x: float
              value at x
        A, B: floats
              Coefficients
    '''
    return A*x + B

def f_2(x, A):
    ''' Calculates output from the function of first order equation y = ax
    
        Arguments
        ----------
        x: float
                value at x
        A: floats
                Coefficient
    '''
    return A*x

def plot_test(y_true,y_pred): 
    ''' Plots predictions vs. observations, data fitting, and error statistics (e.g., RMSE, MAPE, R^2, MAE, etc.)
    
        Arguments
        ----------
        y_true: float
                Observations
        y_pred: float
                Predictions
    '''    
    #plt.figure()
 
    #fitting variable
    x0 = y_true
    y0 = y_pred
    num0 = len(x0)
 
    # drawing the fitting line
    #A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]  # y0 = A1x0 + B1
    #x1 = np.arange(0, num0+1100, 0.01)
    #y1 = A1*x1 + B1
    #plt.plot(x1, y1, "blue")
    #print('slope: %.2f' % A1)
    #print('intercept: %.2f' % B1)
    #font1 = {'size':20}
    #plt.text(10,500,'y=%.2f*x+%.2f'% (A1,B1), font1)
    
    A1 = optimize.curve_fit(f_2, y_pred, y_true)[0] # y0 = A1x0
    x1 = np.arange(0, num0+1100, 0.01)
    y1 = A1*x1
    plt.plot(x1, y1, "red")
    print('slope: %.2f' % A1)
    font1 = {'size':16}
    plt.text(10,575,'y=%.2f*x'% (A1), font1)
    # rmse_value
    rmse_value = rmse(x0,y0)
    plt.text(10,650,'RMSE= %.2f'% rmse_value, font1)
    print('RMSE:%.2f' % rmse_value)
    # MAE_value
    MAE_value = np.mean(np.abs(y0-x0))
    plt.text(10,725,'MAE= %.2f '% MAE_value, font1)
    print('MAE:%.2f' % MAE_value)    
    # MAPE_value
    MAPE_value = np.mean(np.abs((y0-x0)/x0))*100
    plt.text(10,800,'MAPE= %.2f '% MAPE_value+'%', font1)
    print('MAPE:%.2f' % MAPE_value)
    # R2_value
    corr_1 = np.corrcoef(x0.astype('float'),y0.astype('float'))
    R2 = corr_1[0,1]**2
    plt.text(10,875,r'$R^2$= %.2f'% corr_1[0,1]**2, font1)    
    #('R^2:%.2f' % corr_1[0,1]**2)
    # N
    plt.text(10,950,'N= %d'% num0, font1)
    print('N= %d' % num0)
    
    return A1, rmse_value, MAPE_value, R2, num0, MAE_value

def plot_test_log_rsc(y_true,y_pred): 
    ''' Plots predictions vs. observations, data fitting, and error statistics (e.g., RMSE, MAPE, R^2, MAE, etc.)
    
        Arguments
        ----------
        y_true: float
                Observations
        y_pred: float
                Predictions
    '''    
    #plt.figure()
 
    #fitting variable
    x0 = y_true
    y0 = y_pred
    num0 = len(x0)
    
    A1 = optimize.curve_fit(f_2, y_pred, y_true)[0] # y0 = A1x0
    x1 = np.arange(0, num0+110, 0.01)
    y1 = A1*x1
    plt.plot(x1, y1, "red")
    print('slope: %.2f' % A1)
    font1 = {'size':16}
    plt.text(1,6,'y=%.2f*x'% (A1), font1)
    # rmse_value
    rmse_value = rmse(x0,y0)
    plt.text(1,7,'RMSE= %.2f'% rmse_value, font1)
    print('RMSE:%.2f' % rmse_value)
    # MAE_value
    MAE_value = np.mean(np.abs(y0-x0))
    plt.text(1,8,'MAE= %.2f '% MAE_value, font1)
    print('MAE:%.2f' % MAE_value)    
    # MAPE_value
    MAPE_value = np.mean(np.abs((y0-x0)/x0))*100
    plt.text(1,9,'MAPE= %.2f '% MAPE_value+'%', font1)
    print('MAPE:%.2f' % MAPE_value)
    # R2_value
    corr_1 = np.corrcoef(x0.astype('float'),y0.astype('float'))
    R2 = corr_1[0,1]**2
    plt.text(1,10,r'$R^2$= %.2f'% corr_1[0,1]**2, font1)    
    print('R^2:%.2f' % corr_1[0,1]**2)
    # N
    plt.text(1,11,'N= %d'% num0, font1)
    print('N= %d' % num0)
    
    return A1, rmse_value, MAPE_value, R2, num0, MAE_value

def plot_test_LE_T(y_true,y_pred): 
    ''' Plots predictions vs. observations, data fitting, and error statistics (e.g., RMSE, MAPE, R^2, MAE, etc.)
    
        Arguments
        ----------
        y_true: float
                Observations
        y_pred: float
                Predictions
    '''    
    #plt.figure()
 
    #fitting variable
    x0 = y_true
    y0 = y_pred
    num0 = len(x0)
    
    A1 = optimize.curve_fit(f_2, y_pred, y_true)[0] # y0 = A1x0
    x1 = np.arange(0, num0+1000, 0.01)
    y1 = A1*x1
    plt.plot(x1, y1, "red")
    print('slope: %.2f' % A1)
    font1 = {'size':16}
    plt.text(5,500,'y=%.2f*x'% (A1), font1)
    # rmse_value
    rmse_value = rmse(x0,y0)
    plt.text(5,550,'RMSE= %.2f'% rmse_value, font1)
    print('RMSE:%.2f' % rmse_value)
    # MAE_value
    MAE_value = np.mean(np.abs(y0-x0))
    plt.text(5,600,'MAE= %.2f '% MAE_value, font1)
    print('MAE:%.2f' % MAE_value)    
    # MAPE_value
    MAPE_value = np.mean(np.abs((y0-x0)/x0))*100
    plt.text(5,650,'MAPE= %.2f '% MAPE_value+'%', font1)
    print('MAPE:%.2f' % MAPE_value)
    # R2_value
    corr_1 = np.corrcoef(x0.astype('float'),y0.astype('float'))
    R2 = corr_1[0,1]**2
    plt.text(5,700,r'$R^2$= %.2f'% corr_1[0,1]**2, font1)    
    print('R^2:%.2f' % corr_1[0,1]**2)
    # N
    plt.text(5,750,'N= %d'% num0, font1)
    print('N= %d' % num0)
    
    return A1, rmse_value, MAPE_value, R2, num0, MAE_value

def drop_outliers2(y_true, y_cal): 
    ''' Dropping rows with NaN                 
    '''      
    data = np.c_[y_true, y_cal]
    names = ['y_true', 'y_cal']
    df_y = pd.DataFrame(data, columns=names)    
    df_y = df_y.dropna(axis=0, how='any') 
       
    return df_y['y_true'], df_y['y_cal']

# SW_model_LE: calculate the LE using Shutteleworth-Wallace model given rsc and other input variable
def SW_model_LE(auxi, rsc_pre):
    
    #---- SW-Model ----
    # --> LE = LE_e + LE_c = w_s*PM_s + w_c*PM_c
    # --> PM_s = (delta*A + (rho*Cp*VPD - delta*ras*(A - A_s))/(raa + ras))/(delta + Psy*(1 + rss/(raa + ras)))
    # --> PM_c = (delta*A + (rho*Cp*VPD - delta*rac*A_s)/(raa + rac))/(delta + Psy*(1 + rsc/(raa + rac)))
    # --> w_s = 1/(1 + R_s*R_a/(R_c*(R_s + R_a)))
    # --> w_c = 1/(1 + R_c*R_a/(R_s*(R_c + R_a)))
    # --> R_s = (delta + Psy)*ras + Psy*rss
    # --> R_c = (delta + Psy)*rac + Psy*rsc
    # --> R_a = (delta + Psy)*raa
    
    # --> A = Rn - G
    # --> A_s = Rns - G
    # --> Rns = Rn*exp(-Kr*LAI)
    #----- END ----

    Kr = 0.6
    A = auxi[25,:] - auxi[26,:]
    Rns = auxi[25,:]*tf.exp(-Kr*auxi[27,:])
    A_s = Rns - auxi[26,:]
    delta = auxi[17,:]
    rho = auxi[18,:]
    Cp = auxi[20,:]
    Psy = auxi[19,:]
    VPD = auxi[28,:]/10   # Check the units (should be in kPa)
    ras = auxi[21,:]
    rac = auxi[22,:]
    raa = auxi[23,:]
    rss = auxi[16,:]
    PM_s = (delta*A + (rho*Cp*VPD - delta*ras*(A - A_s))/(raa + ras))/(delta + Psy*(1 + rss/(raa + ras)))
    rsc = tf.exp(rsc_pre)  # y = log(rsc)
    tf.clip_by_value(rsc,1e-5,5000)
    PM_c = (delta*A + (rho*Cp*VPD - delta*rac*A_s)/(raa + rac))/(delta + Psy*(1 + rsc/(raa + rac)))
    R_s = (delta + Psy)*ras + Psy*rss
    R_c = (delta + Psy)*rac + Psy*rsc
    R_a = (delta + Psy)*raa
    w_s = 1/(1 + R_s*R_a/(R_c*(R_s + R_a)))
    w_c = 1/(1 + R_c*R_a/(R_s*(R_c + R_a)))
    
    T_pre = w_c*PM_c
    LE_pre = w_s*PM_s + w_c*PM_c
    
    return LE_pre

# SW_model_T: calculate the T (Transpiration) using Shutteleworth-Wallace model given rsc and other input variable
def SW_model_T(auxi, rsc_pre):
    
    #---- SW-Model ----
    # --> LE = LE_e + LE_c = w_s*PM_s + w_c*PM_c
    # --> PM_s = (delta*A + (rho*Cp*VPD - delta*ras*(A - A_s))/(raa + ras))/(delta + Psy*(1 + rss/(raa + ras)))
    # --> PM_c = (delta*A + (rho*Cp*VPD - delta*rac*A_s)/(raa + rac))/(delta + Psy*(1 + rsc/(raa + rac)))
    # --> w_s = 1/(1 + R_s*R_a/(R_c*(R_s + R_a)))
    # --> w_c = 1/(1 + R_c*R_a/(R_s*(R_c + R_a)))
    # --> R_s = (delta + Psy)*ras + Psy*rss
    # --> R_c = (delta + Psy)*rac + Psy*rsc
    # --> R_a = (delta + Psy)*raa
    
    # --> A = Rn - G
    # --> A_s = Rns - G
    # --> Rns = Rn*exp(-Kr*LAI)
    #----- END ----

    Kr = 0.6
    A = auxi[25,:] - auxi[26,:]
    Rns = auxi[25,:]*tf.exp(-Kr*auxi[27,:])
    A_s = Rns - auxi[26,:]
    delta = auxi[17,:]
    rho = auxi[18,:]
    Cp = auxi[20,:]
    Psy = auxi[19,:]
    VPD = auxi[28,:]/10   # Check the units (should be in kPa)
    ras = auxi[21,:]
    rac = auxi[22,:]
    raa = auxi[23,:]
    rss = auxi[16,:]
    PM_s = (delta*A + (rho*Cp*VPD - delta*ras*(A - A_s))/(raa + ras))/(delta + Psy*(1 + rss/(raa + ras)))
    rsc = tf.exp(rsc_pre)  # y = log(rsc)
    tf.clip_by_value(rsc,1e-5,5000)
    PM_c = (delta*A + (rho*Cp*VPD - delta*rac*A_s)/(raa + rac))/(delta + Psy*(1 + rsc/(raa + rac)))
    R_s = (delta + Psy)*ras + Psy*rss
    R_c = (delta + Psy)*rac + Psy*rsc
    R_a = (delta + Psy)*raa
    w_s = 1/(1 + R_s*R_a/(R_c*(R_s + R_a)))
    w_c = 1/(1 + R_c*R_a/(R_s*(R_c + R_a)))
    
    
    T_pre = w_c*PM_c
    LE_pre = w_s*PM_s + w_c*PM_c
    
    return T_pre

def random_mini_batches(X, Y,Auxi,mini_batch_size, seed, shuffle):
    '''produce mini_batches for each batch_train
        Creates a list of random minibatches from (X, Y)
    Arguments
    ----------
        X: input data, of shape (input size, number of examples)
        Y: true "label" vector
        mini_batch_size: size of the mini-batches, integer
        seed: this is only for repriducibility of the results
    Returns
    ----------
        mini_batches: list of synchronous (mini_batch_X, mini_batch_Y)
    '''
 
    m = int(X.shape[1])  # number of training examples

    mini_batches = []
    np.random.seed(seed)
    
    if shuffle == True:
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
    else:
        permutation = list(range(0,m))

    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))
    shuffled_Auxi = Auxi[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(
        m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Auxi = shuffled_Auxi[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y,mini_batch_Auxi)
        mini_batches.append(mini_batch)
 
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Auxi = shuffled_Auxi[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_Auxi)
        mini_batches.append(mini_batch)
 
    return mini_batches
 
def predict_tuning(X, n_hidden, parameters):
    '''
    Make predictions using trained model
    Arguments
    ----------
    X: Input variables
    n_hidden: number of hidden layers
    parameters: Trained parameters of ANN model
    
    Returns
    ---------
    predictions: Prediction of the desirable variable
    '''
    # get the number of input variables and samples 
    (n_x,m) = X.shape    
    # get the parameters which have been trained (W,b)
    if n_hidden==1:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==2:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==3:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==4:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==5:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==6:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        W7 = tf.convert_to_tensor(value=parameters["W7"])
        b7 = tf.convert_to_tensor(value=parameters["b7"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==7:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        W7 = tf.convert_to_tensor(value=parameters["W7"])
        b7 = tf.convert_to_tensor(value=parameters["b7"])
        W8 = tf.convert_to_tensor(value=parameters["W8"])
        b8 = tf.convert_to_tensor(value=parameters["b8"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==8:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        W7 = tf.convert_to_tensor(value=parameters["W7"])
        b7 = tf.convert_to_tensor(value=parameters["b7"])
        W8 = tf.convert_to_tensor(value=parameters["W8"])
        b8 = tf.convert_to_tensor(value=parameters["b8"])
        W9 = tf.convert_to_tensor(value=parameters["W9"])
        b9 = tf.convert_to_tensor(value=parameters["b9"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==9:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        W7 = tf.convert_to_tensor(value=parameters["W7"])
        b7 = tf.convert_to_tensor(value=parameters["b7"])
        W8 = tf.convert_to_tensor(value=parameters["W8"])
        b8 = tf.convert_to_tensor(value=parameters["b8"])
        W9 = tf.convert_to_tensor(value=parameters["W9"])
        b9 = tf.convert_to_tensor(value=parameters["b9"])
        W10 = tf.convert_to_tensor(value=parameters["W10"])
        b10 = tf.convert_to_tensor(value=parameters["b10"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9,
                  "W10": W10,
                  "b10": b10}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==10:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        W7 = tf.convert_to_tensor(value=parameters["W7"])
        b7 = tf.convert_to_tensor(value=parameters["b7"])
        W8 = tf.convert_to_tensor(value=parameters["W8"])
        b8 = tf.convert_to_tensor(value=parameters["b8"])
        W9 = tf.convert_to_tensor(value=parameters["W9"])
        b9 = tf.convert_to_tensor(value=parameters["b9"])
        W10 = tf.convert_to_tensor(value=parameters["W10"])
        b10 = tf.convert_to_tensor(value=parameters["b10"])
        W11 = tf.convert_to_tensor(value=parameters["W11"])
        b11 = tf.convert_to_tensor(value=parameters["b11"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9,
                  "W10": W10,
                  "b10": b10,
                  "W11": W11,
                  "b11": b11}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==11:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        W7 = tf.convert_to_tensor(value=parameters["W7"])
        b7 = tf.convert_to_tensor(value=parameters["b7"])
        W8 = tf.convert_to_tensor(value=parameters["W8"])
        b8 = tf.convert_to_tensor(value=parameters["b8"])
        W9 = tf.convert_to_tensor(value=parameters["W9"])
        b9 = tf.convert_to_tensor(value=parameters["b9"])
        W10 = tf.convert_to_tensor(value=parameters["W10"])
        b10 = tf.convert_to_tensor(value=parameters["b10"])
        W11 = tf.convert_to_tensor(value=parameters["W11"])
        b11 = tf.convert_to_tensor(value=parameters["b11"])
        W12 = tf.convert_to_tensor(value=parameters["W12"])
        b12 = tf.convert_to_tensor(value=parameters["b12"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9,
                  "W10": W10,
                  "b10": b10,
                  "W11": W11,
                  "b11": b11,
                  "W12": W12,
                  "b12": b12}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==12:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        W7 = tf.convert_to_tensor(value=parameters["W7"])
        b7 = tf.convert_to_tensor(value=parameters["b7"])
        W8 = tf.convert_to_tensor(value=parameters["W8"])
        b8 = tf.convert_to_tensor(value=parameters["b8"])
        W9 = tf.convert_to_tensor(value=parameters["W9"])
        b9 = tf.convert_to_tensor(value=parameters["b9"])
        W10 = tf.convert_to_tensor(value=parameters["W10"])
        b10 = tf.convert_to_tensor(value=parameters["b10"])
        W11 = tf.convert_to_tensor(value=parameters["W11"])
        b11 = tf.convert_to_tensor(value=parameters["b11"])
        W12 = tf.convert_to_tensor(value=parameters["W12"])
        b12 = tf.convert_to_tensor(value=parameters["b12"])
        W13 = tf.convert_to_tensor(value=parameters["W13"])
        b13 = tf.convert_to_tensor(value=parameters["b13"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9,
                  "W10": W10,
                  "b10": b10,
                  "W11": W11,
                  "b11": b11,
                  "W12": W12,
                  "b12": b12,
                  "W13": W13,
                  "b13": b13}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==13:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        W7 = tf.convert_to_tensor(value=parameters["W7"])
        b7 = tf.convert_to_tensor(value=parameters["b7"])
        W8 = tf.convert_to_tensor(value=parameters["W8"])
        b8 = tf.convert_to_tensor(value=parameters["b8"])
        W9 = tf.convert_to_tensor(value=parameters["W9"])
        b9 = tf.convert_to_tensor(value=parameters["b9"])
        W10 = tf.convert_to_tensor(value=parameters["W10"])
        b10 = tf.convert_to_tensor(value=parameters["b10"])
        W11 = tf.convert_to_tensor(value=parameters["W11"])
        b11 = tf.convert_to_tensor(value=parameters["b11"])
        W12 = tf.convert_to_tensor(value=parameters["W12"])
        b12 = tf.convert_to_tensor(value=parameters["b12"])
        W13 = tf.convert_to_tensor(value=parameters["W13"])
        b13 = tf.convert_to_tensor(value=parameters["b13"])
        W14 = tf.convert_to_tensor(value=parameters["W14"])
        b14 = tf.convert_to_tensor(value=parameters["b14"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9,
                  "W10": W10,
                  "b10": b10,
                  "W11": W11,
                  "b11": b11,
                  "W12": W12,
                  "b12": b12,
                  "W13": W13,
                  "b13": b13,
                  "W14": W14,
                  "b14": b14}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==14:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        W7 = tf.convert_to_tensor(value=parameters["W7"])
        b7 = tf.convert_to_tensor(value=parameters["b7"])
        W8 = tf.convert_to_tensor(value=parameters["W8"])
        b8 = tf.convert_to_tensor(value=parameters["b8"])
        W9 = tf.convert_to_tensor(value=parameters["W9"])
        b9 = tf.convert_to_tensor(value=parameters["b9"])
        W10 = tf.convert_to_tensor(value=parameters["W10"])
        b10 = tf.convert_to_tensor(value=parameters["b10"])
        W11 = tf.convert_to_tensor(value=parameters["W11"])
        b11 = tf.convert_to_tensor(value=parameters["b11"])
        W12 = tf.convert_to_tensor(value=parameters["W12"])
        b12 = tf.convert_to_tensor(value=parameters["b12"])
        W13 = tf.convert_to_tensor(value=parameters["W13"])
        b13 = tf.convert_to_tensor(value=parameters["b13"])
        W14 = tf.convert_to_tensor(value=parameters["W14"])
        b14 = tf.convert_to_tensor(value=parameters["b14"])
        W15 = tf.convert_to_tensor(value=parameters["W15"])
        b15 = tf.convert_to_tensor(value=parameters["b15"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9,
                  "W10": W10,
                  "b10": b10,
                  "W11": W11,
                  "b11": b11,
                  "W12": W12,
                  "b12": b12,
                  "W13": W13,
                  "b13": b13,
                  "W14": W14,
                  "b14": b14,
                  "W15": W15,
                  "b15": b15}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
    elif n_hidden==15:
        W1 = tf.convert_to_tensor(value=parameters["W1"])
        b1 = tf.convert_to_tensor(value=parameters["b1"])
        W2 = tf.convert_to_tensor(value=parameters["W2"])
        b2 = tf.convert_to_tensor(value=parameters["b2"])
        W3 = tf.convert_to_tensor(value=parameters["W3"])
        b3 = tf.convert_to_tensor(value=parameters["b3"])
        W4 = tf.convert_to_tensor(value=parameters["W4"])
        b4 = tf.convert_to_tensor(value=parameters["b4"])
        W5 = tf.convert_to_tensor(value=parameters["W5"])
        b5 = tf.convert_to_tensor(value=parameters["b5"])
        W6 = tf.convert_to_tensor(value=parameters["W6"])
        b6 = tf.convert_to_tensor(value=parameters["b6"])
        W7 = tf.convert_to_tensor(value=parameters["W7"])
        b7 = tf.convert_to_tensor(value=parameters["b7"])
        W8 = tf.convert_to_tensor(value=parameters["W8"])
        b8 = tf.convert_to_tensor(value=parameters["b8"])
        W9 = tf.convert_to_tensor(value=parameters["W9"])
        b9 = tf.convert_to_tensor(value=parameters["b9"])
        W10 = tf.convert_to_tensor(value=parameters["W10"])
        b10 = tf.convert_to_tensor(value=parameters["b10"])
        W11 = tf.convert_to_tensor(value=parameters["W11"])
        b11 = tf.convert_to_tensor(value=parameters["b11"])
        W12 = tf.convert_to_tensor(value=parameters["W12"])
        b12 = tf.convert_to_tensor(value=parameters["b12"])
        W13 = tf.convert_to_tensor(value=parameters["W13"])
        b13 = tf.convert_to_tensor(value=parameters["b13"])
        W14 = tf.convert_to_tensor(value=parameters["W14"])
        b14 = tf.convert_to_tensor(value=parameters["b14"])
        W15 = tf.convert_to_tensor(value=parameters["W15"])
        b15 = tf.convert_to_tensor(value=parameters["b15"])
        W16 = tf.convert_to_tensor(value=parameters["W16"])
        b16 = tf.convert_to_tensor(value=parameters["b16"])
        params = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5,
                  "W6": W6,
                  "b6": b6,
                  "W7": W7,
                  "b7": b7,
                  "W8": W8,
                  "b8": b8,
                  "W9": W9,
                  "b9": b9,
                  "W10": W10,
                  "b10": b10,
                  "W11": W11,
                  "b11": b11,
                  "W12": W12,
                  "b12": b12,
                  "W13": W13,
                  "b13": b13,
                  "W14": W14,
                  "b14": b14,
                  "W15": W15,
                  "b15": b15,
                  "W16": W16,
                  "b16": b16}
        x = tf.compat.v1.placeholder("float", [n_x, None])
        Z6 = forward_propagation_for_predict_tuning(x, n_hidden, params)
        sess = tf.compat.v1.Session()
        prediction = sess.run(Z6, feed_dict={x: X})
     
    return prediction

def forward_propagation_for_predict_tuning(X, n_hidden, parameters):
    '''
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    Arguments
    ----------
    X -- input dataset placeholder, of shape (input size, number of examples)
    n_hidden -- number of hidden layers
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3", ...
                  the shapes are given in initialize_parameters_tuning
    Returns
    --------
    Zl -- the output of the last LINEAR unit
    '''
    if n_hidden==1:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']   
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Zl = tf.add(tf.matmul(W2, A1), b2)
    elif n_hidden==2:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2) 
        Zl = tf.add(tf.matmul(W3, A2), b3)
    elif n_hidden==3:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Zl = tf.add(tf.matmul(W4, A3), b4)
    elif n_hidden==4:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Zl = tf.add(tf.matmul(W5, A4), b5)
    elif n_hidden==5:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Zl = tf.add(tf.matmul(W6, A5), b6)
    elif n_hidden==6:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Zl = tf.add(tf.matmul(W7, A6), b7)
    elif n_hidden==7:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Zl = tf.add(tf.matmul(W8, A7), b8)
    elif n_hidden==8:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Zl = tf.add(tf.matmul(W9, A8), b9)
    elif n_hidden==9:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Zl = tf.add(tf.matmul(W10, A9), b10)
    elif n_hidden==10:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Zl = tf.add(tf.matmul(W11, A10), b11)
    elif n_hidden==11:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        W12 = parameters['W12']
        b12 = parameters['b12']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Z11 = tf.add(tf.matmul(W11, A10), b11)
        A11 = tf.nn.relu(Z11)
        Zl = tf.add(tf.matmul(W12, A11), b12)
    elif n_hidden==12:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        W12 = parameters['W12']
        b12 = parameters['b12']
        W13 = parameters['W13']
        b13 = parameters['b13']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Z11 = tf.add(tf.matmul(W11, A10), b11)
        A11 = tf.nn.relu(Z11)
        Z12 = tf.add(tf.matmul(W12, A11), b12)
        A12 = tf.nn.relu(Z12)
        Zl = tf.add(tf.matmul(W13, A12), b13)
    elif n_hidden==13:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        W12 = parameters['W12']
        b12 = parameters['b12']
        W13 = parameters['W13']
        b13 = parameters['b13']
        W14 = parameters['W14']
        b14 = parameters['b14']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Z11 = tf.add(tf.matmul(W11, A10), b11)
        A11 = tf.nn.relu(Z11)
        Z12 = tf.add(tf.matmul(W12, A11), b12)
        A12 = tf.nn.relu(Z12)
        Z13 = tf.add(tf.matmul(W13, A12), b13)
        A13 = tf.nn.relu(Z13)
        Zl = tf.add(tf.matmul(W14, A13), b14)
    elif n_hidden==14:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        W12 = parameters['W12']
        b12 = parameters['b12']
        W13 = parameters['W13']
        b13 = parameters['b13']
        W14 = parameters['W14']
        b14 = parameters['b14']
        W15 = parameters['W15']
        b15 = parameters['b15']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Z11 = tf.add(tf.matmul(W11, A10), b11)
        A11 = tf.nn.relu(Z11)
        Z12 = tf.add(tf.matmul(W12, A11), b12)
        A12 = tf.nn.relu(Z12)
        Z13 = tf.add(tf.matmul(W13, A12), b13)
        A13 = tf.nn.relu(Z13)
        Z14 = tf.add(tf.matmul(W14, A13), b14)
        A14 = tf.nn.relu(Z14)
        Zl = tf.add(tf.matmul(W15, A14), b15)
    elif n_hidden==15:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        W12 = parameters['W12']
        b12 = parameters['b12']
        W13 = parameters['W13']
        b13 = parameters['b13']
        W14 = parameters['W14']
        b14 = parameters['b14']
        W15 = parameters['W15']
        b15 = parameters['b15']
        W16 = parameters['W16']
        b16 = parameters['b16']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Z11 = tf.add(tf.matmul(W11, A10), b11)
        A11 = tf.nn.relu(Z11)
        Z12 = tf.add(tf.matmul(W12, A11), b12)
        A12 = tf.nn.relu(Z12)
        Z13 = tf.add(tf.matmul(W13, A12), b13)
        A13 = tf.nn.relu(Z13)
        Z14 = tf.add(tf.matmul(W14, A13), b14)
        A14 = tf.nn.relu(Z14)
        Z15 = tf.add(tf.matmul(W15, A14), b15)
        A15 = tf.nn.relu(Z15)
        Zl = tf.add(tf.matmul(W16, A15), b16)

    return Zl

def create_placeholder(n_x,n_y):
    '''
    create placeholder for X,Y. dimensions:(n_x, n_y)
    '''
    X = tf.compat.v1.placeholder(dtype=tf.float32,shape=[n_x,None],name="X")
    Y = tf.compat.v1.placeholder(dtype=tf.float32,shape=[n_y,None],name="Y")
 
    return X,Y
 
def initialize_parameters_tuning(var_n, n_hidden, n_neurons):
    '''
    Tensorflow initialize parameters
    Argument
    --------
    var_n: number of input variables
    n_hidden: number of hidden layers; 1<=n_hidden<=15
    n_neurons: number of neurons each layer
    Returns
    ---------
    parameters: Initialized parameters for W and b
    '''
    # set the random seed for parameter initialization
    tf.compat.v1.set_random_seed(1)
    # initialize parameters for W, b
    if n_hidden ==1:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[1,1],initializer=tf.compat.v1.zeros_initializer())
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    elif n_hidden ==2:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[1,1],initializer=tf.compat.v1.zeros_initializer())
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}
    elif n_hidden ==3:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4}
    elif n_hidden ==4:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())              
        W5 = tf.compat.v1.get_variable(name="W5",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5}
    elif n_hidden ==5:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6}
    elif n_hidden ==6:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W7 = tf.compat.v1.get_variable(name="W7",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b7 = tf.compat.v1.get_variable(name="b7",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6,
                     "W7":W7,"b7":b7}
    elif n_hidden ==7:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W7 = tf.compat.v1.get_variable(name="W7",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b7 = tf.compat.v1.get_variable(name="b7",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W8 = tf.compat.v1.get_variable(name="W8",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b8 = tf.compat.v1.get_variable(name="b8",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6,
                     "W7":W7,"b7":b7,"W8":W8,"b8":b8}
    elif n_hidden ==8:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W7 = tf.compat.v1.get_variable(name="W7",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b7 = tf.compat.v1.get_variable(name="b7",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W8 = tf.compat.v1.get_variable(name="W8",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b8 = tf.compat.v1.get_variable(name="b8",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W9 = tf.compat.v1.get_variable(name="W9",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b9 = tf.compat.v1.get_variable(name="b9",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6,
                     "W7":W7,"b7":b7,"W8":W8,"b8":b8,"W9":W9,"b9":b9}
    elif n_hidden ==9:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W7 = tf.compat.v1.get_variable(name="W7",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b7 = tf.compat.v1.get_variable(name="b7",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W8 = tf.compat.v1.get_variable(name="W8",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b8 = tf.compat.v1.get_variable(name="b8",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W9 = tf.compat.v1.get_variable(name="W9",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b9 = tf.compat.v1.get_variable(name="b9",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W10 = tf.compat.v1.get_variable(name="W10",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b10 = tf.compat.v1.get_variable(name="b10",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6,
                     "W7":W7,"b7":b7,"W8":W8,"b8":b8,"W9":W9,"b9":b9,"W10":W10,"b10":b10}
    elif n_hidden ==10:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W7 = tf.compat.v1.get_variable(name="W7",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b7 = tf.compat.v1.get_variable(name="b7",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W8 = tf.compat.v1.get_variable(name="W8",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b8 = tf.compat.v1.get_variable(name="b8",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W9 = tf.compat.v1.get_variable(name="W9",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b9 = tf.compat.v1.get_variable(name="b9",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W10 = tf.compat.v1.get_variable(name="W10",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b10 = tf.compat.v1.get_variable(name="b10",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W11 = tf.compat.v1.get_variable(name="W11",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b11 = tf.compat.v1.get_variable(name="b11",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6,
                     "W7":W7,"b7":b7,"W8":W8,"b8":b8,"W9":W9,"b9":b9,"W10":W10,"b10":b10,"W11":W11,"b11":b11}
    elif n_hidden ==11:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W7 = tf.compat.v1.get_variable(name="W7",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b7 = tf.compat.v1.get_variable(name="b7",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W8 = tf.compat.v1.get_variable(name="W8",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b8 = tf.compat.v1.get_variable(name="b8",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W9 = tf.compat.v1.get_variable(name="W9",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b9 = tf.compat.v1.get_variable(name="b9",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W10 = tf.compat.v1.get_variable(name="W10",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b10 = tf.compat.v1.get_variable(name="b10",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W11 = tf.compat.v1.get_variable(name="W11",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b11 = tf.compat.v1.get_variable(name="b11",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W12 = tf.compat.v1.get_variable(name="W12",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b12 = tf.compat.v1.get_variable(name="b12",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6,
                     "W7":W7,"b7":b7,"W8":W8,"b8":b8,"W9":W9,"b9":b9,"W10":W10,"b10":b10,"W11":W11,"b11":b11,
                     "W12":W12,"b12":b12}
    elif n_hidden ==12:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W7 = tf.compat.v1.get_variable(name="W7",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b7 = tf.compat.v1.get_variable(name="b7",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W8 = tf.compat.v1.get_variable(name="W8",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b8 = tf.compat.v1.get_variable(name="b8",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W9 = tf.compat.v1.get_variable(name="W9",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b9 = tf.compat.v1.get_variable(name="b9",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W10 = tf.compat.v1.get_variable(name="W10",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b10 = tf.compat.v1.get_variable(name="b10",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W11 = tf.compat.v1.get_variable(name="W11",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b11 = tf.compat.v1.get_variable(name="b11",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W12 = tf.compat.v1.get_variable(name="W12",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b12 = tf.compat.v1.get_variable(name="b12",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W13 = tf.compat.v1.get_variable(name="W13",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b13 = tf.compat.v1.get_variable(name="b13",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6,
                     "W7":W7,"b7":b7,"W8":W8,"b8":b8,"W9":W9,"b9":b9,"W10":W10,"b10":b10,"W11":W11,"b11":b11,
                     "W12":W12,"b12":b12,"W13":W13,"b13":b13}
    elif n_hidden ==13:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W7 = tf.compat.v1.get_variable(name="W7",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b7 = tf.compat.v1.get_variable(name="b7",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W8 = tf.compat.v1.get_variable(name="W8",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b8 = tf.compat.v1.get_variable(name="b8",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W9 = tf.compat.v1.get_variable(name="W9",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b9 = tf.compat.v1.get_variable(name="b9",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W10 = tf.compat.v1.get_variable(name="W10",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b10 = tf.compat.v1.get_variable(name="b10",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W11 = tf.compat.v1.get_variable(name="W11",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b11 = tf.compat.v1.get_variable(name="b11",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W12 = tf.compat.v1.get_variable(name="W12",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b12 = tf.compat.v1.get_variable(name="b12",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W13 = tf.compat.v1.get_variable(name="W13",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b13 = tf.compat.v1.get_variable(name="b13",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W14 = tf.compat.v1.get_variable(name="W14",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b14 = tf.compat.v1.get_variable(name="b14",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6,
                     "W7":W7,"b7":b7,"W8":W8,"b8":b8,"W9":W9,"b9":b9,"W10":W10,"b10":b10,"W11":W11,"b11":b11,
                     "W12":W12,"b12":b12,"W13":W13,"b13":b13,"W14":W14,"b14":b14}
    elif n_hidden ==14:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W7 = tf.compat.v1.get_variable(name="W7",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b7 = tf.compat.v1.get_variable(name="b7",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W8 = tf.compat.v1.get_variable(name="W8",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b8 = tf.compat.v1.get_variable(name="b8",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W9 = tf.compat.v1.get_variable(name="W9",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b9 = tf.compat.v1.get_variable(name="b9",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W10 = tf.compat.v1.get_variable(name="W10",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b10 = tf.compat.v1.get_variable(name="b10",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W11 = tf.compat.v1.get_variable(name="W11",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b11 = tf.compat.v1.get_variable(name="b11",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W12 = tf.compat.v1.get_variable(name="W12",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b12 = tf.compat.v1.get_variable(name="b12",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W13 = tf.compat.v1.get_variable(name="W13",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b13 = tf.compat.v1.get_variable(name="b13",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W14 = tf.compat.v1.get_variable(name="W14",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b14 = tf.compat.v1.get_variable(name="b14",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W15 = tf.compat.v1.get_variable(name="W15",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b15 = tf.compat.v1.get_variable(name="b15",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6,
                     "W7":W7,"b7":b7,"W8":W8,"b8":b8,"W9":W9,"b9":b9,"W10":W10,"b10":b10,"W11":W11,"b11":b11,
                     "W12":W12,"b12":b12,"W13":W13,"b13":b13,"W14":W14,"b14":b14,"W15":W15,"b15":b15}
    elif n_hidden ==15:
        W1 = tf.compat.v1.get_variable(name="W1",shape=[n_neurons,var_n],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b1 = tf.compat.v1.get_variable(name="b1",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W2 = tf.compat.v1.get_variable(name="W2",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b2 = tf.compat.v1.get_variable(name="b2",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W3 = tf.compat.v1.get_variable(name="W3",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b3 = tf.compat.v1.get_variable(name="b3",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W4 = tf.compat.v1.get_variable(name="W4",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b4 = tf.compat.v1.get_variable(name="b4",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W5 = tf.compat.v1.get_variable(name="W5",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b5 = tf.compat.v1.get_variable(name="b5",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W6 = tf.compat.v1.get_variable(name="W6",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b6 = tf.compat.v1.get_variable(name="b6",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer()) 
        W7 = tf.compat.v1.get_variable(name="W7",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b7 = tf.compat.v1.get_variable(name="b7",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W8 = tf.compat.v1.get_variable(name="W8",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b8 = tf.compat.v1.get_variable(name="b8",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W9 = tf.compat.v1.get_variable(name="W9",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b9 = tf.compat.v1.get_variable(name="b9",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W10 = tf.compat.v1.get_variable(name="W10",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b10 = tf.compat.v1.get_variable(name="b10",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W11 = tf.compat.v1.get_variable(name="W11",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b11 = tf.compat.v1.get_variable(name="b11",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W12 = tf.compat.v1.get_variable(name="W12",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b12 = tf.compat.v1.get_variable(name="b12",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W13 = tf.compat.v1.get_variable(name="W13",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b13 = tf.compat.v1.get_variable(name="b13",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W14 = tf.compat.v1.get_variable(name="W14",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b14 = tf.compat.v1.get_variable(name="b14",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W15 = tf.compat.v1.get_variable(name="W15",shape=[n_neurons,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b15 = tf.compat.v1.get_variable(name="b15",shape=[n_neurons,1],initializer=tf.compat.v1.zeros_initializer())
        W16 = tf.compat.v1.get_variable(name="W16",shape=[1,n_neurons],initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=1))
        b16 = tf.compat.v1.get_variable(name="b16",shape=[1,1],initializer=tf.compat.v1.zeros_initializer()) 
        
        parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4,"W5":W5,"b5":b5,"W6":W6,"b6":b6,
                     "W7":W7,"b7":b7,"W8":W8,"b8":b8,"W9":W9,"b9":b9,"W10":W10,"b10":b10,"W11":W11,"b11":b11,
                     "W12":W12,"b12":b12,"W13":W13,"b13":b13,"W14":W14,"b14":b14,"W15":W15,"b15":b15,
                      "W16":W16,"b16":b16}
 
    return parameters

def forward_propogation_tuning(X, n_hidden, parameters):
    '''
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    Arguments
    ----------
    X -- input dataset placeholder, of shape (input size, number of examples)
    n_hidden -- number of hidden layers
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3", ...
                  the shapes are given in initialize_parameters_tuning
    Returns
    --------
    Zl -- the output of the last LINEAR unit
    '''
    if n_hidden==1:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']   
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Zl = tf.add(tf.matmul(W2, A1), b2)
    elif n_hidden==2:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2) 
        Zl = tf.add(tf.matmul(W3, A2), b3)
    elif n_hidden==3:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Zl = tf.add(tf.matmul(W4, A3), b4)
    elif n_hidden==4:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Zl = tf.add(tf.matmul(W5, A4), b5)
    elif n_hidden==5:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Zl = tf.add(tf.matmul(W6, A5), b6)
    elif n_hidden==6:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Zl = tf.add(tf.matmul(W7, A6), b7)
    elif n_hidden==7:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Zl = tf.add(tf.matmul(W8, A7), b8)
    elif n_hidden==8:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Zl = tf.add(tf.matmul(W9, A8), b9)
    elif n_hidden==9:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Zl = tf.add(tf.matmul(W10, A9), b10)
    elif n_hidden==10:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Zl = tf.add(tf.matmul(W11, A10), b11)
    elif n_hidden==11:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        W12 = parameters['W12']
        b12 = parameters['b12']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Z11 = tf.add(tf.matmul(W11, A10), b11)
        A11 = tf.nn.relu(Z11)
        Zl = tf.add(tf.matmul(W12, A11), b12)
    elif n_hidden==12:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        W12 = parameters['W12']
        b12 = parameters['b12']
        W13 = parameters['W13']
        b13 = parameters['b13']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Z11 = tf.add(tf.matmul(W11, A10), b11)
        A11 = tf.nn.relu(Z11)
        Z12 = tf.add(tf.matmul(W12, A11), b12)
        A12 = tf.nn.relu(Z12)
        Zl = tf.add(tf.matmul(W13, A12), b13)
    elif n_hidden==13:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        W12 = parameters['W12']
        b12 = parameters['b12']
        W13 = parameters['W13']
        b13 = parameters['b13']
        W14 = parameters['W14']
        b14 = parameters['b14']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Z11 = tf.add(tf.matmul(W11, A10), b11)
        A11 = tf.nn.relu(Z11)
        Z12 = tf.add(tf.matmul(W12, A11), b12)
        A12 = tf.nn.relu(Z12)
        Z13 = tf.add(tf.matmul(W13, A12), b13)
        A13 = tf.nn.relu(Z13)
        Zl = tf.add(tf.matmul(W14, A13), b14)
    elif n_hidden==14:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        W12 = parameters['W12']
        b12 = parameters['b12']
        W13 = parameters['W13']
        b13 = parameters['b13']
        W14 = parameters['W14']
        b14 = parameters['b14']
        W15 = parameters['W15']
        b15 = parameters['b15']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Z11 = tf.add(tf.matmul(W11, A10), b11)
        A11 = tf.nn.relu(Z11)
        Z12 = tf.add(tf.matmul(W12, A11), b12)
        A12 = tf.nn.relu(Z12)
        Z13 = tf.add(tf.matmul(W13, A12), b13)
        A13 = tf.nn.relu(Z13)
        Z14 = tf.add(tf.matmul(W14, A13), b14)
        A14 = tf.nn.relu(Z14)
        Zl = tf.add(tf.matmul(W15, A14), b15)
    elif n_hidden==15:
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2'] 
        W3 = parameters['W3']
        b3 = parameters['b3']
        W4 = parameters['W4']
        b4 = parameters['b4']
        W5 = parameters['W5']
        b5 = parameters['b5']
        W6 = parameters['W6']
        b6 = parameters['b6']
        W7 = parameters['W7']
        b7 = parameters['b7']
        W8 = parameters['W8']
        b8 = parameters['b8']
        W9 = parameters['W9']
        b9 = parameters['b9']
        W10 = parameters['W10']
        b10 = parameters['b10']
        W11 = parameters['W11']
        b11 = parameters['b11']
        W12 = parameters['W12']
        b12 = parameters['b12']
        W13 = parameters['W13']
        b13 = parameters['b13']
        W14 = parameters['W14']
        b14 = parameters['b14']
        W15 = parameters['W15']
        b15 = parameters['b15']
        W16 = parameters['W16']
        b16 = parameters['b16']
        Z1 = tf.add(tf.matmul(W1, X), b1)
        A1 = tf.nn.relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)  
        A2 = tf.nn.relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)
        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)
        A4 = tf.nn.relu(Z4)
        Z5 = tf.add(tf.matmul(W5, A4), b5)
        A5 = tf.nn.relu(Z5)
        Z6 = tf.add(tf.matmul(W6, A5), b6)
        A6 = tf.nn.relu(Z6)
        Z7 = tf.add(tf.matmul(W7, A6), b7)
        A7 = tf.nn.relu(Z7)
        Z8 = tf.add(tf.matmul(W8, A7), b8)
        A8 = tf.nn.relu(Z8)
        Z9 = tf.add(tf.matmul(W9, A8), b9)
        A9 = tf.nn.relu(Z9)
        Z10 = tf.add(tf.matmul(W10, A9), b10)
        A10 = tf.nn.relu(Z10)
        Z11 = tf.add(tf.matmul(W11, A10), b11)
        A11 = tf.nn.relu(Z11)
        Z12 = tf.add(tf.matmul(W12, A11), b12)
        A12 = tf.nn.relu(Z12)
        Z13 = tf.add(tf.matmul(W13, A12), b13)
        A13 = tf.nn.relu(Z13)
        Z14 = tf.add(tf.matmul(W14, A13), b14)
        A14 = tf.nn.relu(Z14)
        Z15 = tf.add(tf.matmul(W15, A14), b15)
        A15 = tf.nn.relu(Z15)
        Zl = tf.add(tf.matmul(W16, A15), b16)

    return Zl   

# kling gupta efficiency
def kge(actual, predct):
    """
    a custom loss function based on the Kling Gupta Efficiency
    formula: [1 - sqrt((r-1)**2 + ((stddev_sim/stddev_obs)-1)**2 + ((mean_sim/mean_obs) - 1)**2)]
    reference: Decomposition of the mean squared error and NSE performance criteria: 
               Implications for improving hydrological modelling. (2009). 
               Journal of Hydrology, 377(1–2), 80–91. 
               DOI: https://doi.org/10.1016/j.jhydrol.2009.08.003

    Parameters
    ----------
    actual : tensor
        ground truth data for the predictions to be compared against
    predct : tensor
        predicted data

    Returns
    -------
    (1-kge): scalar
        The loss function to be minimized

    """
    # >>> correlation
    acmean = tf.math.reduce_mean(actual)
    pdmean = tf.math.reduce_mean(predct)
    acmdev, pdmdev = actual - acmean, predct - pdmean
    cornum = tf.math.reduce_mean(tf.multiply(acmdev, pdmdev))        
    corden = tf.math.reduce_std(acmdev) * tf.math.reduce_std(pdmdev)
    corcof = cornum / corden
    cratio = (corcof - 1)**2
    
    # variability ratio
    actstd = tf.math.reduce_std(actual)
    prestd = tf.math.reduce_std(predct)
    stdrat = prestd / actstd
    vratio = (stdrat - 1)**2
    
    # bias ratio (Beta)
    menrat = pdmean / acmean
    bratio = (menrat - 1)**2
    
    kgeval = 1 - tf.math.sqrt(cratio + vratio + bratio)
    retn01 = 1 - kgeval
    
    return retn01
                      
def compute_cost(Z6,auxi,model_num):  
    '''
    Calculates the target function (cost function)
    Arguments
    ---------
    Z6: ANN predictions
    Y: Observation
    auxi: Auxillary data
    model_num
    Returns
    -------
    cost: Cost of the predictions
    '''
    if model_num==1: # Target observed LE
        #cost = tf.sqrt(tf.reduce_mean(input_tensor=(SW_model_LE(auxi,Z6) - auxi[0,:])**2))
        actual = auxi[0,:]  
        predict = SW_model_LE(auxi,Z6)
        cost = kge(actual, predict)
    if model_num==2:  # Target observed LE and TEA-based T estimates
        #cost = tf.sqrt(tf.reduce_mean(input_tensor=(SW_model_LE(auxi,Z6) - auxi[0,:])**2)) +\
        #tf.sqrt(tf.reduce_mean(input_tensor=(SW_model_T(auxi,Z6) - auxi[0,:]*auxi[5,:])**2))
        actual_1 = auxi[0,:] 
        predict_1 = SW_model_LE(auxi,Z6)
        actual_2 = auxi[0,:]*auxi[5,:] 
        predict_2 = SW_model_T(auxi,Z6)
        cost = (kge(actual_1, predict_1) + kge(actual_2, predict_2))/2
    if model_num==3:  # Target observed LE and uWUE-based T estimates
        #cost = tf.sqrt(tf.reduce_mean(input_tensor=(SW_model_LE(auxi,Z6) - auxi[0,:])**2)) +\
        #tf.sqrt(tf.reduce_mean(input_tensor=(SW_model_T(auxi,Z6) - auxi[0,:]*auxi[7,:])**2))
        actual_1 = auxi[0,:] 
        predict_1 = SW_model_LE(auxi,Z6)
        actual_2 = auxi[0,:]*auxi[7,:] 
        predict_2 = SW_model_T(auxi,Z6)
        cost = (kge(actual_1, predict_1) + kge(actual_2, predict_2))/2
    if model_num==4:  # Target observed LE and Liuyang-based T estimates
        #cost = tf.sqrt(tf.reduce_mean(input_tensor=(SW_model_LE(auxi,Z6) - auxi[0,:])**2)) +\
        #tf.sqrt(tf.reduce_mean(input_tensor=(SW_model_T(auxi,Z6) - auxi[0,:]*auxi[24,:])**2))
        actual_1 = auxi[0,:] 
        predict_1 = SW_model_LE(auxi,Z6)
        actual_2 = auxi[0,:]*auxi[24,:] 
        predict_2 = SW_model_T(auxi,Z6)
        cost = (kge(actual_1, predict_1) + kge(actual_2, predict_2))/2
    
    return cost

def model_train_tuning(X_train,Y_train,X_val,Y_val,X_test,Y_test,Auxi_train,Auxi_val,Auxi_test,learning_rate,
          num_epochs,minibatch_size,print_cost,n_hidden,n_neurons,model_num):
    '''
    Backward propogation and update the parameters (w, b)
    '''
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape     #(n_x:number of input variables, m：samples)
    n_y = Y_train.shape[0]      #n_y： number of output variables, 1
    (n_x_val,m_val) = X_val.shape     #(n_x_val: number of input variables, m_val：samples)
    n_y_val = Y_val.shape[0]      #n_y_val：number of output variables, 1
    costs = []
    costs_val = []
    # create placeholder for X,Y. dimensions:(n_x, n_y)
    X,Y = create_placeholder(n_x,n_y)
    n_auxi = Auxi_train.shape[0]
    auxi = tf.compat.v1.placeholder(dtype=tf.float32,shape=[n_auxi,None],name="auxi")  
    # initialize parameters in neural network
    parameters = initialize_parameters_tuning(n_x, n_hidden, n_neurons)
 
    # forward propogation
    Zl = forward_propogation_tuning(X,n_hidden,parameters)
 
    # calculate the target function
    cost = compute_cost(Zl,auxi,model_num)
 
    # backward propogation
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
 
    # initialize all the parameters
    init = tf.compat.v1.global_variables_initializer()    
    
    # create a session to run the tensorflow graph
    with tf.compat.v1.Session() as sess:
        # initialize all the parameters
        sess.run(init) 
        
        # do for each epoch:
        for epoch in range(num_epochs):
            print(epoch)
            epoch_cost = 0      # define a cost for recording each epoch
            
            # number of minibatch for train dataset
            num_minibatches = int(int(m)/minibatch_size)     
            seed = seed + 1
            
            # produce the group of minibatches
            minibatches = random_mini_batches(X_train,Y_train,Auxi_train,
                                              minibatch_size,seed,True)       

            # do for each minibatch 
            for minibatch in minibatches:
                # select each minibatch
                (minibatch_X,minibatch_Y,minibatch_auxi) = minibatch            
                    
                _,minibatch_cost = sess.run([optimizer,cost],
                                            feed_dict={X:minibatch_X,Y:minibatch_Y,auxi:minibatch_auxi})
                # print(minibatch_cost)
                # calculate the whole epoch_cost for each epoch                
                epoch_cost += minibatch_cost/num_minibatches           
                   
            # print cost, %i: Decimal integer placeholder, %f: float placeholder
            if print_cost == True and epoch % 1 == 0:
                print("Cost after epoch %i : %.15f" %(epoch,epoch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)                 
            
            # calculate the validation accuracy
            if model_num==1:
                #correct_prediction_val = (SW_model_LE(auxi,Zl) - Y)**2
                #accuracy_val = tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_val,"float")))
                actual = Y  
                predict = SW_model_LE(auxi,Zl)
                accuracy_val = kge(actual, predict)
            if model_num==2:
                #correct_prediction_val_1 = (SW_model_LE(auxi,Zl) - Y)**2
                #correct_prediction_val_2 = (SW_model_T(auxi,Zl) - Y*auxi[5,:])**2
                #accuracy_val = tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_val_1,"float"))) +\
                #tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_val_2,"float")))
                actual_1 = Y  
                predict_1 = SW_model_LE(auxi,Zl) 
                actual_2 = Y*auxi[5,:]  
                predict_2 = SW_model_T(auxi,Zl) 
                accuracy_val = (kge(actual_1, predict_1) + kge(actual_2, predict_2))/2
            if model_num==3:
                #correct_prediction_val_1 = (SW_model_LE(auxi,Zl) - Y)**2
                #correct_prediction_val_2 = (SW_model_T(auxi,Zl) - Y*auxi[7,:])**2
                #accuracy_val = tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_val_1,"float"))) +\
                #tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_val_2,"float")))
                actual_1 = Y  
                predict_1 = SW_model_LE(auxi,Zl) 
                actual_2 = Y*auxi[7,:]  
                predict_2 = SW_model_T(auxi,Zl) 
                accuracy_val = (kge(actual_1, predict_1) + kge(actual_2, predict_2))/2
            if model_num==4:
                #correct_prediction_val_1 = (SW_model_LE(auxi,Zl) - Y)**2
                #correct_prediction_val_2 = (SW_model_T(auxi,Zl) - Y*auxi[24,:])**2
                #accuracy_val = tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_val_1,"float"))) +\
                #tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_val_2,"float")))
                actual_1 = Y  
                predict_1 = SW_model_LE(auxi,Zl) 
                actual_2 = Y*auxi[24,:]  
                predict_2 = SW_model_T(auxi,Zl) 
                accuracy_val = (kge(actual_1, predict_1) + kge(actual_2, predict_2))/2
                
            # feed X_val, Y_val
            cost_val = accuracy_val.eval({X:X_val,Y:Y_val,auxi:Auxi_val})
            print("val Accuracy:", cost_val)
            print('-----------------------------')
            costs_val.append(cost_val)     
        
            if epoch > 10:
                if (costs_val[epoch] > costs_val[epoch-1]) &  (costs_val[epoch] > costs_val[epoch-2]) & (costs_val[epoch] > costs_val[epoch-3]) & (costs_val[epoch] > costs_val[epoch-4]) & (costs_val[epoch] > costs_val[epoch-5]) & (costs_val[epoch] > costs_val[epoch-6]) & (costs_val[epoch] > costs_val[epoch-7]) & (costs_val[epoch] > costs_val[epoch-8]):
                    break
                    
        # keep the parameters (W,b) into Variables
        parameters = sess.run(parameters)
        print("Parameters have been trained!")
 
        # calculate the final accuracy
        if model_num==1:
            #correct_prediction = (SW_model_LE(auxi,Zl) - Y)**2
            #accuracy = tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction,"float")))
            actual = Y  
            predict = SW_model_LE(auxi,Zl)
            accuracy = kge(actual, predict)
        if model_num==2:
            #correct_prediction_1 = (SW_model_LE(auxi,Zl) - Y)**2
            #correct_prediction_2 = (SW_model_T(auxi,Zl) - Y*auxi[5,:])**2
            #accuracy = tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_1,"float"))) +\
            #tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_2,"float")))
            actual_1 = Y  
            predict_1 = SW_model_LE(auxi,Zl) 
            actual_2 = Y*auxi[5,:]  
            predict_2 = SW_model_T(auxi,Zl) 
            accuracy = (kge(actual_1, predict_1) + kge(actual_2, predict_2))/2
        if model_num==3:
            #correct_prediction_1 = (SW_model_LE(auxi,Zl) - Y)**2
            #correct_prediction_2 = (SW_model_T(auxi,Zl) - Y*auxi[7,:])**2
            #accuracy = tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_1,"float"))) +\
            #tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_2,"float")))
            actual_1 = Y  
            predict_1 = SW_model_LE(auxi,Zl) 
            actual_2 = Y*auxi[7,:]  
            predict_2 = SW_model_T(auxi,Zl) 
            accuracy = (kge(actual_1, predict_1) + kge(actual_2, predict_2))/2
        if model_num==4:
            #correct_prediction_1 = (SW_model_LE(auxi,Zl) - Y)**2
            #correct_prediction_2 = (SW_model_T(auxi,Zl) - Y*auxi[24,:])**2
            #accuracy = tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_1,"float"))) +\
            #tf.sqrt(tf.reduce_mean(input_tensor=tf.cast(correct_prediction_2,"float")))  
            actual_1 = Y  
            predict_1 = SW_model_LE(auxi,Zl) 
            actual_2 = Y*auxi[24,:]  
            predict_2 = SW_model_T(auxi,Zl) 
            accuracy = (kge(actual_1, predict_1) + kge(actual_2, predict_2))/2
 
        # feed X_test, Y_test
        print("Train Accuracy:",accuracy.eval({X:X_train,Y:Y_train,auxi:Auxi_train}))
        print("Validation Accuracy:",accuracy.eval({X:X_val,Y:Y_val,auxi:Auxi_val}))
        print("Test Accuracy:",accuracy.eval({X:X_test,Y:Y_test,auxi:Auxi_test}))
        loss_train = accuracy.eval({X:X_train,Y:Y_train,auxi:Auxi_train})
        loss_val = accuracy.eval({X:X_val,Y:Y_val,auxi:Auxi_val})
        return parameters, costs, costs_val, epoch, loss_train, loss_val
