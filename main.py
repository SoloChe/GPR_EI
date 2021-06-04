# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 22:51:43 2021

@author: cheyi
"""

import numpy as np
import pylab as py
import scipy
from pyDOE import lhs
from GP import GP
from sci_tfem import tfem
from SLHD import SLHD
import warnings
warnings.filterwarnings("ignore")


def EI(mu, sigma, ep, rho):
    f_p = np.min(np.abs(ep - rho))
    f = np.abs(ep - mu)
    Phi = scipy.stats.norm.cdf((f_p - f)/sigma) 
    phi = scipy.stats.norm.pdf((f_p - f)/sigma) 
    ei = sigma*phi + (f_p - f)*Phi 
    return ei

def get_X_train_ini(X1, X2, X, rho, N):
    S = lhs(2, samples = N, criterion = 'maximin')
    # map
    a_1 = (X1[-1] - X1[0]) * S[:,0].reshape([-1,1]) + X1[0]
    b_1 = (X2[-1] - X2[0]) * S[:,1].reshape([-1,1]) + X2[0]
    X_train_ini = np.hstack([a_1, b_1])
    # find the closest in  grid
    X_arg = np.array([np.argmin(np.linalg.norm(i - X, axis = 1, keepdims = True)) for i in X_train_ini])
    X_train_ini = X[X_arg] 
    y_train_ini = rho[X_arg]
    return X_train_ini, y_train_ini

def get_grid(X1, X2):
    X1, X2 = np.meshgrid(X1, X2)
    X1 = X1.reshape([-1,1])
    X2 = X2.reshape([-1,1])
    X = np.hstack([X1, X2]) # 10000 x 1
    return X
    
    
if __name__ == '__main__':
    
    # plot setting
    py.rcParams['xtick.labelsize'] = 12
    py.rcParams['ytick.labelsize'] = 12
    py.rcParams['font.size'] = 12
    py.rcParams["font.weight"] = 'bold'
    py.rcParams["axes.labelweight"] = 'bold'
    py.rcParams['axes.linewidth'] = 2
    
    # parameters
    ksi = 0.02
    k = 200e9
    w = 50e-3
    wn = 300*2*np.pi
    # resolution
    n = 100
    # range
    X_min = np.array([500, 0])
    X_max = np.array([4000, 26e-5])
    # design variable 
    X1 = np.linspace(X_min[0], X_max[0], n)
    X2 = np.linspace(X_min[1], X_max[1], n)
    # design variable grid 100 x 100
    X = get_grid(X1, X2)
    
    # real rho 100 x 100
    rho_ = np.load('real_rho.npy').T
    rho = rho_.reshape((10000,))
    
    # initial training
    X_train_ini, y_train_ini = get_X_train_ini(X1, X2, X, rho, N = 100)
    y_train_ini_color = ['g' if i <= 0 else 'b' for i in y_train_ini]
    # fit
    gp = GP(basis_num = 4, cho = True, stochastic = False, add_noise = True)
    gp.fit(X_train_ini, y_train_ini, rangeX = [X_min, X_max])
    y_pred_ini, y_var = gp.predict(X)
    
    # true rho plot
    py.figure() 
    py.contour(X1, X2, rho_, levels = [0], colors = 'k', linewidths = 3)
    py.contour(X1, X2, y_pred_ini.reshape((100,100)), levels = [0], colors = 'r', linewidths = 3)
    py.scatter(X_train_ini[:,0], X_train_ini[:,1], c = y_train_ini_color)
    py.xlabel(r'$x_2$' + r' $(round/min)$')
    py.ylabel(r'$x_1$' + r' $(m)$')
    py.xlim([500, 4050])
    py.ylim([0.5e-4, 26e-5])
    py.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    py.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,0))
    
    
    #%% Sequential Design
    target = 0 # find boundary 0
    X_train_ = X_train_ini.copy()
    y_train_ = y_train_ini.copy()
    
    slhd = SLHD(5000, 15, 2) # 5000 points in each slice; 20 slices; 2-d
    
    X_added = []
    y_added = []
    for i in range(len(slhd)):
        print(f'At iteration {i+1}')
        slhd_i = (X_max-X_min) * np.array(slhd[i]) + X_min
        X_arg = np.array([np.argmin( np.linalg.norm(i - X, axis = 1, keepdims = True)) for i in slhd_i])
        X_ = np.unique(X[X_arg], axis = 0)
        for _ in range(3):
        
            y_, var_ = gp.predict(X_)
            std_ = np.sqrt(var_)
            I_ = np.array([EI(jj[0], jj[1], target, y_train_) for jj in zip(y_, std_)])
            
            ind_ = I_.flatten().argsort()[-10:] # select 10 largest
                            
            X_train_sq = X_[ind_]
            y_train_sq = np.array([tfem(i[0], i[1], ksi, w, wn, k) for i in X_train_sq])
            y_train_sq_c = ['g' if i <= 0 else 'b' for i in y_train_sq]
            
            
            X_train_ = np.vstack([X_train_, X_train_sq])
            y_train_ = np.concatenate([y_train_, y_train_sq])
            
            X_added.append(X_train_sq)
            y_added += y_train_sq_c
            
            gp.fit(X_train_, y_train_, rangeX = [X_min, X_max])
            y_pred_, _ = gp.predict(X)
        
        X_added_ = np.vstack(X_added)    
        
        py.figure() 
        py.contour(X1, X2, rho_, levels = [0], colors = 'k', linewidths = 3)
        py.contour(X1, X2, y_pred_.reshape((100,100)), levels = [0], colors = 'r', linewidths = 3)
        py.scatter(X_added_[:,0], X_added_[:,1], c = y_added)
        py.xlabel(r'$x_2$' + r' $(round/min)$')
        py.ylabel(r'$x_1$' + r' $(m)$')
        py.xlim([500, 4050])
        py.ylim([0.5e-4, 26e-5])
        py.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
        py.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,0))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    