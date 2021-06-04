# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:57:07 2020

@author: cheyi
"""

import numpy as np
import pylab as py
import numexpr as ne
from numpy.linalg import cholesky, det
from numpy.linalg import inv
from scipy.linalg import lstsq 
from scipy.optimize import minimize
import sobol_seq
import warnings
warnings.filterwarnings("ignore")


#%% GP
class GP:
    def __init__(self, basis_num = 2, stochastic = False, cho = True, add_noise = False, fixed_noise = 1e-8):
        
        self.basis_num = basis_num
        self.stochastic = stochastic
        self.cho = cho
        self.add_noise = add_noise
        self.fixed_noise = fixed_noise
        
    def K(self, para):
        X = self.X_train*para[1:-1]
        X_norm = np.sum(X**2, axis = -1)
        k = ne.evaluate('sigma_f2 * exp(-(A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : X_norm[None,:],
                'C' : np.dot(X, X.T),
                'sigma_f2':para[0]
        })
        if self.stochastic:
            k = k + np.diag(self.ep)
        elif self.add_noise:
            k = k + para[-1] * np.eye(self.X_train.shape[0])
        else: # nugget
            k = k + self.fixed_noise * np.eye(self.X_train.shape[0])
        return k

    def Ks(self, para):
        X1 = self.X_train*para[1:-1]
        X2 = self.X_test*para[1:-1]
        X1_norm = np.sum(X1**2, axis = -1)
        X2_norm = np.sum(X2**2, axis = -1)
        ks = ne.evaluate('sigma_f2 * exp(-(A + B - 2 * C))', {
                'A' : X1_norm[:,None],
                'B' : X2_norm[None,:],
                'C' : np.dot(X1, X2.T),
                'sigma_f2':para[0]
        })
        return ks
    
    def Kss(self, para):
        X = self.X_test*para[1:-1]
        X_norm = np.sum(X**2, axis = -1)
        kss = ne.evaluate('sigma_f2 * exp(-(A + B - 2 * C))', {
                'A' : X_norm[:,None],
                'B' : X_norm[None,:],
                'C' : np.dot(X, X.T),
                'sigma_f2':para[0]
        })
        return kss
    
    def H(self, X):
        if X.shape[1] > 1:
            h = np.hstack([X**i for i in range(self.basis_num+1)])[:,1:]
        else:
            h = np.hstack([X**i for i in range(self.basis_num+1)])
        return h.T

    def beta_hat(self, para):
        h = self.H(self.X_train)
        k = self.K(para)
        try:
            L = cholesky(k)
            self.k_inv = inv(L).T.dot(inv(L))
            betahat = inv((h.dot(self.k_inv).dot(h.T))).dot(h.dot(self.k_inv).dot(self.y_train))
        except:
            self.nugget = np.diag(np.random.normal(0, 1, size = h.shape[0]))
            betahat = inv((h.dot(self.k_inv).dot(h.T)) + self.nugget).dot(h.dot(self.k_inv).dot(self.y_train))
            self.ck_problem = k
            self.para_problem = para
            print('betahat problem!')
            raise
        return betahat
        
    def neg_log_likelihood_basis(self):
        def non_cholesky(para):
            k = self.K(para)
            h = self.H(self.X_train)
            beta = self.beta_hat(para)
            L1 =  0.5 * np.log(det(k)) + \
                  0.5 * (self.y_train - h.T.dot(beta)).T.dot(inv(k).dot(self.y_train - h.T.dot(beta))) + \
                  0.5 * len(self.X_train) * np.log(2*np.pi) 
            return L1
        def with_cholesky(para):
            k = self.K(para)
            h = self.H(self.X_train)
            beta = self.beta_hat(para)
            
            L = cholesky(k)
            
            try:
                Z = lstsq(L, self.y_train - h.T.dot(beta), cond = None, lapack_driver = 'gelsy')[0]
            except:
                self.L_problem = L
                self.k_problem = k
                self.h_problem = h
                self.beta_problem = beta
                raise
                
            L2 = np.sum(np.log(np.diagonal(L))) + \
                         0.5 * Z.T.dot(Z) + \
                         0.5 * len(self.X_train) * np.log(2*np.pi)
            return L2

        if self.cho:
            return with_cholesky
        else:
            return non_cholesky
    
    def fit(self, X_train, y_train, ep = None, rangeX = [], ifscale = True):
        # scale
        self.ifscale = ifscale
        if self.ifscale:
            if len(rangeX) == 0:
                self.minX = np.min(X_train, axis = 0)
                self.maxX = np.max(X_train, axis = 0)
            else:
                self.minX = rangeX[0]
                self.maxX = rangeX[1]

            self.X_train = (X_train - self.minX)/(self.maxX - self.minX)
            self.y_train = y_train.ravel()
        else:
            self.X_train = X_train
            self.y_train = y_train.ravel()
        
        # flattern if not
        if self.stochastic:
            self.ep = ep.ravel()
            
        _ = 1e8
        for i in range(3):
            # print(f'======== In optimizer {i+1} ============')
            self.para_ = np.random.uniform(1e-2, 1e1, size = self.X_train.shape[1] + 2)
            res = minimize(self.neg_log_likelihood_basis(), 
                           self.para_, 
                           bounds = [(1e-5, None) for i in range(self.X_train.shape[1] + 2)],
                           method = 'L-BFGS-B', options = {'maxiter':5000})
            # print(res.message)
            # print(f'optimize success: {res.success}')
            if res.fun < _:
                    # print('para updated')
                    _ = res.fun
                    self.para = res.x
#        print(f'condition number of K = {np.linalg.cond(self.K(self.para))}')
            
    def predict(self, X_test, imse = False, ifcov = False):
        # scale
        if self.ifscale:
            self.X_test = (X_test - self.minX)/(self.maxX - self.minX)
        else:
            self.X_test = X_test
            
        k = self.K(self.para)
        ks = self.Ks(self.para)
        kss = self.Kss(self.para) 
        h = self.H(self.X_train)
        hs = self.H(self.X_test)
        beta = self.beta_hat(self.para)
        
        if self.cho:
            L = cholesky(k)
            k_inv = inv(L).T.dot(inv(L))
            R = hs - h.dot(k_inv).dot(ks)
            alpha = lstsq(L.T, lstsq(L, self.y_train - h.T.dot(beta), cond = None)[0], cond = None, lapack_driver = 'gelsy')[0]
            v = lstsq(L, ks, cond = None, lapack_driver = 'gelsy')[0]
            mu = hs.T.dot(beta) + ks.T.dot(alpha)
            try:
                cov = kss - v.T.dot(v) + R.T.dot(inv(h.dot(k_inv).dot(h.T))).dot(R)
            except:
                cov = kss - v.T.dot(v) + R.T.dot(inv(h.dot(k_inv).dot(h.T) + self.nugget)).dot(R)
                print('cov problem!')
        else: 
            k_inv = inv(k)
            R = hs - h.dot(k_inv).dot(ks)
            mu = ks.T.dot(k_inv).dot(self.y_train) + R.T.dot(beta)
            try:
                cov = kss - ks.T.dot(k_inv).dot(ks) + R.T.dot(inv(h.dot(k_inv).dot(h.T))).dot(R)
            except:
                cov = kss - ks.T.dot(k_inv).dot(ks) + R.T.dot(inv(h.dot(k_inv).dot(h.T) + self.nugget)).dot(R)
        if imse:    
            return np.diag(cov)
        elif ifcov:
            return mu.ravel(), cov
        else:
            return mu.ravel(), np.diag(cov)
   
    def IMSE(self, a, b, x_new = []):
        if len(x_new) != 0:
            x_new = (x_new - self.minX)/(self.maxX - self.minX)
            x_new = x_new.reshape([1,-1])
            self.X_train = np.vstack([self.X_train, x_new])
        ss = sobol_seq.i4_sobol_generate(self.X_train.shape[1], 2000)
        ss = (b - a)*ss + a
        imse = np.mean(self.predict(ss, imse = True)* np.prod(b-a))
        return imse
    
    def IMSE_local(self, a, b, sig_tol, T, x_new = []):
        if len(x_new) != 0:
            x_new = (x_new - self.minX)/(self.maxX - self.minX)
            x_new = x_new.reshape([1,-1])
            self.X_train = np.vstack([self.X_train, x_new])
        ss = sobol_seq.i4_sobol_generate(self.X_train.shape[1], 2000)
        ss = (b - a)*ss + a
        mu, cov = self.predict(ss)
        
        inner = cov * 1/np.sqrt(2*np.pi*(sig_tol**2 + cov)) * np.exp(-0.5*((mu-T)**2/(sig_tol**2 + cov)))
        imse_local = np.mean(inner*np.prod(b-a))
        return imse_local

    #%% test 
def test1():
    X_train = np.array([-5, -4, -2, -1, 1, 2, 3, 5]).reshape(-1, 1)
    y_train = np.sin(X_train)
    X_test = np.linspace(-5, 5, 50).reshape(-1,1) 
    y_test = np.sin(X_test)
   
    gp = GP(basis_num = 2, cho = True, stochastic = False, add_noise = False)
    gp.fit(X_train, y_train)
    yp, covp = gp.predict(X_test)
    std = np.sqrt(covp)

    py.figure()
    py.plot(X_test, yp, 'r')
    py.plot(X_test, y_test, 'k')
    py.plot(X_train, y_train, 'ro')
    py.fill_between(X_test.flat, yp.flat - 2*std, yp.flat + 2*std,
                   color = "#dddddd")

def test2():
    X1 = np.linspace(-1,1,100)
    X2 = np.linspace(-1,1,100)
    x1, x2 = np.meshgrid(X1, X2)
    X = np.hstack([x1.reshape(-1,1), x2.reshape(-1,1)])
    y = x1**2 + x2**2

    X_train = X[::20]
    y_train = y.reshape((-1,1))[::20]

    gp = GP(basis_num = 3, cho = True, stochastic = False, add_noise = False)
    gp.fit(X_train, y_train)
    yp = gp.predict(X)
    yp = yp[0].reshape((100, 100))
    
    py.figure()
    py.contour(X1, X2, y, colors = 'k', linewidths = 3)
    py.contour(X1, X2, yp, colors = 'r', linewidths = 3)
    
if __name__ == '__main__':  
    test1()
    test2()
    
    
    
    
    
    
    
    
    
    