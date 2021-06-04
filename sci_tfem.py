# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 00:22:39 2018

@author: cheyi
"""

import numpy as np
from scipy.integrate import quad as integrate
import pylab as py
# from numba import jit
from tqdm import tqdm

def phi(f, i, tj, w = False):
    if f == 0:
        phi1 = lambda x: 0
        phi2 = lambda x: 0
        phi3 = lambda x: 0
    else:
        if w:
            phi1 = lambda x: f*(1 - 23*(x/tj)**2 + 66*(x/tj)**3 -68*(x/tj)**4 + 24*(x/tj)**5) * ((x/tj) - 1/2)
            phi2 = lambda x: f*(16*(x/tj)**2 - 32*(x/tj)**3 + 16*(x/tj)**4) * ((x/tj) - 1/2)
            phi3 = lambda x: f*(7*(x/tj)**2 - 34*(x/tj)**3 + 52*(x/tj)**4 - 24*(x/tj)**5) * ((x/tj) - 1/2)
        else:
            phi1 = lambda x: f*(1 - 23*(x/tj)**2 + 66*(x/tj)**3 -68*(x/tj)**4 + 24*(x/tj)**5)
            phi2 = lambda x: f*(16*(x/tj)**2 - 32*(x/tj)**3 + 16*(x/tj)**4)
            phi3 = lambda x: f*(7*(x/tj)**2 - 34*(x/tj)**3 + 52*(x/tj)**4 - 24*(x/tj)**5)
    
    dic = {0:phi1, 1:phi2, 2:phi3}
    return dic[i]

def dphi(f, i, tj, w = False):
    if f == 0:
        dphi1 = lambda x: 0
        dphi2 = lambda x: 0
        dphi3 = lambda x: 0
    else:
        if w:
            dphi1 = lambda x: f*(-46*(x/tj)/tj + 66*3*(x/tj)**2/tj - 68*4*(x/tj)**3/tj + 24*5*(x/tj)**4/tj) * ((x/tj) - 1/2)
            dphi2 = lambda x: f*(32*(x/tj)/tj - 32*3*(x/tj)**2/tj + 16*4*(x/tj)**3/tj) * ((x/tj) - 1/2)
            dphi3 = lambda x: f*(14*(x/tj)/tj - 34*3*(x/tj)**2/tj + 52*4*(x/tj)**3/tj - 24*5*(x/tj)**4/tj) * ((x/tj) - 1/2)
        else:
            dphi1 = lambda x: f*(-46*(x/tj)/tj + 66*3*(x/tj)**2/tj - 68*4*(x/tj)**3/tj + 24*5*(x/tj)**4/tj)
            dphi2 = lambda x: f*(32*(x/tj)/tj - 32*3*(x/tj)**2/tj + 16*4*(x/tj)**3/tj)
            dphi3 = lambda x: f*(14*(x/tj)/tj - 34*3*(x/tj)**2/tj + 52*4*(x/tj)**3/tj - 24*5*(x/tj)**4/tj)
    
    dic = {0:dphi1, 1:dphi2, 2:dphi3}
    return dic[i]




# @jit
def matfun(m, n, a, b, ksi, w, wn, k):
    ms = 5.3
    tj = 1/a/m * 60
    B = np.array([[0, 0],
                  [k*w*b/ms, 0]])
    A = np.array([[0, 1],
                  [-(k*w*b/ms + wn**2), -2*ksi*wn]])
    II = np.eye(n)
    N = np.zeros([2,2])
    P = np.zeros([2,2])
    NN = np.zeros([4,6])
    PP = np.zeros([4,6])
    
    for i in range(2):
        for j in range(2):
            ff1 = dphi(II[i,j], 0, tj) 
            ff2 = phi(A[i,j], 0, tj)
            N[i,j] = integrate(ff1, 0, tj)[0] - integrate(ff2, 0, tj)[0]
            gg = phi(B[i,j], 0, tj)
            P[i,j] = integrate(gg, 0, tj)[0]
    NN[0:2,0:2] = N
    PP[0:2,0:2] = P
    
    for i in range(2):
        for j in range(2):
            ff1 = dphi(1, 1, tj) 
            ff2 = phi(A[i,j], 1, tj)
            N[i,j] = integrate(ff1, 0, tj)[0] - integrate(ff2, 0, tj)[0]
            gg = phi(B[i,j], 1, tj)
            P[i,j] = integrate(gg, 0, tj)[0]
    NN[0:2,2:4] = N
    PP[0:2,2:4] = P
    
    for i in range(2):
        for j in range(2):
            ff1 = dphi(II[i,j], 2, tj) 
            ff2 = phi(A[i,j], 2, tj)
            N[i,j] = integrate(ff1, 0, tj)[0] - integrate(ff2, 0, tj)[0]
            gg = phi(B[i,j], 2, tj)
            P[i,j] = integrate(gg, 0, tj)[0]
    NN[0:2,4:] = N
    PP[0:2,4:] = P
    
    for i in range(2):
        for j in range(2):
            ff1 = dphi(II[i,j], 0, tj, w = True) 
            ff2 = phi(A[i,j], 0, tj, w = True)
            N[i,j] = integrate(ff1, 0, tj)[0] - integrate(ff2, 0, tj)[0]
            gg = phi(B[i,j], 0, tj, w = True)
            P[i,j] = integrate(gg, 0, tj)[0]
    NN[2:4,0:2] = N
    PP[2:4,0:2] = P
    
    for i in range(2):
        for j in range(2):
            ff1 = dphi(II[i,j], 1, tj, w = True) 
            ff2 = phi(A[i,j], 1, tj, w = True)
            N[i,j] = integrate(ff1, 0, tj)[0] - integrate(ff2, 0, tj)[0]
            gg = phi(B[i,j], 1, tj, w = True)
            P[i,j] = integrate(gg, 0, tj)[0]
    NN[2:4,2:4] = N
    PP[2:4,2:4] = P
    
    for i in range(2):
        for j in range(2):
            ff1 = dphi(II[i,j], 2, tj, w = True) 
            ff2 = phi(A[i,j], 2, tj, w = True)
            N[i,j] = integrate(ff1, 0, tj)[0] - integrate(ff2, 0, tj)[0]
            gg = phi(B[i,j], 2, tj, w = True)
            P[i,j] = integrate(gg, 0, tj)[0]
    NN[2:4,4:] = N
    PP[2:4,4:] = P
    return NN, PP
    
# @jit
def tfem(a, b, ksi, w, wn, k):
    delay = 1/a*60  
    m = 40 + np.int(np.ceil(150*delay))
    n = 2
    sz = m*4+2
    H = np.zeros([sz,sz])
    G = np.zeros([sz,sz])
    H[0:2,0:2] = np.eye(n)
    G[0:2,sz-2:] = np.eye(n)
    
    NN, PP = matfun(m, n, a, b, ksi, w, wn, k) 
    for i in range(1,m+1):
        H[2+(i-1)*4:6+(i-1)*4, (i-1)*4:6+(i-1)*4] = NN;
        G[2+(i-1)*4:6+(i-1)*4, (i-1)*4:6+(i-1)*4] = PP;
   
    
    V = np.linalg.solve(H,G)
    lam = np.linalg.eigvals(V)
    rho = np.max(np.real(np.log(lam)))/delay
    return rho

if __name__ == '__main__':
    
    py.rcParams['xtick.labelsize'] = 12
    py.rcParams['ytick.labelsize'] = 12
    py.rcParams['font.size'] = 15
    py.rcParams["font.weight"] = 'bold'
    py.rcParams["axes.labelweight"] = 'bold'
    py.rcParams['axes.linewidth'] = 2

    
    # parameters in the simulation
    ksi = 0.02
    k = 200*10**9
    w = 50*10**(-3)
    wn = 300*2*np.pi
    
    # resolution
    n = 100
    
    # range of the simulation
    alist = np.linspace(500, 4000, n)
    blist = np.linspace(0, 26e-5, n)
   
    rho = np.zeros([n,n])
    #simulation start
    for i in tqdm(range(len(alist))):
        for j in range(len(blist)):
            rho[i,j] = tfem(alist[i], blist[j], ksi, w, wn, k)      
    
    
#    rho = np.load('MC_mean_nor_2d.npy')
    py.figure() 
    a = py.contourf(alist, blist, rho.T, 0, colors = ['w', 'grey'])
    b = py.contour(a, levels = a.levels, colors = 'k', linewidths = 3)
#    py.plot(2.5e3, 1e-4, 'go')
#    py.plot(2.5e3, 2e-4, 'bo')
#    py.plot(2.5e3, 1.485e-4, 'yo')
#    py.xlabel(r'$\Omega$' + r' $(round/min)$')
#    py.ylabel(r'$b$' + r' $(m)$')
    py.xlabel(r'$x_2$' + r' $(round/min)$')
    py.ylabel(r'$x_1$' + r' $(m)$')
    py.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
    py.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,0))
    py.subplots_adjust(left = 0.14, right=0.9, top = 0.88, bottom = 0.13)

      
    
    
    
    
    