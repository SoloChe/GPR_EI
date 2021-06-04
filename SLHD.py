# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:32:19 2020

@author: cheyi
"""

'Qian, Peter ZG. "Sliced Latin hypercube designs." Journal of the American Statistical Association 107.497 (2012): 393-399.'

import numpy as np
import pylab as py
import scipy.stats
import pylab as py
import matplotlib.pyplot as plt

def LHD(N, K):
    P = np.zeros([N,K])
    for i in range(K):
        P[:,i] = np.random.permutation(N)     
    Ksi = np.random.uniform(size = [N, K])
    p = (P + Ksi)/N
    X = scipy.stats.uniform.ppf(p)
    return X

def SLHD(m, t, q):
    A = []
    H = []
    for i in range(q):
        Hi = np.array(list(range(1, m*t+1))).reshape([m, t])
        np.apply_along_axis(lambda x: np.random.shuffle(x), 1, Hi)
        np.apply_along_axis(lambda x: np.random.shuffle(x), 0, Hi)
        H.append(Hi)
    for c in range(t):
        Ac = np.zeros([m, q])
        for j in range(q):
            Ac[:,j] = [(i - np.random.random())/(m*t) for i in H[j][:,c]]
        A.append(Ac)
    return A

if __name__ == '__main__':
    
    # 3 slice 2-d with 10 samples on each slice 
    A = SLHD(10,3,2)
    fix, ax = plt.subplots(1)
    py.plot(A[0][:,0], A[0][:,1], 'ro')
    py.plot(A[1][:,0], A[1][:,1], 'b^')
    py.plot(A[2][:,0], A[2][:,1], 'g*')
    major_ticks = np.arange(0, 1, 1/30)
    py.xticks(major_ticks)
    py.yticks(major_ticks)
    py.grid(color = 'k')
    py.xlim([0,1])
    py.ylim([0,1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    