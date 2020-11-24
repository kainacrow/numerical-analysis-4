#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 09:08:04 2018

@author: Kaina
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi
sin = np.sin
I = np.identity
N2 = 128
N1 = 2*N2
delta = 0.01 # big delta
delta_t = 0.01 # little delta
M = 2/delta_t
t = 0.5 # time 

def makeD(N, delta):
    D_matrix = []
    for row in range(N):
        row_array = []
        for col in range(N):
            if(row == col):
                row_array.append(-2)
            elif(row==col+1 or col==row+1):
                row_array.append(1)
            elif(row==N-1 and col==0 or row==0 and col==N-1):
                row_array.append(1)
            else:
                row_array.append(0)
        D_matrix.append(row_array)
    return (1.0/delta**2)*np.asarray(D_matrix)

def makeDsym(N, delta):
    D_matrix = []
    for row in range(N):
        row_array = []
        for col in range(N):
            if(row == 0 and col == 0 or row == N-1 and col == N-1):
                row_array.append(-1)
            elif(row==col+1 or col==row+1):
                row_array.append(1)
            elif(row==N-1 and col==0 or row==0 and col==N-1):
                row_array.append(1)
            elif(row == col):
                row_array.append(-2)
            else:
                row_array.append(0)
        D_matrix.append(row_array)
    return (1.0/delta**2)*np.asarray(D_matrix)

def makeDnon(N, delta):
    D_matrix = []
    for row in range(N):
        row_array = []
        for col in range(N):
            if(row == 0 and col == 1 or row == N-1 and col == N-2):
                row_array.append(2)
            elif(row==col+1 or col==row+1):
                row_array.append(1)
            elif(row==N-1 and col==0 or row==0 and col==N-1):
                row_array.append(1)
            elif(row == col):
                row_array.append(-2)
            else:
                row_array.append(0)
        D_matrix.append(row_array)
    return (1.0/delta**2)*np.asarray(D_matrix)

c_t0 = lambda x: x*sin(pi*x)

def c0(N, delta):
    c0_vector = []
    for i in range(N):
        x = i*delta
        val = x*sin(pi*x)
        c0_vector.append(val)
    return c0_vector

c1 = c0(N1, delta)
c2 = c0(N2, delta)

def BE(N, LDelta, D, c):
    clist = [c]
    A = inv(I(N)-LDelta*D)
    for i in range(int(t/delta_t)):
        clist.append(np.dot(A,clist[-1]))
    return clist

def Y(N):
    ymatrix = []
    for i in range(N):
        r = []
        for j in range(int(t/delta_t)+1):
            y = j*delta_t
            r.append(y)
        ymatrix.append(r)
    return ymatrix

def X(N):
    xmatrix = []
    for i in range(N):
        r = []
        for j in range(int(t/delta_t)+1):
            x = i*delta_t
            r.append(x)
        x = delta_t
        xmatrix.append(r)
    return xmatrix

D1 = makeD(N1, delta)
D2 = makeD(N2, delta)

Dsym1 = makeDsym(N1, delta)
Dsym2 = makeDsym(N2, delta)
Dnon1 = makeDnon(N1, delta)
Dnon2 = makeDnon(N2, delta)

X1 = X(N1)
X2 = X(N2)
Y1 = Y(N1)
Y2 = Y(N2)

BE2 = BE(N2, delta_t, D2, c2) ## bigger set of Cs
BE2 = np.transpose(BE2)
####
#### N1 is true ###
BEnon1 = BE(N1, delta_t, Dnon1, c1)
BEnon1 = np.transpose(BEnon1)
BEnon2 = BE(N2, delta_t, Dnon2, c2)
BEnon2 = np.transpose(BEnon2)
BEsym1 = BE(N1, delta_t, Dsym1, c1)
BEsym1 = np.transpose(BEsym1)
BEsym2 = BE(N2, delta_t, Dsym2, c2)
BEsym2 = np.transpose(BEsym2)
#print BE2
# =============================================================================
# BEsym1 = BE(N, delta_t, Dsym)
# BEsym1 = np.transpose(BEsym1)
# BEnon1 = BE(N, delta_t, Dnon)
# BEnon1 = np.transpose(BEnon1)
# =============================================================================

true_Cs = BE(N1, delta_t, D1, c1)
true_Cs = np.transpose(true_Cs)
#print true_Cs
#true_Cs = [] ## 8193 * 2
current_Cs = [] ## N+1 *2
subset_trueCs = []
for i in range(N1+1): ## need another for loop? for arrays within the main array
    if i % (float(N1+1)/float(N2)) == 0:
        print i
        #print true_Cs[i]
        subset_trueCs.append(true_Cs[i])
print len(subset_trueCs[0])
print len(BE2)
            
def maxNormError(current, true):
    e = 0.0
    for i in range(len(true)):
        print current[i], true[i]
        if(np.any((np.abs(current[i] - true[i]) > e))):
            e = np.abs(current[i]-true[i])
    return e

error = maxNormError(BE2, subset_trueCs)

fig = plt.figure()
plt.plot(error[1:])
plt.xscale('log')
plt.yscale('log')
plt.show()
# =============================================================================
# for i in range(len(subset_trueCs)):
#     print BE2[i]
#     print subset_trueCs[i]
# =============================================================================



# ========================== BE NORMAL d = 0.01 ===============================
# =============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X2, Y2, np.array(BE2), rstride=4, cstride=int(M/50), color="white", shade=False, edgecolor="green")
# ax.view_init(azim=0)
# plt.title("BE NORMAL d = 0.01")
# plt.show()
# =============================================================================
