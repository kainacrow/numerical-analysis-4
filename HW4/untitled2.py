#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:34:20 2018

@author: Kaina
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D

pi = np.pi
sin = np.sin
I = np.identity

delta = 0.01
delta_t = 0.00001
delta_t2 = 2*delta_t
N = 100
D = (1/delta**2)
M = 2/delta_t
M2 = M/2

def makeD():
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
    #print (1.0/delta**2)*np.asarray(D_matrix)
    return (1.0/delta**2)*np.asarray(D_matrix)

D = makeD()
#print D

c_t0 = lambda x: x*sin(pi*x)


def c0():
    c0_vector = []
    for i in range(N):
        x = i*delta
        val = x*sin(pi*x)
        c0_vector.append(val)
    #print c0_vector
    return c0_vector
c0 = c0()
#print c0

#print (I(N)+delta_t*D)
def FE():
    clist = [c0]
    for i in range(int(2/delta_t)):
        clist.append(np.dot(I(N)+delta_t*D,clist[-1]))
        ##new_c_num = np.dot(np.array(I(N)+delta_t*D), np.array(clist[-1]))
        #clist.append(new_c_num)
    #print len(clist[0])
    return list(clist)
FE = FE()
FE = np.transpose(FE)

def Y():
    ymatrix = []
    for i in range(N):
        r = []
        for j in range(int(2/delta_t)+1):
            y = j*delta_t
            r.append(y)
        ymatrix.append(r)
    #print len(ymatrix[0])
    return ymatrix

def X():
    xmatrix = []
    for i in range(N):
        r = []
        for j in range(int(2/delta_t)+1):
            x = i*delta_t
            r.append(x)
        x = delta_t
        xmatrix.append(r)
    #print len(xmatrix[0])
    return xmatrix

X = X()
Y = Y() 
Z = FE

#fig = plt.figure(1,figsize=(20, 15))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, np.array(Z), rstride=N/50, cstride=int(M/100), color='blue')
ax.plot_wireframe(X, Y, np.array(FE), rstride=N/50, cstride=int(M/1000), color='pink')

plt.show()