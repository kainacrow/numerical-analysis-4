#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 10:51:37 2018

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
delta_t = 0.01
delta_t2 = 2*delta_t
#N = 100
D = (1/delta**2)
M = 2/delta_t
M2 = M/2

Nlist = [16, 32, 64]##, 128, 256, 512, 1024, 2048]
Nval = 128##4096

# =============================================================================
# domain = list(np.arange(0, 1, delta))
# domain.append(1)
# time = list(np.arange(0, 2, delta_t))
# time.append(2)
# =============================================================================

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
    #print (1.0/delta**2)*np.asarray(D_matrix)
    return (1.0/delta**2)*np.asarray(D_matrix)

D = makeD(Nval, delta)
for i in Nlist:
    Dn = makeD(Nval, delta)


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
    #print (1.0/delta**2)*np.asarray(D_matrix)
    return (1.0/delta**2)*np.asarray(D_matrix)

Dsym = makeDsym(Nval, delta)
for i in Nlist:
    Dn = makeDsym(i, delta)
#print Dsym

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
    #print (1.0/delta**2)*np.asarray(D_matrix)
    return (1.0/delta**2)*np.asarray(D_matrix)

Dnon = makeDnon(Nval, delta)
for i in Nlist:
    Dn = makeDnon(Nval, delta)
#print Dnon

c_t0 = lambda x: x*sin(pi*x)


def c0(N, delta):
    c0_vector = []
    for i in range(N):
        x = i*delta
        val = x*sin(pi*x)
        c0_vector.append(val)
    #print c0_vector
    return c0_vector
c = c0(Nval, delta)
for i in Nlist:
    c0n = c0(Nval, delta)
#print c0

# =============================================================================
# #print (I(N)+delta_t*D)
# def FE():
#     clist = [c0]
#     for i in range(int(2/dt)):
#         clist.append(np.dot(I(N)+dt*D,clist[-1]))
#         ##new_c_num = np.dot(np.array(I(N)+delta_t*D), np.array(clist[-1]))
#         #clist.append(new_c_num)
#     #print len(clist[0])
#     return list(clist)
# FE = FE()
# FE = np.transpose(FE)
# =============================================================================

def BE(N, LDelta, D):
    clist = [c]
    A = inv(I(N)-LDelta*D) 
    for i in range(int(0.3/delta_t)):
        clist.append(np.dot(A,clist[-1]))
        #print clist[-1]
    #print len(clist[0])
    return clist

BE1 = BE(Nval, delta_t, D)
BE1 = np.transpose(BE1)

for i in Nlist:
    D = makeD(Nval, delta)
    BEn = BE(i, delta_t, D)
    BEn = np.transpose(BEn)
    ##print BEn


# =============================================================================
# BE1 = BE(delta_t)
# BE1 = np.transpose(BE1)
# BE2 = BE(delta_t2)
# BE2 = np.transpose(BE2)
# #print len(BE2[0])
# #print len(BE1[0])
# =============================================================================

def BEsym(N, LDelta):
    clist = [c]
    A = inv(I(N)-LDelta*Dsym)
    for i in range(int(0.3/delta_t)):
        clist.append(np.dot(A,clist[-1]))
        #print clist[-1]
    #print len(clist[0])
    return clist
BEsym1 = BEsym(Nval, delta_t)
BEsym1 = np.transpose(BEsym1)
BEsym2 = BEsym(Nval, delta_t2)
BEsym2 = np.transpose(BEsym2)

def BEnon(N, LDelta):
    clist = [c]
    A = inv(I(N)-LDelta*Dnon)
    for i in range(int(0.3/delta_t)):
        clist.append(np.dot(A,clist[-1]))
        #print clist[-1]
    #print len(clist[0])
    return clist
BEnon1 = BEnon(Nval, delta_t)
BEnon1 = np.transpose(BEnon1)
BEnon2 = BEnon(Nval, delta_t2)
BEnon2 = np.transpose(BEnon2)

def Y(N, delta_t):
    ymatrix = []
    for i in range(N):
        r = []
        for j in range(int(0.3/delta_t)+1):
            y = j*delta_t
            r.append(y)
        ymatrix.append(r)
    #print len(ymatrix[0])
    return ymatrix

def X(N, delta_t):
    xmatrix = []
    for i in range(N):
        r = []
        for j in range(int(0.3/delta_t)+1):
            x = i*delta_t
            r.append(x)
        x = delta_t
        xmatrix.append(r)
    #print len(xmatrix[0])

    return xmatrix




X = X(Nval, delta_t)
print X
Y = Y(Nval, delta_t) 
print Y

# ========================== BE NORMAL d = 0.01 ===============================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.array(BE1), rstride=N/25, cstride=int(M/50), color="white", shade=False, edgecolor="green")
ax.view_init(azim=0)
plt.title("BE NORMAL d = 0.01")
plt.show()
# ========================== BE NORMAL d = 0.02 ===============================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.array(BE1), rstride=N/25, cstride=int(M/50), color="white", shade=False, edgecolor="green")
ax.view_init(azim=0)
plt.title("BE NORMAL d = 0.02")
plt.show()
# ======================= BE SYMMETRICAL d = 0.01 =============================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.array(BEsym1), rstride=N/10, cstride=int(M/10), color="white", shade=False, edgecolor="green")
ax.view_init(azim=0)
plt.title("BE SYMMETRICAL d = 0.01")
plt.show()
# ======================= BE SYMMETRICAL d = 0.02 =============================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.array(BEsym2), rstride=N/10, cstride=int(M/10), color="white", shade=False, edgecolor="green")
ax.view_init(azim=0)
plt.title("BE SYMMETRICAL d = 0.02")
plt.show()
# ======================= BE NON-SYMMETRICAL d = 0.01 =========================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.array(BEnon1), rstride=N/10, cstride=int(M/10), color="white", shade=False, edgecolor="green")
ax.view_init(azim=0)
plt.title("BE NON-SYMMETRICAL d = 0.01")
plt.show()

# ======================= BE NON-SYMMETRICAL d = 0.02 =========================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.array(BEnon2), rstride=N/10, cstride=int(M/10), color="white", shade=False, edgecolor="green")
ax.view_init(azim=0)
plt.title("BE NON-SYMMETRICAL d = 0.02")
plt.show()










# =============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_wireframe(X, Y, np.array(BEnon1), rstride=N/50, cstride=int(M/1000), color='blue')
# ax.plot_wireframe(X, Y, np.array(BEnon2), rstride=N/50, cstride=int(M/1000), color='white')
# plt.show()
# 
# =============================================================================

# =============================================================================
# print len(time)
# print domain
# print len(domain)
# =============================================================================
