#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:13:46 2018

@author: Kaina
"""

def BE():
    clist = [c0]
    for i in range(int(2/delta_t)):
        clist.append(np.dot(inv(I(N)-delta_t*D),clist[-1]))
        #print clist[-1]
    #print len(clist[0])
    return clist
BE = BE()
BE = np.transpose(BE)