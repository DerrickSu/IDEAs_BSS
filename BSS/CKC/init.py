




"""
Initialization of rCKC.
Some codes are written for original CKC.

讀mat檔
scipy.io.loadmat 

讀CSV檔
np.getfromtxt(file , delimter = ",") 

"""
import numpy as np
import random as rd

import rCKC

# Simulation data
# n = 1009
my_seq = np.array([ np.sin(np.pi* i/100) for i in range(1009)  ])
my_rd = np.array(rd.sample(list(range(-20,80)),50)*20+[1,5,2,3,5,8,9,5,1])/50

mix = np.array( [[1 , 0],[0,1]])
Y = (np.c_[my_seq,my_rd]@mix).T

Y = rCKC.Sample_process(Y)

M = 2
# Data end 


cov = Y@Y.T
cov_inv = np.linalg.inv(cov)



# Covariance matrix and diagonalize it
def Obser_eig(y,k=5):
    y = np.array(y)
    #y = (y-y.mean()).reshape((len(y),1))
    Cov = y@y.T/len(y)
    s,Q = linalg.eig(Cov)

    #sort
    idx = s.argsort()[::-1]
    s = s[idx]
    Q = Q[:,idx]
    
    #S_hat = s[:len(s)-k]
    return s,Q
    




# This cov is inverse of covariance matrix.
def norm_one(cov):
    th = 0
    for i in cov.T:
        a = 0
        for j in i:
            a += abs(j)
        if a > th:
            th = a
    return th


act_idx = np.diag( Y.T@ cov_inv @ Y)

max_idx = act_idx.argsort()[::-1][0:M]



"""
orignal CKC

sig = s[len(s)*8//10]
threshold = sig*norm_one(cov_inv)
idx = act_idx < threshold
act_idx[idx] = 0

"""

def MU_init():
    global max_idx
    eval_fun = [ "rCKC.MU( Y[:,max_idx[{0}]] , act_idx[max_idx[{0}]] )".format(i) for i in range(M)]
    for i in eval_fun:
        yield eval(i)

listMU = []
for i in MU_init():
    listMU.append(i)

# 蒐集所有filter
# 測試
"""
f = np.array([eval( "listMU[{0}].f".format(i) ) for i in range(10) ])

t = f[0] @ cov_inv @ Y

s = listMU[0].max_s
"""
fun = ["listMU[{0}].train(Y,cov_inv,1000)".format(i) for i in range(M)] 
for i in fun:
    eval(i)

for i in range(M):
    a = listMU[i]
    print("MU{0}: {1}".format(i,a.n)  )

