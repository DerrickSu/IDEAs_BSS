




"""
Initialization of rCKC.
Some codes are written for original CKC.



"""
import numpy as np

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


sig = s[len(s)*8//10]
threshold = sig*norm_one(cov_inv)


act_idx = Y.T@ cov_inv @ Y

idx = act_idx < threshold

act_idx[idx] = 0






