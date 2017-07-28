

import sys , os , time
import numpy as np
import matplotlib.pyplot as plt
import scipy


from numpy import linalg

pi = np.pi

"""
inverse
numpy.linalg.inv()

matrix.transpose()

np.linalg.svd(a)
a = U S V^T

np.linalg.eig()

實對稱矩陣可正交對角化 
共變異數矩陣亦可

"""
# input
# Y(n): M*K sample . M observations , K delay.
# Further , reshape Y(n) to size of (MK * 1)
# N time stamp
# t means source. L sources.

# tj(n0) == 1 means that jth source is active at n0

# Calculate activity index to find t(nk) == 1 while index is larger than noise.
# Collect all time stamp that t == 1 to G (a set)


# Reshape the data
def process_M(data , K = 4):
    # M observation 
    # obser is M*N
    # K is delay
    
    # Output is Y = [ [Y(1)],
    #                 ........
    #                 [Y(N-K+1)] ]
    
    # Dimension of Y is  (N-K+1) * MK

    # Y(nk) is K delay of time stamp of all observations ( dimension is 1*MK )
    data = np.array(data)
    M , N = data.shape

    x = np.zeros( shape = (N - K+1 ,M*K) )

    for n in range(N-K+1):
        m_k = []

        for m in data:
           m_k.extend( m[ n : n+K ] )

        x[n,:] = m_k

    return x



# Covariance matrix and diagonalize it
def obser_eig(y,k=5):
    y = np.array(y)
    y = (y-y.mean()).reshape((len(y),1))
    Cov = y@y.T
    s,Q = linalg.eig(Cov)

    #sort
    idx = s.argsort()[::-1]
    s = s[idx]
    Q = Q[:,idx]
    
    #S_hat = s[:len(s)-k]
    return s,Q
    
    






if __name__ == "__main__":
    obser = np.arange(1,51).reshape((5,10))
    x = process_M(obser , K = 5)
    





