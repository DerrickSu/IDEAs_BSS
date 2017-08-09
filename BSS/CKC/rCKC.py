

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

"""
## input
## Y(n): M*K sample . M observations , K delay.
## Further , reshape Y(n) to size of (MK * 1)
## N time stamp
## t means source. L sources.
##
## tj(n0) == 1 means that jth source is active at n0
##
## Calculate activity index to find t(nk) == 1 while index is larger than noise.
## Collect all time stamp that t == 1 to G (a set)


# 初始時間D以前，block時間長Q
# 資料應該要紀載時間


block = 250
sample_rate = 1000

D = 0
Q = 250


# Reshape the data
def Sample_process(data , K = 4):
    # M observation ,N times data
    # obser is M*N.
    # K is delay.
    
    # Output is Y = [ [Y(1)],
    #                 ........
    #                 [Y(N-K+1)] ].T
    
    # Dimension of Y is MK * (N-K+1) .(obser * time delay ,time)
    # Row represents time stamp.

    # Y(nk) is K delay of time stamp of all observations ( dimension is 1*MK )
    data = np.array(data)
    M , N = data.shape

    Y = np.zeros( shape = (N - K+1 ,M*K) )

    for n in range(N-K+1):
        m_k = []

        for m in data:
           m_k.extend( m[ n : n+K ] )

        Y[n,:] = m_k

    return Y.T


# Set of MU
class MU:
    def __init__(self):
        # Signal time stamp
        # It's a list of index
        self.__s = []

        # Num of s
        self.__n = 0

        # Filter of this MU
        self.__f = 0

        # Threshold
        self.__th = 0

        # Max of activity index
        self.__max_s = 0
        
    @property
    def s(self):
        return self.__s

    @property
    def n(self):
        return self.__n

    @property
    def f(self):
        return self.__f

    @property
    def th(self):
        return self.__th

    # Use a block to update all
    # 要一邊更新alpha 一邊更新index 選擇penalty最小的 後再加入
    def update( self , YQ , index ):
        n = self.__n
        x = 0
        for i in YQ[:,index]:
            x += i
            self.__s.append(i)
            self.__n += 1
        self.__f = self.__f*n + x
        self.__f /=self.__n
    

    # 更新
    # YQ is next block of signal.
    # Training threshold (jth MU)
    # Just use 0.2*max activity index to 1*activity index by step of 0.2 .
    # 時間在 D ~ D+Q , Q is sample number in a block.
    def train(self , YQ , cov ):
        global block , sample_rate ,D
        act_idx = self.__f.T @ cov @ YQ
        p = 0
        alpha = 0
        for i in range(0.2,1.2,5):
            idx = (np.array(range(0,YQ.shape[1])))[ act_idx > i*self.__max_s ]
            
            time_space = np.linspace(0,block/sample_rate , block)
            t = time_space[idx]
            interval = t[1::] - t[0:-1]

            discharge = 1/np.median(interval)
            if(discharge < 6 )or(discharge > 40):
                d = 1
            else:
                d = 0

            CV = interval.std() / interval.mean()
            penalty = 100*d+CV
            if penalty > p:
                pass
            else:
                p = penalty
                alpha = i
        idx = (np.array(range(0,YQ.shape[1])))[ act_idx > i*self.__max_s ]
        
            
# Coefficient of variation (CV) = std/mean
# Median of discharge rate (pulse per second)
# for interspike intervals in jth MU set.



# Covariance matrix and diagonalize it
def Obser_eig(y,k=5):
    y = np.array(y)
    y = (y-y.mean()).reshape((len(y),1))
    Cov = y@y.T/(len(y)-1)
    s,Q = linalg.eig(Cov)

    #sort
    idx = s.argsort()[::-1]
    s = s[idx]
    Q = Q[:,idx]
    
    #S_hat = s[:len(s)-k]
    return s,Q
    

# Create covariance matrix 
# Y is MK * N . It means the data of all sample from n = 0 to n = N.
def produce_inv_cov(Y):
    return linalg.inv(Y@Y.T/Y.shape[1])


# update covariance matrix
def update_cov(orig , YQ):
    I = np.diag([1]*orig.shape[0])
    inv = linalg.inv(I + YQ.T @ orig @ YQ)

    return orig - orig @ YQ @ inv @ YQ.T @ orig




if __name__ == "__main__":
    obser = np.arange(1,51).reshape((5,10))
    x = Sample_process(obser , K = 5)
    





