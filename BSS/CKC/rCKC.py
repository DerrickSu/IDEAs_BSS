

##import sys , os , time
import numpy as np
##import matplotlib.pyplot as plt
##import scipy


from numpy import linalg

# define warning as error to avoid wasting time for calculation
import warnings
warnings.filterwarnings("error")


"""
This method refer to the paper " Real-Time Motor Unit Identification
From High-Density Surface EMG ".

The main destination is to find MU signal and find which is fatigue or
what kind of motion is act.


Functions:

Sample_process
Obser_eig
init_inv_cov
update_cov


Class:

MU


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



sample_rate = 1000 # Hz

D = 0
Q = 250

cov = [] # The first cov matrix from 0 to D

# Reshape the data
def Sample_process(data , K = 10):
    # data is M*N
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
# (1)不記錄過去的activity index，
# (2)記錄過去activity index，每次更新後皆重新挑選，如此較耗時。
# 目前選用(1)方案，

class MU:
    def __init__(self,f,s):
        # Signal time stamp
        # It's a list of index
        self.__s = []

        # Num of s
        self.__n = 0

        # Filter of this MU
        self.__f = f

        # Max of activity index
        self.__max_s = s
        
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
    def max_s(self):
        return self.__max_s

    # Use a block to update all
    # 要一邊更新alpha 一邊更新index 選擇penalty最小的 後再加入
    def __update( self , YQ , index ):

        global D
        n = self.__n
        x = 0
        for s ,i in zip(YQ[:,index].T,index):
            x += s
            # time stamp is D+i
            self.__s.append(D+i)
            self.__n += 1
        self.__f =  self.__f*n + x
        self.__f /= self.__n


    # 找尋YQ中MU的訊號
    # YQ is next block of signal.
    # Training threshold (jth MU)
    # Just use 0.2*max activity index to 1*activity index by step of 0.2 .
    # 時間在 D ~ D+Q , Q is sample number in a block.
    def train(self , YQ , cov ,block = 250):
        """
        如果在此區間沒抓到MU
        則會出現warning(因interval.std無法計算
        得排除此BUG以免程式無法繼續運作
        """
        global sample_rate ,D
        
        act_idx = self.__f.T @ cov @ YQ
        p = np.inf
        alpha = 0
        for a in np.arange(0.2,1.1,0.1): # alpha training
            idx = np.array(range(0,YQ.shape[1]))[ act_idx >= a*self.__max_s ]

            # Penalty function = 100*d+CV
            # Coefficient of variation (CV) = std/mean
            # Median of discharge rate (pulse per second)
            # for interspike intervals in jth MU set.
            time_space = np.linspace(0,block/sample_rate , block , endpoint = False)            
            t = time_space[idx]
            interval = t[1::] - t[0:-1]
            try:
                discharge = 1/np.median(interval)

                if(discharge < 6 or(discharge > 40)):
                    d = 1
                else:
                    d = 0                
                CV = interval.std() / interval.mean()
            except RuntimeWarning:
                break
            penalty = 10*d+CV
            
#            print("alpha:",a , penalty)

            if penalty < p:
                p = penalty
                alpha = a

#        print("final alpha:",alpha,"\n")

        idx = np.array(range(0,YQ.shape[1]))[ act_idx > alpha*self.__max_s ]
        # update MU information
        self.__update(YQ,idx)
        if act_idx.max() > self.__max_s:
            self.__max_s = act_idx.max()






# Create covariance matrix 
# Y is MK * N . It means the data of all sample from n = 0 to n = N.
def init_inv_cov(Y):
    global cov
    cov = linalg.inv(Y@Y.T/Y.shape[1])


# update covariance matrix
def update_cov( YQ ):
    # Use global var to change its value sychronously.
    global cov
    
    I = np.diag([1]*cov.shape[0])
    inv = linalg.inv(I + YQ.T @ cov @ YQ)
    cov = cov - cov @ YQ @ inv @ YQ.T @ cov







