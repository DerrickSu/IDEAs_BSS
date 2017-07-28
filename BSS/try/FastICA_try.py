#!python3


"""
FastICA is linear and can't process the sparse data well.
SparsePCA doesn't work on sparse data


"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.decomposition import FastICA , PCA, SparsePCA


np.random.seed(0)
n_samples = 2000
time = np.linspace(0,8,n_samples)



s1 = np.sin(2*time)
s2 = np.sign(np.sin(3*time))
s3 = signal.sawtooth(2 * np.pi * time)

noise = np.random.normal(0,0.5,2000)

"""
# sparse
fil_1 = [i for i in range(2000) if ((i>=90) and (i<300)) or ( (i>=800) and (i<1300) )]
s1[fil_1] = [0]*710
fil_2 = [i for i in range(2000) if ((i>=100) and (i<310)) or ( (i>=850) and (i<1350) )]
s2[fil_2] = [0]*710
s3[[i for i in range(2000) if ((i>=70) and (i<280)) or ( (i>=700) and (i<1200) )]] = [0]*710
"""

S = np.c_[s1,s2,s3,noise]
S +=0.2 * np.random.normal(size = S.shape)

# standardize data
S /= S.std(axis = 0)

#A = np.array([[1,0.5,1.1],[0.5 ,3 ,1],[1.5,1,2]])


A = np.array([[1,0.5,1 , 1],[0.5 ,2 ,1,1],[1.5,1,2,1]])


X = S @ A.T


# compute ICA

ica = FastICA(n_components = 3)
S_ = ica.fit_transform(X) # Get the estimated sources
A_ = ica.mixing_ # Get estimated mixing matrix


# compute PCA

pca = PCA(n_components = 3)
H = pca.fit_transform(X) # estimate PCA sources


# plot
from matplotlib import interactive


"""
plt.figure( figsize = (9,6))

models = [X, S, S_, H]

names = [
    'Observations (mixed signal)',
    'True Sources',
    'ICA',
    'PCA'
    ]
colors = ['red' , 'steelblue' , 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(2, 2, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.title("Component Analysis")
plt.show()
"""





plt.figure(figsize = (12,9))
names = ['original']*4 +['MIX']*3+ ['ICA']*4

M = np.array([*S.T,*X.T, *S_.T])
#ch = [1,4,7,2,5,8,3,6,9]
ch = [1,4,7,10,2,5,8,3,6,9,12]


for i , s in enumerate(M ,0):
    plt.subplot(4,3,ch[i])
    plt.title(names[i])
    plt.plot(s)

plt.tight_layout()
plt.show()






