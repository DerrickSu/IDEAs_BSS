#!python3


"""
This is a document of convolution kernel compensation.

First, I want to try some matrix multiplication and how to do the
inverse of matrix.

"""

# Numpy has svd function : numpy.linalg.svd . Then we can do pseudoinverse.
# Scikitlearn can do FastICA.


import time, sys, os
import random as rd
import numpy as np
import sklearn as sk
import matplotlib as mat
import matplotlib.pyplot as plt


from scipy.integrate import odeint
# 積分

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import interactive
from math import *
from numpy import matrix

# sin(pi) # 不為0是因為浮點數問題

rd.seed(time.time())

my_seq = [ sin(pi* i/100) for i in range(1000)  ]
my_rd = np.array(rd.sample(list(range(-20,80)),50)*20)/50
my_rdseq = my_rd*my_seq

my_index = np.array(range(0,1000)) *pi/100


