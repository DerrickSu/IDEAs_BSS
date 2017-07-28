#!python3


import numpy as np

from math import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import interactive


s = np.arange(-2.5 , 2.5 , 0.005)
x = [sin(pi*i) + np.random.normal(0,0.05) for i in s]
y = [cos(pi*i) + np.random.normal(0,0.05)for i in s]
z = s + np.random.normal(0,0.1)


interactive(True)
fig = plt.figure()
ax = fig.gca(projection = "3d")
ax.scatter(x,y,z,marker = ".")
plt.show()


