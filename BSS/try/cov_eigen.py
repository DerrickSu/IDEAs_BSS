




import numpy as np
from sklearn.decomposition import PCA


np.random.seed(1268153)
x = np.random.randint(-5,5,size = (5,1))

X = x@x.T

eig_val , eig_vec = np.linalg.eig(X)
