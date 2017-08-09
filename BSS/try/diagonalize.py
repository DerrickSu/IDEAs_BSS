


import numpy as np
import numpy.linalg as lin



P = np.array(
    [ [ 3 , 2 , 1 ],
      [ 2 , 2 , 3 ],
      [ -4, 1 , -1] ]
    )


s = np.random.randint(1,5,size = (3))

S = np.diag(s)


B = P@S@lin.inv(P)

A = lin.eig(B)
