

"""
Calculate threshold by original CKC method.


    # orignal CKC

    sig = s[len(s)*8//10]
    threshold = sig*norm_one(cov_inv)
    idx = act_idx > threshold    # boolean type
    act_idx[np.invert(idx)] = 0



"""

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
