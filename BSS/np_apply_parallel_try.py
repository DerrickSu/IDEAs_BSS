import timeit
import numpy as np
f = lambda x: x ** 2
vf = np.vectorize(f)

def test_array(x, n):
    t = timeit.timeit(
        'np.array([f(xi) for xi in x])',
        'from __main__ import np, x, f', number=n)
    print('array: ' + str(t))

def test_fromiter(x, n):
    t = timeit.timeit(
        'np.fromiter((f(xi) for xi in x), x.dtype)',
        'from __main__ import np, x, f', number=n)
    print('fromiter: ' + str(t))

def test_vectorized(x, n):
    t = timeit.timeit(
        'vf(x)',
        'from __main__ import x, vf', number=n)
    print('vectorized: ' + str(t))

x = np.array([1, 2, 3, 4, 5])
n = 100000
test_array(x, n)       # 0.616514921188
test_fromiter(x, n)    # 0.585698843002
test_vectorized(x, n)  # 2.6228120327


# numpy.apply_along_axis( func , arrdim , ndarr , *arg , **kwarg )
