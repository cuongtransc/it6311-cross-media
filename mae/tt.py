import numpy as np

K = 10
N = 9
ll = []
for i in xrange(K):
    ar = np.ndarray((N))
    ar.fill(i)
    ll.append(ar)

ss = np.asarray(ll)
print ss.shape

