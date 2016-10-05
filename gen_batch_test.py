import numpy as np
import random
import pdb
import math
# Dummy input matrix
X = np.random.rand(31223, 300)
batch_size = 1024
n_batches = X.shape[0]/float(batch_size)
n_batches = int(math.ceil(n_batches))
end = int(X.shape[0]/float(batch_size)) * batch_size
batch = []
n = 0
for i in xrange(0,n_batches-1):
    batch = X[i*batch_size:(i+1) * batch_size, :]
    print batch.shape
    n += batch.shape[0]
    #pdb.set_trace()

print(X[end: , :].shape)
n += X[end:, :].shape[0]
batch = np.array(batch)
pdb.set_trace()
