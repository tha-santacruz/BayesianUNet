import numpy as np

cf_mat = np.array([[13,0,2],[1,18,4],[2,2,50]])
print(cf_mat)
print(cf_mat.sum(axis=1,keepdims=True))
print(cf_mat/cf_mat.sum(axis=1,keepdims=True))