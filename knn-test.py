import numpy as np
from sklearn.neighbors import KDTree
exit()
X = np.random.rand(10, 3)  # 10 points in 3 dimensions
print X
tree = KDTree(X, leaf_size=2)              
print "-"*40
dist, ind = tree.query(X[0].reshape(1, -1), k=3)
print dist
print ind
