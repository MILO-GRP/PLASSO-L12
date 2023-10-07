import numpy as np

def demo():
    X = np.array(
        [[0.463, 0.319, -0.100, 0.526, 0.535, 0.329, 0.475],
         [0.296, 0.192, 0.058, -0.076, 0.152, 0.313, -0.114],
         [0.196, 0.189, 0.167, -0.280, 0.267, -0.246, 0.164],
         [0.330, 0.357, 0.027, -0.001, 0.118, 0.058, 0.191],
         [0.332, 0.035, -0.002, 0.280, 0.111, -0.043, 0.104],
         [-0.022, -0.026, 0.770, 0.189, 0.196, -0.146, -0.121],
         [-0.217, 0.028, 0.404, 0.359, 0.335, -0.282, -0.235],
         [0.396, 0.297, 0.260, 0.241, 0.193, 0.038, 0.101]]).T # p-by-n
    Y = np.array(
        [[1, 0, 0],
         [1, 1, 0],
         [1, 0, 1],
         [1, 1, 1],
         [0, 1, 0],
         [0, 1, 1],
         [0, 0, 1],
         [0, 0, 1]]) # n-by-k
    return X, Y

def print_sparse_matrix(X):
    r, c = X.shape
    w = (7 + 1) * c
    print(w * '-')
    for i in range(r):
        for j in range(c):
            print('%7.4f' % X[i][j], end=' ')
        print()
    print(w * '-')

def calc_gamma(X, percentage):
    eigVal = np.sort(np.linalg.eigvals(np.dot(X, X.T)))
    gamma = eigVal[int(np.round(len(eigVal) * percentage)) - 1]
    return gamma
