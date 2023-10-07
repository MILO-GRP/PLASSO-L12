#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 17:52:27 2021

@author: Midas
"""

import numpy as np

class PLasso:
    '''
        Probabilistic Lasso (Column-wise L12-norm Regularization)
    '''

    def __init__(self, lamd=1, gamma=0.1, thresh=1e-7, maxiter=10000, eps=1e-9, verbose=False):
        self.X = None # p-by-n
        self.Y = None # n-by-k
        self.W = None # p-by-k
        self.lamd = lamd
        self.gamma = gamma
        self.maxiter = maxiter
        self.thresh = thresh
        self.eps = eps
        self.verbose = verbose

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.W = self.opt_alg(self.X, self.Y, self.lamd, self.gamma)
        return self.W

    def predict(self, X):
        prob = np.dot(self.W.T, self.padding(X))
        y_pred = np.argmax(prob, axis=0) + 1 # 0-based => 1-based
        return y_pred

    def init_w(self, XX, XY, I, gamma):
        return np.dot(np.linalg.inv(XX + gamma * I), XY)

    def cost_fn(self, X, Y, W, lamd):
        loss = np.sum((np.dot(X.T, W) - Y) ** 2)
        reg = np.sum(np.sum(np.abs(W), axis=0) ** 2) # column-wise L12-norm
        return loss + lamd * reg

    def L1(self, x):
        return np.sum(np.abs(x))

    def opt_alg(self, X, Y, lamd, gamma):
        # Optimization: Re-Weighted Algorithm
        # min_{W} ||X'W-Y||_F^2 + lambda*||W||_{1,2}^2 (X: p-by-n, Y: n-by-k, W: p-by-k)
        p, n = X.shape
        k = Y.shape[1]
        XX = np.dot(X, X.T) # p-by-p
        XY = np.dot(X, Y) # p-by-k
        Ip = np.identity(p) # p-by-p
        W = self.init_w(XX, XY, Ip, gamma)
        loss = [self.cost_fn(X, Y, W, lamd)]
        iter = 1
        if self.verbose:
            print('%d-th iteration: loss=%.10f' % (0, loss[-1]))
        while iter <= self.maxiter:
            for j in range(k):
                Wj = W[:, j] # p-by-1 (1d)
                Xyj = XY[:, j:j+1] # p-by-1
                Dj = self.L1(Wj) * np.diag(1.0 / (np.abs(Wj) + self.eps)) # p-by-p
                W[:, j:j+1] = np.dot(np.linalg.inv(XX + lamd * Dj), Xyj)
            loss.append(self.cost_fn(X, Y, W, lamd))
            if self.verbose:
                print('%d-th iteration: loss=%.10f' % (iter, loss[-1]))
            if iter > 1 and abs(loss[-1] - loss[-2]) / abs(loss[-1]) < self.thresh:
                break
            iter += 1
        return W


def main():
    from utils import demo, calc_gamma, print_sparse_matrix
    X, Y = demo()
    r, lamd = calc_gamma(X, 0.3), 0.01
    clf = PLasso(lamd=lamd, gamma=r, thresh=1e-7, maxiter=10000, verbose=False)
    W = clf.fit(X, Y)
    print('lambda = %.2f, the learned weight matrix is: ' % (lamd))
    print_sparse_matrix(W)


if __name__ == '__main__':
    main()
