import os

import numpy as np
import pandas as pd
from numpy.linalg import inv


def load_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'), header=None)
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'), header=None)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), header=None)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), header=None)
    return X_train.values, y_train.values, X_test.values, y_test.values


def kernel(kernel_fn, signature='(i),(i)->()'):
    vectorized_kernel_fn = np.vectorize(kernel_fn, signature=signature)

    def _reshape(x):
        return x if x.ndim > 1 else x[:, np.newaxis]

    def covariance_matrix_computation(X, Y):
        X = _reshape(X)
        Y = _reshape(Y)
        x_indices, y_indices = np.meshgrid(np.arange(X.shape[0]), np.arange(Y.shape[0]), indexing='ij')
        return vectorized_kernel_fn(X[x_indices], Y[y_indices])

    return covariance_matrix_computation


def gaussian(b=1):
    return kernel(lambda x1, x2: np.exp((-1 / b) * np.sum((x1 - x2) ** 2)))

class GaussianProcess(object):

    def __init__(self, x, y, kernel, mean_fn=lambda y: np.mean(y), sigma=0.1):
        self.X = x
        self.y = y
        self.kernel = kernel
        self.mean_fn = mean_fn
        self.sigma = sigma
        self.k_X_X = self.kernel(self.X, self.X)
        self.noise_covariance = self.sigma ** 2 * np.identity(self.X.shape[0])
        self.inv_k_X_X = inv(self.k_X_X + self.noise_covariance)

    def mean(self, x0):
        k_x0_X = self.kernel(x0, self.X)
        return self.mean_fn(self.y) + np.dot(np.dot(k_x0_X, self.inv_k_X_X), (self.y - self.mean_fn(self.y)))

    def covariance(self, x0):
        k_x0_D = self.kernel(x0, self.X)
        return self.kernel(x0, x0) - np.dot(np.dot(k_x0_D, self.inv_k_X_X), k_x0_D.T)

    def std(self, x0):
        cov = self.covariance(x0)
        return np.sqrt(cov.diagonal()).flatten()

def problem_1_part_a():
    pass