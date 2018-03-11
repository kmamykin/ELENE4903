import os

import numpy as np
import pandas as pd
from numpy.linalg import inv
from IPython.display import display
import matplotlib.pyplot as plt


def load_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'), header=None)
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'), header=None)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), header=None)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), header=None)
    return X_train.values, y_train.values, X_test.values, y_test.values


def accuracy_metric(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.shape[0]


def mrse_metric(y_pred, y_true):
    return np.sqrt(np.sum((y_pred - y_true) ** 2) / y_true.shape[0])


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
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw3/hw3-data/gaussian_process')
    gp = GaussianProcess(X_train, y_train, kernel=gaussian(b=10), sigma=0.01)
    predictions = gp.mean(X_test)
    print("Test MRSE: {:.2f}".format(mrse_metric(predictions, y_test)))


def problem_1_part_b():
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw3/hw3-data/gaussian_process')
    bs = [5, 7, 9, 11, 13, 15]
    variances = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    msres = np.zeros([len(bs), len(variances)])
    for b_index, b in enumerate(bs):
        for v_index, v in enumerate(variances):
            gp = GaussianProcess(X_train, y_train, kernel=gaussian(b=b),  sigma=np.sqrt(v))
            predictions = gp.mean(X_test)
            msres[b_index, v_index] = mrse_metric(predictions, y_test)

    table = pd.DataFrame(msres, index=["b={}".format(b) for b in bs], columns=["s^2={}".format(v) for v in variances])
    display(table)

def problem_1_part_d():
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw3/hw3-data/gaussian_process')
    X_one_dim = np.sort(X_train[:, 4].flatten()) # has to be sorted to make sense on the plot
    gp = GaussianProcess(X_one_dim, y_train, kernel=gaussian(b=5),  sigma=np.sqrt(2))
    predictions = gp.mean(X_one_dim).flatten()
    plt.figure(figsize=(16, 3))
    plt.scatter(X_one_dim, y_train[:,0], color='blue')
    plt.scatter(X_test[:, 4], y_test[:,0], color='red')
    plt.plot(X_one_dim, predictions, color='black')

    # std = gp.std(X_one_dim)
    # plt.fill_between(X_one_dim, predictions - 2 * std, predictions + 2 * std, alpha=0.2, color='k')
