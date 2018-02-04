import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from cycler import cycler

plt.rc('axes', prop_cycle=(cycler('color', ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']) +
                           cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.'])))

def pairplot(dataframe):
    g = sns.PairGrid(dataframe, diag_sharey=False)
    g.map_lower(sns.kdeplot, kernel='gau', n_levels=10)
    g.map_upper(plt.scatter, marker='.')
    g.map_diag(sns.kdeplot)
    g.fig.set_size_inches(14,14);


FEATURES = [
    'cylinders',
    'displacement',
    'horsepower',
    'weight',
    'acceleration',
    'year',
    'mpg'
]


def make_dataframe(X, y):
    df = pd.DataFrame(X[:,0:6])
    df[6] = y
    df.columns = FEATURES
    return df


def load_data(data_dir='hw1/hw1-data'):
    """
    Loads data from csv files
    :param data_dir:
    :return: numpy arrays
    """
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'), header=None)
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'), header=None)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), header=None)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), header=None)
    return X_train.values, y_train.values, X_test.values, y_test.values


class RidgeRegression:

    # We can't use the parameter name "lambda" as it is reserved keyword in python. Use "alpha" instead.
    def __init__(self, alpha=None):
        """
        :param alpha: Regularization strength parameter (can't use lambda as param name)
        """
        self.alpha = alpha

    def fit(self, X=None, y=None):
        """
        Fit linear regression with regularization "ridge regression"
        :param X: Input features matrix NxD (D has an extra dimension with 1s)
        :param y: Target vector shaped (N,1)
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert y.shape[1] == 1
        N, D = X.shape
        pseudo_inverse = np.matmul(np.linalg.inv(self.alpha * np.identity(D, dtype=np.float) + np.matmul(X.T, X)), X.T)
        self.W_ = np.matmul(pseudo_inverse, y)
        self.df_ = np.trace(np.matmul(X, pseudo_inverse))
        return self

    @property
    def W(self):
        """
        :return: Vector of W (weights) shaped (D,1)
        """
        return self.W_

    @property
    def df(self):
        """
        :return: Degrees of freedom of self.alpha
        """
        return self.df_

    def predict(self, X=None):
        """
        :param X: Features to make prediction for
        :return: y vector of predictions
        """
        assert X.shape[1] == self.W.shape[0], 'Number of features in X must match'
        return np.matmul(X, self.W)

    def score(self, X=None, y=None):
        """
        :param X:
        :param y:
        :return: Mean Squared Error == (RMSE)**2
        """
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == self.W.shape[0], 'Number of features in X must match'
        return np.sum((self.predict(X) - y)**2)/y.shape[0]


def compute_part1a(max_lambda, X_train, y_train):
    lambdas = np.linspace(0, max_lambda, num=max_lambda, endpoint=False)
    ws = np.zeros((max_lambda, 7))
    dfls = np.zeros(max_lambda)
    for i in range(max_lambda):
        rr = RidgeRegression(lambdas[i]).fit(X_train, y_train)
        ws[i] = rr.W.reshape(-1) # to flat vector
        dfls[i] = rr.df
    return lambdas, ws, dfls, ['dim '+ str(i+1) for i in range(X_train.shape[1])]


def plot_part1a(lambdas, weights, degrees_of_freedom, labels):
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(lambdas, degrees_of_freedom)
    ax.set_xlabel('lambda')
    ax.set_ylabel('df(lambda)')

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(lambdas, weights)
    ax.set_xlabel('lambda')
    ax.set_ylabel('W rr')

    ax = fig.add_subplot(gs[:, 1:])
    for i in range(7):
        ax.plot(degrees_of_freedom, weights[:, i], label=labels[i])
    ax.set_xlabel('df(lambda)')
    ax.set_ylabel('W rr')
    ax.legend(loc='lower left')
    fig.tight_layout()


def part1a():
    X_train, y_train, X_test, y_test = load_data('hw1/hw1-data')
    plot_part1a(*compute_part1a(5000, X_train, y_train))


def compute_part1c(max_lambda, X_train, y_train, X_test, y_test):
    lambdas = np.linspace(0, max_lambda, num=max_lambda, endpoint=False)
    mse_train = np.zeros(max_lambda)
    mse_test = np.zeros(max_lambda)

    for i in range(max_lambda):
        rr = RidgeRegression(lambdas[i]).fit(X_train, y_train)
        mse_train[i] = rr.score(X_train, y_train)
        mse_test[i] = rr.score(X_test, y_test)
    return lambdas, mse_train, mse_test


def plot_part1c(lambdas, mse_train, mse_test):
    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(lambdas, mse_train, label='Train MSE')
    ax.plot(lambdas, mse_test, label='Test MSE')
    ax.set_xlabel('lambda')
    ax.set_ylabel('MSE')
    ax.legend()

    fig.tight_layout()


def part1c():
    X_train, y_train, X_test, y_test = load_data('hw1/hw1-data')
    plot_part1c(*compute_part1c(50, X_train, y_train, X_test, y_test))


def expand_features(X_train, p):
    # X_train has 7 features including the bias which we should not transform
    N, D = X_train.shape
    X_orig = X_train[:,0:6] # this is the original features without bias
    X = np.zeros([N, 0], dtype=np.float)
    for poly_order in range(1, p+1):
        X = np.hstack((np.power(X_orig, poly_order), X))
    return np.hstack((X, np.ones([N, 1], dtype=np.float)))


def compute_part2a(max_lambda, max_p, X_train, y_train, X_test, y_test):
    lambdas = np.linspace(0, max_lambda, num=max_lambda, endpoint=False)
    result = []
    for p in range(1, max_p+1):
        mse_train = np.zeros(max_lambda)
        mse_test = np.zeros(max_lambda)
        X_train_expanded = expand_features(X_train, p)
        X_test_expanded = expand_features(X_test, p)
        for i in range(max_lambda):
            rr = RidgeRegression(lambdas[i]).fit(X_train_expanded, y_train)
            mse_train[i] = rr.score(X_train_expanded, y_train)
            mse_test[i] = rr.score(X_test_expanded, y_test)
        result.append((mse_train, mse_test))
    return lambdas, result


def plot_part2a(lambdas, results):
    fig = plt.figure(figsize=(8, 9))
    gs = gridspec.GridSpec(len(results), 1)

    for subplot in range(3):
        ax = fig.add_subplot(gs[subplot, 0])
        ax.set_title("Polynomial order ={}".format(subplot+1))
        ax.plot(lambdas, results[subplot][0], label='Train MSE')
        ax.plot(lambdas, results[subplot][1], label='Test MSE')
        ax.set_xlabel('lambda')
        ax.set_ylabel('MSE')
        optimal_lambda = np.argmin(results[subplot][1])
        ax.annotate("{}".format(optimal_lambda),
                    xy=(optimal_lambda, results[subplot][1][optimal_lambda]), xycoords='data',
                    xytext=(optimal_lambda, results[subplot][1][optimal_lambda] + 2.5), textcoords='data',
                    arrowprops=dict(facecolor='red', arrowstyle='->'))
        ax.legend()
    fig.tight_layout()


def part2a():
    X_train, y_train, X_test, y_test = load_data('hw1/hw1-data')
    plot_part2a(*compute_part2a(500, 3, X_train, y_train, X_test, y_test))
