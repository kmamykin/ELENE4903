import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from cycler import cycler

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
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
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'), header=None)
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'), header=None)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), header=None)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), header=None)
    return X_train.values, y_train.values, X_test.values, y_test.values


class RidgeRegression:

    # We can't use the parameter name "lambda" as it is reserved keyword in python. Use "alpha" instead.
    def __init__(self, alpha=None):
        self.alpha = alpha

    def fit(self, X=None, y=None):
        """
        Fit linear regression with regularization "ridge regression"
        :param X: Input features matrix NxD (D has an extra dimension with 1s)
        :param y: Target vector shaped (N,1)
        :param alpha: Regularization strength parameter (can't use lambda as param name)
        :return: Vector of W (weights) shaped (D,1)
        """
        assert X.shape[0] == y.shape[0]
        assert y.shape[1] == 1
        N, D = X.shape
        regularized_pseude_inverse = np.matmul(np.linalg.inv(self.alpha * np.identity(D, dtype=np.float) + np.matmul(X.T, X)), X.T)
        self.usv_ = np.linalg.svd(X)
        self.W_ = np.matmul(regularized_pseude_inverse, y)
        self.df_ = np.trace(regularized_pseude_inverse)
        return self

    @property
    def W(self): return self.W_

    @property
    def df(self): return self.df_

    @property
    def usv(self): return self.usv_

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
        :return: RMSE
        """
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == self.W.shape[0], 'Number of features in X must match'
        return np.sqrt(np.sum((self.predict(X) - y)**2)/y.shape[0])


def compute_part1a(max_lambda, X_train, y_train):
    lambdas = np.linspace(0, max_lambda, num=max_lambda, endpoint=False)
    ws = np.zeros((max_lambda, 7))
    dfls = np.zeros(max_lambda)
    dfls2 = np.zeros(max_lambda)
    for i in range(max_lambda):
        rr = RidgeRegression(lambdas[i]).fit(X_train, y_train)
        ws[i] = rr.W.reshape(-1)
        dfls[i] = rr.df
        u, s, v = rr.usv
        dfls2[i] = np.sum(s ** 2 / (lambdas[i] + s ** 2))
    return lambdas, ws, dfls2, ['dim '+ str(i+1) for i in range(7)]


def plot_part1a(lambdas, weights, degrees_of_freedom, labels):
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(lambdas, degrees_of_freedom)
    ax.set_xlabel('$$\lambda$$')
    ax.set_ylabel('$$df(\lambda)$$')

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(lambdas, weights)
    ax.set_xlabel('$$\lambda$$')
    ax.set_ylabel('$$w_{rr}$$')

    ax = fig.add_subplot(gs[:, 1:])
    for i in range(7):
        ax.plot(degrees_of_freedom, weights[:, i], label=labels[i])
    ax.set_xlabel('$$df(\lambda)$$')
    ax.set_ylabel('$$w_{rr}$$')
    ax.legend(loc='lower left')
    fig.tight_layout()


def part1a():
    X_train, y_train, X_test, y_test = load_data('hw1/hw1-data')
    plot_part1a(*compute_part1a(5000, X_train, y_train))


def compute_part1c(max_lambda, X_train, y_train, X_test, y_test):
    lambdas = np.linspace(0, max_lambda, num=max_lambda, endpoint=False)
    rmse_train = np.zeros(max_lambda)
    rmse_test = np.zeros(max_lambda)
    for i in range(max_lambda):
        rr = RidgeRegression(lambdas[i]).fit(X_train, y_train)
        rmse_train[i] = rr.score(X_train, y_train)
        rmse_test[i] = rr.score(X_test, y_test)
    return lambdas, rmse_train, rmse_test


def plot_part1c(lambdas, rmse_train, rmse_test):
    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(lambdas, rmse_train, label='Train')
    ax.plot(lambdas, rmse_test, label='Test')
    ax.set_xlabel('$$\lambda$$')
    ax.set_ylabel('RMSE')
    ax.legend()

    fig.tight_layout()


def part1c():
    X_train, y_train, X_test, y_test = load_data('hw1/hw1-data')
    plot_part1c(*compute_part1c(50, X_train, y_train, X_test, y_test))


def expand_features(X_train, p):
    # X_train has 7 features including the bias which we should not transform
    N, D = X_train.shape
    # print('Expand X_train', X_train.shape, p)
    X_orig = X_train[:,0:6] # this is the original features without bias
    # print('X_orig', X_orig.shape, X_orig[0])
    X = np.zeros([N, 0], dtype=np.float)
    for poly_order in range(1, p+1):
        X = np.hstack((np.power(X_orig, poly_order), X))
        # print('Poly expansion for p=' + str(poly_order), X.shape, X[0])
    return np.hstack((X, np.ones([N, 1], dtype=np.float)))


def compute_part2a(max_lambda, max_p, X_train, y_train, X_test, y_test):
    lambdas = np.linspace(0, max_lambda, num=max_lambda, endpoint=False)
    result = []
    for p in range(1, max_p+1):
        rmse_train = np.zeros(max_lambda)
        rmse_test = np.zeros(max_lambda)
        X_train_expanded = expand_features(X_train, p)
        X_test_expanded = expand_features(X_test, p)
        for i in range(max_lambda):
            rr = RidgeRegression(lambdas[i]).fit(X_train_expanded, y_train)
            rmse_train[i] = rr.score(X_train_expanded, y_train)
            rmse_test[i] = rr.score(X_test_expanded, y_test)
        result.append((rmse_train, rmse_test))
    return lambdas, result


def plot_part2a(lambdas, results):
    fig = plt.figure(figsize=(8, 9))
    gs = gridspec.GridSpec(len(results), 1)

    for subplot in range(3):
        ax = fig.add_subplot(gs[subplot, 0])
        ax.set_title("p={}".format(subplot+1))
        ax.plot(lambdas, results[subplot][0], label="Train")
        ax.plot(lambdas, results[subplot][1], label='Test')
        ax.set_xlabel('$$\lambda$$')
        ax.set_ylabel('RMSE')
        optimal_lambda = np.argmin(results[subplot][1])
        # ax.axvline(x=optimal_lambda, ymin=0, ymax=0.25, color='red')
        ax.annotate("{}".format(optimal_lambda),
                    xy=(optimal_lambda, results[subplot][1][optimal_lambda]), xycoords='data',
                    xytext=(optimal_lambda, results[subplot][1][optimal_lambda] + 0.3), textcoords='data',
                    arrowprops=dict(facecolor='red', arrowstyle='->'))
        ax.legend()
    fig.tight_layout()


def part2a():
    X_train, y_train, X_test, y_test = load_data('hw1/hw1-data')
    plot_part2a(*compute_part2a(500, 3, X_train, y_train, X_test, y_test))
