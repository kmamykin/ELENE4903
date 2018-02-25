import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from numpy.linalg import inv
from scipy.special import expit  # This is only used to check our implementation of sigmoid
from scipy import signal


def load_data(data_dir='./hw2/hw2-data'):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'), header=None)
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'), header=None)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), header=None)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), header=None)
    return X_train.values, y_train.values, X_test.values, y_test.values


def smooth(x, window_size=10):
    """
    Smooth a series with a moving average (simple). Smoothing over the trailing values.
    :param x: (n,)
    :param window_size: int, size of the smoothing window
    :return: (n,)
    """
    window = signal.boxcar(window_size)  # using simple average
    window = window / np.sum(window)  # normalize the window so we don't change the scale of convolved series
    averaged = signal.convolve(x, window, mode='valid') # convolved series will be smaller when using valid mode
    return np.pad(averaged, pad_width=(window_size-1, 0), mode='edge')  # pad averaged on the left


def class_selectors(y):
    """
    Return indices useful to select instances of different classes from the array of features
    :param y: array of class labels shaped (n, 1) with values {0,1}
    :return: A tuple of two boolean array to select class 0 and class 1 from features, both shaped (n)
    """
    return (y == 0).flatten(), (y == 1).flatten()


def bern_log_p(thetas, features):
    """
    Log likelihood of Bernoulli distributed features with given thetas
    :param thetas: Bernoulli parameter for each feature, shaped (1, n_of_features)
    :param features: shaped (n, n_of_features)
    :return: log likelihood of each instance/row, shape (n, 1)
    """
    return np.sum(features * np.log(thetas) + (1 - features) * np.log(1 - thetas), axis=1)


def pareto_log_p(thetas, features):
    """
    Log likelihood of features distributed with Pareto distribution given thetas
    :param thetas: numpy array shaped (1, n_of_features)
    :param features: numpy array shaped (n, n_of_features)
    :return: log likelihood for each instance/row, shape (n, 1)
    """
    return np.sum(np.log(thetas) - (thetas + 1.) * np.log(features), axis=1)


class NaiveBayes(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        bern_feat, pareto_feat = np.split(X, [54], axis=1)
        # print(bern_feat.shape, pareto_feat.shape)
        class_0, class_1 = class_selectors(y)

        self.pi = np.array([np.sum(class_0), np.sum(class_1)]) / (np.sum(class_0) + np.sum(class_1))
        # print('pi', self.pi)

        bern_theta = np.zeros([2, 54])
        bern_theta[0] = np.sum(bern_feat[class_0, :], axis=0) / np.sum(class_0)
        bern_theta[1] = np.sum(bern_feat[class_1, :], axis=0) / np.sum(class_1)
        # print('bern_theta', bern_theta.shape)
        self.bern_theta = bern_theta

        pareto_theta = np.zeros([2, 3])
        pareto_theta[0] = np.sum(class_0) / np.sum(np.log(pareto_feat[class_0, :]), axis=0)
        pareto_theta[1] = np.sum(class_1) / np.sum(np.log(pareto_feat[class_1, :]), axis=0)
        self.pareto_theta = pareto_theta
        # print('pareto_theta', pareto_theta.shape)
        return self

    def predict(self, X):
        bern_feat, pareto_feat = np.split(X, [54], axis=1)
        y_hat_0_log_p = np.log(self.pi[0]) + bern_log_p(self.bern_theta[0], bern_feat) + pareto_log_p(self.pareto_theta[0], pareto_feat)
        y_hat_1_log_p = np.log(self.pi[1]) + bern_log_p(self.bern_theta[1], bern_feat) + pareto_log_p(self.pareto_theta[1], pareto_feat)
        return ((y_hat_0_log_p - y_hat_1_log_p) < 0).astype(int).reshape((-1, 1))


def accuracy_metric(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.shape[0]


def confusion_matrix(y_true, y_pred):
    return pd.DataFrame(np.array([
        np.count_nonzero(np.logical_and(y_true == 0, y_pred == 0)),
        np.count_nonzero(np.logical_and(y_true == 0, y_pred == 1)),
        np.count_nonzero(np.logical_and(y_true == 1, y_pred == 0)),
        np.count_nonzero(np.logical_and(y_true == 1, y_pred == 1))
    ]).reshape((2, 2)), index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1'])


def problem_2_part_a():
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw2/hw2-data')
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    classifier = NaiveBayes()
    predictions = classifier.fit(X_train, y_train).predict(X_test)
    display(confusion_matrix(y_test, predictions))
    print("Accuracy: {0:.2f}".format(accuracy_metric(y_test, predictions)*100))


def problem_2_part_b():
    """
    Plot a stem plot to display differences in learned Bernoulli parameters for each feature (word)
    :return: None
    """
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw2/hw2-data')
    classifier = NaiveBayes()
    theta = classifier.fit(X_train, y_train).bern_theta
    feat_idx = np.arange(54) + 1
    plt.figure(figsize=(14, 3))

    markerline, stemlines, baseline = plt.stem(feat_idx, theta[0], linefmt=':')
    plt.setp(markerline, color='b')
    plt.setp(stemlines, color='black', linewidth=1, linestyle='-')

    markerline, stemlines, baseline = plt.stem(feat_idx, theta[1], linefmt='-')
    plt.setp(markerline, color='r')
    plt.setp(stemlines, color='black', linewidth=1, linestyle='-')

    plt.xlim(0, 55)

    plt.show()


def l1_distance(x1, x2):
    """
    Calculated distance between a pair of observations.
    Simular observations get a small distance value (large similarity), large distance value means small similarity.
    :param x1:
    :param x2:
    :return: distance (scalar)
    """
    # Ideally we would be to normalize the features across dataset, for this assignment the last 3 features
    # with Pareto distribution may overwhelm {0,1} of the Bernoulli features
    return np.sum(np.abs(x1 - x2))


def fast_cartesian_product_eval(X, Y, binary_fn):
    """
    Performs fast evaluation of binary_fn on a cartesian product pairs of elements in X and Y.
    The elements iterated must be in the outer most dimension.
    :param X: numpy array shape (Nx, ...)
    :param Y: numpy array shape (Ny, ...)
    :param binary_fn: Function taking two parameters shaped like X and Y inner dimensions and returning a scalar
    :return: Matrix (Nx, Ny) with values == binary_fn evaluated on the corresponding elements
    """
    # Convert a python function to something numpy can use to quickly iterate over darrays
    vectorized_binary_fn = np.vectorize(binary_fn, signature='(i),(i)->()')
    x_indices, y_indices = np.meshgrid(np.arange(X.shape[0]), np.arange(Y.shape[0]))
    return vectorized_binary_fn(X[x_indices], Y[y_indices])


def majority_vote(y_votes):
    """
    Get majority vote with breaking ties according to around (i.e. 0.5 will be treated as 0)
    :param y_votes: shaped (n_instances, n_votes) with values {0,1}
    :return: (n_instances,1) with values {0,1}
    """
    return np.around(np.average(y_votes, axis=1)).astype(dtype=np.int)


class KNNClassifier(object):
    def __init__(self, k=1, distance=l1_distance):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        # Just remember the training set
        self.X_train = X
        self.y_train = y
        return self

    def nearest_neighbours(self, X):
        # calculate distance to each training example
        distances = fast_cartesian_product_eval(self.X_train, X, self.distance)
        # sort neighbours by distance
        return np.argsort(distances, axis=1)

    def nearest_label(self, neighbours, k):
        # select top k examples and pick predicted class label
        return majority_vote(self.y_train[neighbours[:, :k]])

    def predict(self, X):
        neighbours = self.nearest_neighbours(self, X)
        return self.nearest_label(neighbours, self.k)


def plot_accuracy(ks, accuracies):
    plt.figure(figsize=(10, 3))
    plt.title('k-NN accuracy vs k')
    plt.plot(ks, accuracies)
    plt.xticks(ks)
    plt.show()


def problem_2_part_c():
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw2/hw2-data')
    ks = np.linspace(start=1, stop=20, num=20, dtype=np.int)
    accuracies = np.zeros(ks.shape)
    classifier = KNNClassifier().fit(X_train, y_train)
    # Optimization: do not need to re-calculate distances matrix for each k
    neighbours = classifier.nearest_neighbours(X_test)
    for i, k in enumerate(ks):
        accuracies[i] = accuracy_metric(y_test, classifier.nearest_label(neighbours, k))
    return ks, accuracies


def sigmoid(x):
    """Numerically-stable sigmoid function."""
    is_positive = x >= 0
    positives = np.abs(x)
    probabilities = 1 / (1 + np.exp(-positives))
    return np.where(is_positive, probabilities, 1 - probabilities)
    # Another implementation tried (it has more operations and slightly less performant)
    # x = np.clip(x, -709, 709)
    # positive = x >= 0
    # positive_sigmoid = np.where(positive, 1 / (1 + np.exp(-x)), 0)
    # negative_sigmoid = np.where(~positive, np.exp(x) / (1 + np.exp(x)), 0)
    # return np.where(positive, positive_sigmoid, negative_sigmoid)


def extend_with_bias(x):
    return np.hstack((x, np.ones((x.shape[0], 1))))


def gradient_learning_rate(iteration):
    return 1.0 / (10**5 * np.sqrt(iteration + 1))


def gradient_optimizer(X, y, W):
    likelihoods = sigmoid(y * np.dot(X, W)) # (n, 1)
    objective = np.sum(np.log(likelihoods + 1e-10)) # sum of log likelihoods (scalar)
    gradient = np.sum((1 - likelihoods) * y * X, axis=0).reshape(W.shape)
    return objective, gradient


def newton_learning_rate(iteration):
    return 1.0 / np.sqrt(iteration + 1)


def newton_optimizer(X, y, W):
    likelihoods = sigmoid(y * np.dot(X, W)) # (n, 1)
    objective = np.sum(np.log(likelihoods + 1e-10)) # sum of log likelihoods (scalar)
    gradient = np.sum((1 - likelihoods) * y * X, axis=0).reshape(W.shape)
    z = sigmoid(np.dot(X, W))
    N, D = X.shape
    hessian = np.zeros((N,D,D))
    for i in range(N):
        x = X[i].reshape([-1,1])
        hessian[i] = z[i]*(1-z[i])*np.dot(x, x.T)
    hessian = -1*np.sum(hessian, axis=0)
    return objective, -1 * np.dot(inv(hessian), gradient)


class LogisticRegression(object):

    def __init__(self, max_iterations, method, progress_callback_fn):
        self.max_iterations = max_iterations
        self.learning_rate_fn = newton_learning_rate if method == 'newton' else gradient_learning_rate
        self.optimizer_fn = newton_optimizer if method == 'newton' else gradient_optimizer
        self.progress_callback_fn = progress_callback_fn

    def fit(self, X_train, y_train):
        # Initialize training process
        X = extend_with_bias(X_train)
        N, D = X.shape
        # Convert y class labels vector from {0,1} to {-1,1}
        y = np.where(y_train == 1, 1, -1)
        self.W = np.zeros((D, 1))  # np.random.randn(D, 1) * 0.01
        # print('y', y)
        # print('y', y.shape)
        # print('X', X.shape)
        # print('W', self.W.shape)
        for iteration in range(self.max_iterations):
            learning_rate = self.learning_rate_fn(iteration)
            # We are maximizing the reward in this homework instead of minimizing the cost as often defined in literature
            # so instead of calling this a cost or loss, we call this an objective
            objective, update = self.optimizer_fn(X, y, self.W)
            # update weights
            self.W = self.W + learning_rate * update
            self.progress_callback_fn(iteration, objective, self)
        return self

    def predict(self, X):
        X = extend_with_bias(X)
        return (sigmoid(np.dot(X, self.W)) > 0.5).astype(dtype=np.int)


def train_logistic_regression(method, max_iterations):
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw2/hw2-data')
    iterations = []
    objectives = []
    train_accuracies = []
    test_accuracies = []

    def record_progress(iteration, objective, classifier):
        iterations.append(iteration)
        objectives.append(objective)
        train_accuracies.append(accuracy_metric(y_train, classifier.predict(X_train)))
        test_accuracies.append(accuracy_metric(y_test, classifier.predict(X_test)))

    classifier = LogisticRegression(
        max_iterations=max_iterations,
        method=method,
        progress_callback_fn=record_progress
    ).fit(X_train, y_train)

    return np.array(iterations), np.array(objectives), np.array(train_accuracies), np.array(test_accuracies), method


def plot_learning_progress(iterations, objectives, train_accuracies, test_accuracies, method):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(211)
    ax1.set_title('Optimization objective')
    ax1.plot(iterations, objectives, label='Objective')
    ax1.legend()
    # print(objectives[-10:])
    smoothing_window = 3 if method == 'newton' else 100
    train_accuracies_smoothed = smooth(train_accuracies, window_size=smoothing_window)
    test_accuracies_smoothed = smooth(test_accuracies, window_size=smoothing_window)
    ax2 = plt.subplot(212)
    ax2.set_title('Accuracies')
    ax2.plot(iterations, 100*train_accuracies_smoothed, label='Train accuracy (smoothed)')
    # ax2.plot(iterations[s], 100*train_accuracies, label='Train accuracy')
    ax2.plot(iterations, 100*test_accuracies_smoothed, label='Test accuracy (smoothed)')
    # ax2.plot(iterations[s], 100*test_accuracies, label='Test accuracy')
    ax2.annotate("Train: {0:.2f}\nTest: {0:.2f}".format(train_accuracies_smoothed[-1]*100, test_accuracies_smoothed[-1]*100),
                 xy=(.5, .5),
                 xycoords='axes fraction',
                 arrowprops=None)
    ax2.set_xlabel('Iteration')
    ax2.legend()
    plt.show()


def problem_2_part_d():
    plot_learning_progress(*train_logistic_regression('gradient', 10000))


def problem_2_part_e():
    plot_learning_progress(*train_logistic_regression('newton', 100))


class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.inputs = np.array([-710, -709, -10., -6., -2., 0, 2., 6., 10., 709, 710])
        self.expected = np.array([0, 0, 0.000045, 0.002473, 0.119203, 0.5, 0.880797, 0.9975274, 0.9999546, 1.0, 1.0])

    @unittest.skip('')
    def test_sigmoid_with_expit(self):
        np.testing.assert_allclose(sigmoid(self.inputs), expit(self.inputs), atol=1e-06)

    @unittest.skip('')
    def test_sigmoid_for_known_values(self):
        np.testing.assert_allclose(sigmoid(self.inputs), self.expected, atol=1e-06)

    @unittest.skip('')
    def test_expit_for_known_values(self):
        np.testing.assert_allclose(expit(self.inputs), self.expected, atol=1e-06)

    def test_newton_optimizer(self):
        X = np.array([
            [1., 2.],
            [2., 1.],
            [1.1, 2.1],
            [2.1, 1.1]])
        y = np.array([1, -1, 1, -1]).reshape([-1, 1])
        W = np.array([1., 1.]).reshape([-1, 1])
        # print(X.shape, y.shape, W.shape)
        objective, update = newton_optimizer(X, y, W)
        np.testing.assert_allclose(objective, -6.3770, atol=1e-4)
        np.testing.assert_allclose(update, [[ 15.652592],[-8.499538]], atol=1e-4)

if __name__ == '__main__':
    unittest.main()
