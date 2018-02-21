import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def load_data(data_dir='./hw2/hw2-data'):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'), header=None)
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'), header=None)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), header=None)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), header=None)
    return X_train.values, y_train.values, X_test.values, y_test.values


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
        pareto_theta[0] = np.sum(class_0)/np.sum(np.log(pareto_feat[class_0, :]), axis=0)
        pareto_theta[1] = np.sum(class_1)/np.sum(np.log(pareto_feat[class_1, :]), axis=0)
        self.pareto_theta = pareto_theta
        # print('pareto_theta', pareto_theta.shape)
        return self

    def predict(self, X):
        bern_feat, pareto_feat = np.split(X, [54], axis=1)
        y_hat_0_log_p = np.log(self.pi[0]) + bern_log_p(self.bern_theta[0], bern_feat) + pareto_log_p(self.pareto_theta[0], pareto_feat)
        y_hat_1_log_p = np.log(self.pi[1]) + bern_log_p(self.bern_theta[1], bern_feat) + pareto_log_p(self.pareto_theta[1], pareto_feat)
        return ((y_hat_0_log_p - y_hat_1_log_p) < 0).astype(int).reshape((-1, 1))


def accuracy_metric(y_true, y_pred):
    return 100.0 * np.sum(y_true == y_pred) / y_true.shape[0]


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
    print("Accuracy: {0:.2f}".format(accuracy_metric(y_test, predictions)))



def problem_2_part_b():
    """
    Plot a stem plot to display differences in learned Bernoulli parameters for each feature (word)
    :return: None
    """
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw2/hw2-data')
    classifier = NaiveBayes()
    theta = classifier.fit(X_train, y_train).bern_theta
    feat_idx = np.arange(54) + 1
    fig = plt.figure(figsize=(14, 3))

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
    # Here we are disregarding the features that follow Pareto distribution, they are scaled differently
    # and kNN performs just fine without them. Possible improvement would be to normalize those features across dataset
    return np.sum(np.abs(x1[:54]-x2[:54]))


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
        self.distance=distance

    def fit(self, X, y):
        # Just remember the training set
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        # calculate distance to each training example
        distances = fast_cartesian_product_eval(self.X_train, X, self.distance)
        # sort and select top k examples
        top_k_indices = np.argsort(distances, axis=1)[:,:self.k]
        # pick predicted class label and break ties
        return majority_vote(self.y_train[top_k_indices])


def plot_accuracy(ks, accuracies):
    plt.figure(figsize=(10, 3))
    plt.title('k-NN accuracy vs k')
    plt.plot(ks, accuracies)
    plt.show()


def problem_2_part_c():
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw2/hw2-data')
    ks = np.linspace(start=1, stop=20, num=20, dtype=np.int)
    accuracies = np.zeros(ks.shape)
    for i, k in enumerate(ks):
        predictions = KNNClassifier(k=k).fit(X_train, y_train).predict(X_test)
        accuracies[i] = accuracy_metric(y_test, predictions)
    return ks, accuracies
