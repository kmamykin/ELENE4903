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


def confusion_matrix(y_true, y_pred):
    return pd.DataFrame(np.array([
        np.count_nonzero(np.logical_and(y_true == 0, y_pred == 0)),
        np.count_nonzero(np.logical_and(y_true == 0, y_pred == 1)),
        np.count_nonzero(np.logical_and(y_true == 1, y_pred == 0)),
        np.count_nonzero(np.logical_and(y_true == 1, y_pred == 1))
    ]).reshape((2, 2)), index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1'])


def display_confusion_matrix(confusion_matrix):
    display(confusion_matrix)
    accuracy = 100.0 * np.sum(confusion_matrix.values * np.eye(2)) / np.sum(confusion_matrix.values)
    print("Accuracy: {0:.2f}".format(accuracy))


def problem_2_part_a():
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw2/hw2-data')
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    classifier = NaiveBayes()
    predictions = classifier.fit(X_train, y_train).predict(X_test)
    display_confusion_matrix(confusion_matrix(y_test, predictions))


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
