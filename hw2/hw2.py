import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from IPython.display import display, HTML


def load_data(data_dir = './hw2/hw2-data'):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'), header=None)
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'), header=None)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), header=None)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), header=None)
    return X_train.values, y_train.values, X_test.values, y_test.values


def class_indices(y):
    class_0_indices, _ = np.where(y == 0)
    class_1_indices, _ = np.where(y == 1)
    return class_0_indices, class_1_indices


def class_selectors(y):
    return (y == 0).flatten(), (y == 1).flatten()


class NaiveBayes(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        bern_feat, pareto_feat = np.split(X, [54], axis=1)
        print(bern_feat.shape, pareto_feat.shape)
        class_0, class_1 = class_selectors(y)

        self.pi = np.array([np.sum(class_0), np.sum(class_1)])/(np.sum(class_0) + np.sum(class_1))
        print('pi', self.pi)

        bern_theta = np.zeros([2, 54])
        bern_theta[0] = np.sum(bern_feat[class_0,:], axis=0)/np.sum(class_0)
        bern_theta[1] = np.sum(bern_feat[class_1,:], axis=0)/np.sum(class_1)
        print('bern_theta', bern_theta.shape)
        self.bern_theta = bern_theta
        return self

    def predict(self, X):
        bern_feat, pareto_feat = np.split(X, [54], axis=1)
        # y_hat = np.zeros([X.shape[0], 2])
        y_hat_0 = self.pi[0] * np.prod((self.bern_theta[0]**bern_feat)*(1-self.bern_theta[0])**(1-bern_feat), axis=1)
        print('y_hat_0.shape', y_hat_0.shape)
        # print('y_hat_0', y_hat_0)
        y_hat_1 = self.pi[1] * np.prod((self.bern_theta[1]**bern_feat)*(1-self.bern_theta[1])**(1-bern_feat), axis=1)
        # print('y_hat_1', y_hat_1)
        return (np.log(y_hat_0/y_hat_1) < 0).astype(int).reshape((-1, 1))


def confusion_matrix(y_true, y_pred):
    return pd.DataFrame(np.array([
        np.count_nonzero(np.logical_and(y_true == 0, y_pred == 0)),
        np.count_nonzero(np.logical_and(y_true == 0, y_pred == 1)),
        np.count_nonzero(np.logical_and(y_true == 1, y_pred == 0)),
        np.count_nonzero(np.logical_and(y_true == 1, y_pred == 1))
    ]).reshape((2,2)), index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1'])


def display_confusion_matrix(confusion_matrix):
    display(confusion_matrix)
    # display(HTML(confusion_matrix.to_html()))
    accuracy = np.sum(confusion_matrix.values*np.eye(2))/np.sum(confusion_matrix.values)
    print("Accuracy: {0:.2f}".format(accuracy))


def problem_2_part_a():
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw2/hw2-data')
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    classifier = NaiveBayes()
    predictions = classifier.fit(X_train, y_train).predict(X_test)
    # print(predictions, y_test)
    display_confusion_matrix(confusion_matrix(y_test, predictions))
