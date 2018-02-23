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
    # Ideally we would be to normalize the features across dataset, for this assignment the last 3 features
    # with Pareto distribution may overwhelm {0,1} of the Bernoulli features
    return np.sum(np.abs(x1-x2))


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

    def nearest_neighbours(self, X):
        # calculate distance to each training example
        distances = fast_cartesian_product_eval(self.X_train, X, self.distance)
        # sort neighbours by distance
        return np.argsort(distances, axis=1)

    def nearest_label(self, neighbours, k):
        # select top k examples and pick predicted class label
        return majority_vote(self.y_train[neighbours[:,:k]])

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


# def sigmoid(x):
#     return np.exp(x)/(1 + np.exp(x))
def sigmoid(x):
    "Numerically-stable sigmoid function."
    pos = np.where(x >=0, x, 0)
    pos = 1 / (1 + np.exp(-pos))
    # if x is less than zero then z will be small, denom can't be
    # zero because it's 1+z.
    neg = np.where(x < 0, x, 0)
    z = np.exp(neg)
    neg = z / (1 + z)
    return pos + neg

def extend_with_bias(x):
    return np.hstack((x, np.ones((x.shape[0], 1))))


class LogisticRegression(object):

    def __init__(self, max_iterations, learning_rate_fn, progress_callback_fn):
        self.max_iterations = max_iterations
        self.learning_rate_fn = learning_rate_fn
        self.progress_callback_fn = progress_callback_fn

    def fit(self, X_train, y_train):
        # Initialize training process
        X = extend_with_bias(X_train)
        N, D = X.shape
        y = np.where(y_train == 1, 1, -1)
        self.W = np.random.randn(D, 1)#np.zeros((D, 1))
        print('y', y)
        print('y', y.shape)
        print('X', X.shape)
        print('W', self.W.shape)
        for iteration in range(self.max_iterations):
            learning_rate = self.learning_rate_fn(iteration)
            likelihoods = y*sigmoid(np.dot(X, self.W))
            # print('likelihood', likelihoods.shape)
            loss = np.sum(np.log(likelihoods)) # sum of log likelihoods
            gradient = np.sum((1 - likelihoods) * y * X, axis=0).reshape(self.W.shape)
            # print('gradient', (1 - likelihoods).shape, ((1 - likelihoods) * y).shape, ((1 - likelihoods) * y * X).shape, gradient.shape)
            # update weights
            self.W = self.W + learning_rate * gradient
            self.progress_callback_fn(iteration, loss, self)
        return self

    def predict(self, X):
        X = extend_with_bias(X)
        return (sigmoid(np.dot(X, self.W)) > 0.5).astype(dtype=np.int)


def problem_2_part_d():
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw2/hw2-data')
    iterations = []
    losses = []
    train_accuracies = []
    test_accuracies = []

    def learning_rate(iteration):
        return 1.0/(10e5*np.sqrt(iteration+1))

    def record_progress(iteration, loss, classifier):
        iterations.append(iteration)
        losses.append(loss)
        train_accuracies.append(accuracy_metric(y_train, classifier.predict(X_train)))
        test_accuracies.append(accuracy_metric(y_test, classifier.predict(X_test)))

    classifier = LogisticRegression(
        max_iterations=10000,
        learning_rate_fn=learning_rate,
        progress_callback_fn=record_progress
    ).fit(X_train, y_train)

    return iterations, losses, train_accuracies, test_accuracies


def plot_learning_progress(iterations, losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(10, 4))
    plt.title('Learning progress')
    plt.plot(iterations, losses, label='Loss')
    plt.plot(iterations, train_accuracies, label='Train accuracy')
    plt.plot(iterations, test_accuracies, label='Test accuracy')
    plt.xlabel('Iteration')
    plt.legend()
    plt.show()
