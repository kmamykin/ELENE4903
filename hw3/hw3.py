import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from numpy.linalg import inv


def load_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'), header=None)
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'), header=None)
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'), header=None)
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'), header=None)
    return X_train.values, y_train.values, X_test.values, y_test.values


# ****************************** Problem 1 ************************************

def accuracy_metric(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    return np.sum(y_pred == y_true) / y_true.shape[0]


def error_metric(y_pred, y_true):
    return 1 - accuracy_metric(y_pred, y_true)


def rmse_metric(y_pred, y_true):
    """Root Mean Square Error metric between predictions and ground truth"""
    assert y_pred.shape == y_true.shape
    n, _ = y_true.shape
    return np.sqrt(np.sum((y_pred - y_true) ** 2) / n)


def kernel(kernel_fn, signature='(i),(i)->()'):
    """
    Fast computation of the covariance matrix using numpy vectorized implementation
    :param kernel_fn: binary function (x1, x2) -> R, where x1 and x2 are vectors
    :param signature: numpy ufunc signature, default is for kernel params taking 1d arrays and producing a scalar
    :return: A function to compute covariance matrix, (X(Nx,d), Y(Ny,d)) -> array (Nx, Ny)
    """
    vectorized_kernel_fn = np.vectorize(kernel_fn, signature=signature)

    def _reshape(x):
        return x if x.ndim > 1 else x[:, np.newaxis]

    def covariance_matrix_computation(X, Y):
        # Ensure X and Y are at least 2 dimensional
        X = _reshape(X)
        Y = _reshape(Y)
        # Get indices for an element wise evaluation of the kernel fn
        x_indices, y_indices = np.meshgrid(np.arange(X.shape[0]), np.arange(Y.shape[0]), indexing='ij')
        return vectorized_kernel_fn(X[x_indices], Y[y_indices])

    return covariance_matrix_computation


def gaussian(b=1):
    """
    Gaussian kernel
    :param b: length scale parameter
    :return: Function to compute covariance matrix
    """
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
    print("Test RMSE: {:.2f}".format(rmse_metric(predictions, y_test)))


def problem_1_part_b():
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw3/hw3-data/gaussian_process')
    bs = [5, 7, 9, 11, 13, 15]
    variances = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    rmses = np.zeros([len(bs), len(variances)])
    for b_index, b in enumerate(bs):
        for v_index, v in enumerate(variances):
            gp = GaussianProcess(X_train, y_train, kernel=gaussian(b=b), sigma=np.sqrt(v))
            predictions = gp.mean(X_test)
            rmses[b_index, v_index] = rmse_metric(predictions, y_test)

    table = pd.DataFrame(rmses, index=["b={}".format(b) for b in bs],
                         columns=["$\sigma^2$={}".format(v) for v in variances])
    display(table)


def problem_1_part_d():
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw3/hw3-data/gaussian_process')
    X_one_dim = np.sort(X_train[:, 4].flatten())  # has to be sorted to make sense on the plot
    gp = GaussianProcess(X_one_dim, y_train, kernel=gaussian(b=5), sigma=np.sqrt(2))
    predictions = gp.mean(X_one_dim).flatten()
    plt.figure(figsize=(16, 8))
    plt.title('Scatterplot of X[:,4],y and GP mean prediction')
    plt.scatter(X_train[:, 4], y_train[:, 0], color='blue', label='Training')
    # plt.scatter(X_test[:, 4], y_test[:, 0], color='red', label='Testing')
    plt.plot(X_one_dim, predictions, color='black')
    plt.legend()

    # std = gp.std(X_one_dim)
    # plt.fill_between(X_one_dim, predictions - 2 * std, predictions + 2 * std, alpha=0.2, color='k')


# ****************************** Problem 2 ************************************

def extend_with_bias(x):
    return np.hstack((x, np.ones((x.shape[0], 1))))


def normalize(a):
    return a / np.sum(a)


def training_error_upper_bound(epsilons):
    return np.exp(-2 * np.sum((0.5 - epsilons) ** 2))


class LeastSquaresClassifier(object):

    @staticmethod
    def fit(X, y):
        """Trains and returns an instance of LS classifier"""
        X = extend_with_bias(X)
        W = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)
        return LeastSquaresClassifier(W)

    def __init__(self, W):
        self.W = W

    def predict(self, X):
        X = extend_with_bias(X)
        return np.sign(np.dot(X, self.W))

    def flip(self):
        """Flips vector W in the opposite direction and returns a LS classifier with that W"""
        return LeastSquaresClassifier(-1 * self.W)


class WeightedBootstrapSampler(object):
    """
    Class with ability to perform weighted sampling for a dataset
    An instance keeps the weighted probabilities of each datapoint in the dataset.
    It also keeps track of the frequencies that each datapoint that was selected over the course of the training process.
    """
    @staticmethod
    def initialize(n, X, y):
        """
        :param n: how many datapoints to sample, we use X's number of points
        :param X: X for the full dataset
        :param y: y for the full dataset
        :return: an instance of sampler
        """
        return WeightedBootstrapSampler(n, normalize(np.ones([X.shape[0]])), X, y, np.array([], dtype=np.int))

    def __init__(self, n, probabilities, X, y, sampled_indices):
        self.n = n
        self.probabilities = probabilities
        self.X = X
        self.y = y
        self.sampled_indices = sampled_indices

    def sample(self):
        """
        Samples the dataset according to the weights distribution for each point
        :return: (Samples from X, Samples from y)
        """
        indices = np.random.choice(np.arange(self.X.shape[0]), size=self.n, p=self.probabilities, replace=True)
        self.sampled_indices = np.append(self.sampled_indices, indices)
        return self.X[indices], self.y[indices]

    def weighted_error(self, misclassified: np.ndarray):
        """Weighted error for misclassified datapoints"""
        return np.sum(self.probabilities[misclassified.flatten()])

    def rescaled(self, factors):
        """Returns the sampler with re-scaled weight probabilities."""
        return WeightedBootstrapSampler(self.n, normalize(self.probabilities * factors.flatten()), self.X, self.y,
                                        self.sampled_indices)

    def histogram(self):
        """
        Aggregates counts of each datapoint at a particular index being sampled.
        :return: (max(sampled_indices)+1) array of times each point was sampled.
        """
        return np.bincount(self.sampled_indices)


class AdaBoostEnsembleClassifier(object):

    def __init__(self, alphas=np.array([]), learners=np.array([])):
        self.alphas = alphas
        self.learners = learners

    def boosted(self, alpha, learner):
        return AdaBoostEnsembleClassifier(np.append(self.alphas, [alpha]), np.append(self.learners, [learner]))

    def predict(self, X):
        # Evaluate predictions of all learners
        learner_predictions = np.hstack([learner.predict(X) for learner in self.learners])
        # predictions will be (N, t), alphas reshaped (t, 1),
        predictions = np.sign(np.dot(learner_predictions, self.alphas.reshape((-1, 1))))
        return predictions


def train_boosted_classifier(T):
    """
    Trains an AdaBoost classifier
    :param T: Number of iterations
    :return: (classifier, DataFrame, sampler)
    """
    X_train, y_train, X_test, y_test = load_data(data_dir='./hw3/hw3-data/boosting')

    progress = pd.DataFrame(columns=['training_error', 'testing_error', 'learner_error', 'epsilon', 'alpha'])
    N, _ = X_train.shape
    sampler = WeightedBootstrapSampler.initialize(N, X_train, y_train)
    classifier = AdaBoostEnsembleClassifier()
    for t in range(T):
        bootstrap_X, bootstrap_y = sampler.sample()
        learner = LeastSquaresClassifier.fit(bootstrap_X, bootstrap_y) # train on bootstrapped data
        learner_predictions = learner.predict(X_train) # predict on the original dataset
        learner_error = error_metric(learner_predictions, y_train)
        epsilon = sampler.weighted_error(learner_predictions != y_train)
        if epsilon > 0.5: # if the learner did poorly, flip it to change all predictions to the opposites
            learner = learner.flip()
            learner_predictions = learner.predict(X_train)
            learner_error = error_metric(learner_predictions, y_train)
            epsilon = sampler.weighted_error(learner_predictions != y_train)

        alpha = 0.5 * np.log((1 - epsilon) / epsilon)
        factors = np.exp(-1 * alpha * y_train * learner_predictions)
        sampler = sampler.rescaled(factors)
        classifier = classifier.boosted(alpha, learner)
        training_error = error_metric(classifier.predict(X_train), y_train)
        testing_error = error_metric(classifier.predict(X_test), y_test)
        values = [training_error, testing_error, learner_error, epsilon, alpha]
        progress.loc[t] = values
    return classifier, progress, sampler


def problem_2_part_a(classifier, progress, sampler):
    plt.figure(figsize=(16, 3))
    plt.plot(progress['training_error'], color='blue', label='Training error')
    plt.plot(progress['testing_error'], color='green', label='Testing error')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.legend()


def problem_2_part_b(classifier, progress, sampler):
    upper_bound = [training_error_upper_bound(progress.loc[:t, 'epsilon'].values) for t in
                   range(progress['epsilon'].size)]
    plt.figure(figsize=(16, 3))
    plt.plot(range(progress['epsilon'].size), upper_bound, color='blue', label='Upper bound')
    plt.legend()
    plt.xlabel('Iteration')
    


def problem_2_part_c(classifier, progress, sampler):
    hist = sampler.histogram()
    plt.figure(figsize=(16, 3))
    markerline, stemlines, baseline = plt.stem(np.arange(hist.shape[0]), hist)
    plt.setp(markerline, visible=False)
    plt.setp(stemlines, color='blue', linewidth=1, linestyle='-')
    plt.setp(baseline, visible=False)
    plt.ylabel('Sampling Counts')
    plt.xlabel('Observation index')


def problem_2_part_d(classifier, progress, sampler):
    plt.figure(figsize=(16, 3))
    plt.plot(progress['epsilon'], color='blue', label='Epsilon')
    plt.plot(progress['alpha'], color='green', label='Alpha')
    plt.legend()
    plt.xlabel('Iteration')


if __name__ == '__main__':
    problem_2_part_a(*train_boosted_classifier(100))
