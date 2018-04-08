import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


# ************************* Problem 1 *******************************


def generate_gaussian_mixture_data(n=5):
    pi = np.array([.2, .5, .3])
    centroids = np.array([[0, 0], [3, 0], [0, 3]])
    cov = np.array([[1, 0], [0, 1]])
    d0 = np.random.multivariate_normal(centroids[0], cov, size=n)
    d1 = np.random.multivariate_normal(centroids[1], cov, size=n)
    d2 = np.random.multivariate_normal(centroids[2], cov, size=n)
    assignment = np.random.choice(3, size=n, p=pi, replace=True)
    data = np.stack((d0, d1, d2))
    return data[assignment, np.arange(n)], centroids


def k_means_loss(data, centroids, assignments):
    return np.sum((data - centroids[assignments, :]) ** 2)


def better_assignments(data, centroids):
    # data is Nx2, centroids are Kx2, Use new dim expansion and broadcast
    # (N,1,2) - (1,K,2) -> (N,K,2)
    coord_diffs = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.argmin(np.sum(coord_diffs ** 2, axis=2), axis=1)


def better_centroids(data, assignments, K):
    centroids = np.zeros((K, 2))
    for k in range(K):
        in_k_cluster = (assignments == k).flatten()
        centroids[k] = np.average(data[in_k_cluster], axis=0)
    return centroids


def learn_k_means(data, K, iterations):
    centroids = np.random.randn(K, 2)
    assignments = np.zeros(data.shape[0])
    loss_history = np.zeros(iterations)
    for iteration in range(iterations):
        assignments = better_assignments(data, centroids)
        centroids = better_centroids(data, assignments, K)
        loss_history[iteration] = k_means_loss(data, centroids, assignments)
    return centroids, assignments, loss_history


def k_means_experiment(n=100, iterations=20):
    data, true_centroids = generate_gaussian_mixture_data(n=n)
    experiment = {'data': data, 'true_centroids': true_centroids, 'K': {}}
    for k in [2, 3, 4, 5]:
        experiment['K'][k] = learn_k_means(data, K=k, iterations=iterations)
    return experiment


def plot_problem1_a(experiment):
    fig = plt.figure(1, figsize=(16, 4))
    for k in [2, 3, 4, 5]:
        _, _, loss_history = experiment['K'][k]
        plt.plot(loss_history, label="K = {}".format(k))
    plt.title("K-means iterations loss")
    plt.xticks(np.arange(20))
    plt.legend()
    plt.show()


def plot_problem1_b(experiment):
    fig = plt.figure(1, figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-3, 6)
    ax.set_ylim(-3, 6)
    ax.grid(linestyle='-', linewidth=1, alpha=0.1)
    ax.scatter(experiment['data'][:, 0], experiment['data'][:, 1], color='black', alpha=0.2, marker='.', linewidths=0.1)
    # ax.scatter(experiment['true_centroids'][:, 0], experiment['true_centroids'][:, 1], marker='+', linewidths=2, label='True')
    for i, k in enumerate([3, 5]):
        centroids, assignments, _ = experiment['K'][k]
        ax.scatter(centroids[:, 0], centroids[:, 1], s=50, marker='o', label="K = {}".format(k))
    plt.legend()
    plt.show()


# ************************* Problem 2 *******************************


def load_data(data_dir='./hw4/hw4-data'):
    with open(os.path.join(data_dir, 'movies.txt'), encoding='utf-8') as f:
        lines = f.read()
    # movies = lines
    movies = [l for l in lines.strip().split('\n')]
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'), header=None)
    # Convert 1-based indices to 0-based
    ratings.iloc[:, :2] = ratings.iloc[:, :2] - 1
    ratings.columns = pd.Index(['user_id', 'movie_id', 'rating'])
    ratings_test = pd.read_csv(os.path.join(data_dir, 'ratings_test.csv'), header=None)
    ratings_test.iloc[:, :2] = ratings_test.iloc[:, :2] - 1
    ratings_test.columns = pd.Index(['user_id', 'movie_id', 'rating'])
    return movies, ratings, ratings_test


def select_movie_data(ratings, user_embeddings, movie_id):
    selection = ratings['movie_id'] == movie_id
    user_ids = ratings.loc[selection, 'user_id']
    targets = ratings.loc[selection, 'rating'].values.reshape((-1, 1))
    return user_embeddings[user_ids], targets


def select_user_data(ratings, movie_embeddings, user_id):
    selection = ratings['user_id'] == user_id
    movie_ids = ratings.loc[selection, 'movie_id']
    targets = ratings.loc[selection, 'rating'].values.reshape((-1, 1))
    return movie_embeddings[movie_ids], targets


def ridge_regression_solution(X, y):
    sigma_aquared = 0.25
    l = 1.0
    N, D = X.shape
    diag = l * sigma_aquared * np.identity(D, dtype=np.float)
    return np.dot(np.dot(np.linalg.inv(diag + np.dot(X.T, X)), X.T), y).flatten()


def predict(rows, cols, user_embeddings, movie_embeddings):
    selected_users = user_embeddings[rows]
    selected_movies = movie_embeddings[cols]
    # element-wise product with a sum == row-wise dot product
    return np.sum(selected_users * selected_movies, axis=1)


def rmse_metric(y_true, y_pred):
    return np.sqrt(np.average((y_pred.flatten() - y_true.flatten()) ** 2))


class SparseMatrixFactorization(object):

    def __init__(self, ratings, movies, params):
        self.ratings = ratings
        self.movies = movies
        self.params = params
        d = self.params['d']
        n_of_users = ratings.loc[:, 'user_id'].max() + 1
        n_of_movies = ratings.loc[:, 'movie_id'].max() + 1
        # both embeddings start random, so it does not matter from which we start coordinate ascend
        self.user_embeddings = np.random.randn(n_of_users, d)
        # this is a transpose of the shape in lectures, but more convenient for calculations
        self.movie_embeddings = np.random.randn(n_of_movies, d)

    def update(self):
        for i in range(self.user_embeddings.shape[0]):
            features, targets = select_user_data(self.ratings, self.movie_embeddings, i)
            self.user_embeddings[i] = ridge_regression_solution(features, targets)
        for j in range(self.movie_embeddings.shape[0]):
            features, targets = select_movie_data(self.ratings, self.user_embeddings, j)
            self.movie_embeddings[j] = ridge_regression_solution(features, targets)
        return self

    def train(self, iterations=100):
        history = np.zeros(iterations)
        for iteration in range(iterations):
            self.update()
            log_p = self.log_likelihood()
            history[iteration] = log_p
        return history[-1], history

    def log_likelihood(self):
        ii = self.ratings['user_id'].values
        jj = self.ratings['movie_id'].values
        M = self.ratings['rating'].values
        log_p_data = (-1 / (2 * self.params['sigma'] ** 2)) * np.sum(
            (M - np.sum(self.user_embeddings[ii] * self.movie_embeddings[jj], axis=1)) ** 2)
        log_p_prior_users = (-1 * self.params['lambda'] / 2.0) * np.sum(self.user_embeddings ** 2)
        log_p_prior_movies = (-1 * self.params['lambda'] / 2.0) * np.sum(self.movie_embeddings ** 2)
        return log_p_data + log_p_prior_users + log_p_prior_movies

    def rmse(self, ratings_test):
        predicted_ratings = predict(ratings_test['user_id'].values, ratings_test['movie_id'].values,
                                    self.user_embeddings, self.movie_embeddings)
        return rmse_metric(ratings_test['rating'].values, predicted_ratings)

    def similar_movies(self, movie, first=10):
        movie_idx = self.movies.index(movie)
        query_embedding = self.movie_embeddings[movie_idx].reshape((1, -1))
        # Using Euclidean distance
        # target embedding will be broadcasted to match dims
        distances = np.sqrt(np.sum((self.movie_embeddings - query_embedding) ** 2, axis=1))
        closest_idx = np.argsort(distances, axis=None)
        return [self.movies[i] for i in closest_idx[1:first + 1]]  # index 0 is the movie (closest to itself), ignore it


def factorize_ratings(runs=10, iterations=100):
    movies, ratings, ratings_test = load_data('./hw4/hw4-data')
    params = {'sigma': 0.5, 'lambda': 1.0, 'd': 10}
    curves = pd.DataFrame()
    final_stats = pd.DataFrame(columns=['train objective', 'test RMSE'])
    models = []
    for run in range(runs):
        model = SparseMatrixFactorization(ratings, movies, params)
        objective, training_history = model.train(iterations=iterations)
        rmse = model.rmse(ratings_test)
        curves[run] = pd.Series(training_history)
        final_stats.loc[run] = [objective, rmse]
        models.append(model)

        # print("run: {} completed. LL: {:.8f}, RMSE: {:.8f}".format(run, objective, rmse))
    # print("Best run: {}".format(final_stats['train objective'].argmax()))
    best_model = models[final_stats['train objective'].argmax()]
    return best_model, curves, final_stats.sort_values(by=['train objective'], ascending=False)


def problem2_a1(model, curves, final_stats):
    plt.figure(figsize=(16, 5))
    for i in range(len(curves.columns)):
        plt.plot(curves.iloc[1:, i], label='run ' + str(i))
    plt.ylabel('Train objective')
    plt.xlabel('Iteration')
    plt.legend()


def problem2_a2(model, curves, final_stats):
    display(final_stats)


def problem2_b(model, curves, final_stats):
    query_movies = [
        'Star Wars (1977)',
        'My Fair Lady (1964)',
        'GoodFellas (1990)'
    ]
    df = pd.DataFrame()
    for movie in query_movies:
        df[movie] = pd.Series(model.similar_movies(movie, first=10))
        # print("{}:{}".format(movie, model.similar_movies(movie, first=10)))
    display(df)


if __name__ == '__main__':
    pass
