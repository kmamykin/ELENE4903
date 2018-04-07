import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from numpy.linalg import inv
from scipy import sparse


def load_data(data_dir='./hw4/hw4-data'):
    with open(os.path.join(data_dir, 'movies.txt'), encoding='utf-8') as f:
        lines = f.read()
    # movies = lines
    movies = [l for l in lines.strip().split('\n')]
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'), header=None)
    # Convert 1-based indices to 0-based
    ratings.iloc[:,:2] = ratings.iloc[:,:2] - 1
    ratings.columns = pd.Index(['user_id', 'movie_id', 'rating'])
    ratings_test = pd.read_csv(os.path.join(data_dir, 'ratings_test.csv'), header=None)
    ratings_test.iloc[:,:2] = ratings_test.iloc[:,:2] - 1
    ratings_test.columns = pd.Index(['user_id', 'movie_id', 'rating'])
    return movies, ratings, ratings_test


def initialize_matrix_factorization(ratings, d=10):
    n_of_users = ratings.loc[:, 'user_id'].max() + 1
    n_of_movies = ratings.loc[:, 'movie_id'].max() + 1
    # both embeddings start random, so it does not matter from which we start coordinate ascend
    user_embeddings = np.random.randn(n_of_users, d)
    movie_embeddings = np.random.randn(n_of_movies, d) # this is a transpose of the shape in lectures, but more convenient for calculations
    return user_embeddings, movie_embeddings


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
        self.movie_embeddings = np.random.randn(n_of_movies, d) # this is a transpose of the shape in lectures, but more convenient for calculations

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
        log_p_data = (-1 / (2 * self.params['sigma'] ** 2)) * np.sum((M - np.sum(self.user_embeddings[ii] * self.movie_embeddings[jj], axis=1)) ** 2)
        log_p_prior_users = (-1 * self.params['lambda'] / 2.0) * np.sum(self.user_embeddings ** 2)
        log_p_prior_movies = (-1 * self.params['lambda'] / 2.0) * np.sum(self.movie_embeddings ** 2)
        return log_p_data + log_p_prior_users + log_p_prior_movies

    def rmse(self, ratings_test):
        predicted_ratings = predict(ratings_test['user_id'].values, ratings_test['movie_id'].values, self.user_embeddings, self.movie_embeddings)
        return rmse_metric(ratings_test['rating'].values, predicted_ratings)

    def similar_movies(self, movie, first=10):
        movie_idx = self.movies.index(movie)
        query_embedding = self.movie_embeddings[movie_idx].reshape((1, -1))
        # Using Euclidean distance
        # target embedding will be broadcasted to match dims
        distances = np.sqrt(np.sum((self.movie_embeddings - query_embedding) ** 2, axis=1))
        closest_idx = np.argsort(distances, axis=None)
        return [self.movies[i] for i in closest_idx[1:first+1]]  # index 0 is the movie (closest to itself), ignore it


def factorize_ratings():
    movies, ratings, ratings_test = load_data('./hw4/hw4-data')
    params = { 'sigma': 0.5, 'lambda': 1.0, 'd': 10 }
    curves = pd.DataFrame()
    final_stats = pd.DataFrame(columns=['train objective', 'test RMSE'])
    models = []
    for run in range(10):
        model = SparseMatrixFactorization(ratings, movies, params)
        objective, training_history = model.train(iterations=100)
        rmse = model.rmse(ratings_test)
        curves[run] = pd.Series(training_history)
        final_stats.loc[run] = [objective, rmse]
        models.append(model)

        print("run: {} completed. LL: {:.8f}, RMSE: {:.8f}".format(run, objective, rmse))
    print("Best run: {}".format(final_stats['train objective'].argmax()))
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