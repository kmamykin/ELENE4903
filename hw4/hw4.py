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


def joined_log_likelihood(params, ratings, user_embeddings, movie_embeddings):
    ii = ratings['user_id'].values
    jj = ratings['movie_id'].values
    M = ratings['rating'].values
    log_p_data = (-1/(2*params['sigma']**2)) * np.sum((M - np.sum(user_embeddings[ii] * movie_embeddings[jj], axis=1))**2)
    log_p_prior_users = (-1*params['lambda']/2.0) * np.sum(user_embeddings ** 2)
    log_p_prior_movies = (-1*params['lambda']/2.0) * np.sum(movie_embeddings ** 2)
    return log_p_data + log_p_prior_users + log_p_prior_movies


def predict(rows, cols, user_embeddings, movie_embeddings):
    selected_users = user_embeddings[rows]
    selected_movies = movie_embeddings[cols]
    # element-wise product with a sum == row-wise dot product
    return np.sum(selected_users * selected_movies, axis=1)


def rmse_metric(y_true, y_pred):
    n = y_true.flatten().shape[0]
    return np.sqrt(np.sum((y_pred.flatten() - y_true.flatten()) ** 2) / n)


def similar_movies(movie, movies, movie_embeddings):
    movie_idx = movies.index(movie)
    query_embedding = movie_embeddings[movie_idx].reshape((1, -1))
    # Using Euclidean distance
    # target embedding will be broadcasted to match dims
    # print(movie_embeddings.shape, query_embedding.shape, (movie_embeddings * query_embedding).shape)
    distances = np.sqrt(np.sum((movie_embeddings - query_embedding)**2, axis=1))
    closest_idx = np.argsort(distances, axis=None)
    # print('distances', distances.shape, distances[:5], distances[-5:])
    # print('closest_idx', closest_idx[:5], closest_idx[-5:])
    # print('close/far dist', distances[closest_idx[:5]], distances[closest_idx[-5:]])
    return [movies[i] for i in closest_idx[1:11]] # index 0 is the movie (closest to itself), ignore it


if __name__ == '__main__':
    movies, ratings, ratings_test = load_data('./hw4-data')
    params = { 'sigma': 0.5, 'lambda': 1.0 }
    user_embeddings, movie_embeddings = initialize_matrix_factorization(ratings, d=10)
    for iteration in range(100):
        for i in range(user_embeddings.shape[0]):
            features, targets = select_user_data(ratings, movie_embeddings, i)
            user_embeddings[i] = ridge_regression_solution(features, targets)
        for j in range(movie_embeddings.shape[0]):
            features, targets = select_movie_data(ratings, user_embeddings, j)
            movie_embeddings[j] = ridge_regression_solution(features, targets)
        log_p = joined_log_likelihood(params, ratings, user_embeddings, movie_embeddings)
        predicted_ratings = predict(ratings_test['user_id'].values, ratings_test['movie_id'].values, user_embeddings, movie_embeddings)
        rmse = rmse_metric(ratings_test['rating'].values, predicted_ratings)
        print("Iteration: {}, log likelihood: {:.8f}, RMSE: {:.2f}".format(iteration, log_p, rmse))

    query_movies = [
        'Star Wars (1977)',
        'My Fair Lady (1964)',
        'GoodFellas (1990)'
    ]
    for movie in query_movies:
        print("{}:{}".format(movie, similar_movies(movie, movies, movie_embeddings)))