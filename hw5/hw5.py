import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from scipy import sparse

import pandas as pd
from IPython.display import display


def read_scores_file(data_dir='./hw5/hw5-data'):
    scores = pd.read_csv(os.path.join(data_dir, 'CFB2017_scores.csv'), header=None)
    with open(os.path.join(data_dir, 'TeamNames.txt'), encoding='utf-8') as f:
        lines = f.read()
    teams = [l for l in lines.strip().split('\n')]
    return teams, scores


def construct_transition_matrix(scores):
    N_teams = max(scores.iloc[:,0].max(), scores.iloc[:,2].max())
    M = np.zeros((N_teams, N_teams), dtype=np.float)
    for row in scores.itertuples(index=False):
        i, points_i, j, points_j = row
        i, j = i - 1, j - 1 # 1 based indices to 0 based
        assert points_i != points_j, 'No ties allowed in the tournament!'
        team_a_win_bonus = 1. if points_i > points_j else 0
        team_b_win_bonus = 1. if points_i < points_j else 0
        M[i,i] = M[i,i] + team_a_win_bonus + points_i/(points_i+points_j)
        M[j,j] = M[j,j] + team_b_win_bonus + points_j/(points_i+points_j)
        M[i,j] = M[i,j] + team_b_win_bonus + points_j/(points_i+points_j)
        M[j,i] = M[j,i] + team_a_win_bonus + points_i/(points_i+points_j)
    normalizing_row_sum = np.sum(M, axis=1).reshape((-1, 1))
    return M/normalizing_row_sum


def stationary_distribution(M):
    # linalg.eig is computing right eigenvectors
    # state probabilities are multiplied from the left (probs @ M)
    # So we need to transpose M before taking eigen decomposition
    evals, evecs = linalg.eig(M.T)
    max_eval_idx = np.argmax(evals) # eigenvalues are not sorted
    eig_vec = np.real(evecs[:, max_eval_idx]).reshape((-1, 1)) # is a column vector
    eig_val = np.real(evals[max_eval_idx])
    sdist = (eig_vec / np.sum(eig_vec)).reshape((1, -1)) # is a row vector
    assert np.allclose(M.T @ eig_vec, eig_val * eig_vec) # ensure proper eigen vector/value
    assert np.allclose(sdist @ M, sdist) # ensure proper stationary distribution
    return sdist


def top_teams(teams, w, top=25):
    flat_w = w.flatten()
    indices = np.argsort(flat_w)
    indices = indices[::-1] # reverse
    top_indices = indices[:top]
    return np.array(teams)[top_indices], flat_w[top_indices]


def make_report_columns(checkpoints):
    return pd.MultiIndex.from_tuples([("t={}".format(i), l) for i in checkpoints for l in ['teams', 'p']])


def markov_chain_experiment():
    teams, scores = read_scores_file(data_dir='./hw5/hw5-data')
    M = construct_transition_matrix(scores)
    N, _ = M.shape
    w_infinity = stationary_distribution(M)
    checkpoints = [10, 100, 1000, 10000]
    max_iterations = checkpoints[-1]
    w = np.ones((1, N)) / N # Initial uniform distribution
    team_rankings = pd.DataFrame(columns=make_report_columns(checkpoints))
    norm_p1_trend = np.zeros(max_iterations)
    for t in range(1, max_iterations+1):
        w = w @ M
        assert np.allclose(np.sum(w), 1.0)
        norm_p1_trend[t-1] = linalg.norm(w - w_infinity, ord=1)
        if t in checkpoints:
            checkpoint_index = checkpoints.index(t)
            check_teams, check_w = top_teams(teams, w, top=25)
            team_rankings.iloc[:, checkpoint_index * 2] = check_teams
            team_rankings.iloc[:, checkpoint_index * 2 + 1] = check_w

    # print(top_teams(teams, w, top=25))
    return dict(team_rankings=team_rankings, norm_p1_trend=norm_p1_trend)


def problem_1_a(experiment):
    display(experiment['team_rankings'])


def problem_1_b(experiment):
    plt.plot(experiment['norm_p1_trend'])



# ************************** PROBLEM 1 (Non-negative Matrix Factorization) ******************


def load_word_frequency_matrix(data_dir='./hw5/hw5-data'):
    split_line = lambda line: [tuple(values.split(':')) for values in line.split(',')]
    with open(os.path.join(data_dir, 'nyt_data.txt'), encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    a = [[int(word_cound), int(word_idx) - 1, doc_idx] for doc_idx, line in enumerate(lines) for word_idx, word_cound in split_line(line)]
    data = np.array(a)
    matrix = sparse.coo_matrix((data[:, 0], (data[:, 1], data[:, 2])))
    with open(os.path.join(data_dir, 'nyt_vocab.dat'), encoding='utf-8') as f:
        vocab = f.read().strip().split('\n')
    return matrix, vocab


def plot_histogram(W, H):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(W, bins=100)
    ax2.hist(H, bins=100)
    plt.show()

def divergence_objective(X, W, H):
    iidx, jidx, x = X.row, X.col, X.data
    wh = W @ H
    print(x.shape, wh[iidx,jidx].shape)
    print(x[:5], wh[iidx,jidx][:5])
    plot_histogram(W, H)
    return np.sum(x*np.log(1/wh[iidx,jidx]) + wh[iidx,jidx])


def nonnegative_matrix_factorization(X, rank, iterations):
    N, M = X.shape
    W = np.random.rand(N, rank)# + 1.0 # Uniform [1,2)
    H = np.random.rand(rank, M)# + 1.0 # Uniform [1,2)
    objectives = np.zeros(iterations)
    for iteration in range(iterations):
        purple = X.multiply(1 / (W @ H))
        pink = W.T / np.sum(W.T, axis=1, keepdims=True)
        H = H * (pink @ purple)
        turquoise = H.T / np.sum(H.T, axis=0, keepdims=True)
        W = W * (purple @ turquoise)
        objectives[iteration] = divergence_objective(X, W, H)
        print(objectives[iteration])
    return W, H, objectives


def normalize_w_h(W, H):
    a = np.sum(W, axis=0) # 1-D array
    return W / a.reshape((1, -1)), H * a.reshape((-1, 1))


def top_words_in_topics(W, top):
    return np.flip(np.argsort(W, axis=0)[-top:,:], axis=0)


def nmf_experiment():
    X, vocab = load_word_frequency_matrix()
    W, H, objectives = nonnegative_matrix_factorization(X, rank=25, iterations=10)
    W, H = normalize_w_h(W, H)
    topic_words_idx = top_words_in_topics(W, top=10)
    topic_words = np.array(vocab)[topic_words_idx]
    return X, W, H, objectives, topic_words


def problem_2_a(X, W, H, objectives, topic_words):
    plt.plot(objectives)


def problem_2_b(X, W, H, objectives, topic_words):
    print(topic_words)