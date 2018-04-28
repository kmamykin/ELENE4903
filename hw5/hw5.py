import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg

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


def markov_chain_experiment():
    teams, scores = read_scores_file(data_dir='./hw5/hw5-data')
    M = construct_transition_matrix(scores)
    N, _ = M.shape
    w_infinity = stationary_distribution(M)
    checkpoint_iterations = [10, 100, 1000, 10000]
    max_iterations = checkpoint_iterations[-1]
    w = np.ones((1, N)) / N # Initial uniform distribution
    team_rankings = pd.DataFrame(columns=["t={}".format(i) for i in checkpoint_iterations])
    norm_p1_trend = np.zeros(max_iterations)
    for t in range(1, max_iterations+1):
        w = w @ M
        assert np.allclose(np.sum(w), 1.0)
        norm_p1_trend[t-1] = linalg.norm(w - w_infinity, ord=1)
        if t in checkpoint_iterations:
            column = checkpoint_iterations.index(t)
            team_rankings.iloc[:,column] = top_teams(teams, w, top=25)[0]

    # print(top_teams(teams, w, top=25))
    return dict(team_rankings=team_rankings, norm_p1_trend=norm_p1_trend)


def problem_1_a(experiment):
    display(experiment['team_rankings'])


def problem_1_b(experiment):
    plt.plot(experiment['norm_p1_trend'])
