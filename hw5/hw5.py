import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, svd
import pandas as pd
from IPython.display import display


def read_scores_file(data_dir='./hw5/hw5-data'):
    scores = pd.read_csv(os.path.join(data_dir, 'CFB2017_scores.csv'), header=None)
    with open(os.path.join(data_dir, 'TeamNames.txt'), encoding='utf-8') as f:
        lines = f.read()
    teams = [l for l in lines.strip().split('\n')]
    return np.array(teams), scores


def construct_transition_matrix(scores):
    N_teams = max(scores.iloc[:,0].max(), scores.iloc[:,2].max())
    M = np.zeros((N_teams, N_teams), dtype=np.float)
    for row in scores.itertuples(index=False):
        i, j = row[0] - 1, row[2] - 1 # 1 based indices to 0 based
        points_i, points_j = row[1], row[3]
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
    U, S, _ = svd(M.T)
    return (U[0] / np.sum(U[0]))#.reshape((1, -1))


def top_teams(teams, w, top=25):
    indices = np.argsort(w)
    indices = indices[::-1] # reverse
    print(indices)
    return teams[indices[:top]], w[indices[:top]]
