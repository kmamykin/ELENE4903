import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from scipy import sparse

import pandas as pd
from IPython.display import display, HTML
import dominate
from dominate.tags import *
from dominate.util import raw

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
    # print(np.flip(np.sort(evals)[-5:], axis=0))
    sdist = (eig_vec / np.sum(eig_vec)).reshape((1, -1)) # is a row vector
    assert np.allclose(M.T @ eig_vec, eig_val * eig_vec) # ensure proper eigen vector/value
    assert np.allclose(sdist @ M, sdist) # ensure proper stationary distribution
    return sdist


def top_teams(teams, w, top=25):
    flat_w = w.flatten()
    indices = np.flip(np.argsort(flat_w), axis=0) # in descending order now
    top_indices = indices[:top]
    return np.array(teams)[top_indices], flat_w[top_indices]


def make_report_columns(checkpoints):
    return pd.MultiIndex.from_tuples([("t={}".format(i), l) for i in checkpoints for l in ['team', 'prop']])


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
        norm_p1_trend[t-1] = linalg.norm(w - w_infinity, axis=1, ord=1)
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
    plt.figure(figsize=(12,3))
    plt.plot(experiment['norm_p1_trend'])
    plt.title("Convergence of $w_t$ to $w_{\infty}$")
    plt.xlabel("Iteration $t$")
    plt.ylabel("$||w_t-w_{\infty}||_1$")


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


def divergence_objective(X, W, H):
    x_i, x_j, x_value = X.row, X.col, X.data
    WH = W @ H
    assert x_value.shape == WH[x_i, x_j].shape
    return np.sum(x_value * np.log(1 / WH[x_i, x_j]) + WH[x_i, x_j])


def h_update(X, W, H):
    purple = X.multiply(1 / ((W @ H) + 1e-16))
    pink = W.T / np.sum(W.T, axis=1, keepdims=True)
    return H * (pink @ purple)


def w_update(X, W, H):
    purple = X.multiply(1 / ((W @ H) + 1e-16))
    turquoise = H.T / np.sum(H.T, axis=0, keepdims=True)
    return W * (purple @ turquoise)


def normalize_w_h(W, H):
    a = np.sum(W, axis=0) # 1-D array
    return W / a.reshape((1, -1)), H * a.reshape((-1, 1))


def nonnegative_matrix_factorization(X, rank, iterations):
    N, M = X.shape
    W = np.random.rand(N, rank) + 1.0 # Uniform [1,2)
    H = np.random.rand(rank, M) + 1.0 # Uniform [1,2)
    objectives = np.zeros(iterations)
    for iteration in range(iterations):
        H = h_update(X, W, H)
        W = w_update(X, W, H)
        objectives[iteration] = divergence_objective(X, W, H)
    W, H = normalize_w_h(W, H)
    return W, H, objectives


def top_words(W, vocab, top):
    """
    :param W: (word_idx, topics) matrix
    :param vocab: list of words
    :param top: number of top words to return
    :return: words: (#topics, top) str, weights: (#topics, top) float
    """
    word_idx = np.flip(np.argsort(W, axis=0), axis=0)[:top, :]
    words = np.array(vocab)[word_idx]
    weights = np.flip(np.sort(W, axis=0), axis=0)[:top, :]
    # print(word_idx.shape, words.shape, weights.shape)
    return np.transpose(words), np.transpose(weights)


def nmf_experiment():
    X, vocab = load_word_frequency_matrix()
    W, H, objectives = nonnegative_matrix_factorization(X, rank=25, iterations=100)
    return X, vocab, W, H, objectives


def problem_2_a(X, vocab, W, H, objectives):
    plt.figure(figsize=(12,3))
    plt.plot(np.arange(1, len(objectives)+1), objectives)
    plt.title("Divergence training objective")
    plt.xlabel("Iteration $t$")
    plt.ylabel("$D ( X \| W H )$")


def html_report(cells, word_count, words, weights):
    doc = dominate.document()
    with doc.head:
        with style():
            raw(""".table { width: 100%; display: flex; flex-wrap: wrap; }""")
            raw(""".cell { width: 19%; border: 1px black solid; padding: 1em; box-sizing: border-box; }""")
            raw(""".word { display: flex; justify-content: space-between; }""")
    with doc:
        with div(cls='table'):
            for i in range(cells):
                with div(cls='cell'):
                    for w in range(word_count):
                        with div(cls='word'):
                            span(words[i, w])
                            span('{:.5f}'.format(weights[i,w]))
    return doc.render(pretty=False)


def problem_2_b(X, vocab, W, H, objectives):
    pd.set_option('display.max_colwidth', 1000)
    words, weights = top_words(W, vocab, top=10)
    markup = html_report(25, 10, words, weights)
    # print(markup)
    display(HTML(markup))

    # join = lambda a: ''.join(a)
    # table_cell = lambda i,j: f"""<td align='center'>{ '<br/>'.join(topic_words[i * 5 + j, :])}</td>"""
    # table_row = lambda i: f"""<tr>{ join([table_cell(i,j) for j in range(5)]) }</tr>"""
    # markup = f"""<table border='1' cellpadding='10px'>{(
    #     join([table_row(i) for i in range(5)])
    # )}</table>"""


