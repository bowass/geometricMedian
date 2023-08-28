import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import torch
from geom_median.torch import compute_geometric_median


def generate_data(shape=(100, 2), percent_outliers=0.2) -> np.ndarray:
    n, d = shape
    A, _ = make_blobs(n_samples=[int(n*(1 - percent_outliers)), int(n*(0.5*percent_outliers)), int(n*(0.5*percent_outliers))],
                      centers=None, n_features=d, random_state=0, cluster_std=0.4)
    return A


def plot_data(A: np.ndarray, median: np.ndarray, step=1, show_data=False) -> None:
    """
    Input:
        1. A: nXd matrix
        2. array (or list) of medians in R^d
        3. step the step size for the medians (plot every $step median)
        4. show_data = False <=> plot only medians
    Output: plots.
    """
    n, d = A.shape
    if d == 1:
        if show_data:
            plt.scatter(A[:, 0], label='data')
        for i in range(0, len(median), step):
            plt.scatter(median[i][0], label=f'median {i}')
    elif d == 2:
        if show_data:
            plt.scatter(A[:, 0], A[:, 1], label='data')
        # plt.plot(median[:, 0], median[:, 1], '-o', label=f'median')
        for i in range(0, len(median), step):
            plt.scatter(median[i][0], median[i][1], label=f'median {i}')
    elif d == 3:
        if show_data:
            plt.scatter(A[:, 0], A[:, 1], A[:, 2], label='data')
        for i in range(0, len(median), step):
            plt.scatter(median[i][0], plotting the mediansmedian[i][1], median[i][2], label=f'median {i}')
    else:
        raise ValueError("Required d<=3")
    plt.legend((['data'] if show_data else []) + [f'median {i}' for i in range(0, len(median), step)])
    plt.show()
