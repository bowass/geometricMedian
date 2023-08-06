import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import torch
from geom_median.torch import compute_geometric_median

def generate_data(shape=(100, 2), percent_outliers=0.2) -> np.ndarray:
    """
    implementing in a clearer way
    GAUSSIAN NOISE
    """
    # noise = np.random.normal(mu, )
    #
    #
    n, d = shape
    # n_outlier = int((n * percent_outliers))
    # n -= n_outlier
    # std = var * np.random.rand(n, d) * np.random.choice((1, -1), (n, d))
    # # data = mu + std
    #
    #
    # # out_std = var_outlier * np.random.rand(n_outlier, d)
    # # outliers = mu*10 + var + out_std
    #
    # data = np.random.normal(mu, var, size=(n, d))
    # outliers = np.random.normal(2*mu, var_outlier, size=(n_outlier, d))
    #
    # data = np.concatenate((data, outliers))
    # np.random.shuffle(data)
    # return data
    A, _ = make_blobs(n_samples=[int(n*(1 - percent_outliers)), int(n*(0.5*percent_outliers)), int(n*(0.5*percent_outliers))], centers=None, n_features=d, random_state=8)
    return A

def plot_data(A: np.ndarray, median: np.ndarray, step=1, show_data=False) -> None:
    """
    TODO: implement in a cleaner way + better UI?
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
            plt.scatter(median[i][0], median[i][1], median[i][2], label=f'median {i}')
    else:
        raise ValueError("Required d<=3")
    plt.legend((['data'] if show_data else []) + [f'median {i}' for i in range(0, len(median), step)])
    plt.show()


if __name__ == "__main__":
    n, d = 10000, 2
    np.random.seed(1234)
    # A = generate_data(100, 100, 1000, (10000, 2), 0.3)
    A, _ = make_blobs(n_samples=[8000, 1000, 1000], centers=None, n_features=2, random_state=8)
    print(A.shape)
    plot_data(A, np.array([np.mean(A, axis=0)]), step=1, show_data=True)
    cmp_out = compute_geometric_median(torch.from_numpy(A), np.ones(n))
    plot_data(A, np.array([np.mean(A, axis=0), torch.Tensor.numpy(cmp_out.median)]), show_data=True)