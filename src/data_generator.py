import matplotlib.pyplot as plt
import numpy as np


def generate_data(median=500, var=100, var_outlier=10, shape=(100, 2), percent_outliers=0.2) -> np.ndarray:
    """
    TODO: implement in a cleaner way
    """
    n, d = shape
    n_outlier = int((n * percent_outliers))
    n -= n_outlier
    std = var * np.random.rand(n, d) * np.random.choice((1, -1), (n, d))
    data = median + std

    out_std = var_outlier * np.random.rand(n_outlier, d)
    outliers = median*10 + var + out_std

    data = np.concatenate((data, outliers))
    np.random.shuffle(data)
    return data


def plot_data(A: np.ndarray, median: list, step=1) -> None:
    """
    TODO: implement in a cleaner way + better UI?
    """
    n, d = A.shape
    if d == 1:
        plt.scatter(A[:, 0], label='data')
        for i in range(0, len(median), step):
            plt.scatter(median[i][0], label=f'median {i}')
    elif d == 2:
        plt.scatter(A[:, 0], A[:, 1], label='data')
        for i in range(0, len(median), step):
            plt.scatter(median[i][0], median[i][1], label=f'median {i}')
    elif d == 3:
        plt.scatter(A[:, 0], A[:, 1], A[:, 2], label='data')
        for i in range(0, len(median), step):
            plt.scatter(median[i][0], median[i][1], median[i][2], label=f'median {i}')
    else:
        raise ValueError("Required d<=3")
    plt.legend(['data'] + [f'median {i}' for i in range(0, len(median), step)])
    plt.show()


def main():
    n, d = 100, 2
    A = generate_data(100, 120, 100, (1000, 2), 0.2)
    plot_data(A, [np.zeros(d), np.array([500, 500]), np.array([4, 200])], step=3)


if __name__ == "__main__":
    main()
