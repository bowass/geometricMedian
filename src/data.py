import random
from geomedian import GeometricMedian
import matplotlib.pyplot as plt
import numpy as np


def generate_data(median=500, var=100, var_outlier=10, shape=(100, 2), percent_outliers=0.2) -> np.ndarray:
    n, d = shape
    n_outlier = int((n * percent_outliers))
    std = var * np.random.rand(n, d) * np.random.choice((1, -1), (n, d))
    data = median + std

    lower_std = var_outlier * np.random.rand(n_outlier, d)
    lower_outliers = median*10 + var + lower_std

    # upper_std = var_outlier * np.random.rand(n_outlier, d)
    # upper_outliers = median + var + upper_std
    data = np.concatenate((data, lower_outliers))
    np.random.shuffle(data)
    return data


def plot_data(A: np.ndarray, median: list) -> None:
    """
    plots A with the median
    requires d <= 3
    """
    n, d = A.shape
    if d == 1:
        plt.scatter(A[:, 0])
        plt.scatter(median[0], label='median')
    elif d == 2:
        plt.scatter(A[:, 0], A[:, 1])
        plt.scatter(median[0][0], median[0][1], label='median0')
        plt.scatter(median[1][0], median[1][1], label='median1')
    elif d == 3:
        plt.scatter(A[:, 0], A[:, 1], A[:, 2])
        plt.scatter(median[0], median[1], median[2], label='median')
    else:
        raise ValueError("Required d<=3")
    plt.show()


def main():
    n, d = 100, 2
    A = generate_data(100, 120, 100, (1000, 2), 0.2)
    gm = GeometricMedian(A)
    # median = gm.AccurateMedian(0.3)  # should not work yet
    plot_data(A, [np.zeros(d), np.array([500, 500])])


if __name__ == "__main__":
    main()
