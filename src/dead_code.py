import numpy as np
import numpy.linalg as LA
from utils import g_t

"""
This file contains dead and unused code:
    - functions and parameters defined in the paper but found no use
    - naive implementations, vectorized versions in utils.py
"""


def calc_f_i(a: np.ndarray, x: np.ndarray) -> float:
    """
    Input:
        1. point x with shape (d, 1)
        2. data sample with shape (1, d)
    Output: f^{i} (x)
    Source: section 2.3, page 6
    """
    return LA.norm(a - x.T)


def g_t_i(x: np.ndarray, a: np.ndarray, t: float) -> np.ndarray:
    """
    Input:
        1. point x with shape (d, 1)
        2. data sample with shape (1, d)
        3. path parameter t
    Output: g_{t} (x)
    Source: section 2.3, page 6
    """
    return np.sqrt(1 + (t * LA.norm(x - a)) ** 2)


def f_t_i_x(x: np.ndarray, a: np.ndarray, t: float):
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: f_{t}^{i} (x)
    Source: section 2.3, page 6
    """
    return g_t_i(x, a, t) - np.log(1 + g_t_i(x, a, t))


def PowerMethod(A: np.ndarray, k: int) -> np.ndarray:
    """
    Input:
        1. A (PSD)
        2. k - exp.
    Output: minimal eigenvalue of the hessian and its corresponding eigenvector (approx.)
    Source: algorithm 5, page 24
    """
    x = np.random.normal(0, 1, size=(A.shape[1], 1))  # x is dX1
    y = LA.matrix_power(A, k) @ x
    return np.zeros(A.shape[1]) if LA.norm(y) == 0 else y / LA.norm(y)


def calc_hessian(x: np.ndarray, A: np.ndarray, t: float, products) -> np.ndarray:
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: the hessian of f_{t} (x)
    note: calculated in O(d^2) instead of O(n) explicit for-loop (d^2 << n)
    Source: lemma 13, page 13
    """
    n, d = A.shape
    At = np.zeros((d, d))
    g = g_t(x, A, t)
    z = 1 / (g * ((1 + g) ** 2))
    zsum = np.sum(z)

    for i in range(d):
        for j in range(d):
            At[i, j] = zsum*x[i]*x[j] - x[j] * z @ A[:, i] - x[i] * z @ A[:, j] + z @ products[min(i, j)][max(i, j)]

    At = (t**2 * np.sum(1/(1+g))) * np.eye(d) - At
    return At
