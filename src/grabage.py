import numpy as np
import numpy.linalg as LA
from utils import g_t

"""
this file contains dead, unused code
"""


def calc_f_i(a: np.ndarray, x: np.ndarray) -> float:
    """
    Input:
        1. point x with shape (d, 1)
        2. data sample with shape (1, d)
    Output: f^{i} (x)
    TODO:   :: DOCUMENT
            :: delete if found useless
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
    TODO:   :: DOCUMENT
            :: can we remove it and only use g_t?
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
    TODO:   :: DOCUMENT
            :: delete? seems useless
    Source: section 2.3, page 6
    """
    return g_t_i(x, a, t) - np.log(1 + g_t_i(x, a, t))


def calc_grad_ft(x: np.ndarray, A: np.ndarray, t: float) -> np.ndarray:
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: the gradient of f_{t} (x)
    TODO:   :: DOCUMENT
            :: speed up if possible (hopefully vectorize this code correctly)
            :: delete if found useless
    Source: lemma 13, page 13
    """
    n, d = A.shape
    result = np.zeros(d)
    for i in range(n):
        result += (t * t * (x - A[i])) / (1 + g_t_i(x, A[i], t))
    return result


def w_t(x: np.ndarray, A: np.ndarray, t: float):
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: the weight of x (yes? idk)
    TODO:   :: DOCUMENT
    Source: section 2.3, page 6
    """
    return np.sum(1 / (1 + g_t(x, A, t)))


def PowerMethod(A: np.ndarray, k: int) -> np.ndarray:
    """
    Input:
        1. ...
    Output: maximal eigenvalue of the hessian and its corresponding eigenvector
    TODO:   :: this function has no use for now (using faster eigenvector/value computation with numpy)
            :: delete when sure
    Source: algorithm 5, page 24
    """
    x = np.random.normal(0, 1, size=(A.shape[1], 1))  # x is dX1
    y = LA.matrix_power(A, k) @ x
    return np.zeros(A.shape[1]) if LA.norm(y) == 0 else y / LA.norm(y)
