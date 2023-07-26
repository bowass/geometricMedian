import numpy as np
from numpy import linalg as LA
from typing import Callable

# oops -- TMP
EYE = (np.eye(2))


def calc_f_i(a: np.ndarray, x: np.ndarray) -> float:
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A with shape (n, d)
    Output: f^{i} (x)
    TODO:   :: real explanation
            :: delete if found useless
    Source: section 2.3, page 6
    """
    return LA.norm(a - x.T)


def g_t_i(x: np.ndarray, a: np.ndarray, t: float) -> np.ndarray:
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A with shape (n, d)
        3. path parameter t
    Output: g_{t} (x)
    TODO:   :: real explanation
            :: can we remove it and only use g_t_i?
    Source: section 2.3, page 6
    """
    return np.sqrt(1 + (t * LA.norm(x - a)) ** 2)


def g_t(x: np.ndarray, A: np.ndarray, t: float) -> np.ndarray:
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A with shape (n, d)
        3. path parameter t
    Output: g_{t} (x), vectorized version of g_t_i
    TODO:   :: real explanation
    Source: section 2.3, page 6
    """
    return np.sqrt(1 + (t * LA.norm(x.T - A, axis=1)) ** 2)


def calc_f(A: np.ndarray, x: np.ndarray) -> float:
    """
    Input:
        1. matrix A with shape (n, d)
        2. point x with shape (d, 1)
    Output: f(x): euclidian
    TODO:   :: find out PyTypeChecker is giving this sus warning?
    Source: equation 1.1, page 3
    """
    return np.sum(LA.norm(A - x.T, axis=1))


def calc_grad_ft(x: np.ndarray, A: np.ndarray, t: float) -> np.ndarray:
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: the gradient of f_{t} (x)
    TODO:   :: add comments regarding explanation and true meaning
            :: speed up if possible (hopefully vectorize this code correctly)
            :: delete if found useless
    Source: lemma 13, page 13
    """
    n, d = A.shape
    result = np.zeros(d)
    for i in range(n):
        result += (t*t * (x - A[i])) / (1 + g_t_i(x, A[i], t))
    return result


def calc_hessian(x: np.ndarray, A: np.ndarray, t: float) -> np.ndarray:
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: the hessian of f_{t} (x)
    TODO:   :: add comments regarding explanation and true meaning
            :: speed up if possible (hopefully vectorize this code correctly)
    Source: lemma 13, page 13
    """
    n, d = A.shape
    result = np.zeros((d, d))
    g = g_t(x, A, t)
    prod1 = (t*t) / (1 + g)
    for i in range(n):
        diff = np.reshape(x.T - A[i], (d, 1))
        result += prod1[i] * (EYE - t*t*diff @ diff.T / (g[i] * (1 + g[i])))
    return result


def f_t_i_x(x: np.ndarray, a: np.ndarray, t: float):
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: f_{t}^{i} (x)
    TODO:   :: add comments regarding explanation and true meaning
            :: delete? seems useless
    Source: section 2.3, page 6
    """
    return g_t_i(x, a, t) - np.log(1 + g_t_i(x, a, t))


def f_t_x(x: np.ndarray, A: np.ndarray, t: float):
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: f_{t} (x)
    TODO:   :: add comments regarding explanation and true meaning
    Source: section 2.3, page 6
    """
    g = g_t(x, A, t)
    return np.sum(g - np.log(1 + g))


def main1():
    # A = np.random.random((2, 2))
    # x = np.random.random((2, ))
    t = 5.5
    A = np.array([[2, 4, 9], [3, 9, 2]])
    x = np.array([6, 1, 7])
    print(calc_hessian(x, A, t))


if __name__ == "__main__":
    main1()


def w_t(x: np.ndarray, A: np.ndarray, t: float):
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: the weight of x (yes? idk)
    TODO:   :: add comments regarding explanation and true meaning
    Source: section 2.3, page 6
    """
    return np.sum(1/(1 + g_t(x, A, t)))


def t_i(f: float, i: float) -> float:
    """
    Input:
        1. f
        2. i
    Output: t_i yk
    TODO:   :: add comments regarding explanation
    Source: algorithm 1, page 8
    """
    return (1 / (400 * f)) * ((1 + 1 / 600) ** (i - 1))


def matrix_norm(x: np.ndarray, A: np.ndarray):
    """
    Input:
        1. point x with shape (d, 1)
        2. symmetric positive semi-definite matrix A (all eigenvalues are non-negative)
        3. target accuracy epsilon
    Output: norm of x with respect to A
    TODO:   :: nothing much
    Source: section 2.1, page 5
    """
    return np.sqrt(x.T @ A @ x)


def PowerMethod(A: np.ndarray, k: int) -> np.ndarray:
    """
    Input:
        1. ...
    Output: maximal eigenvalue of the hessian and its corresponding eigenvector
    TODO:   :: this function has no use for now (using faster eigenvector/value computation with numpy
            :: delete when sure
    Source: algorithm 5, page 24
    """
    x = np.random.normal(0, 1, size=(A.shape[1], 1))  # x is dX1
    y = LA.matrix_power(A, k) @ x
    return np.zeros(A.shape[1]) if LA.norm(y) == 0 else y / LA.norm(y)


def ApproxMinEig(x: np.ndarray, A: np.ndarray, t: float, eps: float) -> (float, np.ndarray):
    """
    Input:
        1. point x with shape (d, 1)
        2. path parameter t (redundant! using np.eig instead of PowerMethod)
        3. target accuracy epsilon
    Output: maximal eigenvalue of the hessian and its corresponding eigenvector
    TODO:   :: optimize - do we need to calc all eigenvalues? any real impact?
            :: ...
    Source: algorithm 2, page 9
    """
    # n, d = A.shape
    # At = np.zeros((d, d))
    # for i in range(n):
    #     g = g_t_i(x, A[i], t)
    #     At += (t ** 4 * (x - A[i]) @ (x - A[i]).T) / (((1 + g) ** 2) * g)
    #
    # k = int(np.floor(np.log(A.shape[0] / eps)) + 10)
    # u = PowerMethod(At, k)
    # lmbda = u.T @ calc_hessian(x, A, t) @ u
    # return lmbda, u
    eigenvalues, eigenvectors = LA.eig(calc_hessian(x, A, t))
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues[0], eigenvectors[0]


def OneDimMinimizer(l: float, u: float, eps: float, g: Callable[[float], float], L: float) -> float:
    """
    Input:
        1. interval [l, u] and target error epsilon,
        2. evaluation oracle g : R -> R
        3. Lipschitz bound L > 0
    Output: TODO: understand.
    Source: algorithm 8, page 37
    """
    x = yl = l
    gx = g(x)
    yu = u
    limit = int(np.ceil(np.emath.logn(3 / 2, L * (u - l) / eps)))
    for i in range(limit):
        zl = (2 * yl + yu) / 3
        zu = (yl + 2 * yu) / 3
        gzl = g(zl)
        gzu = g(zu)
        if gzl <= gzu:
            yu = zu
            if gzl <= gx:
                x = zl
                gx = gzl
        else:
            yl = zl
            if gzu <= gx:
                x = zu
                gx = gzu
    print("done with OneDim, x =", x)
    return x
