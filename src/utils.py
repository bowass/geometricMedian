import numpy as np
from numpy import linalg as LA
from typing import Callable


def calc_f_i(a: np.ndarray, x: np.ndarray) -> float:
    return LA.norm(a - x)


def g_t_i(x: np.ndarray, a: np.ndarray, t: float) -> np.ndarray:
    """
    defined in page 6, section 2.3
    """
    return np.sqrt(1 + (t * LA.norm(x - a)) ** 2)


def g_t(x: np.ndarray, A: np.ndarray, t: float) -> np.ndarray:
    """
    defined in page 6, section 2.3
    """
    return np.sqrt(1 + (t * LA.norm(x - A, axis=1)) ** 2)


def calc_f(A: np.ndarray, x: np.ndarray) -> float:
    """
    defined in page 3, equation 1.1
    """
    # noinspection PyTypeChecker
    return np.sum(LA.norm(A - x, axis=1))


def calc_grad_ft(x: np.ndarray, A: np.ndarray, t: float) -> np.ndarray:
    """
    page 13, lemma 13
    """
    n, d = A.shape
    result = np.zeros(d)
    for i in range(n):
        result += (t*t * (x - A[i])) / (1 + g_t_i(x, A[i], t))
    return result
    # print(result)
    # tp = []
    # for i in range(n):
    #     tp.append(1 + g_t_i(x, A[i], t))
    # return np.sum((t*t*(x - A))/(1 + g_t(x, A, t)).T, axis=1)


def calc_hessian(x: np.ndarray, A: np.ndarray, t: float) -> np.ndarray:
    """
    page 13, lemma 13
    TODO: (x-A)(x-A).T shape??? something is weird
    """
    n, d = A.shape
    result = np.zeros((d, d))
    for i in range(n):
        g = g_t_i(x, A[i], t)
        result += ((t*t) / (1 + g)) * (np.identity(d) - (t*t * (x - A[i]) @ (x - A[i]).T)/(g * (1 + g)))
    return result
    # g = g_t(x, A, t)
    # prod1 = (t*t) / (1 + g)
    # prod2 = 1 - (t*t * (x-A)@(x-A).T)/(g*(1 + g))
    # return prod1 * prod2


def f_t_i_x(x: np.ndarray, a: np.ndarray, t: float):
    return g_t_i(x, a, t) - np.log(1 + g_t_i(x, a, t))


def f_t_x(x: np.ndarray, A: np.ndarray, t: float):
    """
    f_{t}(x), page 6, section 2.3
    """
    g = g_t(x, A, t)
    return np.sum(g - np.log(1 + g))


def main1():
    A =np.random.random((2, 2))
    x = np.random.random((2, ))
    t = 5.5
    print(f_t_x(x, A, t))


if __name__ == "__main__":
    main1()


def w_t(x: np.ndarray, A: np.ndarray, t: float):
    """
    defined in page 6, section 2.3
    """
    return np.sum(1/(1 + g_t(x, A, t)))


def t_i(f: float, i: float) -> float:
    """
    t_i: defined in algorithm 1, page 8
    """
    return (1 / (400 * f)) * ((1 + 1 / 600) ** (i - 1))


def matrix_norm(x: np.ndarray, A: np.ndarray):
    """
    :param x: a vector
    :param A: symmetric positive semi-definite matrix
    :return: ||x||_A; page 5, section 2.1
    """
    return np.sqrt(x.T @ A @ x)


# is there ANY usage to this?
def PowerMethod(A: np.ndarray, k: int) -> np.ndarray:
    """
    algorithm 5, page 24
    """
    x = np.random.normal(0, 1, size=(A.shape[1], 1))  # x is dX1
    y = LA.matrix_power(A, k) @ x
    return np.zeros(A.shape[1]) if LA.norm(y) == 0 else y / LA.norm(y)


def ApproxMinEig(x: np.ndarray, A: np.ndarray, t: float, eps: float) -> (float, np.ndarray):
    """
    algorithm 2, page 9
    TODO: fix code
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
    algorithm 8, page 37
        - improved by 40% !!!
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
    print("done with OneDim")
    return x
