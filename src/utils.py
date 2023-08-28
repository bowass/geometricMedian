import numpy as np
from numpy import linalg as LA
from typing import Callable
import cvxpy as cp

count_trivial_sol = 0
count_func_calls = 0


def g_t(x: np.ndarray, A: np.ndarray, t: float) -> np.ndarray:
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A with shape (n, d)
        3. path parameter t
    Output: g_{t} (x)
    Source: section 2.3, page 6
    """
    return np.sqrt(1 + (t * LA.norm(x.T - A, axis=1)) ** 2)


def calc_f(A: np.ndarray, x: np.ndarray) -> float:
    """
    Input:
        1. matrix A with shape (n, d)
        2. point x with shape (d, 1)
    Output: f(x): sum of euclidian distances
    Source: equation 1.1, page 3
    """
    return np.sum(LA.norm(A - x.T, axis=1))


def f_t_x(x: np.ndarray, A: np.ndarray, t: float):
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: f_{t} (x)
    Source: section 2.3, page 6
    """
    g = g_t(x, A, t)
    return np.sum(g - np.log(1 + g))


def t_i(f: float, i: float) -> float:
    """
    Input:
        1. f
        2. i
    Output: t_i yk
    Source: algorithm 1, page 8
    """
    # changed 1/400*f to 1/4*f
    # return (1 / (400 * f)) * ((1 + 1 / 460) ** (i - 1))
    return (1 / (0.4 * f)) * ((1 + 1 / 50) ** (i - 1))


def matrix_norm(x: np.ndarray, A: np.ndarray):
    """
    Input:
        1. point x with shape (d, 1)
        2. symmetric positive semi-definite matrix A (all eigenvalues are non-negative)
        3. target accuracy epsilon
    Output: norm of x with respect to A
    """
    return np.sqrt(x.T @ A @ x)


def ApproxMinEig(x: np.ndarray, A: np.ndarray, t: float, eps: float, products) -> (float, np.ndarray):
    """
    Input:
        1. point x with shape (d, 1)
        2. path parameter t (redundant! using np.eig instead of PowerMethod)
        3. target accuracy epsilon
        4. products - matrix with [i,j] = A[:, i] @ A[:, j]
    Output: minimal eigenvalue of the hessian and its corresponding eigenvector
    Source: algorithm 2, page 9
    """
    n, d = A.shape
    At = np.zeros((d, d))
    g = g_t(x, A, t)
    z = 1 / (g * ((1 + g) ** 2))
    zsum = np.sum(z)

    # calculate hessian
    for i in range(d):
        for j in range(d):
            At[i, j] = zsum*x[i]*x[j] - x[j] * z @ A[:, i] - x[i] * z @ A[:, j] + z @ products[min(i, j)][max(i, j)]
    # full hessian
    At = (t**2 * np.sum(1/(1+g))) * np.eye(d) - At

    # print("calculating hessian...")
    # n, d = A.shape
    # result = np.zeros((d, d))
    # g = g_t(x, A, t)
    # prod1 = (t * t) / (1 + g)
    # for i in range(n):
    #     diff = np.reshape(x.T - A[i], (d, 1))
    #     result += prod1[i] * (np.eye(d) - t * t * diff @ diff.T / (g[i] * (1 + g[i])))
    # At = result
    # print("calculated hessian!")

    # At = At * (t ** 4) / (g * (1 + g) ** 2)
    # for i in range(n):
    #     At += (t ** 4 * (x - A[i]) @ (x - A[i]).T) / (((1 + g[i]) ** 2) * g[i])

    # k = int(np.floor(np.log(A.shape[0] / eps)) + 10)
    # u = PowerMethod(At, k)
    # return 0, u
    # lmbda = u.T @ calc_hessian(x, A, t) @ u
    # return lmbda, u
    eigenvalues, eigenvectors = LA.eig(At)
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
    Output: TODO: ?????????
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
    return x


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
    g = np.expand_dims(g_t(x, A, t), 1)
    return t*t * np.sum((1/(1 + g) * (x.T - A)), axis=0)


def w_t(x: np.ndarray, A: np.ndarray, t: float):
    """
    Input:
        1. point x with shape (d, 1)
        2. matrix A
        3. path parameter t
    Output: the weight of x TODO: ????
    Source: section 2.3, page 6
    """
    return np.sum(1 / (1 + g_t(x, A, t)))


def minimize_local_center(y: np.ndarray, z: np.ndarray, v: np.ndarray, alpha: float) -> (np.ndarray, int):
    """
    lemma 32, redundent!
    """
    global count_func_calls, count_trivial_sol
    zy = z - y
    count_func_calls += 1
    sols = []
    oops = 0
    # trivial case
    print(f"z = {z}, y = {y}, alpha = {alpha}")
    if LA.norm(zy) ** 2 < alpha:
        count_trivial_sol += 1
        sols.append(z)
        oops = 1
        return z, 1

    Q = np.eye(y.shape[0]) - v @ v.T
    Qzy = Q @ zy
    t = LA.norm(v) ** 2
    c1 = LA.norm(Qzy) ** 2
    c2 = np.squeeze(v.T @ Qzy) ** 2
    coeff = np.array([alpha, -2 * alpha * t, alpha * t ** 2 - c1, -2 * c1 * t - 2 * c2, c1 * t ** 2 + c2 * t])
    etas = np.roots(coeff)
    lambdas = etas - 1
    tmpQz = Q @ z
    sols = np.concatenate(sols, [LA.inv(Q + lmbd * np.eye(y.shape[0])) @ (tmpQz + lmbd * y) for lmbd in lambdas])
    mini = -1
    best = 0
    # find min more efficient
    for x, lmbd in zip(sols, lambdas):
        cost = matrix_norm(x - z, Q) ** 2 + lmbd * LA.norm(x - y) ** 2
        if mini == -1 or mini > cost:
            mini = cost
            best = x
    return best, oops


if __name__ == "__main__":
    pass
