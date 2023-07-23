import numpy as np

from utils import *
from scipy.optimize import minimize

"""
    TODO:
        1. SPEEDM
        2. test + shapes
        3. comments
        4. naive solution to compare with the approximation
"""

def minimize_local_center(y: np.ndarray, z: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    """
    lemma 32, page 35
    TODO: format
    """
    # trivial case
    if LA.norm(z - y) ** 2 < alpha:
        return z
    d = y.shape[0]
    I = np.identity(d)
    Q = I - v @ v.T
    t = LA.norm(v) ** 2 # is is norm2?
    c1 = LA.norm(Q @ (z - y)) ** 2
    c2 = (v.T @ Q @ (z-y)) ** 2 # should be a scalar!
    coeff = np.array([alpha,  -2 * alpha * t, alpha * t ** 2 - c1, -2 * c1 * t - 2 * c2, c1 * t ** 2 + c2 * t])
    etas = np.roots(coeff)
    lambdas = etas - 1
    sols = []
    for lmbd in lambdas:
        sols.append(LA.inv(Q + lmbd * I) @ (Q @ z + lmbd * y))
    mini = -1
    best = 0
    for x, lmbd in zip(sols, lambdas):
        cost = matrix_norm(x - z, Q) ** 2 + lmbd * LA.norm(x - y) ** 2
        if mini == -1 or mini > cost:
            mini = cost
            best = x
    return best


class GeometricMedian:
    """
    find geometric median in O(nd*log^3(n/eps))
    main function is AccurateMedian
    """
    def __init__(self, A: np.ndarray):
        self.A = A  # nXd
        self.n, self.d = A.shape
        self.f_star = None
        self.eps_star = None
        self.medians = None

    def LocalCenter(self, y: np.ndarray, t: float, eps: float) -> np.ndarray:
        # print("At LocalCenter")
        """
        algorithm 3, page 10
        - first line in the paper has x as a parameter for some reason!
          TODO:
            - v @ v.T shape: 1X1 or dXd (here should be dXd but on hessian its weird)
            - v @ v.T shape: 1X1 or dXd (here should be dXd but on hessian its weird)
            - the bound is circular, original code is wrong (also assumes d=2)
        """
        _, v = ApproxMinEig(y, self.A, t, eps)
        # lmbda, v = ApproxMinEig(y, self.A, t, eps)
        # prod = t * t * w_t(y, self.A, t)
        # Q = prod * np.identity(self.d) - (prod - lmbda) * v @ v.T
        xt = y
        k = int(np.ceil(64 * np.log(1 / eps)))
        for i in range(1, k + 1):
            # print("im here!", i)
            xt = minimize_local_center(y, xt, v, alpha=1/(49*t))
        return xt

    def LineSearch(self, y: np.ndarray, t: float, t_tag: float, u: np.ndarray, eps: float) -> np.ndarray:
        """
        algorithm 4, page 10
        - t is unused for some reason!
        """
        # print("At LineSearch")
        epsO = (eps * self.eps_star / (160 * self.n ** 2)) ** 2
        l = -6 * self.f_star
        u = 6 * self.f_star
        oracle = lambda alpha: f_t_x(self.LocalCenter(y + alpha * u, t_tag, epsO), self.A, t_tag)
        alpha_tag = OneDimMinimizer(l, u, epsO, oracle, t_tag * self.n)
        return self.LocalCenter(y + alpha_tag*u, t_tag, epsO)

    def AccurateMedian(self, eps: float) -> np.ndarray:
        """
        algorithm 1, page 8
        :return: (1+eps)-approximate geometric median with constant probability
        works in O(nd*log^3(n/eps)) time
        """
        print("At AccurateMedian")
        n, d = self.A.shape
        x = np.mean(self.A, axis=0)
        self.f_star = calc_f(self.A, x)
        self.eps_star = eps / 3
        self.medians = [x]

        eps_v = ((self.eps_star / 7 * n) ** 2) / 8
        eps_c = np.power(eps_v / 36, 3 / 2)

        x = self.LineSearch(x, t_i(self.f_star, 1), t_i(self.f_star, 1), np.zeros(d), eps_c)
        self.medians.append(x)
        # max i such that t_i <= t_star
        k = int(np.floor(1 + (np.log(n / self.eps_star) + np.log(800))/np.log(1 + 1/600))) // 1000
        print("AccurateMedian k is", k)
        for i in range(1, k + 1):
            _, u = ApproxMinEig(x, self.A, t_i(self.f_star, i), eps_v)
            x = self.LineSearch(x, t_i(self.f_star, i), t_i(self.f_star, i + 1), u, eps_c)
            self.medians.append(x)
        return x


"""
also add functions to plot it nicely like michael cohen did
- speed up hessian && gradient
- eig ? PowerMethod
- visualization -- show all medians
- add k as a parameter
"""
