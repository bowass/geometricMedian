from utils import *


class GeometricMedian:
    """
    find geometric median in O(nd*log^3(n/eps))
    main function is AccurateMedian
    maybe make constructor with eps parameter?
    """

    def __init__(self, A: np.ndarray, n_iter=None):
        self.A = A  # nXd
        self.n, self.d = A.shape
        self.f_star = None
        self.eps_star = None
        self.medians = None
        self.n_iter = n_iter
        self.products = np.zeros((self.d, self.d, self.n))
        for i in range(self.d):
            if i % 50 == 0:
                print(f"starting {i}'th iteration!")
            for j in range(i, self.d):
                self.products[i][j] = A[:, i] * A[:, j]
        # self.products = [[tmp[i] @ tmp[j] for i in range(self.d)] for j in range(self.d)]

    def LocalCenter(self, y: np.ndarray, t: float, eps: float) -> np.ndarray:
        """
        algorithm 3, page 10
        - in the paper: x as a parameter for some reason
        """
        _, v = ApproxMinEig(y, self.A, t, eps, self.products)
        v = np.reshape(v, (self.d, 1))
        # print(v.shape, y.shape)
        # lmbda, v = ApproxMinEig(y, self.A, t, eps)
        # prod = t * t * w_t(y, self.A, t)
        # Q = prod * np.identity(self.d) - (prod - lmbda) * v @ v.T
        xt = y
        k = int(np.ceil(64 * np.log(1 / eps)))
        for i in range(k):
            xt = minimize_local_center(y, xt, v, alpha=1 / (49 * t))
        return xt

    def LineSearch(self, y: np.ndarray, t: float, t_tag: float, u: np.ndarray, eps: float) -> np.ndarray:
        """
        algorithm 4, page 10
        - t is unused for some reason!
        """
        epsO = (eps * self.eps_star / (160 * self.n ** 2)) ** 2
        l = -6 * self.f_star
        u = 6 * self.f_star

        def oracle(alpha):
            return f_t_x(self.LocalCenter(y + alpha * u, t_tag, epsO), self.A, t_tag)

        alpha_tag = OneDimMinimizer(l, u, epsO, oracle, t_tag * self.n)
        return self.LocalCenter(y + alpha_tag * u, t_tag, epsO)

    def AccurateMedian(self, eps: float) -> np.ndarray:
        """
        Input:
            1. target accuracy epsilon
        Output: computes (1+eps)-approximate geometric median
            in O(nd*log^3(n/eps)) time
        Source: algorithm 1, page 8
        """
        print("At AccurateMedian")
        n, d = self.A.shape
        x = np.mean(self.A, axis=0)
        x = np.reshape(x, (d, 1))
        self.f_star = calc_f(self.A, x)
        self.eps_star = eps / 3
        self.medians = [x]

        eps_v = ((self.eps_star / 7 * n) ** 2) / 8
        eps_c = np.power(eps_v / 36, 3 / 2)

        self.medians.append(x)
        x = self.LineSearch(x, t_i(self.f_star, 1), t_i(self.f_star, 1), np.zeros(d), eps_c)
        self.medians.append(x)
        # max i such that t_i <= t_star (// 1000 is tmp and is due to high computation time)
        if self.n_iter is None:
            self.n_iter = int(np.floor(1 + (np.log(n / self.eps_star) + np.log(800)) / np.log(1 + 1 / 600))) // 250
        print("AccurateMedian k is", self.n_iter)
        for i in range(1, self.n_iter + 1):
            _, u = ApproxMinEig(x, self.A, t_i(self.f_star, i), eps_v, self.products)
            x = self.LineSearch(x, t_i(self.f_star, i), t_i(self.f_star, i + 1), u, eps_c)
            self.medians.append(x)
            # print(f"{len(self.medians)}'th median is {x}")
        return x
