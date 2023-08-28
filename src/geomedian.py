from utils import *
import cvxpy as cp


class GeometricMedian:

    def __init__(self, A: np.ndarray, n_iter=None):
        self.A = A
        self.n, self.d = A.shape
        self.f_star = None
        self.eps_star = None
        self.medians = None
        self.n_iter = n_iter
        self.products = np.zeros((self.d, self.d, self.n))
        for i in range(self.d):
            if i % 50 == 0:
                print(f"processing data... {i}/{self.d}")
            for j in range(i, self.d):
                self.products[i][j] = A[:, i] * A[:, j]

    def LocalCenter(self, y: np.ndarray, t: float, eps: float) -> np.ndarray:
        """
        Source: algorithm 3, page 10
        notes: might be able to speed up by defining the minimization problem a single time (instead of every iteration)
        """
        lmbda, v = ApproxMinEig(y, self.A, t, eps, self.products)
        prod = t * t * w_t(y, self.A, t)
        v = np.expand_dims(v, 1)
        Q = prod * np.identity(self.d) - (prod - lmbda) * v @ v.T
        xt = y
        k = int(np.ceil(32*np.log(1/eps)))
        x = cp.Variable((self.d, 1))

        constraints = [cp.norm(x - y) <= 1 / (49 * t)]

        for i in range(k):
            # next iteration
            grad = calc_grad_ft(xt, self.A, t)
            objective = cp.Minimize(grad @ x + 4 * cp.quad_form(x-xt, Q))
            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=False)
            if np.allclose(xt, x.value):
                return x.value
            xt = x.value
        return xt

    def LineSearch(self, y: np.ndarray, t: float, t_tag: float, u: np.ndarray, eps: float) -> np.ndarray:
        """
        Source: algorithm 4, page 10
        """
        epsO = (eps * self.eps_star / (160 * self.n ** 2)) ** 2
        l = -6 * self.f_star
        u = 6 * self.f_star

        def oracle(alpha):
            return f_t_x(self.LocalCenter(y + 2*alpha * u, t, epsO), self.A, t_tag)

        alpha_tag = OneDimMinimizer(l, u, epsO, oracle, t_tag * self.n)
        return self.LocalCenter(y + alpha_tag * u, t_tag, epsO)

    def AccurateMedian(self, eps: float, verbose: float = 0) -> np.ndarray:
        """
        Input:
            1. target accuracy epsilon
        Output: (1+eps)-approximate geometric median
        notes: theoretically around O(nd*log^3(n/eps) time. VERY SLOW IN PRACTICE
        Source: algorithm 1, page 8
        """
        if verbose > 0:
            print("Starting...")
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

        # max i such that t_i <= t_star
        if self.n_iter is None:
            self.n_iter = int(np.floor(1 + (np.log(n / self.eps_star) + np.log(800)) / np.log(1 + 1 / 460)))
        print("AccurateMedian k is", self.n_iter)
        for i in range(1, self.n_iter + 1):
            _, u = ApproxMinEig(x, self.A, t_i(self.f_star, i), eps_v, self.products)
            x = self.LineSearch(x, t_i(self.f_star, i), t_i(self.f_star, i + 1), u, eps_c)
            self.medians.append(x)
            if verbose > 0 and i % int(self.n_iter*verbose) == 0:
                print(f"Completed {i}/{self.n_iter} iterations. Current cost is {calc_f(self.A, x)}")
        return x

    def AccurateMedianV2(self, eps: float, verbose: float = 0) -> np.ndarray:
        """
        faster & more accurate median
        verbose: prints progress message every %verbose
        """
        if verbose > 0:
            print("Starting...")
        n, d = self.A.shape

        # mean is 2-approximation
        x = np.reshape(np.mean(self.A, axis=0), (d, 1))
        self.f_star = calc_f(self.A, x)
        self.eps_star = eps / 3
        self.medians = [x]

        eps_v = ((self.eps_star / 7 * n) ** 2) / 8
        eps_c = np.power(eps_v / 36, 3 / 2)
        epsO = (eps_c * self.eps_star / (160 * self.n ** 2)) ** 2

        x = self.LocalCenter(x, t_i(self.f_star, 1), epsO)
        self.medians.append(x)

        # max i such that t_i <= t_star
        if self.n_iter is None:
            self.n_iter = int(np.floor(1 + (np.log(n / self.eps_star) + np.log(800)) / np.log(1 + 1 / 50)) // 2)
        for i in range(self.n_iter):
            x = self.LocalCenter(x, t_i(self.f_star, i+1), epsO)
            self.medians.append(x)
            if verbose > 0 and i % int(self.n_iter*verbose) == 0:
                print(f"Completed {i}/{self.n_iter} iterations. Current cost is {calc_f(self.A, x)}")
        return x
