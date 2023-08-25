from utils import *
import matplotlib.pyplot as plt


class GeometricMedian:
    """
    find geometric median in O(nd*log^3(n/eps))
    main function is AccurateMedian
    """

    def __init__(self, A: np.ndarray, n_iter=None):
        self.A = A
        self.n, self.d = A.shape
        self.f_star = None
        self.eps_star = None
        self.medians = None
        self.n_iter = n_iter
        self.products = np.zeros((self.d, self.d, self.n))
        self.trivial_calls_minimize_local_center = 0
        self.total_calls_minimize_local_center = 0
        for i in range(self.d):
            if i % 50 == 0:
                print(f"processing data... {i}/{self.d}")
            for j in range(i, self.d):
                self.products[i][j] = A[:, i] * A[:, j]
        # self.products = [[tmp[i] @ tmp[j] for i in range(self.d)] for j in range(self.d)]

    def LocalCenter(self, y: np.ndarray, t: float, eps: float) -> np.ndarray:
        """
        algorithm 3, page 10
        - in the paper: x as a parameter for some reason
        notes: we call minimize_local_center and calculate the same values all over again, for no reason!
        """
        # _, v = ApproxMinEig(y, self.A, t, eps, self.products)
        # v = np.reshape(v, (self.d, 1))
        # print(v.shape, y.shape)
        lmbda, v = ApproxMinEig(y, self.A, t, eps, self.products)
        prod = t * t * w_t(y, self.A, t)
        v = np.expand_dims(v, 1)
        Q = prod * np.identity(self.d) - (prod - lmbda) * v @ v.T
        # print(prod, lmbda, v@v.T)
        xt = y
        # iftach proved we can replace 64 with 32 w/o any loss
        k = int(np.ceil(32*np.log(1/eps)))
        x = cp.Variable((self.d, 1))
        # print(f"{k} calls to minimize_local_center")
        # z = cp.Parameter((self.d, 1))
        # objective = cp.Minimize(calc_grad_ft(z, self.A, t) @ x + 4 * cp.quad_form(x - z, Q))
        # constraints = [cp.norm(x - y) <= 1 / (49 * t)]
        # prob = cp.Problem(objective, constraints)

        '''
        calc_grad content:
            n, d = A.shape
            g = np.expand_dims(g_t(x, A, t), 1)
            grad = t*t * np.sum((1/(1 + g) * (x.T - A)), axis=0)
            grad @ x + 4 * cp.quad_from(x-xt, Q)
        '''
        constraints = [cp.norm(x - y) <= 1 / (49 * t)]
        # define the problem with parameters
        # warm_start=True?

        for i in range(k):
            # xt, oops = minimize_local_center(y, xt, v, alpha=1 / (49 * t))
            # if oops == 1:
            #     self.trivial_calls_minimize_local_center += 1
            #     self.total_calls_minimize_local_center += 1
            #     return xt
            # self.total_calls_minimize_local_center += 1
            # tmpi = actual_minimze_local_center(y, xt, self.A, Q, t, 1/(49*t))
            # if np.allclose(tmpi, xt):
            #     print("OOOOO")
            #     break
            # xt = tmpi
            # print(xt)
            # print("I AM HERE")
            # print("eigvals are", np.linalg.eigvals(Q) >= 0, "Q=", Q)
            grad = calc_grad_ft(xt, self.A, t)
            objective = cp.Minimize(grad @ x + 4 * cp.quad_form(x-xt, Q))
            prob = cp.Problem(objective, constraints)
            # z.value = xt
            result = prob.solve(verbose=False)
            if np.allclose(xt, x.value):
                break
            xt = x.value
            # print(f"result={result} is", x.value)
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
            # changed alpha to 2*alpha
            return f_t_x(self.LocalCenter(y + 2*alpha * u, t, epsO), self.A, t_tag)

        alpha_tag = OneDimMinimizer(l, u, epsO, oracle, t_tag * self.n)
        return self.LocalCenter(y + alpha_tag * u, t_tag, epsO)

    def AccurateMedian(self, eps: float) -> np.ndarray:
        """
        Input:
            1. target accuracy epsilon
        Output: computes (1+eps)-approximate geometric median
            in O(nd*log^3(n/eps)) time - is it less now? also, at least nd^2 preprocessing time
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
        epsO = (eps_c * self.eps_star / (160 * self.n ** 2)) ** 2 # tmp

        self.medians.append(x)
        # x = self.LineSearch(x, t_i(self.f_star, 1), t_i(self.f_star, 1), np.zeros(d), eps_c)
        x = self.LocalCenter(x, t_i(self.f_star, 1), epsO)
        self.medians.append(x)
        # max i such that t_i <= t_star (modified)
        if self.n_iter is None:
            self.n_iter = int(np.floor(1 + (np.log(n / self.eps_star) + np.log(800)) / np.log(1 + 1 / 460))) # TMP CHANGED 367 to 50
        print("AccurateMedian k is", self.n_iter)
        for i in range(1, self.n_iter + 1):
            # _, u = ApproxMinEig(x, self.A, t_i(self.f_star, i), eps_v, self.products)
            # x = self.LineSearch(x, t_i(self.f_star, i), t_i(self.f_star, i + 1), u, eps_c)
            x = self.LocalCenter(x, t_i(self.f_star, i), epsO)
            self.medians.append(x)
            if i % 100 == 0:
                print(f"iteration {i}/{self.n_iter}")
            if i % 1000 == 0:
                plt.plot(range(len(self.medians)), [calc_f(self.A, med) for med in self.medians], '-o')
                plt.show()
            # print(f"{len(self.medians)}'th median is {x}")
        return x
