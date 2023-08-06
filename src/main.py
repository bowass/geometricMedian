import matplotlib.pyplot as plt

from geomedian import GeometricMedian
from data_generator import *
from utils import calc_f
import torch
from geom_median.torch import compute_geometric_median


def main():
    # n, d = 100000, 100
    n, d = 10000, 2
    A = generate_data((n, d), 0.2)
    gm = GeometricMedian(A)
    gm.AccurateMedian(0.1)  # works but SLOW
    plot_data(A, np.array(gm.medians), 100, show_data=True)
    print(gm.medians)
    plt.plot(range(len(gm.medians)), [calc_f(A, med) for med in gm.medians], '-o')
    plt.xlabel("iteration")
    plt.ylabel("total cost")
    plt.show()
    # numerical weighted geometric median
    # cmp_out = compute_geometric_median(torch.from_numpy(A), np.ones(n))
    # plot_data(A, [torch.Tensor.numpy(cmp_out.median)])


if __name__ == "__main__":
    main()
