from geomedian import GeometricMedian
from data_generator import *
import torch
from geom_median.torch import compute_geometric_median


def main():
    # n, d = 100000, 100
    n, d = 100000, 2
    A = generate_data(50, 100, 10, (n, d), 0.15)
    gm = GeometricMedian(A)
    gm.AccurateMedian(0.3)  # works but SLOW
    plot_data(A, gm.medians, 6)
    print(gm.medians)

    # numerical weighted geometric median
    # cmp_out = compute_geometric_median(torch.from_numpy(A), np.ones(n))
    # plot_data(A, [torch.Tensor.numpy(cmp_out.median)])


if __name__ == "__main__":
    main()
