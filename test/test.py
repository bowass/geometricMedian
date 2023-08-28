from geomedian import GeometricMedian
from data_generator import *
import time
from utils import calc_f


def test(n=1000, d=10, eps=0.1):
    data = generate_data((n, d), 0.2)
    gm = GeometricMedian(data)
    s1 = time.time()
    gm.AccurateMedianV2(eps, verbose=0.1)
    e1 = time.time()
    s2 = time.time()
    shit = torch.Tensor.numpy(compute_geometric_median(torch.from_numpy(data)).median)
    e2 = time.time()
    print(f"Ours: {e1-s1:.03f} seconds, final cost of {calc_f(data, gm.medians[-1])}")
    print(f"Against: {e2-s2:.03f} seconds, final cost of {calc_f(data, shit)}")
    return gm, shit, data


def main():
    # n, d = 100, 3
    # A = generate_data((n, d), 0.2)
    # gm = GeometricMedian(A)
    # gm.AccurateMedianV2(eps=0.01, verbose=0.1)
    # plot_data(A, np.array(gm.medians), 100, show_data=True)
    test(100, 3)


if __name__ == "__main__":
    main()
