from src.geomedian import GeometricMedian
from src.data_generator import *
import time
from src.utils import calc_f


def test(n=1000, d=10, eps=0.1):
    data = generate_data((n, d), 0.2)
    gm = GeometricMedian(data)
    s1 = time.time()
    gm.AccurateMedianV2(eps, verbose=0.1)
    e1 = time.time()
    s2 = time.time()
    weis = torch.Tensor.numpy(compute_geometric_median(torch.from_numpy(data)).median)
    e2 = time.time()
    print(f"Ours: {e1-s1:.03f} seconds, final cost of {calc_f(data, gm.medians[-1])}")
    print(f"Against: {e2-s2:.03f} seconds, final cost of {calc_f(data, weis)}")
    return gm, weis, data


if __name__ == "__main__":
    test(100, 5)
