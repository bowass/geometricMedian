import random
from geomedian import GeometricMedian
from data import *


def main():
    n, d = 100, 2
    A = generate_data(50, 100, 10, (1000, 2), 0.15)
    gm = GeometricMedian(A)
    median = gm.AccurateMedian(0.01)  # should work
    plot_data(A, gm.medians)


if __name__ == "__main__":
    main()
