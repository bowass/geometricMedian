from geomedian import GeometricMedian
from data_generator import *


def main():
    n, d = 100, 2
    A = generate_data(50, 100, 10, (n, d), 0.15)
    gm = GeometricMedian(A)
    gm.AccurateMedian(0.01)  # works but SLOW
    # plot_data(A, gm.medians)


if __name__ == "__main__":
    main()
