from geomedian import GeometricMedian
from data_generator import *


def main():
    n, d = 100, 5

    # generate data as a numpy array
    A = generate_data((n, d), 0.2)

    # initialize + preprocessing
    gm = GeometricMedian(A)

    # AccurateMedianV2 is faster
    gm.AccurateMedianV2(eps=0.01, verbose=0.1)

    # plot the data together with the computed approximated medians (for d <= 3)
    if d <= 3:
        plot_data(A, np.array(gm.medians), 100, show_data=True)


if __name__ == "__main__":
    main()
