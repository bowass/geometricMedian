# Geometric Median
Implementation of a (1+$`\varepsilon`$)-approximation of a geometric median in **nearly** linear time, based on [this paper](https://arxiv.org/abs/1606.05225).
There are two variations of the algorithm: `AccurateMedian` (an implementation of the original algorithm) and `AccurateMedianV2` (a faster and more practical version of the algorithm).

## Usage
```python
from geomedian import GeometricMedian
from data_generator import *

n, d = 1000, 3

# generate data as a numpy array
A = generate_data((n, d), 0.2)

# initialization + preprocessing
gm = GeometricMedian(A)

# AccurateMedianV2 is faster
gm.AccurateMedianV2(eps=0.01, verbose=0.1)

# plot the data together with the
# computed approximated medians (for d <= 3)
if d <= 3:
    plot_data(A, gm.medians, step=100, show_data=True)
```

## TODO
1. ...