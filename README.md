# Geometric Median
Implementation of a (1+$`\varepsilon`$)-approximation of a geometric median in nearly linear time, based on [this paper](https://arxiv.org/abs/1606.05225).
The code is very slow at the moment: this algorithm converges at a very slow pace: $`1 + \frac{\log\left(\frac{3n}{\varepsilon}\right)}{\log\left(1 + \frac{1}{600}\right)}`$ iterations of the main algorithm, more than $`4000`$ iterations even for $`\log\frac{n}{\varepsilon}=0`$.

## TODO
1. speed up code (current image shows a computation of 1/1000 of the needed iterations for 1.01-approximation of 100 samples)
   1. accelerate ```minimize_local_center``` (takes a large portion of the total running time)
   2. reduce heavy, repeating computations + vectorize code
2. improve visualization (for $`d \le 3`$) + UI
3. compare with [geom-median](https://github.com/krishnap25/geom_median)
