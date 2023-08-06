# Geometric Median
Implementation of a (1+$`\varepsilon`$)-approximation of a geometric median in nearly linear time, based on [this paper](https://arxiv.org/abs/1606.05225).
The code is very slow at the moment: this algorithm converges at a very slow pace: $`1 + \frac{\log\left(\frac{3n}{\varepsilon}\right)}{\log\left(1 + \frac{1}{600}\right)}`$ iterations of the main algorithm, more than $`4000`$ iterations even for $`\log\frac{n}{\varepsilon}=0`$.

## TODO
1. Speed Up
   1. hyperparameter tuning
2. Visualization
   1. add more parameters to plotted results
      1. direction of each search
      2. its length
      3. trust region ball
   2. zoom in (don't have to plot all data)
   3. compare with OPT
3. Testing
   1. Gaussian distribution of the data
   2. Add noise
