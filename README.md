# Geometric Median
Implementation of a (1+$`\varepsilon`$)-approximation of a geometric median in nearly linear time, based on [this paper](https://arxiv.org/abs/1606.05225).
The code is very slow at the moment: this algorithm converges at a very slow pace: $`1 + \frac{\log\left(\frac{3n}{\varepsilon}\right)}{\log\left(1 + \frac{1}{600}\right)}`$ iterations of the main algorithm, more than $`4000`$ iterations even for $`\log\frac{n}{\varepsilon}=0`$.

## TODO
1. speed up with hyperparameter tuning
2. visualization - combine everything to a single graph (show TR ball radius, search length and direction in a single graph)
3. clean dead code
4. calculate hessian with coresets?
