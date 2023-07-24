# Geometric Median
Implementation of an (1+$`\varepsilon`$)-approximation of a geometric median in nearly linear time, based on [this paper](https://arxiv.org/abs/1606.05225).
The code is very slow at the moment (despite almost linear time complexity) and takes a long time to converge.

## TODO
1. speed up code
   1. accelerate ```minimize_local_center``` by faster root finding (```scipy``` instead of ```numpy```)
   2. reduce #calls for ```OneDimMinimizer```
2. vectorize for loops 
3. visualization (for `d <= 3`)
4. deterministic algorithm to find median for comparison
5. UI