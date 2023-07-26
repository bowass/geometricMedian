# Geometric Median
Implementation of an (1+$`\varepsilon`$)-approximation of a geometric median in nearly linear time, based on [this paper](https://arxiv.org/abs/1606.05225).
The code is very slow at the moment (despite almost linear time complexity) and takes a long time to converge.

## TODO
1. speed up code (current image shows a computation of 1/1000 of the needed iterations for 1.01-approximation)
   1. accelerate ```minimize_local_center```
   2. minimize norm computation ()
2. vectorize for loops
3. improve visualization (for $`d \le 3`$) + UI
4. compare with [geom-median](https://github.com/krishnap25/geom_median)
