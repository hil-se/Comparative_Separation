import numpy as np
from collections import Counter
from scipy.stats import norm

p1 = 0.3
p2 = 0.6
n = 1000
r = 10000

xbar = []
for _ in range(r):
    c1 = np.random.choice([1, 0], size=n, p=[p1, 1-p1])
    c2 = np.random.choice([1, 0], size=n, p=[p2, 1-p2])
    x = c1 * c2
    xbar.append(np.mean(x))
mu = np.mean(xbar)
var = np.var(xbar)

print("mu: %f, var: %f" %(mu, var))

m = p1*p2
v = m*(1-m) / n

print("estimated mu: %f, var: %f" %(m, v))
